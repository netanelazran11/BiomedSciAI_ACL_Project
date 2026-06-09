#!/usr/bin/env python3
"""
compare_datasets_fingerprint.py
=================================
Rigorous cross-dataset comparison: MethylLlama (19k h5ad) vs MethylGPT (21k parquet).

Sample identity is verified via methylation FINGERPRINT matching:
  For every sample in MethylLlama valid/test, find its nearest neighbour in
  MethylGPT valid/test by cosine similarity over ALL shared CpG sites.
  A cosine similarity > 0.9999 means the samples are identical.

This is far more reliable than age-based matching (many samples share the same age).

Outputs (--outdir):
  fingerprint_report.html   — slide-style visual report
  fingerprint_summary.txt   — plain-text digest
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
_BASE  = "/sci/labs/benjamin.yakir/netanel.azran"
_DATA  = f"{_BASE}/data"
_REPOS = f"{_BASE}/repos"

LLAMA_H5AD = (
    f"{_DATA}/data_methyl_finetune_19k_h5ad/"
    "finetuning_19608_clean_stratified_no_outliers.h5ad"
)
GPT_PARQUET_DIR = f"{_REPOS}/MethylGPT-Thesis/data/finetuning_data_21k"
GPT_CPG_MAPPING = f"{GPT_PARQUET_DIR}/cpg_mapping"


# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────
def load_llama(h5ad_path: str):
    """
    Returns:
      X_by_split : dict split -> np.ndarray (n, n_cpgs) float32
      ids_by_split: dict split -> list[str]
      ages_by_split: dict split -> np.ndarray float32
      cpg_ids     : list[str]  length = n_cpgs
    """
    print(f"\n[MethylLlama] loading {h5ad_path}")
    try:
        import scanpy as sc
        adata = sc.read_h5ad(h5ad_path)
    except Exception as e:
        print(f"  scanpy failed ({e}), trying h5py…")
        import h5py, anndata as ad, scipy.sparse as sp
        with h5py.File(h5ad_path, "r") as f:
            X_grp = f["X"]
            if isinstance(X_grp, h5py.Dataset):
                X = X_grp[()].astype(np.float32)
            else:
                data    = X_grp["data"][()]
                indices = X_grp["indices"][()]
                indptr  = X_grp["indptr"][()]
                shape   = tuple(f["X"].attrs.get("shape", [f["obs/_index"].shape[0],
                                                             f["var/_index"].shape[0]]))
                X = sp.csr_matrix((data, indices, indptr), shape=shape).toarray().astype(np.float32)

            def _read_grp(grp, n):
                idx_key = "_index" if "_index" in grp else list(grp.keys())[0]
                idx = [x.decode() if isinstance(x, bytes) else str(x) for x in grp[idx_key][:]]
                cols = {}
                for k in grp.keys():
                    if k == idx_key:
                        continue
                    try:
                        v = grp[k]
                        if isinstance(v, h5py.Dataset) and v.ndim == 1 and len(v) == n:
                            raw = v[()]
                            cols[k] = np.array([x.decode() if isinstance(x, bytes) else x for x in raw])
                    except Exception:
                        pass
                return idx, pd.DataFrame(cols, index=idx)

            obs_idx, obs = _read_grp(f["obs"], X.shape[0])
            var_idx, var = _read_grp(f["var"], X.shape[1])
        adata = __import__("anndata").AnnData(X=X, obs=obs, var=var)

    print(f"  shape: {adata.n_obs:,} × {adata.n_vars:,}")

    obs = adata.obs.copy()
    obs.index = obs.index.astype(str)
    cpg_ids = list(adata.var.index.astype(str))

    # dense X
    import scipy.sparse
    X_dense = adata.X.toarray().astype(np.float32) if scipy.sparse.issparse(adata.X) else np.asarray(adata.X, dtype=np.float32)

    split_col = next((c for c in ("split", "Split", "set") if c in obs.columns), None)
    print(f"  split column: {split_col}")

    X_by_split, ids_by_split, ages_by_split = {}, {}, {}
    for sp in ("train", "valid", "test"):
        mask = (obs[split_col] == sp).values if split_col else np.ones(len(obs), dtype=bool)
        X_by_split[sp]    = X_dense[mask]
        ids_by_split[sp]  = obs.index[mask].tolist()
        ages_by_split[sp] = pd.to_numeric(obs["age"][mask], errors="coerce").values if "age" in obs.columns else np.full(mask.sum(), np.nan)
        print(f"  LL {sp}: {mask.sum():,} samples")

    return X_by_split, ids_by_split, ages_by_split, cpg_ids


def load_cpg_mapping(mapping_dir: str):
    """
    Load CpG probe IDs from the cpg_mapping/ directory.
    Returns ordered list[str] of probe names (cg…), length = n_cpgs in parquet.
    Returns None if not found.
    """
    import pyarrow.parquet as pq
    p = Path(mapping_dir)
    if not p.exists():
        print(f"  [WARN] cpg_mapping dir not found: {mapping_dir}")
        return None
    for fname in sorted(p.iterdir()):
        try:
            if fname.suffix == ".csv":
                df = pd.read_csv(fname)
            elif fname.suffix == ".parquet":
                df = pq.read_table(fname).to_pandas()
            else:
                continue
            # Find the probe ID column
            for col in ("cpg_id", "probe_id", "CpG", "name", "id", "cpg"):
                if col in df.columns:
                    ids = df[col].astype(str).tolist()
                    break
            else:
                ids = df.iloc[:, 0].astype(str).tolist()
            # Reject pure integers
            sample = ids[:20]
            if all(s.isdigit() for s in sample):
                print(f"  [WARN] cpg_mapping has integer indices — ignoring")
                return None
            print(f"  CpG mapping loaded: {len(ids):,} probes from {fname.name}")
            return ids
        except Exception as e:
            print(f"  [WARN] {fname}: {e}")
    return None


def load_gpt_split(parquet_dir: str, split: str, col_indices: list):
    """
    Load a single MethylGPT parquet split, extracting only the CpG columns
    at col_indices (list of integer positions in the 'data' array).
    Returns (X: np.ndarray, ages: np.ndarray, ids: list[str]).
    """
    import pyarrow.parquet as pq
    f = Path(parquet_dir) / f"{split}.parquet"
    if not f.exists():
        print(f"  [WARN] {f} not found")
        return None, None, None

    pf = pq.ParquetFile(f)
    schema_names = pf.schema_arrow.names
    scalar_cols  = [c for c in schema_names if c != "data"]
    n_rows = pf.metadata.num_rows
    col_indices_set = set(col_indices)
    n_shared = len(col_indices)

    print(f"  GPT {split}: {n_rows:,} rows, extracting {n_shared:,} shared CpG columns…")

    X_list    = []
    ages_list = []
    ids_list  = []

    for batch in pf.iter_batches(batch_size=512):
        # Extract data array
        data_col = batch.column("data")
        rows = data_col.to_pylist()
        mat = np.array([[r[i] for i in col_indices] for r in rows], dtype=np.float32)
        X_list.append(mat)

        # Extract scalar cols
        meta = batch.select(scalar_cols).to_pandas()
        if "age" in meta.columns:
            ages_list.append(pd.to_numeric(meta["age"], errors="coerce").values)
        if "id" in meta.columns:
            ids_list.extend(meta["id"].astype(str).tolist())
        elif "sample_id" in meta.columns:
            ids_list.extend(meta["sample_id"].astype(str).tolist())
        else:
            # row index as id
            ids_list.extend([str(i) for i in range(len(meta))])

    X    = np.concatenate(X_list, axis=0)
    ages = np.concatenate(ages_list, axis=0) if ages_list else np.full(len(X), np.nan)
    print(f"  GPT {split}: loaded X={X.shape}, ages={ages.shape}")
    return X, ages, ids_list


# ─────────────────────────────────────────────────────────────────────────────
# CpG site overlap
# ─────────────────────────────────────────────────────────────────────────────
def analyze_cpg_overlap(ll_cpg_ids: list, gpt_cpg_ids: list):
    ll_set = set(ll_cpg_ids)
    gp_set = set(gpt_cpg_ids)
    shared    = ll_set & gp_set
    only_ll   = ll_set - gp_set
    only_gpt  = gp_set - ll_set
    print(f"\n[CpG overlap]")
    print(f"  MethylLlama: {len(ll_set):,} probes")
    print(f"  MethylGPT:   {len(gp_set):,} probes")
    print(f"  Shared:      {len(shared):,}  ({100*len(shared)/max(len(ll_set),1):.1f}% of LL)")
    print(f"  Only LL:     {len(only_ll):,}")
    print(f"  Only GPT:    {len(only_gpt):,}")
    print(f"  LL ⊆ GPT:    {len(only_ll)==0}")
    return {
        "ll_n":           len(ll_set),
        "gpt_n":          len(gp_set),
        "shared_n":       len(shared),
        "only_ll":        len(only_ll),
        "only_gpt":       len(only_gpt),
        "ll_subset_gpt":  len(only_ll) == 0,
        "pct_ll":         100*len(shared)/max(len(ll_set),1),
        "pct_gpt":        100*len(shared)/max(len(gp_set),1),
        "shared_ids":     shared,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Fingerprint matching
# ─────────────────────────────────────────────────────────────────────────────
def cosine_similarity_batch(A: np.ndarray, B: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """
    Returns matrix S of shape (n_A, n_B) where S[i,j] = cosine_sim(A[i], B[j]).
    Handles NaN by replacing with 0 before normalisation.
    """
    A = np.nan_to_num(A, nan=0.0).astype(np.float32)
    B = np.nan_to_num(B, nan=0.0).astype(np.float32)

    # L2-normalise rows
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)

    n_A = A_norm.shape[0]
    S = np.empty((n_A, B_norm.shape[0]), dtype=np.float32)
    for start in range(0, n_A, batch_size):
        end = min(start + batch_size, n_A)
        S[start:end] = A_norm[start:end] @ B_norm.T
    return S


def fingerprint_match(ll_X: np.ndarray, gpt_X: np.ndarray,
                      ll_ids: list, gpt_ids: list,
                      ll_ages: np.ndarray, gpt_ages: np.ndarray,
                      split: str, threshold: float = 0.9999):
    """
    For every sample in ll_X, find its nearest neighbour in gpt_X by cosine similarity.
    Returns a summary dict.
    """
    print(f"\n  [{split}] fingerprint matching: LL={len(ll_ids):,} × GPT={len(gpt_ids):,}")
    if len(ll_ids) == 0 or len(gpt_ids) == 0:
        return {"note": "empty split"}

    S = cosine_similarity_batch(ll_X, gpt_X)

    best_sim   = S.max(axis=1)          # (n_ll,)
    best_idx   = S.argmax(axis=1)       # (n_ll,)
    best_gpt_id = [gpt_ids[i] for i in best_idx]
    best_gpt_age = gpt_ages[best_idx]

    # Count exact / near-exact matches
    exact   = (best_sim >= threshold).sum()
    near    = ((best_sim >= 0.999) & (best_sim < threshold)).sum()
    partial = ((best_sim >= 0.99)  & (best_sim < 0.999)).sum()
    poor    = (best_sim < 0.99).sum()

    print(f"  [{split}] best-match sim: min={best_sim.min():.6f}  mean={best_sim.mean():.6f}  max={best_sim.max():.6f}")
    print(f"  [{split}] ≥{threshold} (IDENTICAL): {exact:,} / {len(ll_ids):,}  ({100*exact/max(len(ll_ids),1):.1f}%)")
    print(f"  [{split}] [0.999, {threshold}):   {near:,}")
    print(f"  [{split}] [0.99, 0.999):  {partial:,}")
    print(f"  [{split}] < 0.99 (NO MATCH): {poor:,}")

    # Age consistency for matched pairs
    age_diff = np.abs(ll_ages - best_gpt_age)
    age_consistent = np.nansum(age_diff < 0.1) if not np.all(np.isnan(age_diff)) else 0

    # Histogram of best similarities
    bins   = [0.0, 0.9, 0.99, 0.999, 0.9999, 1.0001]
    labels = ["<0.9", "0.9-0.99", "0.99-0.999", "0.999-0.9999", "≥0.9999"]
    counts, _ = np.histogram(best_sim, bins=bins)

    return {
        "split":            split,
        "ll_n":             len(ll_ids),
        "gpt_n":            len(gpt_ids),
        "sim_min":          float(best_sim.min()),
        "sim_mean":         float(best_sim.mean()),
        "sim_max":          float(best_sim.max()),
        "sim_median":       float(np.median(best_sim)),
        "exact_match_n":    int(exact),
        "exact_match_pct":  float(100*exact/max(len(ll_ids),1)),
        "near_match_n":     int(near),
        "partial_match_n":  int(partial),
        "no_match_n":       int(poor),
        "sim_hist_counts":  counts.tolist(),
        "sim_hist_labels":  labels,
        "threshold":        threshold,
        "age_consistent_n": int(age_consistent),
        "verdict": (
            f"IDENTICAL SPLIT — {exact:,}/{len(ll_ids):,} samples matched exactly (cosine ≥ {threshold})"
            if exact == len(ll_ids) else
            f"PARTIAL OVERLAP — {exact:,}/{len(ll_ids):,} samples matched exactly"
            if exact > 0 else
            "NO OVERLAP — no sample with cosine ≥ 0.9999"
        ),
        # Per-sample detail (for diagnosis, first 20)
        "sample_detail": [
            {"ll_id": ll_ids[i], "gpt_id": best_gpt_id[i],
             "sim": float(best_sim[i]),
             "ll_age": float(ll_ages[i]) if not np.isnan(ll_ages[i]) else None,
             "gpt_age": float(best_gpt_age[i]) if not np.isnan(best_gpt_age[i]) else None}
            for i in range(min(20, len(ll_ids)))
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# HTML report
# ─────────────────────────────────────────────────────────────────────────────
CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', Arial, sans-serif; background: #eef0f4;
       color: #1e2535; font-size: 14px; }
.slide { width: 1280px; min-height: 720px; margin: 40px auto;
         background: #fff; border-radius: 16px; padding: 46px 56px;
         box-shadow: 0 4px 24px rgba(0,0,0,.10);
         border: 1px solid #dde1ea; }
.slide-title { font-size: 24px; font-weight: 700; margin-bottom: 28px;
               color: #1a2340; border-bottom: 2px solid #dde3f0; padding-bottom: 12px; }
.grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
.grid3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; }
.panel { background:#f7f9fc; border:1px solid #dde3ef; border-radius:10px;
         padding:18px 20px; }
.panel-title { font-size:12px; font-weight:700; text-transform:uppercase;
               letter-spacing:.8px; color:#5a6888; margin-bottom:14px; }
.stat-card { border-radius:10px; padding:18px 14px; text-align:center; }
.stat-card .s-val { font-size:30px; font-weight:800; line-height:1; margin-bottom:6px; }
.stat-card .s-label { font-size:11px; text-transform:uppercase; letter-spacing:.8px; font-weight:600; }
.stat-card .s-sub { font-size:11px; margin-top:4px; opacity:.75; }
.sc-green  { background:#edf7ed; border:1.5px solid #4a9a4a; color:#1a5a1a; }
.sc-blue   { background:#eef3ff; border:1.5px solid #7a9fe0; color:#1a3a80; }
.sc-amber  { background:#fff8ee; border:1.5px solid #cc9040; color:#6a3a00; }
.sc-red    { background:#fff0f0; border:1.5px solid #cc4040; color:#6a0a0a; }
.sc-purple { background:#f4eeff; border:1.5px solid #8a60c8; color:#3a1a70; }
.callout-ok  { background:#edf7ed; border-left:4px solid #4a9a4a; color:#1a4a1a;
               border-radius:8px; padding:12px 16px; font-size:13px; margin-top:14px; }
.callout-warn { background:#fff8ee; border-left:4px solid #cc9040; color:#5a3a00;
                border-radius:8px; padding:12px 16px; font-size:13px; margin-top:14px; }
.callout-bad  { background:#fff0f0; border-left:4px solid #cc4040; color:#5a0a0a;
                border-radius:8px; padding:12px 16px; font-size:13px; margin-top:14px; }
table { width:100%; border-collapse:collapse; font-size:13px; }
th { background:#f0f3f8; color:#4a5a78; font-size:11px; text-transform:uppercase;
     letter-spacing:.6px; padding:7px 10px; text-align:left; font-weight:700; }
td { padding:6px 10px; border-bottom:1px solid #edf0f6; }
tr:last-child td { border-bottom:none; }
.td-r { text-align:right; }
.mono { font-family:'Courier New',monospace; font-size:12px; }
.bar-row { display:flex; align-items:center; gap:8px; margin-bottom:6px; }
.bar-label { font-size:11px; color:#4a5a78; width:90px; text-align:right;
             flex-shrink:0; font-family:monospace; }
.bar-track { flex:1; height:18px; background:#eef1f8; border-radius:4px; overflow:hidden; }
.bar-fill  { height:100%; border-radius:4px; }
.bar-val   { font-size:11px; color:#6a7890; width:80px; flex-shrink:0; font-family:monospace; }
.b-green { background:#4a9a60; } .b-blue { background:#5a8de0; }
.b-amber { background:#cc8830; } .b-red   { background:#cc5040; }
"""


def _color_for_sim(s):
    if s >= 0.9999: return "#1a7a1a"
    if s >= 0.999:  return "#cc8830"
    if s >= 0.99:   return "#cc5040"
    return "#8a0a0a"


def _callout_cls(exact_pct):
    if exact_pct >= 99: return "callout-ok"
    if exact_pct >= 50: return "callout-warn"
    return "callout-bad"


def _sim_hist_bars(counts, labels, total):
    colors = ["b-red", "b-red", "b-amber", "b-amber", "b-green"]
    html = ""
    mx = max(counts) if max(counts) > 0 else 1
    for i, (c, lbl) in enumerate(zip(counts, labels)):
        pct = 100 * c / max(total, 1)
        w   = 100 * c / mx
        html += (f'<div class="bar-row">'
                 f'<div class="bar-label">{lbl}</div>'
                 f'<div class="bar-track"><div class="bar-fill {colors[i]}" style="width:{w:.1f}%"></div></div>'
                 f'<div class="bar-val">{c:,} ({pct:.1f}%)</div>'
                 f'</div>')
    return html


def build_html(cpg_result: dict, match_results: dict) -> str:
    # ── Slide 1: CpG site overlap ────────────────────────────────────────────
    cpg = cpg_result
    subset_badge = ('<span style="color:#1a5a1a;font-weight:700">✓ Full subset</span>'
                    if cpg["ll_subset_gpt"] else
                    '<span style="color:#8a0a0a;font-weight:700">✗ NOT a subset</span>')
    slide1 = f"""
<div class="slide">
  <div class="slide-title">Slide 1 — CpG Site Overlap</div>
  <div class="grid3" style="margin-bottom:22px">
    <div class="stat-card sc-blue">
      <div class="s-val">{cpg['ll_n']:,}</div>
      <div class="s-label">MethylLlama CpGs</div>
    </div>
    <div class="stat-card sc-purple">
      <div class="s-val">{cpg['gpt_n']:,}</div>
      <div class="s-label">MethylGPT CpGs</div>
    </div>
    <div class="stat-card sc-green">
      <div class="s-val">{cpg['shared_n']:,}</div>
      <div class="s-label">Shared (used for fingerprinting)</div>
      <div class="s-sub">{cpg['pct_ll']:.1f}% of LL &nbsp;·&nbsp; {cpg['pct_gpt']:.1f}% of GPT</div>
    </div>
  </div>
  <div class="panel">
    <div class="panel-title">Interpretation</div>
    <p>MethylLlama ⊆ MethylGPT: {subset_badge}</p>
    <p style="margin-top:8px">Only in MethylLlama: <strong>{cpg['only_ll']:,}</strong> &nbsp;·&nbsp;
       Only in MethylGPT: <strong>{cpg['only_gpt']:,}</strong></p>
    <p style="margin-top:8px">All <strong>{cpg['shared_n']:,}</strong> shared CpG sites were used as the
       methylation fingerprint for sample identity verification below.</p>
  </div>
</div>
"""

    # ── Slides 2-3: Fingerprint matching per split ───────────────────────────
    split_slides = ""
    for sp in ("valid", "test"):
        r = match_results.get(sp, {})
        if "note" in r and r.get("ll_n", 0) == 0:
            split_slides += f'<div class="slide"><div class="slide-title">Split: {sp} — empty</div></div>'
            continue

        hist_html = _sim_hist_bars(r["sim_hist_counts"], r["sim_hist_labels"], r["ll_n"])
        cls = _callout_cls(r["exact_match_pct"])

        sample_rows = ""
        for s in r.get("sample_detail", []):
            sim_col = _color_for_sim(s["sim"])
            match_icon = "✓" if s["sim"] >= r["threshold"] else ("~" if s["sim"] >= 0.999 else "✗")
            sample_rows += (f'<tr>'
                            f'<td class="mono">{s["ll_id"][:30]}</td>'
                            f'<td class="mono">{s["gpt_id"][:30]}</td>'
                            f'<td class="td-r" style="color:{sim_col};font-weight:700">{s["sim"]:.6f}</td>'
                            f'<td class="td-c">{match_icon}</td>'
                            f'<td class="td-r">{s["ll_age"] if s["ll_age"] is not None else "—"}</td>'
                            f'<td class="td-r">{s["gpt_age"] if s["gpt_age"] is not None else "—"}</td>'
                            f'</tr>')

        split_slides += f"""
<div class="slide">
  <div class="slide-title">Slide — {sp.capitalize()} Split: Methylation Fingerprint Matching</div>
  <div class="grid2" style="margin-bottom:20px">
    <div>
      <div class="grid2" style="gap:12px;margin-bottom:14px">
        <div class="stat-card sc-blue">
          <div class="s-val">{r['ll_n']:,}</div><div class="s-label">MethylLlama samples</div>
        </div>
        <div class="stat-card sc-purple">
          <div class="s-val">{r['gpt_n']:,}</div><div class="s-label">MethylGPT samples</div>
        </div>
      </div>
      <div class="grid2" style="gap:12px">
        <div class="stat-card sc-green">
          <div class="s-val">{r['exact_match_n']:,}</div>
          <div class="s-label">Exact matches</div>
          <div class="s-sub">cosine ≥ {r['threshold']}</div>
          <div class="s-sub">{r['exact_match_pct']:.1f}% of LL {sp}</div>
        </div>
        <div class="stat-card {'sc-green' if r['no_match_n']==0 else 'sc-red'}">
          <div class="s-val">{r['no_match_n']:,}</div>
          <div class="s-label">No match found</div>
          <div class="s-sub">cosine &lt; 0.99</div>
        </div>
      </div>
      <div class="{cls}" style="margin-top:14px">
        <strong>Verdict:</strong> {r['verdict']}
      </div>
    </div>
    <div class="panel">
      <div class="panel-title">Similarity distribution (all LL {sp} samples)</div>
      {hist_html}
      <p style="margin-top:12px;font-size:12px;color:#5a6888">
        Min={r['sim_min']:.6f} &nbsp;·&nbsp; Median={r['sim_median']:.6f}
        &nbsp;·&nbsp; Mean={r['sim_mean']:.6f} &nbsp;·&nbsp; Max={r['sim_max']:.6f}
      </p>
    </div>
  </div>
  <div class="panel">
    <div class="panel-title">First 20 samples — nearest GPT neighbour</div>
    <table>
      <tr>
        <th>MethylLlama ID</th><th>Best GPT match</th>
        <th class="td-r">Cosine sim</th><th class="td-c">Match?</th>
        <th class="td-r">LL age</th><th class="td-r">GPT age</th>
      </tr>
      {sample_rows}
    </table>
  </div>
</div>
"""

    # ── Slide 4: Summary verdict ─────────────────────────────────────────────
    v_valid = match_results.get("valid", {}).get("verdict", "N/A")
    v_test  = match_results.get("test",  {}).get("verdict", "N/A")
    pct_v   = match_results.get("valid", {}).get("exact_match_pct", 0)
    pct_t   = match_results.get("test",  {}).get("exact_match_pct", 0)
    overall_cls = _callout_cls(min(pct_v, pct_t))

    slide_summary = f"""
<div class="slide">
  <div class="slide-title">Slide 4 — Summary: Are the Evaluation Sets the Same?</div>
  <div class="grid2" style="margin-bottom:20px">
    <div class="panel">
      <div class="panel-title">Validation split</div>
      <p style="font-size:20px;font-weight:800;color:{_color_for_sim(pct_v/100)}">{pct_v:.1f}% exact</p>
      <p style="margin-top:8px;font-size:13px">{v_valid}</p>
    </div>
    <div class="panel">
      <div class="panel-title">Test split</div>
      <p style="font-size:20px;font-weight:800;color:{_color_for_sim(pct_t/100)}">{pct_t:.1f}% exact</p>
      <p style="margin-top:8px;font-size:13px">{v_test}</p>
    </div>
  </div>
  <div class="{overall_cls}">
    <strong>Comparability of MedAE scores:</strong><br>
    {'Both valid and test evaluation sets contain the same biological samples. MedAE scores are directly comparable.'
     if min(pct_v, pct_t) >= 99 else
     'Evaluation sets differ. MethylLlama and MethylGPT MedAE scores are NOT evaluated on identical samples — direct comparison must be qualified.'}
  </div>
  <div class="panel" style="margin-top:20px">
    <div class="panel-title">Method: cosine similarity fingerprint matching</div>
    <p>For each MethylLlama sample, the methylation vector at all <strong>{cpg['shared_n']:,} shared CpG sites</strong>
       was compared against every MethylGPT sample in the same split using cosine similarity.</p>
    <p style="margin-top:8px">A cosine similarity ≥ 0.9999 is treated as an identical biological sample
       (NaN values replaced with 0 before normalisation).</p>
    <p style="margin-top:8px;color:#5a6888">This is far more reliable than age-based matching,
       which fails when multiple samples share the same age value.</p>
  </div>
</div>
"""

    return f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<title>Dataset Fingerprint Comparison</title>
<style>{CSS}</style></head>
<body>
{slide1}
{split_slides}
{slide_summary}
</body></html>
"""


# ─────────────────────────────────────────────────────────────────────────────
# Text summary
# ─────────────────────────────────────────────────────────────────────────────
def build_txt(cpg_result: dict, match_results: dict) -> str:
    lines = ["=" * 70,
             "DATASET FINGERPRINT COMPARISON REPORT",
             "MethylLlama (19k h5ad)  vs  MethylGPT (21k parquet)",
             "=" * 70, ""]

    cpg = cpg_result
    lines += ["CpG SITE OVERLAP",
              "-" * 40,
              f"  MethylLlama : {cpg['ll_n']:,} probes",
              f"  MethylGPT   : {cpg['gpt_n']:,} probes",
              f"  Shared      : {cpg['shared_n']:,}  ({cpg['pct_ll']:.1f}% of LL, {cpg['pct_gpt']:.1f}% of GPT)",
              f"  LL ⊆ GPT    : {cpg['ll_subset_gpt']}",
              f"  Only in LL  : {cpg['only_ll']:,}",
              f"  Only in GPT : {cpg['only_gpt']:,}", ""]

    for sp in ("valid", "test"):
        r = match_results.get(sp, {})
        if not r:
            lines += [f"{sp.upper()} SPLIT — not found", ""]
            continue
        lines += [f"{sp.upper()} SPLIT — FINGERPRINT MATCH",
                  "-" * 40,
                  f"  MethylLlama samples : {r.get('ll_n', '?'):,}",
                  f"  MethylGPT samples   : {r.get('gpt_n', '?'):,}",
                  f"  Exact matches (≥{r.get('threshold',0.9999)}) : {r.get('exact_match_n','?'):,}  ({r.get('exact_match_pct',0):.1f}%)",
                  f"  Near matches (≥0.999)  : {r.get('near_match_n','?'):,}",
                  f"  Partial (≥0.99)        : {r.get('partial_match_n','?'):,}",
                  f"  No match (<0.99)       : {r.get('no_match_n','?'):,}",
                  f"  Sim min/median/mean/max: {r.get('sim_min',0):.6f} / {r.get('sim_median',0):.6f} / {r.get('sim_mean',0):.6f} / {r.get('sim_max',0):.6f}",
                  f"  VERDICT: {r.get('verdict','?')}", ""]

    lines += ["=" * 70]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama_h5ad",    default=LLAMA_H5AD)
    ap.add_argument("--gpt_parquet",   default=GPT_PARQUET_DIR)
    ap.add_argument("--gpt_cpg_map",   default=GPT_CPG_MAPPING)
    ap.add_argument("--outdir",        default="dataset_fingerprint_outputs")
    ap.add_argument("--threshold",     type=float, default=0.9999,
                    help="Cosine similarity threshold for 'exact' match")
    ap.add_argument("--splits",        default="valid,test",
                    help="Comma-separated splits to compare (default: valid,test)")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    splits_to_run = [s.strip() for s in args.splits.split(",")]

    # ── 1. Load MethylLlama ────────────────────────────────────────────────
    ll_X_by_split, ll_ids_by_split, ll_ages_by_split, ll_cpg_ids = load_llama(args.llama_h5ad)

    # ── 2. Load GPT CpG mapping ───────────────────────────────────────────
    gpt_cpg_ids = load_cpg_mapping(args.gpt_cpg_map)
    if gpt_cpg_ids is None:
        print("\n[FATAL] Cannot load GPT CpG IDs from cpg_mapping/. Aborting.")
        sys.exit(1)

    # ── 3. CpG overlap + shared column indices ────────────────────────────
    cpg_result = analyze_cpg_overlap(ll_cpg_ids, gpt_cpg_ids)
    shared_ids = cpg_result["shared_ids"]

    # Index maps: position of each shared CpG in LL and GPT arrays
    ll_cpg_pos  = {cid: i for i, cid in enumerate(ll_cpg_ids)}
    gpt_cpg_pos = {cid: i for i, cid in enumerate(gpt_cpg_ids)}

    shared_sorted = sorted(shared_ids)
    ll_cols  = np.array([ll_cpg_pos[c]  for c in shared_sorted], dtype=np.int32)
    gpt_cols = [gpt_cpg_pos[c] for c in shared_sorted]

    print(f"\n  Using {len(shared_sorted):,} shared CpGs for fingerprinting")

    # ── 4. Fingerprint match per split ────────────────────────────────────
    match_results = {}
    for sp in splits_to_run:
        ll_X    = ll_X_by_split.get(sp)
        ll_ids  = ll_ids_by_split.get(sp, [])
        ll_ages = ll_ages_by_split.get(sp, np.array([]))

        if ll_X is None or len(ll_ids) == 0:
            print(f"\n  [{sp}] MethylLlama split not found — skipping")
            match_results[sp] = {"note": "not found", "ll_n": 0, "gpt_n": 0}
            continue

        # Subset LL X to shared columns
        ll_X_shared = ll_X[:, ll_cols]

        # Load GPT split (only shared columns)
        gpt_X, gpt_ages, gpt_ids = load_gpt_split(args.gpt_parquet, sp, gpt_cols)
        if gpt_X is None:
            match_results[sp] = {"note": "GPT split missing", "ll_n": len(ll_ids), "gpt_n": 0}
            continue

        match_results[sp] = fingerprint_match(
            ll_X_shared, gpt_X,
            ll_ids, gpt_ids,
            ll_ages, gpt_ages,
            split=sp, threshold=args.threshold,
        )

    # ── 5. Write outputs ──────────────────────────────────────────────────
    html = build_html(cpg_result, match_results)
    txt  = build_txt(cpg_result, match_results)

    html_path = outdir / "fingerprint_report.html"
    txt_path  = outdir / "fingerprint_summary.txt"
    html_path.write_text(html)
    txt_path.write_text(txt)

    print(f"\n{'='*60}")
    print(f"Outputs written to {outdir}/")
    print(f"  {html_path.name}")
    print(f"  {txt_path.name}")
    print(f"{'='*60}")
    print(txt)


if __name__ == "__main__":
    main()
