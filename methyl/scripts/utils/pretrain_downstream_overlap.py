#!/usr/bin/env python3
"""
pretrain_downstream_overlap.py
================================
Which downstream (AltuMAge) profiles are inside the pretraining corpus, and in
which pretraining split (train / valid / held-out)?  READ-ONLY, CPU only.

Two methods, with different standing
  A. by accession (PRIMARY). A GEO accession identifies one sample. If the
     pretraining h5ad carries sample ids -- as the obs index or in any obs
     column -- overlap is the exact intersection of accessions. Nothing else
     is needed and nothing is inferred.
  B. identical profile (SECONDARY, lower bound only). Similar samples are not
     the same sample: two blood profiles from different people correlate above
     0.98. So similarity is never used as evidence. A pair counts only if the
     two profiles are numerically IDENTICAL on every shared CpG -- the rule this
     project already uses for duplicates (cosine >= 0.9999; the same samples in
     the 19k and 21k files scored 0.99995-1.00004) plus mean |difference| <= 0.005.
     A profile that fails this rule is reported as "undetermined", not as
     "absent": different preprocessing of the same sample would also fail it.

Pretraining split membership is recomputed exactly as the training code does
(bmfm_methylation/shared/data_module.py, auto-split): rng(42).permutation(n),
first 80% train, next 10% valid, last 10% held-out.

Outputs (--outdir):
  structure.json            what the pretraining h5ad contains (keys, shapes, ids or not)
  overlap_per_sample.csv    one row per AltuMAge profile: best match, scores, split
  overlap_summary.json      counts by AltuMAge split, benchmark test set, study, threshold

Usage (cluster):
  python scripts/utils/pretrain_downstream_overlap.py \
      --pretrain /sci/.../methylgpt_pretrain_type3.h5ad \
      --downstream /sci/.../altumage_21k_3way.h5ad \
      --test_ids outputs/kfold_splits/test_ids.npy \
      --outdir outputs/pretrain_overlap
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrain", required=True)
    p.add_argument("--downstream", required=True)
    p.add_argument("--test_ids", default="outputs/kfold_splits/test_ids.npy")
    p.add_argument("--outdir", default="outputs/pretrain_overlap")
    p.add_argument("--k_cpgs", type=int, default=2000, help="CpGs used for the first-pass search")
    p.add_argument("--chunk", type=int, default=512)
    p.add_argument("--cos_identical", type=float, default=0.9999,
                   help="cosine on ALL shared CpGs required to call two profiles identical "
                        "(the project's duplicate rule, analyze_full_dataset.py)")
    p.add_argument("--mad_identical", type=float, default=0.005,
                   help="maximum mean absolute beta difference for identical profiles")
    return p.parse_args()


def describe_h5ad(path):
    """Keys, shapes and a peek at obs -- so the origin of the matrix is on record."""
    out = {"path": path}
    with h5py.File(path, "r") as f:
        out["top_level"] = list(f.keys())
        x = f["X"]
        out["X"] = ({"kind": "dense", "shape": list(x.shape), "dtype": str(x.dtype)}
                    if isinstance(x, h5py.Dataset) else {"kind": "group", "keys": list(x.keys())})
        for grp in ("obs", "var", "uns"):
            if grp in f:
                g = f[grp]
                out[grp] = {}
                for k in g.keys():
                    v = g[k]
                    if isinstance(v, h5py.Dataset):
                        peek = [x.decode() if isinstance(x, bytes) else (x.item() if hasattr(x, "item") else str(x))
                                for x in v[:3]] if v.shape and v.shape[0] else []
                        out[grp][k] = {"shape": list(v.shape), "dtype": str(v.dtype), "first3": peek}
                    else:
                        out[grp][k] = {"group_keys": list(v.keys())}
    return out


def main():
    a = parse_args()
    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)

    print("[0/6] Structure of the pretraining h5ad", flush=True)
    structure = describe_h5ad(a.pretrain)
    (outdir / "structure.json").write_text(json.dumps(structure, indent=2, default=str))
    print(json.dumps({k: structure[k] for k in ("top_level", "X")}, default=str), flush=True)
    print("      obs keys:", list(structure.get("obs", {}).keys()), flush=True)

    import scanpy as sc
    from bmfm_methylation.shared.data_module import _read_h5ad_robust

    print("[1/6] Loading pretraining matrix (about 33 GB)", flush=True)
    pre = _read_h5ad_robust(a.pretrain)
    P = pre.X.toarray() if hasattr(pre.X, "toarray") else np.asarray(pre.X)
    pre_cpgs = list(pre.var["cpg_id"]) if "cpg_id" in pre.var.columns else list(pre.var_names)
    n_pre = P.shape[0]
    # any non-numeric index is an identifier (the corpus holds GSM, TCGA and cohort ids such as TWPID...)
    looks_like_ids = lambda v: bool(np.mean([not str(x).isdigit() for x in v[:1000]]) > 0.5)
    pre_ids, id_source = np.asarray(pre.obs_names).astype(str), "obs index"
    if not looks_like_ids(pre_ids):
        for col in pre.obs.columns:
            vals = np.asarray(pre.obs[col]).astype(str)
            if looks_like_ids(vals):
                pre_ids, id_source = vals, f"obs['{col}']"
                break
    ids_look_real = looks_like_ids(pre_ids)
    print(f"      pretraining: {n_pre:,} x {P.shape[1]:,} | sample ids present: {ids_look_real} "
          f"(source: {id_source if ids_look_real else 'none found'}; first: {pre_ids[:3].tolist()})", flush=True)

    # split membership, exactly as the training DataModule defines it
    perm = np.random.default_rng(42).permutation(n_pre)
    tr_end, va_end = int(0.8 * n_pre), int(0.9 * n_pre)
    split_of_row = np.empty(n_pre, dtype=object)
    split_of_row[perm[:tr_end]] = "train"
    split_of_row[perm[tr_end:va_end]] = "valid"
    split_of_row[perm[va_end:]] = "heldout"
    print(f"      pretraining split sizes: train {tr_end:,} / valid {va_end - tr_end:,} / held-out {n_pre - va_end:,}", flush=True)

    print("[2/6] Loading downstream (AltuMAge) matrix", flush=True)
    dn = sc.read_h5ad(a.downstream)
    A = dn.X.toarray() if hasattr(dn.X, "toarray") else np.asarray(dn.X)
    dn_cpgs = list(dn.var_names)
    dn_ids = np.asarray(dn.obs_names).astype(str)
    print(f"      downstream: {A.shape[0]:,} x {A.shape[1]:,}", flush=True)

    print("[3/6] Shared CpGs and first-pass panel", flush=True)
    pre_col = {c: i for i, c in enumerate(pre_cpgs)}
    shared = [(j, pre_col[c]) for j, c in enumerate(dn_cpgs) if c in pre_col]
    dn_sh = np.array([s[0] for s in shared]); pre_sh = np.array([s[1] for s in shared])
    print(f"      shared CpGs: {len(shared):,} of {len(dn_cpgs):,} downstream / {len(pre_cpgs):,} pretraining", flush=True)

    samp = np.random.default_rng(0).choice(n_pre, size=min(5000, n_pre), replace=False)
    miss_pre = np.isnan(P[np.sort(samp)][:, pre_sh]).mean(0)
    miss_dn = np.isnan(A[:, dn_sh]).mean(0)
    var_dn = np.nanvar(A[:, dn_sh], axis=0)
    ok = (miss_pre < 0.01) & (miss_dn < 0.01)
    order = np.argsort(-np.where(ok, var_dn, -1.0))[: a.k_cpgs]
    k_dn, k_pre = dn_sh[order], pre_sh[order]
    print(f"      first-pass panel: {len(order):,} CpGs with <1% missing in both, highest downstream variance", flush=True)

    Pk = P[:, k_pre].astype(np.float32)
    mu = np.nanmean(Pk, axis=0)
    Pk = np.where(np.isnan(Pk), mu, Pk) - mu
    Pk /= (np.linalg.norm(Pk, axis=1, keepdims=True) + 1e-9)
    Ak = A[:, k_dn].astype(np.float32)
    Ak = np.where(np.isnan(Ak), mu, Ak) - mu
    Ak /= (np.linalg.norm(Ak, axis=1, keepdims=True) + 1e-9)

    print("[4/6] Nearest pretraining profile for every downstream profile", flush=True)
    best = np.empty(len(Ak), dtype=np.int64); best_cos = np.empty(len(Ak), dtype=np.float32)
    second_cos = np.empty(len(Ak), dtype=np.float32)
    for s in range(0, len(Ak), a.chunk):
        S = Ak[s:s + a.chunk] @ Pk.T
        top2 = np.argpartition(-S, 1, axis=1)[:, :2]
        v = np.take_along_axis(S, top2, axis=1)
        o = np.argsort(-v, axis=1)
        top2 = np.take_along_axis(top2, o, axis=1); v = np.take_along_axis(v, o, axis=1)
        best[s:s + a.chunk], best_cos[s:s + a.chunk], second_cos[s:s + a.chunk] = top2[:, 0], v[:, 0], v[:, 1]
        if (s // a.chunk) % 5 == 0:
            print(f"      {min(s + a.chunk, len(Ak)):,}/{len(Ak):,}", flush=True)

    print("[5/6] Verifying each best pair on ALL shared CpGs", flush=True)
    r_full = np.full(len(A), np.nan, dtype=np.float32); mad = np.full(len(A), np.nan, dtype=np.float32)
    cos_full = np.full(len(A), np.nan, dtype=np.float32)
    n_cmp = np.zeros(len(A), dtype=np.int64)
    for i in range(len(A)):
        x = A[i, dn_sh].astype(np.float64); y = P[best[i], pre_sh].astype(np.float64)
        m = np.isfinite(x) & np.isfinite(y)
        n_cmp[i] = int(m.sum())
        if n_cmp[i] > 100:
            r_full[i] = np.corrcoef(x[m], y[m])[0, 1]
            mad[i] = np.abs(x[m] - y[m]).mean()
            cos_full[i] = float(x[m] @ y[m] / (np.linalg.norm(x[m]) * np.linalg.norm(y[m]) + 1e-12))

    # identity, not similarity: numerically the same profile on every shared CpG
    is_match = (cos_full >= a.cos_identical) & (mad <= a.mad_identical)
    obs = dn.obs
    df = pd.DataFrame({
        "sample_id": dn_ids,
        "downstream_split": obs["split"].astype(str).values if "split" in obs.columns else "",
        "dataset": obs["dataset"].astype(str).values if "dataset" in obs.columns else "",
        "tissue_type": obs["tissue_type"].astype(str).values if "tissue_type" in obs.columns else "",
        "best_pretrain_row": best, "best_pretrain_id": pre_ids[best] if ids_look_real else "",
        "first_pass_cos": best_cos, "second_cos": second_cos, "margin": best_cos - second_cos,
        "cos_all_shared": cos_full, "r_all_shared": r_full, "mean_abs_diff": mad, "n_cpgs_compared": n_cmp,
        "fingerprint_match": is_match,
        "pretrain_split_of_match": np.where(is_match, split_of_row[best], ""),
    })
    if ids_look_real:
        pre_id_set = set(pre_ids.tolist())
        df["id_match"] = df["sample_id"].isin(pre_id_set)
        id_row = {s: i for i, s in enumerate(pre_ids)}
        df["pretrain_split_of_id"] = [split_of_row[id_row[s]] if s in id_row else "" for s in df["sample_id"]]
    test_ids = set(np.load(a.test_ids, allow_pickle=True).astype(str).tolist()) if Path(a.test_ids).exists() else set()
    df["in_benchmark_test_set"] = df["sample_id"].isin(test_ids)
    df.to_csv(outdir / "overlap_per_sample.csv", index=False)

    print("[6/6] Summary", flush=True)
    def tab(mask):
        sub = df[mask]
        m = sub[sub.fingerprint_match]
        return {"n": int(len(sub)), "matched": int(len(m)),
                "matched_pct": round(100 * len(m) / max(len(sub), 1), 1),
                "match_in_pretrain_split": m["pretrain_split_of_match"].value_counts().to_dict()}
    summary = {
        "pretraining_rows": int(n_pre), "downstream_rows": int(len(A)), "shared_cpgs": int(len(shared)),
        "pretraining_ids_present": ids_look_real,
        "primary_method": "accession intersection" if ids_look_real else "NOT AVAILABLE: no sample ids in the pretraining h5ad",
        "identical_profile_rule": f"cosine on all shared CpGs >= {a.cos_identical} and mean |diff| <= {a.mad_identical} "
                                  "(a lower bound on overlap; failing it means undetermined, not absent)",
        "nearest_profile_is_not_evidence": {f"nearest has cosine>={t}": int((cos_full >= t).sum()) for t in (0.9999, 0.999, 0.995, 0.99)},
        "score_quantiles": {"r_all_shared": {q: float(np.nanquantile(r_full, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
                             "margin": {q: float(np.nanquantile(best_cos - second_cos, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
                             "mean_abs_diff_of_matches": float(np.nanmedian(mad[is_match])) if is_match.any() else None},
        "all_downstream": tab(np.ones(len(df), bool)),
        "by_downstream_split": {s: tab((df.downstream_split == s).values) for s in sorted(df.downstream_split.unique())},
        "benchmark_test_set": tab(df.in_benchmark_test_set.values),
        "by_study": {s: tab((df.dataset == s).values) for s in df.dataset.value_counts().index[:40]},
    }
    if ids_look_real:
        summary["id_vs_fingerprint"] = pd.crosstab(df.id_match, df.fingerprint_match).to_dict()
    (outdir / "overlap_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    if ids_look_real:
        def idtab(mask):
            sub = df[mask]; m = sub[sub.id_match]
            return {"n": int(len(sub)), "in_pretraining": int(len(m)), "pct": round(100 * len(m) / max(len(sub), 1), 1),
                    "pretrain_split": m["pretrain_split_of_id"].value_counts().to_dict()}
        summary["BY_ACCESSION"] = {"all_downstream": idtab(np.ones(len(df), bool)),
                                   "by_downstream_split": {s: idtab((df.downstream_split == s).values) for s in sorted(df.downstream_split.unique())},
                                   "benchmark_test_set": idtab(df.in_benchmark_test_set.values)}
        (outdir / "overlap_summary.json").write_text(json.dumps(summary, indent=2, default=str))
        print("BY ACCESSION (primary):", json.dumps(summary["BY_ACCESSION"], indent=2, default=str))
    print(json.dumps({k: summary[k] for k in ("primary_method", "identical_profile_rule", "nearest_profile_is_not_evidence", "all_downstream",
                                                "by_downstream_split", "benchmark_test_set")}, indent=2, default=str))
    print(f"Saved -> {outdir}/", flush=True)


if __name__ == "__main__":
    main()
