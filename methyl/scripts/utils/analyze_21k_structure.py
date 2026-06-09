#!/usr/bin/env python3
"""
analyze_21k_structure.py
=========================
Two questions before deciding how to handle the 21k dataset:

Q1. Outlier samples — are ALL 630 removed samples negative-age?
    Or are there other reasons they were removed?
    Also: are there ANY samples in the 21k with negative age that we DIDN'T remove?

Q2. Extra 1,760 CpGs — are they NaN for every sample?
    Or just some samples? What is the NaN rate per CpG?
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

_BASE = "/sci/labs/benjamin.yakir/netanel.azran"
_DATA = f"{_BASE}/data"

ALT_H5AD   = f"{_DATA}/data_methyl_21k_h5ad/altumage_21k_3way.h5ad"
LLAMA_H5AD = (f"{_DATA}/data_methyl_finetune_19k_h5ad/"
              "finetuning_19608_clean_stratified_no_outliers.h5ad")
OUTLIERS_CSV = (f"{_BASE}/repos/BMFM-RNA/methyl/"
                "dataset_fingerprint_outputs/outliers.csv")


def load_h5ad_full(path, label):
    """Load full h5ad including X matrix."""
    import scipy.sparse
    print(f"\n[{label}] loading {path}")
    try:
        import scanpy as sc
        adata = sc.read_h5ad(path)
    except Exception as e:
        print(f"  scanpy failed: {e}")
        import h5py, anndata as ad
        with h5py.File(path, "r") as f:
            X_grp = f["X"]
            if isinstance(X_grp, h5py.Dataset):
                X = X_grp[()].astype(np.float32)
            else:
                data    = X_grp["data"][()]
                indices = X_grp["indices"][()]
                indptr  = X_grp["indptr"][()]
                n_obs = len(f["obs"]["_index"])
                n_var = len(f["var"]["_index"])
                X = scipy.sparse.csr_matrix(
                    (data, indices, indptr), shape=(n_obs, n_var)
                ).toarray().astype(np.float32)

            def read_grp(grp, n):
                idx_key = "_index" if "_index" in grp else list(grp.keys())[0]
                idx = [x.decode() if isinstance(x, bytes) else str(x)
                       for x in grp[idx_key][:]]
                cols = {}
                for k in grp.keys():
                    if k == idx_key: continue
                    try:
                        v = grp[k]
                        if isinstance(v, h5py.Dataset) and v.ndim == 1 and len(v) == n:
                            raw = v[()]
                            cols[k] = [x.decode() if isinstance(x, bytes) else x for x in raw]
                        elif isinstance(v, h5py.Group) and "categories" in v:
                            cats  = [x.decode() if isinstance(x, bytes) else str(x)
                                     for x in v["categories"][()]]
                            codes = v["codes"][()]
                            cols[k] = [cats[c] if c >= 0 else None for c in codes]
                    except Exception:
                        pass
                return idx, pd.DataFrame(cols, index=idx)

            obs_idx, obs = read_grp(f["obs"], X.shape[0])
            var_idx, var = read_grp(f["var"], X.shape[1])
        adata = __import__("anndata").AnnData(X=X, obs=obs, var=var)

    if scipy.sparse.issparse(adata.X):
        adata.X = adata.X.toarray().astype(np.float32)
    adata.obs.index = adata.obs.index.astype(str)
    adata.var.index = adata.var.index.astype(str)
    print(f"  shape: {adata.n_obs:,} × {adata.n_vars:,}")
    return adata


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alt_h5ad",     default=ALT_H5AD)
    ap.add_argument("--llama_h5ad",   default=LLAMA_H5AD)
    ap.add_argument("--outliers_csv", default=OUTLIERS_CSV)
    ap.add_argument("--outdir",       default="dataset_fingerprint_outputs")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ── Load outliers CSV ────────────────────────────────────────────────────
    outliers_df = pd.read_csv(args.outliers_csv)
    outlier_ids = set(outliers_df["sample_id"].astype(str))
    print(f"\nOutliers CSV: {len(outlier_ids):,} samples")

    # ── Load AltumAge 21k (full X) ───────────────────────────────────────────
    alt = load_h5ad_full(args.alt_h5ad, "AltumAge 21k")
    alt_obs = alt.obs.copy()
    alt_obs["age_numeric"] = pd.to_numeric(alt_obs["age"], errors="coerce")

    # ── Load MethylLlama 19k var names (to identify shared vs extra CpGs) ───
    import scanpy as sc
    llama = sc.read_h5ad(args.llama_h5ad)
    llama_cpgs = set(llama.var.index.astype(str))
    alt_cpgs   = list(alt.var.index.astype(str))
    extra_cpg_mask = np.array([c not in llama_cpgs for c in alt_cpgs])
    shared_cpg_mask = ~extra_cpg_mask
    extra_cpg_idx  = np.where(extra_cpg_mask)[0]
    shared_cpg_idx = np.where(shared_cpg_mask)[0]

    print(f"\n  Shared CpGs (in both):  {shared_cpg_mask.sum():,}")
    print(f"  Extra CpGs (21k only):  {extra_cpg_mask.sum():,}")

    # ════════════════════════════════════════════════════════════════════════
    # Q1: Age distribution of outliers vs kept samples
    # ════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("Q1: OUTLIER SAMPLE AGE ANALYSIS")
    print(f"{'='*60}")

    in_outliers  = alt_obs.index.isin(outlier_ids)
    kept_obs     = alt_obs[~in_outliers]
    removed_obs  = alt_obs[in_outliers]

    kept_ages    = kept_obs["age_numeric"].dropna()
    removed_ages = removed_obs["age_numeric"].dropna()

    print(f"\nKEPT samples ({len(kept_obs):,}):")
    print(f"  age range : [{kept_ages.min():.2f}, {kept_ages.max():.2f}]")
    print(f"  negative  : {(kept_ages < 0).sum():,}")
    print(f"  > 120     : {(kept_ages > 120).sum():,}")
    print(f"  0–120     : {((kept_ages >= 0) & (kept_ages <= 120)).sum():,}")

    print(f"\nREMOVED samples ({len(removed_obs):,}):")
    print(f"  age range : [{removed_ages.min():.2f}, {removed_ages.max():.2f}]")
    print(f"  negative  : {(removed_ages < 0).sum():,}")
    print(f"  > 120     : {(removed_ages > 120).sum():,}")
    print(f"  0–120     : {((removed_ages >= 0) & (removed_ages <= 120)).sum():,}")

    # Are there any samples in 21k with negative age that are NOT in outliers?
    neg_age_all    = alt_obs[alt_obs["age_numeric"] < 0]
    neg_age_kept   = kept_obs[kept_obs["age_numeric"] < 0]
    above120_kept  = kept_obs[kept_obs["age_numeric"] > 120]

    print(f"\nSAFETY CHECK — samples in 21k with age < 0:")
    print(f"  Total in 21k          : {len(neg_age_all):,}")
    print(f"  In outliers CSV       : {neg_age_all.index.isin(outlier_ids).sum():,}")
    print(f"  NOT in outliers (kept): {len(neg_age_kept):,}")
    if len(neg_age_kept) > 0:
        print(f"  !! KEPT samples with negative age:")
        print(neg_age_kept[["age_numeric","tissue_type","dataset"]].to_string())

    print(f"\nSAFETY CHECK — samples in 21k with age > 120:")
    print(f"  In kept set: {len(above120_kept):,}")
    if len(above120_kept) > 0:
        print(above120_kept[["age_numeric","tissue_type","dataset"]].to_string())

    # Age distribution of removed samples — all reasons
    print(f"\nREMOVED samples age breakdown:")
    bins = [-10, -0.001, 0, 10, 20, 50, 100, 120, 200]
    labels = ["<0 (negative)", "0", "0–10", "10–20", "20–50", "50–100", "100–120", ">120"]
    for i in range(len(labels)):
        lo, hi = bins[i], bins[i+1]
        n = ((removed_ages > lo) & (removed_ages <= hi)).sum()
        if n > 0:
            print(f"  {labels[i]:20s}: {n:,}")

    # Tissue breakdown of removed samples
    if "tissue_type" in removed_obs.columns:
        print(f"\nREMOVED samples tissue breakdown:")
        print(removed_obs["tissue_type"].value_counts().to_string())

    # ════════════════════════════════════════════════════════════════════════
    # Q2: NaN rate of extra 1,760 CpGs
    # ════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("Q2: NaN RATE OF EXTRA 1,760 CpGs (21k only, not in 19k)")
    print(f"{'='*60}")

    X_extra  = alt.X[:, extra_cpg_idx]   # (n_samples, 1760)
    X_shared = alt.X[:, shared_cpg_idx]  # (n_samples, 19608)

    nan_rate_extra  = np.isnan(X_extra).mean(axis=0)   # per CpG NaN rate
    nan_rate_shared = np.isnan(X_shared).mean(axis=0)

    print(f"\nSHARED 19,608 CpGs NaN rate:")
    print(f"  mean  : {nan_rate_shared.mean():.4f}")
    print(f"  max   : {nan_rate_shared.max():.4f}")
    print(f"  % with any NaN : {(nan_rate_shared > 0).mean()*100:.1f}%")
    print(f"  % fully NaN    : {(nan_rate_shared == 1).mean()*100:.1f}%")

    print(f"\nEXTRA 1,760 CpGs NaN rate (21k only):")
    print(f"  mean  : {nan_rate_extra.mean():.4f}")
    print(f"  max   : {nan_rate_extra.max():.4f}")
    print(f"  min   : {nan_rate_extra.min():.4f}")
    print(f"  % with any NaN     : {(nan_rate_extra > 0).mean()*100:.1f}%")
    print(f"  % NaN > 50%        : {(nan_rate_extra > 0.5).mean()*100:.1f}%")
    print(f"  % NaN > 90%        : {(nan_rate_extra > 0.9).mean()*100:.1f}%")
    print(f"  % fully NaN (100%) : {(nan_rate_extra == 1.0).mean()*100:.1f}%")

    # Histogram of NaN rates for extra CpGs
    print(f"\n  NaN rate distribution for extra 1,760 CpGs:")
    hist_bins = [0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.001]
    labels_h  = ["0%", "0-1%", "1-10%", "10-25%", "25-50%", "50-75%", "75-90%", "90-99%", "100%"]
    counts, _ = np.histogram(nan_rate_extra, bins=hist_bins)
    for lbl, cnt in zip(labels_h[1:], counts):
        bar = "█" * int(cnt / max(counts) * 30)
        print(f"  {lbl:10s}: {cnt:4d}  {bar}")

    # Per-sample NaN rate for extra CpGs
    nan_per_sample_extra = np.isnan(X_extra).mean(axis=1)
    print(f"\n  Per-sample NaN rate across extra 1,760 CpGs:")
    print(f"  mean  : {nan_per_sample_extra.mean():.4f}")
    print(f"  % samples with ALL extra CpGs NaN : {(nan_per_sample_extra == 1.0).mean()*100:.1f}%")
    print(f"  % samples with NO  extra CpGs NaN : {(nan_per_sample_extra == 0.0).mean()*100:.1f}%")

    # Save NaN profile to CSV
    nan_profile = pd.DataFrame({
        "cpg_id":    [alt_cpgs[i] for i in extra_cpg_idx],
        "nan_rate":  nan_rate_extra,
        "all_nan":   nan_rate_extra == 1.0,
        "mostly_nan": nan_rate_extra > 0.9,
    })
    nan_path = outdir / "extra_cpg_nan_profile.csv"
    nan_profile.to_csv(nan_path, index=False)
    print(f"\n  NaN profile saved: {nan_path}")

    print(f"\n{'='*60}")
    print("CONCLUSION HINTS")
    print(f"{'='*60}")
    print(f"  Use outliers.csv IDs directly (not age filter) — exact and safe")
    fully_nan = (nan_rate_extra == 1.0).sum()
    high_nan  = (nan_rate_extra > 0.9).sum()
    print(f"  Extra CpGs fully NaN: {fully_nan:,}/{len(nan_rate_extra):,}")
    print(f"  Extra CpGs >90% NaN : {high_nan:,}/{len(nan_rate_extra):,}")
    if high_nan > len(nan_rate_extra) * 0.9:
        print("  → Extra 1,760 CpGs are mostly NaN → safest to EXCLUDE them, use 19,608 only")
    else:
        print("  → Extra CpGs have meaningful data → imputation or inclusion may be worthwhile")


if __name__ == "__main__":
    main()
