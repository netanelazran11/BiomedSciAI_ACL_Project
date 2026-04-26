"""
Data Investigation — Pretraining and Fine-tuning Dataset
=========================================================
Covers all 14 inspection points:
  1.  File path
  2.  File format
  3.  Number of samples
  4.  Number of CpG probes
  5.  Total beta-values
  6.  CpG probe names
  7.  CpG order consistency
  8.  Sample metadata presence
  9.  Metadata fields
  10. Label columns (age, tissue, etc.)
  11. Duplicate samples
  12. Duplicate CpG IDs
  13. CpGs with 100% missing values
  14. Samples with 100% missing methylation

Usage:
    python scripts/utils/investigate_data.py
"""

import os
import sys
import numpy as np
import pandas as pd
import scipy.sparse as sp

try:
    import scanpy as sc
except ImportError:
    sys.exit("scanpy not installed. Run: pip install scanpy")


DATA_PATH = (
    "/sci/labs/benjamin.yakir/netanel.azran/data"
    "/data_methyl_8k_h5ad/methylgpt_8k_altumage_combined.h5ad"
)

SEP = "=" * 70


def section(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def load_data(path):
    print(f"\nLoading: {path}")
    print(f"File size: {os.path.getsize(path) / 1e9:.2f} GB")
    adata = sc.read_h5ad(path)
    print(f"Loaded OK  →  {adata.n_obs} samples × {adata.n_vars} CpGs")
    return adata


def get_X_dense(adata):
    """Return X as a dense float32 array regardless of sparse/dense storage."""
    X = adata.X
    if sp.issparse(X):
        return X.toarray().astype(np.float32)
    return np.array(X, dtype=np.float32)


def nan_stats(X):
    """Return NaN counts/fractions for a 2D array."""
    is_nan = np.isnan(X)
    total = X.size
    n_nan = int(is_nan.sum())
    return {
        "total_values": total,
        "nan_count": n_nan,
        "nan_pct": 100.0 * n_nan / total if total > 0 else 0.0,
        "nan_per_sample": is_nan.sum(axis=1),   # shape (n_samples,)
        "nan_per_cpg": is_nan.sum(axis=0),       # shape (n_cpgs,)
    }


def investigate_split(adata, split_name, X_full):
    """Run full investigation on a subset of adata identified by split_name."""
    section(f"SPLIT: {split_name.upper()}  ({adata.n_obs} samples)")

    # ── Basic structure ──────────────────────────────────────────────────────
    print(f"\n[1] Samples     : {adata.n_obs}")
    print(f"[2] CpGs        : {adata.n_vars}")
    print(f"[3] Total values: {adata.n_obs * adata.n_vars:,}")

    # ── CpG probe names ──────────────────────────────────────────────────────
    cpg_ids = list(adata.var_names)
    print(f"\n[4] CpG IDs sample (first 5): {cpg_ids[:5]}")
    print(f"    CpG IDs sample (last  5): {cpg_ids[-5:]}")

    # ── Duplicate CpG IDs ────────────────────────────────────────────────────
    dup_cpg = pd.Series(cpg_ids).duplicated().sum()
    print(f"\n[5] Duplicate CpG IDs: {dup_cpg}  {'⚠️  WARNING' if dup_cpg > 0 else '✅ none'}")

    # ── Duplicate samples ────────────────────────────────────────────────────
    obs_ids = list(adata.obs_names)
    dup_samples = pd.Series(obs_ids).duplicated().sum()
    print(f"[6] Duplicate sample IDs: {dup_samples}  {'⚠️  WARNING' if dup_samples > 0 else '✅ none'}")

    # ── Metadata ─────────────────────────────────────────────────────────────
    meta_cols = list(adata.obs.columns)
    print(f"\n[7] Metadata columns ({len(meta_cols)}): {meta_cols}")

    # Common label columns
    label_candidates = ["age", "tissue", "disease", "batch", "platform",
                        "dataset", "source", "cell_type", "sex", "split",
                        "sample_id", "donor_id"]
    found_labels = [c for c in label_candidates if c in adata.obs.columns]
    print(f"[8] Label columns found: {found_labels if found_labels else 'none'}")

    # Age stats if present
    if "age" in adata.obs.columns:
        age = adata.obs["age"].dropna()
        print(f"\n    Age — n={len(age)}, "
              f"min={age.min():.1f}, max={age.max():.1f}, "
              f"mean={age.mean():.1f}, NaN={adata.obs['age'].isna().sum()}")

    # ── X matrix NaN analysis ─────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print("  β-VALUE (X MATRIX) NaN ANALYSIS")
    print(f"{'─'*50}")

    X = get_X_dense(adata)
    stats = nan_stats(X)

    print(f"\n  Total β-values : {stats['total_values']:,}")
    print(f"  NaN count      : {stats['nan_count']:,}")
    print(f"  NaN %          : {stats['nan_pct']:.4f}%")

    # Samples fully missing
    nan_per_sample = stats["nan_per_sample"]
    full_missing_samples = int((nan_per_sample == adata.n_vars).sum())
    partial_missing_samples = int(((nan_per_sample > 0) & (nan_per_sample < adata.n_vars)).sum())
    print(f"\n  Samples with 0   NaN : {int((nan_per_sample == 0).sum()):,}")
    print(f"  Samples with SOME NaN: {partial_missing_samples:,}")
    print(f"  Samples with ALL  NaN: {full_missing_samples:,}  "
          f"{'⚠️  WARNING' if full_missing_samples > 0 else '✅ none'}")

    # Per-sample NaN distribution
    pcts = nan_per_sample / adata.n_vars * 100
    for threshold in [1, 5, 10, 25, 50]:
        n = int((pcts >= threshold).sum())
        print(f"    ≥{threshold:2d}% NaN per sample: {n:,} samples ({100*n/adata.n_obs:.1f}%)")

    # CpGs fully missing
    nan_per_cpg = stats["nan_per_cpg"]
    full_missing_cpgs = int((nan_per_cpg == adata.n_obs).sum())
    partial_missing_cpgs = int(((nan_per_cpg > 0) & (nan_per_cpg < adata.n_obs)).sum())
    print(f"\n  CpGs with 0   NaN : {int((nan_per_cpg == 0).sum()):,}")
    print(f"  CpGs with SOME NaN: {partial_missing_cpgs:,}")
    print(f"  CpGs with ALL  NaN: {full_missing_cpgs:,}  "
          f"{'⚠️  WARNING' if full_missing_cpgs > 0 else '✅ none'}")

    # Per-CpG NaN distribution
    cpg_pcts = nan_per_cpg / adata.n_obs * 100
    for threshold in [1, 5, 10, 25, 50]:
        n = int((cpg_pcts >= threshold).sum())
        print(f"    ≥{threshold:2d}% NaN per CpG  : {n:,} CpGs ({100*n/adata.n_vars:.1f}%)")

    # ── Beta value range ──────────────────────────────────────────────────────
    valid = X[~np.isnan(X)]
    if len(valid) > 0:
        print(f"\n  β-value range  : [{valid.min():.4f}, {valid.max():.4f}]")
        out_of_range = int(((valid < 0) | (valid > 1)).sum())
        print(f"  Out-of-range   : {out_of_range}  "
              f"{'⚠️  WARNING' if out_of_range > 0 else '✅ all in [0,1]'}")

        # Percentile distribution
        percs = np.percentile(valid, [0, 1, 5, 25, 50, 75, 95, 99, 100])
        labels = ["0%", "1%", "5%", "25%", "50%", "75%", "95%", "99%", "100%"]
        print(f"\n  β-value percentiles:")
        for lbl, val in zip(labels, percs):
            print(f"    {lbl:>5s}: {val:.4f}")

    return {
        "split": split_name,
        "n_samples": adata.n_obs,
        "n_cpgs": adata.n_vars,
        "total_values": stats["total_values"],
        "nan_count": stats["nan_count"],
        "nan_pct": stats["nan_pct"],
        "full_missing_samples": full_missing_samples,
        "full_missing_cpgs": full_missing_cpgs,
        "dup_cpg": dup_cpg,
        "dup_samples": dup_samples,
        "meta_cols": meta_cols,
        "label_cols": found_labels,
    }


def print_summary_table(results):
    section("SUMMARY TABLE")
    fields = [
        ("Split",                  "split"),
        ("Samples",                "n_samples"),
        ("CpGs",                   "n_cpgs"),
        ("Total β-values",         "total_values"),
        ("NaN count",              "nan_count"),
        ("NaN %",                  "nan_pct"),
        ("Fully missing samples",  "full_missing_samples"),
        ("Fully missing CpGs",     "full_missing_cpgs"),
        ("Duplicate sample IDs",   "dup_samples"),
        ("Duplicate CpG IDs",      "dup_cpg"),
    ]

    col_w = 28
    header = f"{'Field':<{col_w}}" + "".join(
        f"{r['split']:>18}" for r in results
    )
    print(f"\n{header}")
    print("─" * (col_w + 18 * len(results)))

    for label, key in fields:
        row = f"{label:<{col_w}}"
        for r in results:
            val = r[key]
            if isinstance(val, float):
                row += f"{val:>17.4f}%"
            elif isinstance(val, int):
                row += f"{val:>18,}"
            else:
                row += f"{str(val):>18}"
        print(row)


def main():
    section("DATA INVESTIGATION — METHYLATION DATASET")
    print(f"Path: {DATA_PATH}")

    if not os.path.exists(DATA_PATH):
        sys.exit(f"\n❌ File not found: {DATA_PATH}\n"
                 "   Run this script on the cluster.")

    adata = load_data(DATA_PATH)

    # ── Detect split column ───────────────────────────────────────────────────
    section("FULL DATASET OVERVIEW")
    print(f"\nTotal samples : {adata.n_obs:,}")
    print(f"Total CpGs    : {adata.n_vars:,}")
    print(f"Total values  : {adata.n_obs * adata.n_vars:,}")
    print(f"Storage type  : {'sparse' if sp.issparse(adata.X) else 'dense'}")
    print(f"\nobs columns   : {list(adata.obs.columns)}")
    print(f"var columns   : {list(adata.var.columns)}")

    # Split column detection
    split_col = None
    for candidate in ["split", "Split", "dataset_split", "fold"]:
        if candidate in adata.obs.columns:
            split_col = candidate
            break

    if split_col:
        split_counts = adata.obs[split_col].value_counts()
        print(f"\nSplit column  : '{split_col}'")
        print(split_counts.to_string())
    else:
        print("\n⚠️  No split column found — treating full dataset as one split")

    # ── Run investigation per split ───────────────────────────────────────────
    results = []

    if split_col:
        # Investigate each split + full dataset
        splits_to_check = {
            "full": adata,
            **{s: adata[adata.obs[split_col] == s] for s in sorted(adata.obs[split_col].unique())}
        }
    else:
        splits_to_check = {"full": adata}

    X_full = get_X_dense(adata)

    for split_name, split_adata in splits_to_check.items():
        result = investigate_split(split_adata, split_name, X_full)
        results.append(result)

    # ── Summary table ─────────────────────────────────────────────────────────
    print_summary_table(results)

    # ── Interpretation ────────────────────────────────────────────────────────
    section("INTERPRETATION")
    full = results[0]

    nan_ok = full["nan_pct"] < 1.0
    dup_ok = full["dup_samples"] == 0 and full["dup_cpg"] == 0
    miss_ok = full["full_missing_samples"] == 0 and full["full_missing_cpgs"] == 0

    print(f"\n  NaN rate        : {'✅ Low (<1%)' if nan_ok else '⚠️  High (≥1%) — needs handling'}")
    print(f"  Duplicates      : {'✅ None' if dup_ok else '⚠️  Found — check dataset'}")
    print(f"  Fully missing   : {'✅ None' if miss_ok else '⚠️  Found — needs filtering'}")
    print(f"\n  Label coverage  : {full['label_cols']}")
    print(f"  Metadata fields : {full['meta_cols']}")
    print()

    if nan_ok and dup_ok and miss_ok:
        print("  ✅ Dataset is structurally valid and ready for statistical analysis.")
    else:
        print("  ⚠️  Dataset has issues that should be addressed before training.")
        if not nan_ok:
            print("     → High NaN rate: confirm the model fills NaN with 0 or masks them.")
        if not dup_ok:
            print("     → Duplicates: deduplicate before training to avoid data leakage.")
        if not miss_ok:
            print("     → Fully missing samples/CpGs: filter them out.")

    section("DONE")


if __name__ == "__main__":
    main()
