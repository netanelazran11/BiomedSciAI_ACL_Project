#!/usr/bin/env python3
"""
full_data_nan_analysis.py
=========================
Load the full MethylGPT data matrix (11,453 × 49,156) and analyze:
  1. NaN fraction per CpG position, per split (valid / test / train)
  2. Which positions correspond to MethylLlama's 19,608 CpGs
  3. How many MethylGPT samples have zero NaN in those 19,608 positions
     (i.e., could have been in MethylLlama's dataset)
  4. Age-based overlap: for zero-NaN samples in MethylGPT valid/test,
     how many ages match MethylLlama's valid/test ages?

Memory: valid ~268 MB, test ~909 MB, train ~1.07 GB. Total ~2.25 GB.
"""

import argparse
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Cluster paths
# ─────────────────────────────────────────────────────────────────────────────
_BASE = "/sci/labs/benjamin.yakir/netanel.azran"
LLAMA_H5AD    = f"{_BASE}/data/data_methyl_finetune_19k_h5ad/finetuning_19608_clean_stratified_no_outliers.h5ad"
GPT_H5AD      = f"{_BASE}/data/data_methyl_finetune_49k_h5ad/finetuning_49k.h5ad"
GPT_PARQUET   = f"{_BASE}/repos/MethylGPT-Thesis/data/finetuning_data_49k"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_var_names_h5ad(path: str) -> list:
    """Load only var (CpG) names from h5ad without reading X."""
    import h5py
    with h5py.File(path, "r") as f:
        var_grp = f["var"]
        idx_key = "_index" if "_index" in var_grp else list(var_grp.keys())[0]
        names = [x.decode() if isinstance(x, bytes) else str(x)
                 for x in f["var"][idx_key][:]]
    return names


def load_parquet_split_full(parquet_dir: str, split: str):
    """
    Load full data matrix for one split from parquet.
    Returns (data: np.ndarray shape (N, 49156), ages: np.ndarray, ids: list)
    """
    import pyarrow.parquet as pq
    f = Path(parquet_dir) / f"{split}.parquet"
    print(f"  Loading {split}.parquet ...", flush=True)

    # Load scalar columns first
    tbl_meta = pq.read_table(f, columns=["id", "age"])
    ids  = tbl_meta["id"].to_pylist()
    ages = np.array(tbl_meta["age"].to_pylist(), dtype=np.float32)

    # Load data column
    tbl_data = pq.read_table(f, columns=["data"])
    rows = tbl_data["data"].to_pylist()
    data = np.array(rows, dtype=np.float32)  # (N, 49156)
    print(f"    shape: {data.shape}  NaN total: {np.isnan(data).sum():,}", flush=True)
    return data, ages, ids


def load_llama_split_info(h5ad_path: str):
    """Load MethylLlama obs metadata (split, age, id) without loading X."""
    import h5py
    result = {}
    with h5py.File(h5ad_path, "r") as f:
        obs = f["obs"]
        idx_key = obs.attrs.get("_index", "_index")
        if idx_key not in obs:
            idx_key = [k for k in obs.keys() if not k.startswith("__")][0]
        sample_ids = [x.decode() if isinstance(x, bytes) else str(x)
                      for x in obs[idx_key][:]]

        # Read age
        ages_raw = obs["age"][()]
        ages = np.array(ages_raw, dtype=np.float32)

        # Read split column
        split_col = None
        for c in ("split", "Split", "set"):
            if c in obs:
                split_col = c
                break
        if split_col is None:
            print("  [WARN] No split column found in MethylLlama h5ad obs")
            return {}

        grp = obs[split_col]
        if "categories" in grp:
            cats  = [x.decode() if isinstance(x, bytes) else str(x)
                     for x in grp["categories"][()]]
            codes = grp["codes"][()]
            splits = np.array([cats[c] if c >= 0 else "" for c in codes])
        else:
            splits = np.array([x.decode() if isinstance(x, bytes) else str(x)
                               for x in grp[()]])

    for sp in ("train", "valid", "test"):
        mask = splits == sp
        result[sp] = {
            "ids":      [sample_ids[i] for i in np.where(mask)[0]],
            "ages":     np.round(ages[mask], 4),
            "n":        int(mask.sum()),
        }
    return result


def report_split(split: str, data: np.ndarray, ages: np.ndarray, ids: list,
                 llama_mask: np.ndarray, llama_split_info: dict,
                 gpt_cpg_names: list, llama_cpg_set: set):
    """Full analysis for one split."""
    N, C = data.shape
    is_nan = np.isnan(data)

    print(f"\n{'='*60}")
    print(f"  Split: {split.upper()}   ({N:,} samples × {C:,} CpGs)")
    print(f"{'='*60}")

    # ── 1. NaN overview ───────────────────────────────────────────────────────
    nan_per_cpg    = is_nan.mean(axis=0)          # (C,)  NaN fraction per CpG
    nan_per_sample = is_nan.mean(axis=1)          # (N,)  NaN fraction per sample

    nan_in_llama  = nan_per_cpg[llama_mask].mean()
    nan_out_llama = nan_per_cpg[~llama_mask].mean() if (~llama_mask).any() else float("nan")
    nan_overall   = nan_per_cpg.mean()

    print(f"\n  NaN fraction:")
    print(f"    Overall                   : {nan_overall:.4f} ({nan_overall:.1%})")
    print(f"    In MethylLlama positions  : {nan_in_llama:.4f} ({nan_in_llama:.1%})")
    print(f"    Outside MethylLlama pos.  : {nan_out_llama:.4f} ({nan_out_llama:.1%})")

    # ── 2. Samples with zero NaN in MethylLlama positions (usable samples) ───
    llama_nan_per_sample = is_nan[:, llama_mask].sum(axis=1)  # (N,) NaN count in llama cols
    zero_nan_mask = llama_nan_per_sample == 0
    n_zero_nan    = int(zero_nan_mask.sum())
    pct_zero_nan  = 100 * n_zero_nan / max(N, 1)

    print(f"\n  Samples with 0 NaN in MethylLlama's {llama_mask.sum():,} CpG positions:")
    print(f"    Count : {n_zero_nan:,} / {N:,} ({pct_zero_nan:.1f}%)")
    print(f"    These samples COULD have been in MethylLlama's dataset")

    # ── 3. NaN distribution across samples ────────────────────────────────────
    bins = [0, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1.01]
    labels = ["0%", "<1%", "<10%", "<30%", "<50%", "<70%", "<90%", "≤100%"]
    hist, _ = np.histogram(nan_per_sample, bins=bins)
    print(f"\n  NaN fraction distribution across {N:,} samples:")
    for lbl, cnt in zip(labels, hist):
        bar = "█" * int(40 * cnt / max(N, 1))
        print(f"    {lbl:6s} : {cnt:5,} ({100*cnt/max(N,1):5.1f}%)  {bar}")

    # ── 4. NaN distribution across CpG positions ──────────────────────────────
    n_always_measured = int((nan_per_cpg < 0.01).sum())
    n_low_nan         = int((nan_per_cpg < 0.05).sum())
    n_high_nan        = int((nan_per_cpg > 0.9).sum())
    in_llama_low_nan  = int(((nan_per_cpg < 0.05) & llama_mask).sum())

    print(f"\n  CpG position NaN summary ({C:,} positions):")
    print(f"    Always measured (<1% NaN)  : {n_always_measured:,} ({100*n_always_measured/C:.1f}%)")
    print(f"    Low NaN (<5%)              : {n_low_nan:,} ({100*n_low_nan/C:.1f}%)")
    print(f"    High NaN (>90%)            : {n_high_nan:,} ({100*n_high_nan/C:.1f}%)")
    print(f"    MethylLlama CpGs with <5% NaN: {in_llama_low_nan:,} / {llama_mask.sum():,}")

    # ── 5. Age-based overlap with MethylLlama split ───────────────────────────
    sp_info = llama_split_info.get(split, {})
    ll_ages = sp_info.get("ages", np.array([]))
    if len(ll_ages) > 0:
        gpt_ages_r4 = np.round(ages, 4)
        ll_ages_r4  = np.round(ll_ages, 4)

        ll_ctr  = Counter(ll_ages_r4.tolist())
        gpt_ctr = Counter(gpt_ages_r4.tolist())
        shared_ages   = set(ll_ctr.keys()) & set(gpt_ctr.keys())
        shared_count  = sum(min(ll_ctr[a], gpt_ctr[a]) for a in shared_ages)
        unique_shared = len(shared_ages)
        pct_ll = 100 * shared_count / max(len(ll_ages), 1)
        pct_gp = 100 * shared_count / max(len(ages), 1)

        # Among zero-NaN GPT samples, how many match LL ages?
        gpt_zero_nan_ages = gpt_ages_r4[zero_nan_mask]
        gpt_zn_ctr = Counter(gpt_zero_nan_ages.tolist())
        shared_zn   = set(ll_ctr.keys()) & set(gpt_zn_ctr.keys())
        shared_zn_count = sum(min(ll_ctr[a], gpt_zn_ctr[a]) for a in shared_zn)
        pct_ll_zn = 100 * shared_zn_count / max(len(ll_ages), 1)

        print(f"\n  Age-based overlap with MethylLlama {split} ({len(ll_ages):,} samples):")
        print(f"    All GPT {split} samples   : shared_by_age={shared_count:,}  "
              f"({pct_ll:.1f}% of LL / {pct_gp:.1f}% of GPT)  unique_ages={unique_shared:,}")
        print(f"    Zero-NaN GPT samples only : shared_by_age={shared_zn_count:,}  "
              f"({pct_ll_zn:.1f}% of LL)  — these are the most likely true overlaps")

    return {
        "split":           split,
        "n_samples":       N,
        "n_cpgs":          C,
        "nan_overall":     float(nan_overall),
        "nan_in_llama":    float(nan_in_llama),
        "nan_out_llama":   float(nan_out_llama),
        "n_zero_nan":      n_zero_nan,
        "pct_zero_nan":    float(pct_zero_nan),
        "n_always_measured": n_always_measured,
        "n_low_nan_cpgs":  n_low_nan,
        "in_llama_low_nan": in_llama_low_nan,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--llama_h5ad",  default=LLAMA_H5AD)
    p.add_argument("--gpt_h5ad",    default=GPT_H5AD)
    p.add_argument("--gpt_parquet", default=GPT_PARQUET)
    p.add_argument("--splits",      default="valid,test,train",
                   help="Comma-separated splits to analyze")
    p.add_argument("--outdir",      default="dataset_comparison_outputs/full_nan_analysis")
    args = p.parse_args()

    splits = [s.strip() for s in args.splits.split(",")]
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Full Data NaN Analysis — MethylGPT vs MethylLlama CpGs")
    print("=" * 60)

    # ── Load CpG name lists ───────────────────────────────────────────────────
    print(f"\nLoading MethylGPT CpG names from {args.gpt_h5ad} ...")
    gpt_cpg_names = load_var_names_h5ad(args.gpt_h5ad)
    print(f"  MethylGPT: {len(gpt_cpg_names):,} CpG names")

    print(f"\nLoading MethylLlama CpG names from {args.llama_h5ad} ...")
    llama_cpg_names = load_var_names_h5ad(args.llama_h5ad)
    llama_cpg_set   = set(llama_cpg_names)
    print(f"  MethylLlama: {len(llama_cpg_names):,} CpG names")

    # Build boolean mask: which of the 49,156 GPT columns are in MethylLlama
    llama_mask = np.array([name in llama_cpg_set for name in gpt_cpg_names], dtype=bool)
    n_in_llama = llama_mask.sum()
    print(f"\n  MethylLlama CpGs found in GPT column space: {n_in_llama:,} / {len(gpt_cpg_names):,}")
    print(f"  Position range of MethylLlama CpGs in GPT: "
          f"{np.where(llama_mask)[0].min()}–{np.where(llama_mask)[0].max()}")
    print(f"  Are they contiguous? {np.diff(np.where(llama_mask)[0]).max() == 1}")

    # ── Load MethylLlama split info (no X) ───────────────────────────────────
    print(f"\nLoading MethylLlama split metadata ...")
    llama_split_info = load_llama_split_info(args.llama_h5ad)
    for sp, v in llama_split_info.items():
        print(f"  {sp}: {v['n']:,} samples")

    # ── Analyze each split ────────────────────────────────────────────────────
    results = []
    for split in splits:
        data, ages, ids = load_parquet_split_full(args.gpt_parquet, split)
        r = report_split(split, data, ages, ids,
                         llama_mask, llama_split_info,
                         gpt_cpg_names, llama_cpg_set)
        results.append(r)
        del data  # free memory before next split

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Split':8s}  {'N':>6s}  {'NaN%':>6s}  {'NaN in LL pos':>13s}  "
          f"{'NaN out LL pos':>14s}  {'Zero-NaN samples':>16s}")
    for r in results:
        print(f"  {r['split']:6s}  {r['n_samples']:>6,}  {r['nan_overall']:>6.1%}  "
              f"{r['nan_in_llama']:>13.1%}  {r['nan_out_llama']:>14.1%}  "
              f"{r['n_zero_nan']:>6,} ({r['pct_zero_nan']:.1f}%)")

    # ── Save text report ──────────────────────────────────────────────────────
    lines = []
    lines.append("Full NaN Analysis — MethylGPT (49k) vs MethylLlama (19k) CpG positions")
    lines.append(f"MethylLlama CpGs in GPT space: {n_in_llama:,} / {len(gpt_cpg_names):,}")
    lines.append("")
    for r in results:
        lines.append(f"Split: {r['split']}")
        lines.append(f"  Samples        : {r['n_samples']:,}")
        lines.append(f"  NaN overall    : {r['nan_overall']:.3%}")
        lines.append(f"  NaN in LL CpGs : {r['nan_in_llama']:.3%}")
        lines.append(f"  NaN out LL CpGs: {r['nan_out_llama']:.3%}")
        lines.append(f"  Zero-NaN samp. : {r['n_zero_nan']:,} ({r['pct_zero_nan']:.1f}%)")
        lines.append(f"  Low-NaN CpGs(<5%): {r['n_low_nan_cpgs']:,}  "
                     f"of which in LL: {r['in_llama_low_nan']:,}")
        lines.append("")
    txt_path = out / "full_nan_analysis.txt"
    txt_path.write_text("\n".join(lines))
    print(f"\n  Saved → {txt_path}")
    print(f"\n{'='*60}")
    print("Done.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
