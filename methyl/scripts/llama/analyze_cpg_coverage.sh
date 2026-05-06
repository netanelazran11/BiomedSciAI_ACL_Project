#!/bin/bash -l
#SBATCH --job-name=analyze-cpg-coverage
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00

#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/%x_%j.out
#SBATCH --error=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/%x_%j.err

set -euo pipefail

REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
DATA_DIR="/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_pretrain_type3_h5ad"
PRETRAIN_H5AD="${DATA_DIR}/methylgpt_pretrain_type3.h5ad"
FINETUNE_H5AD="/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_finetune_49k_h5ad/finetuning_49k.h5ad"
OUT_DIR="${REPO}/outputs/cpg_coverage_analysis"

mkdir -p "${OUT_DIR}"

cd "${REPO}"
source bmfm_methyl_env/bin/activate

echo "============================================================"
echo "CpG Coverage Analysis"
echo "============================================================"
echo "Job: ${SLURM_JOB_ID} | Host: $(hostname) | Time: $(date)"
echo "Pretrain data: ${PRETRAIN_H5AD}"
echo "Finetune data: ${FINETUNE_H5AD}"
echo "Output:        ${OUT_DIR}"
echo "============================================================"

python3 - <<PY
import anndata
import numpy as np
import pandas as pd
import scipy.sparse

out_dir = "${OUT_DIR}"

def analyze(h5ad_path, label):
    print(f"\n{'='*60}")
    print(f"Dataset: {label}")
    print(f"File:    {h5ad_path}")
    print('='*60)

    adata = anndata.read_h5ad(h5ad_path)
    n_cells, n_cpgs = adata.n_obs, adata.n_vars
    print(f"Shape: {n_cells} cells × {n_cpgs} CpGs")

    X = adata.X
    if scipy.sparse.issparse(X):
        X = X.toarray()
    X = X.astype(np.float32)

    # Per-CpG: count non-NaN cells
    measured = ~np.isnan(X)                    # [n_cells, n_cpgs] bool
    coverage = measured.sum(axis=0)            # [n_cpgs] int

    always_nan  = (coverage == 0).sum()
    always_meas = (coverage == n_cells).sum()
    partial     = ((coverage > 0) & (coverage < n_cells)).sum()

    print(f"\nPer-CpG coverage:")
    print(f"  Always measured (all {n_cells} cells): {always_meas:>8,}  ({100*always_meas/n_cpgs:.1f}%)")
    print(f"  Partially measured (1–{n_cells-1} cells): {partial:>8,}  ({100*partial/n_cpgs:.1f}%)")
    print(f"  Always NaN    (0 cells):             {always_nan:>8,}  ({100*always_nan/n_cpgs:.1f}%)")

    if partial > 0:
        print(f"\n  Partial-coverage breakdown:")
        for pct in [99, 95, 90, 75, 50]:
            n = (coverage >= n_cells * pct / 100).sum()
            print(f"    ≥{pct:3d}% cells: {n:>8,} CpGs")

    # Per-cell: how many CpGs are measured?
    per_cell = measured.sum(axis=1)
    print(f"\nPer-cell measurement count:")
    print(f"  Min:    {per_cell.min():>8,}")
    print(f"  Median: {int(np.median(per_cell)):>8,}")
    print(f"  Max:    {per_cell.max():>8,}")
    print(f"  All cells same count: {per_cell.min() == per_cell.max()}")

    # Save the list of always-measured CpG site IDs
    cpg_names = np.array(adata.var_names)
    measured_cpgs = cpg_names[coverage == n_cells]
    partial_cpgs  = cpg_names[(coverage > 0) & (coverage < n_cells)]
    nan_cpgs      = cpg_names[coverage == 0]

    tag = label.lower().replace(" ", "_")
    pd.DataFrame({"cpg_site": measured_cpgs}).to_csv(
        f"{out_dir}/cpgs_always_measured_{tag}.csv", index=False
    )
    pd.DataFrame({"cpg_site": nan_cpgs}).to_csv(
        f"{out_dir}/cpgs_always_nan_{tag}.csv", index=False
    )
    pd.DataFrame({"cpg_site": cpg_names, "coverage_count": coverage,
                  "coverage_pct": coverage / n_cells * 100}).to_csv(
        f"{out_dir}/cpgs_full_coverage_{tag}.csv", index=False
    )

    print(f"\nSaved:")
    print(f"  {out_dir}/cpgs_always_measured_{tag}.csv  ({len(measured_cpgs)} CpGs)")
    print(f"  {out_dir}/cpgs_always_nan_{tag}.csv       ({len(nan_cpgs)} CpGs)")
    print(f"  {out_dir}/cpgs_full_coverage_{tag}.csv    (all {n_cpgs} CpGs with coverage)")

    return always_meas, always_nan, partial

pretrain_meas, pretrain_nan, pretrain_partial = analyze("${PRETRAIN_H5AD}", "pretrain")
finetune_meas, finetune_nan, finetune_partial = analyze("${FINETUNE_H5AD}", "finetune")

print(f"\n{'='*60}")
print("SUMMARY")
print('='*60)
print(f"Pretrain — measured: {pretrain_meas:,}  always-NaN: {pretrain_nan:,}  partial: {pretrain_partial:,}")
print(f"Finetune — measured: {finetune_meas:,}  always-NaN: {finetune_nan:,}  partial: {finetune_partial:,}")

# Check overlap between pretrain and finetune measured sets
import anndata as ad
pt = ad.read_h5ad("${PRETRAIN_H5AD}")
ft = ad.read_h5ad("${FINETUNE_H5AD}")

pt_X = pt.X.toarray() if scipy.sparse.issparse(pt.X) else pt.X.astype(np.float32)
ft_X = ft.X.toarray() if scipy.sparse.issparse(ft.X) else ft.X.astype(np.float32)

pt_measured = set(pt.var_names[~np.isnan(pt_X).all(axis=0)])
ft_measured = set(ft.var_names[~np.isnan(ft_X).all(axis=0)])

overlap = pt_measured & ft_measured
print(f"\nCpGs measured in BOTH pretrain and finetune: {len(overlap):,}")
print(f"Only in pretrain: {len(pt_measured - ft_measured):,}")
print(f"Only in finetune: {len(ft_measured - pt_measured):,}")

pd.DataFrame({"cpg_site": sorted(overlap)}).to_csv(
    "${OUT_DIR}/cpgs_pretrain_AND_finetune_measured.csv", index=False
)
print(f"\nSaved intersection: ${OUT_DIR}/cpgs_pretrain_AND_finetune_measured.csv  ({len(overlap)} CpGs)")
print(f"\n{'='*60}")
print(f"RECOMMENDATION: use SUBSET_K={len(overlap)} (intersection of both datasets)")
print('='*60)
PY

echo "============================================================"
echo "Analysis finished: $(date)"
echo "Results: ${OUT_DIR}/"
echo "============================================================"
