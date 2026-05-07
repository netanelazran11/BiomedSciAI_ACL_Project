#!/bin/bash -l
#SBATCH --job-name=convert-parquet-h5ad
#SBATCH --partition=glacier,glacier-k,catfish,catfish-k,salmon,salmon-k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00

#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/%x_%j.out
#SBATCH --error=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/%x_%j.err

set -euo pipefail

REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"

# Input: parquet dataset (original 21k data)
PARQUET_DIR="${PARQUET_DIR:-/sci/labs/benjamin.yakir/netanel.azran/data/altumage_21k_parquet}"
CPG_MAP="${PARQUET_DIR}/cpg_mapping/probe_ids_type3_21k.csv"

# Reference: existing 49k h5ad (for CpG comparison only)
H5AD_49K="${H5AD_49K:-/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_finetune_49k_h5ad/finetuning_49k.h5ad}"

# Output: new correct h5ad
OUT_DIR="${REPO}/outputs/h5ad_21k"
OUT_H5AD="${OUT_DIR}/finetuning_21k.h5ad"

mkdir -p "${OUT_DIR}"

cd "${REPO}"
source bmfm_methyl_env/bin/activate

echo "============================================================"
echo "PARQUET → H5AD CONVERSION (21k CpGs, zero NaN)"
echo "============================================================"
echo "Job: ${SLURM_JOB_ID} | Host: $(hostname) | Time: $(date)"
echo "Parquet dir: ${PARQUET_DIR}"
echo "CpG map:     ${CPG_MAP}"
echo "Output:      ${OUT_H5AD}"
echo "============================================================"

python3 - <<PY
import numpy as np
import pandas as pd
import anndata as ad
import h5py
import scipy.sparse

PARQUET_DIR = "${PARQUET_DIR}"
CPG_MAP     = "${CPG_MAP}"
H5AD_49K    = "${H5AD_49K}"
OUT_H5AD    = "${OUT_H5AD}"

# ── 1. Load CpG mapping ──────────────────────────────────────────────────────
cpg_df   = pd.read_csv(CPG_MAP)
cpg_ids  = cpg_df['illumina_probe_id'].tolist()
n_cpgs   = len(cpg_ids)
print(f"CpG sites in parquet mapping: {n_cpgs:,}")
assert len(set(cpg_ids)) == n_cpgs, "Duplicate CpG IDs in mapping file!"

# ── 2. Compare with existing 49k h5ad ───────────────────────────────────────
print(f"\nComparing with existing h5ad: {H5AD_49K}")
with h5py.File(H5AD_49K, "r") as f:
    var     = f["var"]
    idx_key = var.attrs.get("_index", "_index")
    if idx_key not in var:
        idx_key = list(var.keys())[0]
    h5ad_cpgs_all = set(np.array(var[idx_key]).astype(str))

parquet_set = set(cpg_ids)
overlap     = parquet_set & h5ad_cpgs_all
only_parq   = parquet_set - h5ad_cpgs_all
only_h5ad   = h5ad_cpgs_all - parquet_set

print(f"  Parquet CpGs:            {len(parquet_set):,}")
print(f"  h5ad CpGs (all 49k):     {len(h5ad_cpgs_all):,}")
print(f"  Overlap:                 {len(overlap):,}")
print(f"  Only in parquet:         {len(only_parq):,}  ← CpGs in original data but not in h5ad")
print(f"  Only in h5ad (not parq): {len(only_h5ad):,}  ← extra columns added to h5ad")

if only_parq:
    print(f"\n  Sample CpGs only in parquet (not in h5ad): {list(only_parq)[:10]}")
if only_h5ad:
    # Separate NaN-padded from real: how many of the h5ad-only are always NaN
    print(f"  These are the {len(only_h5ad):,} extra NaN-padded columns in the 49k h5ad")

# ── 3. Load all parquet splits ───────────────────────────────────────────────
print(f"\nLoading parquet splits from {PARQUET_DIR} ...")
dfs = {}
for split in ["train", "valid", "test"]:
    dfs[split] = pd.read_parquet(f"{PARQUET_DIR}/{split}.parquet")
    print(f"  {split}: {len(dfs[split]):,} samples")

# ── 4. Build X matrix and obs ────────────────────────────────────────────────
print("\nBuilding data matrix...")
rows_X   = []
rows_obs = []
uid      = 0

for split in ["train", "valid", "test"]:
    df = dfs[split]
    for _, row in df.iterrows():
        beta = np.array(row["data"], dtype=np.float32)
        assert len(beta) == n_cpgs, f"Beta length mismatch: {len(beta)} vs {n_cpgs}"
        assert not np.isnan(beta).any(), f"NaN found in sample {uid}"
        rows_X.append(beta)
        rows_obs.append({
            "cell_id":       f"{split}_{row['id']}",
            "original_id":   str(row["id"]),
            "age":           float(row["age"]),
            "split":         split,
        })
        uid += 1

X   = np.stack(rows_X, axis=0)   # [n_samples, n_cpgs]
obs = pd.DataFrame(rows_obs)
obs.index = obs["cell_id"]
obs = obs.drop(columns=["cell_id"])

print(f"  X shape: {X.shape}")
print(f"  dtype:   {X.dtype}")
print(f"  NaN:     {np.isnan(X).sum()}")
print(f"  min:     {X.min():.4f}  max: {X.max():.4f}")

# ── 5. Build var ─────────────────────────────────────────────────────────────
var = pd.DataFrame({"illumina_probe_id": cpg_ids})
var.index = cpg_ids

# ── 6. Verify splits are clean ───────────────────────────────────────────────
print("\nSplit verification:")
for split in ["train", "valid", "test"]:
    mask = obs["split"] == split
    ages = obs.loc[mask, "age"].values
    print(f"  {split}: {mask.sum():,} samples | age mean={ages.mean():.1f} std={ages.std():.1f} min={ages.min():.0f} max={ages.max():.0f}")

# Cross-split ID check (original IDs within each split should be unique)
for split in ["train", "valid", "test"]:
    ids = obs[obs["split"] == split]["original_id"].tolist()
    assert len(ids) == len(set(ids)), f"Duplicate original IDs in {split}!"
print("  No duplicate IDs within any split ✓")

# No cross-split leakage check (beta fingerprint)
print("  Checking cross-split data leakage...")
split_masks = {s: obs["split"] == s for s in ["train", "valid", "test"]}
for s1, s2 in [("train", "valid"), ("train", "test"), ("valid", "test")]:
    X1 = X[split_masks[s1]]
    X2 = X[split_masks[s2]]
    fp1 = set(map(tuple, X1[:, :8].tolist()))
    fp2 = set(map(tuple, X2[:, :8].tolist()))
    n_overlap = len(fp1 & fp2)
    print(f"    {s1} ∩ {s2} fingerprint overlap: {n_overlap} samples {'⚠️  LEAKAGE' if n_overlap > 0 else '✓'}")

# ── 7. Create AnnData and save ───────────────────────────────────────────────
print(f"\nCreating AnnData ({X.shape[0]:,} cells × {X.shape[1]:,} CpGs)...")
adata = ad.AnnData(X=X, obs=obs, var=var)
adata.obs_names_make_unique()

print(f"Saving to {OUT_H5AD} ...")
adata.write_h5ad(OUT_H5AD, compression="gzip")

import os
size_mb = os.path.getsize(OUT_H5AD) / 1e6
print(f"Done. File size: {size_mb:.1f} MB")

# ── 8. Summary ───────────────────────────────────────────────────────────────
print(f"""
============================================================
CONVERSION COMPLETE
============================================================
Output:       {OUT_H5AD}
Shape:        {adata.n_obs:,} cells × {adata.n_vars:,} CpGs
NaN:          0 (clean)
obs columns:  {list(adata.obs.columns)}
var columns:  {list(adata.var.columns)}
Splits:       {dict(adata.obs['split'].value_counts())}

vs old h5ad:
  Old: 11,453 cells × 49,156 CpGs (29,548 always-NaN columns)
  New: {adata.n_obs:,} cells × {adata.n_vars:,} CpGs (zero NaN)
============================================================
""")
PY

echo "============================================================"
echo "Conversion finished: $(date)"
echo "Output: ${OUT_H5AD}"
echo "============================================================"
