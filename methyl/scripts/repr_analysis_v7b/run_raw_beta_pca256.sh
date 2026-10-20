#!/bin/bash -l
#SBATCH --job-name=raw-beta-pca256
#SBATCH --partition=salmon
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/%x_%j.out
#SBATCH --error=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/%x_%j.err

# Raw-beta PCA-256 baseline for Fig. 3 (no GPU work; same partition/flags as the July raw-vs-model job).
# Usage (from methyl/): sbatch scripts/repr_analysis_v7b/run_raw_beta_pca256.sh
set -euo pipefail
REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
H5AD="${REPO}/../../../data/data_methyl_21k_h5ad/altumage_21k_3way.h5ad"
cd "${REPO}"
source ablation_infonce/env.sh
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
[ -f "${H5AD}" ] || { echo "ERROR: not found: ${H5AD}"; exit 1; }
[ -f figures/v7b_pretrain_cls/metadata.csv ] || { echo "ERROR: metadata.csv missing"; exit 1; }
"${PY}" scripts/repr_analysis_v7b/raw_beta_pca256.py --h5ad "${H5AD}" --dir figures/v7b_pretrain_cls
ls -la figures/v7b_pretrain_cls/raw_pca256.npy figures/v7b_pretrain_cls/raw_pca256_info.json
echo "DONE: $(date)"
