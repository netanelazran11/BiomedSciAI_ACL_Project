#!/bin/bash -l
#SBATCH --job-name=v7b-raw-umap
#SBATCH --partition=salmon
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00

#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/%x_%j.out
#SBATCH --error=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/%x_%j.err

# Raw-vs-model UMAP (MethylGPT Fig 3d-f). No GPU; needs RAM for the raw matrix
# (10,988 x 21,368) + PCA. Run as a job, not on the login node (which OOM-kills).

set -euo pipefail
REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
cd "${REPO}"
source bmfm_methyl_env/bin/activate
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

python scripts/repr_analysis_v7b/raw_vs_model_umap.py \
    --h5ad "${REPO}/../../../data/data_methyl_21k_h5ad/altumage_21k_3way.h5ad" \
    --dir "${REPO}/figures/v7b_pretrain_cls"

echo "DONE: $(date) — figures/v7b_pretrain_cls/pub/raw_vs_model_umap.png"
