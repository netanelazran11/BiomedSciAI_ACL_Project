#!/bin/bash -l
#SBATCH --job-name=ablation-rope-subset
#SBATCH --partition=salmon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00

#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/%x_%j.out
#SBATCH --error=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/%x_%j.err

# ─────────────────────────────────────────────────────────────────────────────
# Genomic RoPE ablation — Step 0: build the fixed 5,000-sample pretrain subset
# used by BOTH Run A (genomic RoPE) and Run B (no genomic RoPE). CPU-only,
# read-only on the source h5ad, writes a new file — does not touch anything
# in the existing pipeline.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
SOURCE="/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_pretrain_type3_h5ad/methylgpt_pretrain_type3.h5ad"

cd "${REPO}"
source /etc/profile.d/modules.sh 2>/dev/null || true
module purge 2>/dev/null || true
module load spack/all 2>/dev/null || true
source bmfm_methyl_env/bin/activate
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"

python scripts/utils/create_pretrain_subset_h5ad.py \
    --source "${SOURCE}" \
    --n_samples 5000 \
    --seed 42 \
    --outdir outputs/ablation_rope

echo "DONE: $(date)"
