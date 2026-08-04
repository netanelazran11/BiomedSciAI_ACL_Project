#!/bin/bash -l
#SBATCH --job-name=v7b-two-view-consistency
#SBATCH --partition=salmon
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00

#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/%x_%j.out
#SBATCH --error=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/%x_%j.err

# ─────────────────────────────────────────────────────────────────────────────
# Standalone re-run of two_view_consistency.py (now also saves the normalized
# View1/View2 embeddings + full similarity matrix, not just summary stats) so
# the contrastive-alignment claim can be shown visually (histogram of positive
# vs negative pair similarities, similarity-matrix heatmap) instead of just
# two numbers.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
CKPT="${REPO}/outputs/pretrain-llama-wced/llama-6L-all49k-r0.5-w0.05-genomic-45468861/checkpoints/epoch=85-recon=0.0552-pcc=0.9713.ckpt"
DATA="${REPO}/../../../data/data_methyl_21k_h5ad/altumage_21k_3way.h5ad"
TOKENIZER="${REPO}/tokenizer_llama_pretrain49k"
GENOMIC_RANK="${REPO}/outputs/cpg_genomic_sort/cpg_genomic_rank_finetune.npy"
OUTDIR="${REPO}/figures/v7b_pretrain_cls"

cd "${REPO}"
source /etc/profile.d/modules.sh 2>/dev/null || true
module purge 2>/dev/null || true
module load spack/all 2>/dev/null || true
module load cuda/12.3.2-gcc-5bv3kyh 2>/dev/null || true
source bmfm_methyl_env/bin/activate
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

python scripts/repr_analysis_v7b/two_view_consistency.py \
    --checkpoint "${CKPT}" --data "${DATA}" --tokenizer "${TOKENIZER}" \
    --genomic_rank "${GENOMIC_RANK}" --n_samples 2000 --outdir "${OUTDIR}"

echo "DONE: $(date) — two_view_v1n.npy / two_view_v2n.npy / two_view_simmatrix.npy in ${OUTDIR}/"
