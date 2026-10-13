#!/bin/bash -l
#SBATCH --job-name=view-heldout-full
#SBATCH --partition=salmon
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=8:00:00
#SBATCH --array=0-6
#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/%x_%A_%a.out
#SBATCH --error=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/%x_%A_%a.err
# ─────────────────────────────────────────────────────────────────────────────
# FINAL cross-view consistency for Fig. 2a/b: the whole held-out partition of the
# pretraining corpus (16,912 profiles), overlap and disjoint views, 3 seeds each, plus one
# negative control (pattern_only: measured-CpG pattern kept, values replaced by per-CpG means).
# Seven array tasks, one (condition, seed) each, about 2 h per task.
# Inference only. Writes to figures/v7b_pretrain_cls/view_consistency_heldout_full/.
#
# Usage:   sbatch scripts/repr_analysis_v7b/run_view_consistency_heldout_full.sh
# Then:    python scripts/repr_analysis_v7b/view_consistency_heldout_full.py --merge
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

CONDS=(overlap overlap overlap disjoint disjoint disjoint pattern_only)
SEEDS=(0 1 2 0 1 2 0)
COND="${CONDS[$SLURM_ARRAY_TASK_ID]}"
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"

REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
CKPT="${REPO}/outputs/pretrain-llama-wced/llama-6L-all49k-r0.5-w0.05-genomic-45468861/checkpoints/epoch=85-recon=0.0552-pcc=0.9713.ckpt"
DATA="/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_pretrain_type3_h5ad/methylgpt_pretrain_type3.h5ad"
TOKENIZER="${REPO}/tokenizer_llama_pretrain49k"
GENOMIC_RANK="${REPO}/outputs/cpg_genomic_sort/cpg_genomic_rank.npy"
OUTDIR="${REPO}/figures/v7b_pretrain_cls/view_consistency_heldout_full"

cd "${REPO}"
source /etc/profile.d/modules.sh 2>/dev/null || true
module purge 2>/dev/null || true
module load spack/all 2>/dev/null || true
module load cuda/12.3.2-gcc-5bv3kyh 2>/dev/null || true
source bmfm_methyl_env/bin/activate
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

[ -f "${CKPT}" ]         || { echo "ERROR: checkpoint not found: ${CKPT}"; exit 1; }
[ -f "${DATA}" ]         || { echo "ERROR: pretrain h5ad not found: ${DATA}"; exit 1; }
[ -f "${GENOMIC_RANK}" ] || { echo "ERROR: genomic rank not found: ${GENOMIC_RANK}"; exit 1; }
[ -d "${TOKENIZER}" ]    || { echo "ERROR: tokenizer not found: ${TOKENIZER}"; exit 1; }

echo "============================================================"
echo "Cross-view consistency, full held-out partition"
echo "Array ${SLURM_ARRAY_JOB_ID} task ${SLURM_ARRAY_TASK_ID}: condition=${COND} seed=${SEED}"
echo "Host: $(hostname)  Time: $(date)"
echo "============================================================"

python scripts/repr_analysis_v7b/view_consistency_heldout_full.py \
    --checkpoint "${CKPT}" --data "${DATA}" --split test \
    --tokenizer "${TOKENIZER}" --genomic_rank "${GENOMIC_RANK}" \
    --condition "${COND}" --seed "${SEED}" --n_samples 0 \
    --outdir "${OUTDIR}"

echo "DONE task ${SLURM_ARRAY_TASK_ID}: $(date)"
