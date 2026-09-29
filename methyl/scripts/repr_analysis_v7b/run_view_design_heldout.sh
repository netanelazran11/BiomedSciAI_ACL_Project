#!/bin/bash -l
#SBATCH --job-name=view-design-heldout
#SBATCH --partition=salmon
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=4:00:00
#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/%x_%j.out
#SBATCH --error=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/%x_%j.err
# ─────────────────────────────────────────────────────────────────────────────
# Cross-view consistency on the HELD-OUT partition of the pretraining corpus
# (overlap vs disjoint views, 5 seeds). Inference only; same checkpoint and
# same analysis code as run_view_design_eval.sh, different profiles.
# Writes to figures/v7b_pretrain_cls/view_design_heldout/ -- the AltuMAge
# results in view_design/ are not touched.
#
# Usage: sbatch scripts/repr_analysis_v7b/run_view_design_heldout.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
CKPT="${REPO}/outputs/pretrain-llama-wced/llama-6L-all49k-r0.5-w0.05-genomic-45468861/checkpoints/epoch=85-recon=0.0552-pcc=0.9713.ckpt"
DATA="/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_pretrain_type3_h5ad/methylgpt_pretrain_type3.h5ad"
TOKENIZER="${REPO}/tokenizer_llama_pretrain49k"
GENOMIC_RANK="${REPO}/outputs/cpg_genomic_sort/cpg_genomic_rank.npy"
OUTDIR="${REPO}/figures/v7b_pretrain_cls/view_design_heldout"
N_SAMPLES="${N_SAMPLES:-2000}"
N_SEEDS="${N_SEEDS:-5}"

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
echo "View-design evaluation on HELD-OUT pretraining profiles"
echo "Job: ${SLURM_JOB_ID}  Host: $(hostname)  Time: $(date)"
echo "Profiles: ${N_SAMPLES}  Seeds: ${N_SEEDS}"
echo "============================================================"

python scripts/repr_analysis_v7b/view_design_eval_heldout.py \
    --checkpoint "${CKPT}" --data "${DATA}" --split test \
    --tokenizer "${TOKENIZER}" --genomic_rank "${GENOMIC_RANK}" \
    --n_samples "${N_SAMPLES}" --n_seeds "${N_SEEDS}" \
    --outdir "${OUTDIR}"

echo "DONE: $(date) -> ${OUTDIR}/"
