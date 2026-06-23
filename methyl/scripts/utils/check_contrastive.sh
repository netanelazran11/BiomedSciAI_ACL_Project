#!/bin/bash -l
#SBATCH --job-name=check-contrastive
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/%x_%j.out
#SBATCH --error=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/%x_%j.err

set -euo pipefail

REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
mkdir -p "${REPO}/logs_llama-wced"
cd "${REPO}"
source bmfm_methyl_env/bin/activate
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

# ── Paths ──────────────────────────────────────────────────────────────────
CKPT="${REPO}/outputs/pretrain-wced-bmfm/wced-contrastive-k8000-w0.1-44206138/pretrain/checkpoints/epoch=epoch=190-val_loss=validation/loss=0.1264.ckpt"
DATA="/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_pretrain_type3_h5ad/methylgpt_pretrain_type3.h5ad"
TOKENIZER="${REPO}/tokenizer_llama_pretrain49k"
OUTDIR="${REPO}/outputs/diagnostics/contrastive_check_${SLURM_JOB_ID}"
mkdir -p "${OUTDIR}"

echo "============================================================"
echo " MethylLlama — Contrastive Loss Diagnostic"
echo " Job : ${SLURM_JOB_ID}  Host: $(hostname)  Time: $(date)"
echo "============================================================"
echo " Checkpoint : ${CKPT}"
echo " Data       : ${DATA}"
echo " Outdir     : ${OUTDIR}"
echo "============================================================"

python scripts/utils/check_contrastive.py \
    --checkpoint  "${CKPT}"      \
    --h5ad        "${DATA}"      \
    --tokenizer   "${TOKENIZER}" \
    --outdir      "${OUTDIR}"    \
    --n_samples   512            \
    --batch_size  64             \
    --subset_k    49156          \
    --input_ratio 0.5            \
    --temperature 0.1

echo ""
echo "============================================================"
echo " DONE: $(date)"
echo " Results → ${OUTDIR}/"
echo "   contrastive_report.txt   — all metrics"
echo "   pos_sim.npy              — per-sample positive similarity"
echo "   neg_sim.npy              — per-sample negative similarity"
echo "============================================================"
