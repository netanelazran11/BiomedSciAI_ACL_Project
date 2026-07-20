#!/bin/bash -l
#SBATCH --job-name=v7b-attention
#SBATCH --partition=salmon
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00

#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/%x_%j.out
#SBATCH --error=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/%x_%j.err

# ─────────────────────────────────────────────────────────────────────────────
# V7b attention analysis — genomic-RoPE correct (the attention_v5 redo).
#   A. CLS attention selectivity (entropy/gini/top-k) per layer/head
#   B. Genomic distance-decay (does attention concentrate on nearby CpGs?) —
#      the real Genomic-RoPE validation.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
cd "${REPO}"
source bmfm_methyl_env/bin/activate
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

CKPT="${CKPT:-${REPO}/outputs/pretrain-llama-wced/llama-6L-all49k-r0.5-w0.05-genomic-45468861/checkpoints/epoch=85-recon=0.0552-pcc=0.9713.ckpt}"
DATA="${DATA:-${REPO}/../../../data/data_methyl_21k_h5ad/altumage_21k_3way.h5ad}"
TOKENIZER="${REPO}/tokenizer_llama_pretrain49k"
GENOMIC_RANK="${REPO}/outputs/cpg_genomic_sort/cpg_genomic_rank_finetune.npy"
OUTDIR="${REPO}/figures/v7b_attention"

echo "=== V7b attention analysis | job ${SLURM_JOB_ID:-local} | $(date) ==="
python scripts/repr_analysis_v7b/attention_analysis_v7b.py \
    --checkpoint "${CKPT}" --data "${DATA}" --tokenizer "${TOKENIZER}" \
    --genomic_rank "${GENOMIC_RANK}" --outdir "${OUTDIR}" \
    --max_samples 512 --batch_size 16 --n_query 32

echo "=== DONE: $(date) — outputs in ${OUTDIR}/ ==="
