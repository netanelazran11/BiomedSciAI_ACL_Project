#!/bin/bash -l
#SBATCH --job-name=v7b-finetune-cls
#SBATCH --partition=salmon
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=3:00:00

#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/%x_%j.out
#SBATCH --error=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/%x_%j.err

# ─────────────────────────────────────────────────────────────────────────────
# V7b FINE-TUNED CLS extraction + representation analysis.
# Sibling of run_pretrain_cls.sh — uses the IDENTICAL DATA / TOKENIZER /
# GENOMIC_RANK as that script, so the pretrain-vs-finetune comparison uses the
# same sample set and the same CpG site set; only the encoder weights differ
# (pretrain vs fine-tuned fold4, the best-performing fold, test MedAE=3.144yr).
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
cd "${REPO}"
source bmfm_methyl_env/bin/activate
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

CKPT="${CKPT:-${REPO}/outputs/finetune-llama-small/llama-v7b-kfold-fold4-ep300-45586014/checkpoints/epoch=138-val_medae=2.6875.ckpt}"
DATA="${DATA:-${REPO}/../../../data/data_methyl_21k_h5ad/altumage_21k_3way.h5ad}"
TOKENIZER="${REPO}/tokenizer_llama_pretrain49k"
GENOMIC_RANK="${REPO}/outputs/cpg_genomic_sort/cpg_genomic_rank_finetune.npy"
OUTDIR="${REPO}/figures/v7b_finetuned_cls"

echo "============================================================"
echo " V7b fine-tuned CLS analysis | job ${SLURM_JOB_ID:-local} | $(date)"
echo " Checkpoint : ${CKPT}"
echo " Data       : ${DATA}"
echo " Genomic rank: ${GENOMIC_RANK}"
echo " Outdir     : ${OUTDIR}"
echo "============================================================"

[ -f "${CKPT}" ] || { echo "ERROR: checkpoint not found: ${CKPT}"; exit 1; }
[ -f "${GENOMIC_RANK}" ] || { echo "ERROR: genomic rank not found: ${GENOMIC_RANK}"; exit 1; }

python scripts/repr_analysis_v7b/extract_finetune_cls.py \
    --checkpoint   "${CKPT}" \
    --data         "${DATA}" \
    --tokenizer    "${TOKENIZER}" \
    --genomic_rank "${GENOMIC_RANK}" \
    --outdir       "${OUTDIR}" \
    --batch_size   32 \
    --age_col      age \
    --label_cols   tissue_type sex dataset \
    --split_col    split

echo "============================================================"
echo " DONE: $(date) — outputs in ${OUTDIR}/"
echo "============================================================"
