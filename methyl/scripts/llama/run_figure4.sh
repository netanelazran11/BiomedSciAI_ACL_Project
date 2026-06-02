#!/bin/bash -l
#SBATCH --job-name=figure4-age-pca
#SBATCH --partition=salmon
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2:00:00

#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/%x_%j.out
#SBATCH --error=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/%x_%j.err

# ─────────────────────────────────────────────────────────────────────────────
# run_figure4.sh  —  Before vs After fine-tuning: PCA colored by age
#
# DATA (both steps use the SAME h5ad):
#   /sci/.../finetuning_19608_clean_stratified_no_outliers.h5ad
#
# STEP 1 — Extract fine-tuned CLS embeddings
#   Model : MethylationAgeRegressorLlama (encoder updated from epoch 10)
#   Output: outputs/repr_analysis/finetune_extract_JOBID/embeddings_cls.npy
#
# STEP 2 — Figure 4: PCA colored by age (before vs after fine-tuning)
#   Before: cls_probing_44905909/embeddings_cls.npy  (pretrained model)
#   After : finetune_extract_JOBID/embeddings_cls.npy (fine-tuned model)
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
LOGDIR="${REPO}/logs_llama-wced"
mkdir -p "${LOGDIR}"

cd "${REPO}"
source bmfm_methyl_env/bin/activate
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# ─────────────────────────────────────────────────────────────────────────────
# Paths — DATA
# ─────────────────────────────────────────────────────────────────────────────

# The 19k finetune h5ad — used for BOTH embedding extractions
DATA="/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_finetune_19k_h5ad/finetuning_19608_clean_stratified_no_outliers.h5ad"
TOKENIZER="${REPO}/tokenizer_llama_pretrain49k"
EXT_META="${REPO}/data/pretrain_metadata.csv.gz"

# ─────────────────────────────────────────────────────────────────────────────
# Paths — PRETRAINED embeddings (already extracted, no GPU needed)
# Model: WCED pretrained checkpoint, frozen encoder, on 19k finetune data
# ─────────────────────────────────────────────────────────────────────────────
PRETRAINED_BASE="${REPO}/outputs/repr_analysis/cls_probing_44905909"
PRETRAINED_NPY="${PRETRAINED_BASE}/embeddings_cls.npy"
PRETRAINED_META="${PRETRAINED_BASE}/metadata.csv"

# ─────────────────────────────────────────────────────────────────────────────
# Paths — FINE-TUNED checkpoint
# Model: MethylationAgeRegressorLlama, encoder unfrozen at epoch 10,
#        trained for 128 epochs total (best: epoch=127, MedAE=3.5625yr)
# ─────────────────────────────────────────────────────────────────────────────
FINETUNE_CKPT="${REPO}/outputs/finetune-llama-small/llama-small-ft-v5-cls-huber-ep300-wu500-scratch-44895876/checkpoints/epoch=127-val_medae=3.5625.ckpt"

# ─────────────────────────────────────────────────────────────────────────────
# Output dirs
# ─────────────────────────────────────────────────────────────────────────────
EXTRACT_OUTDIR="${REPO}/outputs/repr_analysis/finetune_extract_${SLURM_JOB_ID}"
FIGURE_OUTDIR="${REPO}/outputs/repr_analysis/figure4_${SLURM_JOB_ID}"
FINETUNED_NPY="${EXTRACT_OUTDIR}/embeddings_cls.npy"

echo "============================================================"
echo " Figure 4: Before vs After Fine-tuning (PCA by Age)"
echo " Job : ${SLURM_JOB_ID}  Host: $(hostname)  Time: $(date)"
echo "============================================================"
echo " DATA (both steps)    : ${DATA}"
echo " Pretrained CLS npy   : ${PRETRAINED_NPY}"
echo " Fine-tune checkpoint : ${FINETUNE_CKPT}"
echo " Fine-tuned CLS outdir: ${EXTRACT_OUTDIR}"
echo " Figure outdir        : ${FIGURE_OUTDIR}"
echo "============================================================"

# Validate
for f in "${DATA}" "${TOKENIZER}" "${PRETRAINED_NPY}" "${PRETRAINED_META}" "${FINETUNE_CKPT}"; do
    if [ ! -e "${f}" ]; then
        echo "ERROR: required file/dir not found: ${f}"
        exit 1
    fi
done

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Extract CLS from the FINE-TUNED model on the 19k finetune data
# This gives us embeddings from the trained age-prediction model.
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " STEP 1: Extracting fine-tuned CLS embeddings"
echo " Model : fine-tuned (epoch=127, MedAE=3.5625yr)"
echo " Data  : 19k finetune h5ad"
echo "============================================================"
mkdir -p "${EXTRACT_OUTDIR}"

python scripts/repr_analysis/cls_probing_analysis.py \
    --checkpoint     "${FINETUNE_CKPT}"    \
    --ckpt_type      finetune              \
    --data           "${DATA}"             \
    --tokenizer      "${TOKENIZER}"        \
    --metadata       "${EXT_META}"         \
    --metadata_id_col GSM_ID               \
    --outdir         "${EXTRACT_OUTDIR}"   \
    --batch_size     64                    \
    --device         cuda                  \
    --skip_probing                         \
    --label_cols     tissue sex dataset    \
    --age_col        age

echo ""
echo " STEP 1 done. Fine-tuned CLS saved to: ${FINETUNED_NPY}"

if [ ! -f "${FINETUNED_NPY}" ]; then
    echo "ERROR: fine-tuned embeddings not found at ${FINETUNED_NPY}"
    exit 1
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: PCA visualization — before vs after fine-tuning
# Both embedding files must be row-aligned to PRETRAINED_META (metadata.csv)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " STEP 2: Generating Figure 4 (PCA colored by age)"
echo " Before FT : ${PRETRAINED_NPY}"
echo " After  FT : ${FINETUNED_NPY}"
echo "============================================================"

# NOTE: both npy files must have the SAME row order as PRETRAINED_META.
# cls_probing_analysis.py saves embeddings in h5ad obs order.
# pretrained_npy is also in h5ad obs order (from the same cls_probing run).
# If the two runs used the same h5ad file, the order is guaranteed identical.

python scripts/repr_analysis/figure4_age_pca.py \
    --pretrained_npy   "${PRETRAINED_NPY}"   \
    --finetuned_npy    "${FINETUNED_NPY}"    \
    --metadata_csv     "${PRETRAINED_META}"  \
    --ext_metadata     "${EXT_META}"         \
    --ext_id_col       GSM_ID               \
    --age_col          age                  \
    --outdir           "${FIGURE_OUTDIR}"   \
    --dpi              200

echo ""
echo "============================================================"
echo " ALL DONE: $(date)"
echo " Outputs → ${FIGURE_OUTDIR}/figures/"
echo "   figure4_age_pca.png      2×2 panel: pretrained vs fine-tuned, age vs tissue"
echo "   figure4_age_pca.pdf      PDF version"
echo "   pretrained_age.png       individual panel"
echo "   finetuned_age.png        individual panel"
echo "   pretrained_tissue.png    individual panel"
echo "   finetuned_tissue.png     individual panel"
echo "============================================================"
