#!/bin/bash -l
#SBATCH --job-name=v7b-kfold-testeval
#SBATCH --partition=salmon
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00

#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/%x_%j.out
#SBATCH --error=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/%x_%j.err

# ─────────────────────────────────────────────────────────────────────────────
# Standalone test-eval for k-fold folds whose TRAINING converged but whose job
# hit the 48h limit before the auto test-eval step ran.
#
# For each fold: find best val_medae checkpoint, evaluate on the fixed test set
# (split=='test', 2149 samples) with the correct fold age-normalization.
#
# Usage:
#   CHECKPOINT=<pretrain ep85 ckpt> bash scripts/llama/testeval_kfold.sh
#   (or set FOLDS="0 1 2" to run a subset)
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

CHECKPOINT="${CHECKPOINT:?ERROR: set CHECKPOINT to the ep85 pretrain ckpt}"
FOLDS="${FOLDS:-0 1 2 3 4}"

REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
DATA="${REPO}/../../../data/data_methyl_21k_h5ad/altumage_21k_3way.h5ad"
KFOLD_DIR="${REPO}/outputs/kfold_splits"
GENOMIC_RANK_FT_NPY="${REPO}/outputs/cpg_genomic_sort/cpg_genomic_rank_finetune.npy"
TOKENIZER_PATH="${REPO}/tokenizer_llama_pretrain49k"
OUTROOT="${REPO}/outputs/finetune-llama-small"

WANDB_ENTITY="netanelazran11-hebrew-university-of-jerusalem"
WANDB_PROJECT="finetune-llama-small"
WANDB_GROUP="v7b-kfold-5fold"

source /etc/profile.d/modules.sh 2>/dev/null || true
module purge 2>/dev/null || true
module load spack/all 2>/dev/null || true
module load cuda/12.3.2-gcc-5bv3kyh 2>/dev/null || true
cd "${REPO}"
source bmfm_methyl_env/bin/activate
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

for FOLD in ${FOLDS}; do
    # locate the fold's output dir (any job id)
    OUTDIR=$(ls -d "${OUTROOT}"/llama-v7b-kfold-fold${FOLD}-ep*-* 2>/dev/null | head -1)
    if [ -z "${OUTDIR}" ] || [ ! -d "${OUTDIR}/checkpoints" ]; then
        echo "SKIP fold ${FOLD}: no output dir/checkpoints found"
        continue
    fi
    # best val_medae checkpoint = lowest value in filename
    BEST_CKPT=$(ls "${OUTDIR}/checkpoints/"epoch=*-val_medae=*.ckpt 2>/dev/null \
        | sed -E 's/.*val_medae=([0-9.]+)\.ckpt/\1 &/' | sort -n | head -1 | cut -d' ' -f2-)
    if [ -z "${BEST_CKPT}" ]; then
        echo "SKIP fold ${FOLD}: no val_medae checkpoint"
        continue
    fi
    FOLD_TRAIN_NPY="${KFOLD_DIR}/fold_${FOLD}_train.npy"
    FOLD_VAL_NPY="${KFOLD_DIR}/fold_${FOLD}_val.npy"

    echo "============================================================"
    echo "Fold ${FOLD} TEST EVAL"
    echo "  dir  : ${OUTDIR}"
    echo "  ckpt : ${BEST_CKPT}"
    echo "============================================================"

    python -m bmfm_methylation.llama.finetune_llama \
        "data_path='${DATA}'" \
        "checkpoint_path='${CHECKPOINT}'" \
        "tokenizer_path='${TOKENIZER_PATH}'" \
        "output_directory='${OUTDIR}'" \
        "+eval_checkpoint='${BEST_CKPT}'" \
        data_module.subset_k=49156 \
        data_module.fixed_subset_seed=42 \
        data_module.max_length=21369 \
        data_module.batch_size=32 \
        data_module.num_workers=8 \
        data_module.filter_age_outliers=true \
        "data_module.duplicate_pairs_csv='${REPO}/dataset_fingerprint_outputs/duplicate_pairs.csv'" \
        "+data_module.fold_train_ids_npy='${FOLD_TRAIN_NPY}'" \
        "+data_module.fold_val_ids_npy='${FOLD_VAL_NPY}'" \
        wced_input_ratio=1.0 \
        wced_genomic_rank_path="${GENOMIC_RANK_FT_NPY}" \
        finetune.head_hidden_size=128 \
        finetune.head_dropout=0.1 \
        finetune.pooling=cls \
        finetune.loss_type=huber \
        precision="16-mixed" \
        track_wandb.enabled=true \
        track_wandb.project="${WANDB_PROJECT}" \
        track_wandb.entity="${WANDB_ENTITY}" \
        track_wandb.name="llama-v7b-kfold-fold${FOLD}-testeval" \
        "+track_wandb.group='${WANDB_GROUP}'"

    echo "Fold ${FOLD} test eval done."
done

echo "============================================================"
echo "ALL TEST EVALS DONE: $(date)"
echo "============================================================"
