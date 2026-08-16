#!/bin/bash -l
#SBATCH --job-name=kfold-test-predictions
#SBATCH --partition=salmon
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1:00:00

#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/%x_%j.out
#SBATCH --error=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/%x_%j.err

# ─────────────────────────────────────────────────────────────────────────────
# Per-sample test predictions for one V7b k-fold checkpoint, for a paired
# subject-level bootstrap comparison against MethylGPT (extracted separately
# in the MethylGPT-thesis repo, same fixed 2,149-sample test set, joined by
# real GSM sample ID). Inference-only, isolated script -- does not modify or
# resubmit any training job.
#
# Usage: FOLD=0 sbatch scripts/llama/run_kfold_test_predictions.sh
#   (submit once per fold: FOLD=0,1,2,3,4)
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

FOLD="${FOLD:?ERROR: set FOLD=0..4}"

REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
DATA="${REPO}/../../../data/data_methyl_21k_h5ad/altumage_21k_3way.h5ad"
TOKENIZER="${REPO}/tokenizer_llama_pretrain49k"
GENOMIC_RANK="${REPO}/outputs/cpg_genomic_sort/cpg_genomic_rank_finetune.npy"
TEST_IDS="${REPO}/outputs/kfold_splits/test_ids.npy"
DUP_CSV="${REPO}/dataset_fingerprint_outputs/duplicate_pairs.csv"
OUTDIR="${REPO}/outputs/bootstrap_predictions/methyllama"

# Best-val_medae checkpoint per fold -- verified this session (matches WandB
# fold*-testeval runs exactly). See scripts/repr_analysis/validate_kfold_test_predictions.py
# for the recorded official metrics each of these must reproduce.
declare -A CKPT=(
    [0]="${REPO}/outputs/finetune-llama-small/llama-v7b-kfold-fold0-ep300-45586010/checkpoints/epoch=92-val_medae=2.6250.ckpt"
    [1]="${REPO}/outputs/finetune-llama-small/llama-v7b-kfold-fold1-ep300-45586011/checkpoints/epoch=137-val_medae=2.5940.ckpt"
    [2]="${REPO}/outputs/finetune-llama-small/llama-v7b-kfold-fold2-ep300-45586012/checkpoints/epoch=131-val_medae=2.6875.ckpt"
    [3]="${REPO}/outputs/finetune-llama-small/llama-v7b-kfold-fold3-ep300-45586013/checkpoints/epoch=137-val_medae=2.7188.ckpt"
    [4]="${REPO}/outputs/finetune-llama-small/llama-v7b-kfold-fold4-ep300-45586014/checkpoints/epoch=138-val_medae=2.6875.ckpt"
)
CKPT_PATH="${CKPT[$FOLD]:?ERROR: no checkpoint registered for FOLD=$FOLD}"

cd "${REPO}"
source /etc/profile.d/modules.sh 2>/dev/null || true
module purge 2>/dev/null || true
module load spack/all 2>/dev/null || true
module load cuda/12.3.2-gcc-5bv3kyh 2>/dev/null || true
source bmfm_methyl_env/bin/activate
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo "Fold: ${FOLD}"
echo "Checkpoint: ${CKPT_PATH}"
[ -f "${CKPT_PATH}" ] || { echo "ERROR: checkpoint not found"; exit 1; }
[ -f "${TEST_IDS}" ] || { echo "ERROR: test_ids.npy not found"; exit 1; }

python scripts/repr_analysis/extract_kfold_test_predictions.py \
    --fold "${FOLD}" \
    --checkpoint "${CKPT_PATH}" \
    --data "${DATA}" \
    --tokenizer "${TOKENIZER}" \
    --genomic_rank "${GENOMIC_RANK}" \
    --test_ids "${TEST_IDS}" \
    --duplicate_pairs_csv "${DUP_CSV}" \
    --outdir "${OUTDIR}" \
    --model_name "MethylLlamaV7b" \
    --determinism_check

echo "DONE fold ${FOLD}: $(date)"
