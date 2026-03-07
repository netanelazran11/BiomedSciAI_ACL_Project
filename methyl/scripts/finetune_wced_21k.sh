#!/bin/bash -l
#SBATCH --job-name=finetune-wced-21k
#SBATCH --partition=goldfish
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00

#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# -------------------------
# Paths
# -------------------------
REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
LOGDIR="${REPO}/logs"

DATA="/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_21k_h5ad/altumage_21k_combined.h5ad"

# 21k tokenizer (auto-built during pretraining)
TOKENIZER_PATH="${REPO}/tokenizer_21k"

# -------------------------
# WCED-21k pretrained checkpoint
# UPDATE THIS: find the best epoch in your 21k pretrain run output
# e.g. outputs/pretrain-wced-21k/<run_name>/pretrain/checkpoints/epoch=...ckpt
# -------------------------
CHECKPOINT="${CHECKPOINT:-REPLACE_WITH_21K_PRETRAIN_CHECKPOINT_PATH}"

# -------------------------
# 21k-specific settings
# -------------------------
# Must match decoder vocab_size from 21k pretraining (21368 CpGs)
SUBSET_K="${SUBSET_K:-21368}"

# Input ratio: 4000 / 21368 = 0.187
# → each view sees ~4000 CpGs (within max_position_embeddings=8002)
# → CRITICAL: do NOT use 0.5 (would produce 10684 tokens, exceeding max pos emb)
INPUT_RATIO="${INPUT_RATIO:-0.187}"

# max_length = ~4000 tokens + 2 (CLS + SEP)
# Do NOT use SUBSET_K+2 here (that would be 21370 >> max_position_embeddings)
MAX_LENGTH=4002

# -------------------------
# Fine-tuning hyperparameters
# -------------------------
LEARNING_RATE="${LEARNING_RATE:-1e-3}"
BATCH_SIZE="${BATCH_SIZE:-16}"
ACCUMULATE_GRAD="${ACCUMULATE_GRAD:-4}"    # Effective batch = 64
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-300}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-60}"
HEAD_DROPOUT="${HEAD_DROPOUT:-0.1}"
RECON_WEIGHT="${RECON_WEIGHT:-0.1}"

# -------------------------
# W&B
# -------------------------
WANDB_ENTITY="netanelazran11-hebrew-university-of-jerusalem"
WANDB_PROJECT="finetune-wced-21k"
WANDB_RUN_NAME="wced-21k-finetune-k${SUBSET_K}-${SLURM_JOB_ID}"

# -------------------------
# Output directory
# -------------------------
OUTROOT="${REPO}/outputs/${WANDB_PROJECT}"
OUTDIR="${OUTROOT}/${WANDB_RUN_NAME}"

mkdir -p "${LOGDIR}"
mkdir -p "${OUTDIR}"

echo "============================================================"
echo "WCED-21k FINE-TUNING"
echo "============================================================"
echo "Job started:   $(date)"
echo "Host:          $(hostname)"
echo "JobID:         ${SLURM_JOB_ID}"
echo "============================================================"
echo "Checkpoint:    ${CHECKPOINT}"
echo "Data:          ${DATA}"
echo "Tokenizer:     ${TOKENIZER_PATH}"
echo "CpG vocab:     ${SUBSET_K} (must match decoder vocab_size)"
echo "Input/view:    ${INPUT_RATIO} × ${SUBSET_K} ≈ $(python3 -c "print(int(${SUBSET_K} * ${INPUT_RATIO}))") CpGs"
echo "Max length:    ${MAX_LENGTH} (within max_position_embeddings=8002)"
echo "Head LR:       ${LEARNING_RATE}  |  Encoder+Decoder LR: $(python3 -c "print(${LEARNING_RATE} * 0.01)")"
echo "Batch:         ${BATCH_SIZE} × ${ACCUMULATE_GRAD} = $((BATCH_SIZE * ACCUMULATE_GRAD)) effective"
echo "Recon weight:  ${RECON_WEIGHT}"
echo "W&B project:   ${WANDB_PROJECT}"
echo "W&B run:       ${WANDB_RUN_NAME}"
echo "Output:        ${OUTDIR}"
echo "============================================================"

# -------------------------
# Modules
# -------------------------
source /etc/profile.d/modules.sh 2>/dev/null || source /usr/share/modules/init/bash 2>/dev/null || true

module purge
module load spack/all
module load cuda/12.3.2-gcc-5bv3kyh

# -------------------------
# Env
# -------------------------
cd "${REPO}"
source bmfm_methyl_env/bin/activate

export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

python - <<'PY'
import torch
torch.set_float32_matmul_precision("medium")
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
PY

# -------------------------
# Run WCED-21k Fine-tuning
# -------------------------
python -m bmfm_methylation.finetune_wced_21k \
    data_path="${DATA}" \
    "checkpoint_path='${CHECKPOINT}'" \
    output_directory="${OUTDIR}" \
    tokenizer_path="${TOKENIZER_PATH}" \
    fields=methylation_21k \
    finetune_epochs=${FINETUNE_EPOCHS} \
    data_module.subset_k="${SUBSET_K}" \
    data_module.wced_input_ratio="${INPUT_RATIO}" \
    data_module.fixed_subset="false" \
    data_module.fixed_subset_seed="42" \
    data_module.max_length=${MAX_LENGTH} \
    data_module.batch_size=${BATCH_SIZE} \
    data_module.num_workers=4 \
    accumulate_grad_batches=${ACCUMULATE_GRAD} \
    trainer.learning_rate=${LEARNING_RATE} \
    regression_head.dropout=${HEAD_DROPOUT} \
    recon_weight=${RECON_WEIGHT} \
    freeze_encoder=false \
    early_stopping.patience=${EARLY_STOP_PATIENCE} \
    track_wandb.enabled=true \
    track_wandb.project="${WANDB_PROJECT}" \
    track_wandb.entity="${WANDB_ENTITY}" \
    track_wandb.name="${WANDB_RUN_NAME}"

echo "============================================================"
echo "WCED-21k Fine-tuning finished: $(date)"
echo "============================================================"
echo "Checkpoint: ${OUTDIR}"
echo "============================================================"
echo "Next steps:"
echo "  1. Check WandB — val/mae and val/r2"
echo "  2. Compare R² with 8k baseline (0.9327)"
echo "  3. Best checkpoint: ${OUTDIR}/finetune_wced_21k/checkpoints/"
echo "============================================================"
