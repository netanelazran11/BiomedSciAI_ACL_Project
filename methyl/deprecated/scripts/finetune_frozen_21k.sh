#!/bin/bash -l
#SBATCH --job-name=finetune-frozen-21k
#SBATCH --partition=goldfish
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00

#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# -------------------------
# Paths
# -------------------------
REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
LOGDIR="${REPO}/logs"

DATA="/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_21k_h5ad/altumage_21k_combined.h5ad"
TOKENIZER_PATH="${REPO}/tokenizer_21k"

# Best 21k WCED pretrain checkpoint (epoch=270, val_loss=0.2574)
CHECKPOINT="${CHECKPOINT:-/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/outputs/pretrain-wced-21k/wced-21k-contrastive-w0.1-44223479/pretrain/checkpoints/epoch=epoch=270-val_loss=validation/loss=0.2574.ckpt}"

# -------------------------
# Method 1 style settings
# -------------------------
# 8000 fixed CpGs from the 21k vocabulary — within max_position_embeddings=8002
# Same CpGs every batch → stable gradients for the head
SUBSET_K="${SUBSET_K:-8000}"
MAX_LENGTH=8002   # 8000 + CLS + SEP

# -------------------------
# Fine-tuning hyperparameters
# -------------------------
LEARNING_RATE="${LEARNING_RATE:-1e-3}"       # Head only — can use aggressive LR
BATCH_SIZE="${BATCH_SIZE:-16}"
ACCUMULATE_GRAD="${ACCUMULATE_GRAD:-4}"      # Effective batch = 64
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-300}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-60}"
HEAD_DROPOUT="${HEAD_DROPOUT:-0.1}"

# -------------------------
# W&B
# -------------------------
WANDB_ENTITY="netanelazran11-hebrew-university-of-jerusalem"
WANDB_PROJECT="finetune-frozen-21k"
WANDB_RUN_NAME="frozen-21k-k${SUBSET_K}-${SLURM_JOB_ID}"

# -------------------------
# Output directory
# -------------------------
OUTROOT="${REPO}/outputs/${WANDB_PROJECT}"
OUTDIR="${OUTROOT}/${WANDB_RUN_NAME}"

mkdir -p "${LOGDIR}"
mkdir -p "${OUTDIR}"

echo "============================================================"
echo "FROZEN WCED-21k FINE-TUNING — Method 1 style"
echo "============================================================"
echo "Job started:   $(date)"
echo "Host:          $(hostname)"
echo "JobID:         ${SLURM_JOB_ID}"
echo "============================================================"
echo "Checkpoint:    ${CHECKPOINT}"
echo "Data:          ${DATA}"
echo "Tokenizer:     ${TOKENIZER_PATH}"
echo "CpG input:     ${SUBSET_K} FIXED (same every batch)"
echo "Max length:    ${MAX_LENGTH}"
echo "Encoder:       FROZEN — zero gradient"
echo "Head LR:       ${LEARNING_RATE}"
echo "Batch:         ${BATCH_SIZE} × ${ACCUMULATE_GRAD} = $((BATCH_SIZE * ACCUMULATE_GRAD)) effective"
echo "W&B project:   ${WANDB_PROJECT}"
echo "W&B run:       ${WANDB_RUN_NAME}"
echo "Output:        ${OUTDIR}"
echo "============================================================"
echo "Expected improvement: val/r2 > 0.9037 (Method 2 result)"
echo "Target:              val/r2 ~ 0.93+ (matching 8k Method 1)"
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
# Run Frozen Fine-tuning
# -------------------------
python -m bmfm_methylation.finetune_frozen_21k \
    data_path="${DATA}" \
    "checkpoint_path='${CHECKPOINT}'" \
    output_directory="${OUTDIR}" \
    tokenizer_path="${TOKENIZER_PATH}" \
    fields=methylation_21k \
    finetune_epochs=${FINETUNE_EPOCHS} \
    freeze_encoder=true \
    data_module.subset_k="${SUBSET_K}" \
    data_module.fixed_subset="true" \
    data_module.fixed_subset_seed="42" \
    data_module.max_length=${MAX_LENGTH} \
    data_module.batch_size=${BATCH_SIZE} \
    data_module.num_workers=4 \
    accumulate_grad_batches=${ACCUMULATE_GRAD} \
    trainer.learning_rate=${LEARNING_RATE} \
    regression_head.dropout=${HEAD_DROPOUT} \
    early_stopping.patience=${EARLY_STOP_PATIENCE} \
    track_wandb.enabled=true \
    track_wandb.project="${WANDB_PROJECT}" \
    track_wandb.entity="${WANDB_ENTITY}" \
    track_wandb.name="${WANDB_RUN_NAME}"

echo "============================================================"
echo "Frozen 21k fine-tuning finished: $(date)"
echo "============================================================"
echo "Checkpoint: ${OUTDIR}/finetune_frozen_21k/checkpoints/"
echo "============================================================"
echo "Compare val/r2 with:"
echo "  8k Method 1 (frozen + fixed): 0.9327"
echo "  21k Method 2 (WCED decoder):  0.9037  ← current best 21k"
echo "============================================================"
