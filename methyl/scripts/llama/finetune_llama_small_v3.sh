#!/bin/bash -l
#SBATCH --job-name=finetune-llama-small-v3
#SBATCH --partition=goldfish
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00

#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/%x_%j.out
#SBATCH --error=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/%x_%j.err

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Fine-tuning MethylLlama-Small v3 — bug fixes only, reverted bad changes
#
# V1 (44695049) baseline: test/r2=0.862, test/mae=6.81yr
# V2 (44705936) result:   test/r2=0.823, test/mae=7.50yr  ← worse
#
# V2 failures:
#   - CLS pooling hurt: CLS trained for reconstruction, not age; mean preserves beta signal
#   - head_dropout=0.2 caused underfitting (val≈train at epoch 150)
#   - encoder_lr=5e-6 too conservative, encoder barely adapted
#
# V3 keeps only the confirmed improvements:
#   1. Warmup fixed: LambdaLR warmup+cosine (was CosineAnnealingLR, no warmup)
#   2. Scheduler fixed: encoder pre-registered, base_lr updated at unfreeze
#   3. Head simplified: 256→256→128→1 (removed 64D bottleneck, kept from v2)
#   4. unfreeze_epoch=20 (more head warmup than v1's 10, kept from v2)
#   5. encoder_lr=1e-5 (middle ground: v1=5e-5 broken, v2=5e-6 too low)
#
# Reverted to v1 values:
#   - pooling: mean (CLS hurt performance)
#   - head_dropout: 0.1 (0.2 caused underfitting)
# ─────────────────────────────────────────────────────────────────────────────
REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
LOGDIR="${REPO}/logs_llama-wced"

DATA="/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_finetune_19k_h5ad/finetuning_19608_clean.h5ad"

CHECKPOINT="${CHECKPOINT:-${REPO}/outputs/pretrain-llama-wced/llama-small-all49k-r0.5-w0.0-44450919/checkpoints/epoch=98-val_loss=0.0059.ckpt}"
TOKENIZER_PATH="${REPO}/tokenizer_llama_pretrain49k"

# ─────────────────────────────────────────────────────────────────────────────
# Data settings
# ─────────────────────────────────────────────────────────────────────────────
SUBSET_K="${SUBSET_K:-49156}"
INPUT_RATIO="${INPUT_RATIO:-1.0}"

# ─────────────────────────────────────────────────────────────────────────────
# Fine-tuning settings — v3
# ─────────────────────────────────────────────────────────────────────────────
LR="${LR:-1e-4}"
ENCODER_LR="${ENCODER_LR:-1e-5}"         # middle ground: v1=5e-5 (broken), v2=5e-6 (too low)
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
BATCH_SIZE="${BATCH_SIZE:-8}"
ACCUM="${ACCUM:-16}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-150}"
EARLY_STOP="${EARLY_STOP:-30}"
FREEZE_ENCODER="${FREEZE_ENCODER:-true}"
UNFREEZE_EPOCH="${UNFREEZE_EPOCH:-20}"    # kept from v2: more head warmup than v1's 10
WARMUP_STEPS="${WARMUP_STEPS:-500}"       # kept from v2: properly implemented now
RECON_WEIGHT="${RECON_WEIGHT:-0.0}"
HEAD_HIDDEN="${HEAD_HIDDEN:-256}"
HEAD_DROPOUT="${HEAD_DROPOUT:-0.1}"       # reverted: 0.2 caused underfitting in v2
POOLING="${POOLING:-mean}"                # reverted: CLS hurt performance in v2
LOSS_TYPE="${LOSS_TYPE:-mse}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

# ─────────────────────────────────────────────────────────────────────────────
# WandB
# ─────────────────────────────────────────────────────────────────────────────
WANDB_ENTITY="netanelazran11-hebrew-university-of-jerusalem"
WANDB_PROJECT="finetune-llama-small"
WANDB_RUN_NAME="llama-small-ft-v3-${POOLING}-enc${ENCODER_LR}-uf${UNFREEZE_EPOCH}-${SLURM_JOB_ID}"

OUTROOT="${REPO}/outputs/${WANDB_PROJECT}"
OUTDIR="${OUTROOT}/${WANDB_RUN_NAME}"

mkdir -p "${LOGDIR}" "${OUTDIR}"

echo "============================================================"
echo "METHYLLAMA-SMALL FINE-TUNING v3 (bug fixes, reverted bad changes)"
echo "============================================================"
echo "Job: ${SLURM_JOB_ID} | Host: $(hostname) | Time: $(date)"
echo "Checkpoint: ${CHECKPOINT}"
echo "Pooling: ${POOLING} | Loss: ${LOSS_TYPE}"
echo "encoder_lr=${ENCODER_LR} | unfreeze_epoch=${UNFREEZE_EPOCH}"
echo "head_dropout=${HEAD_DROPOUT} | warmup_steps=${WARMUP_STEPS}"
echo "Output: ${OUTDIR}"
echo "============================================================"

# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────
source /etc/profile.d/modules.sh 2>/dev/null || source /usr/share/modules/init/bash 2>/dev/null || true
module purge
module load spack/all
module load cuda/12.3.2-gcc-5bv3kyh

cd "${REPO}"
source bmfm_methyl_env/bin/activate

export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# ─────────────────────────────────────────────────────────────────────────────
# Fine-tuning
# ─────────────────────────────────────────────────────────────────────────────
python -m bmfm_methylation.llama.finetune_llama \
    "data_path='${DATA}'" \
    "checkpoint_path='${CHECKPOINT}'" \
    "tokenizer_path='${TOKENIZER_PATH}'" \
    "output_directory='${OUTDIR}'" \
    data_module.subset_k="${SUBSET_K}" \
    data_module.fixed_subset_seed=42 \
    data_module.max_length=19609 \
    data_module.batch_size="${BATCH_SIZE}" \
    data_module.num_workers=8 \
    wced_input_ratio="${INPUT_RATIO}" \
    finetune.head_hidden_size="${HEAD_HIDDEN}" \
    finetune.head_dropout="${HEAD_DROPOUT}" \
    finetune.learning_rate="${LR}" \
    finetune.encoder_lr="${ENCODER_LR}" \
    finetune.weight_decay="${WEIGHT_DECAY}" \
    finetune.warmup_steps="${WARMUP_STEPS}" \
    finetune.freeze_encoder="${FREEZE_ENCODER}" \
    finetune.unfreeze_encoder_epoch="${UNFREEZE_EPOCH}" \
    finetune.recon_weight="${RECON_WEIGHT}" \
    finetune.pooling="${POOLING}" \
    finetune.loss_type="${LOSS_TYPE}" \
    finetune_epochs="${FINETUNE_EPOCHS}" \
    accumulate_grad_batches="${ACCUM}" \
    gradient_clip_val=1.0 \
    early_stop_patience="${EARLY_STOP}" \
    precision="16-mixed" \
    track_wandb.enabled=true \
    track_wandb.project="${WANDB_PROJECT}" \
    track_wandb.entity="${WANDB_ENTITY}" \
    track_wandb.name="${WANDB_RUN_NAME}" \
    ${RESUME_CHECKPOINT:+"resume_checkpoint='${RESUME_CHECKPOINT}'"}

echo "============================================================"
echo "v3 fine-tuning finished: $(date)"
echo "Checkpoint: ${OUTDIR}/checkpoints/"
echo "============================================================"
