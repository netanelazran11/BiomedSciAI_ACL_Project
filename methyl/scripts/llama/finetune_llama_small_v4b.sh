#!/bin/bash -l
#SBATCH --job-name=finetune-llama-small-v4b
#SBATCH --partition=goldfish
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00

#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/%x_%j.out
#SBATCH --error=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/%x_%j.err

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Fine-tuning MethylLlama-Small V4b — warmstart from V4 best checkpoint
#
# V4 results (44744875): test/mae=6.49yr, test/medae=4.68yr, test/r2=0.881
#   - Best val/mae: 6.307yr @ epoch 117
#   - Best val/medae: 4.426yr @ epoch 88
#   - LR hit cosine floor at epoch ~145 → no room to learn in final epochs
#
# V4b strategy: warmstart (weights only, no optimizer) + fresh 300-epoch LR
#
# Changes from V4:
#   1. WARMSTART_WEIGHTS: load V4 epoch-117 weights, fresh optimizer from epoch 0
#      (vs RESUME_CHECKPOINT which inherits dead LR from cosine floor)
#   2. FINETUNE_EPOCHS 150→300: full cosine schedule over 300 epochs
#   3. LOSS_TYPE mse→huber: delta=5yr/age_std≈0.186 z-scores (correctly computed)
#      Errors <5yr: quadratic (same as MSE). Errors >5yr: linear (down-weights outliers)
#   4. EARLY_STOP 50→100: more room for mid-training plateaus with 300-epoch budget
#   5. WARMUP_STEPS 100→500: slower LR ramp-in to protect V4's already-trained weights
#   6. DATA: outlier-free file (298 samples removed: blood whole batch effect + extremes)
#
# Unchanged from V4:
#   - Architecture: 256D × 4L × 4H, RoPE, SwiGLU, RMSNorm
#   - Pooling: mean over all CpG tokens
#   - LR: 1e-4 (head), 2e-5 (encoder after unfreeze)
#   - Batch: 32 × accum 4 = eff batch 128
#   - Unfreeze encoder: epoch 10
#   - Weight decay: 0.01
#   - All 19,608 CpGs used (SUBSET_K=49156 > n_cpgs → no subsampling)
# ─────────────────────────────────────────────────────────────────────────────

REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
LOGDIR="${REPO}/logs_llama-wced"

# Outlier-free stratified split (298 samples with mean_beta >3σ removed)
DATA="/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_finetune_19k_h5ad/finetuning_19608_clean_stratified_no_outliers.h5ad"

# WCED pretrain checkpoint (used to reconstruct encoder architecture only)
CHECKPOINT="${CHECKPOINT:-${REPO}/outputs/pretrain-llama-wced/llama-small-all49k-r0.5-w0.0-44450919/checkpoints/epoch=98-val_loss=0.0059.ckpt}"
TOKENIZER_PATH="${REPO}/tokenizer_llama_pretrain49k"

# ─────────────────────────────────────────────────────────────────────────────
# Data settings — unchanged from V4
# ─────────────────────────────────────────────────────────────────────────────
SUBSET_K="${SUBSET_K:-49156}"      # > 19608 CpGs in file → uses ALL CpGs, no subsampling
INPUT_RATIO="${INPUT_RATIO:-1.0}"  # all CpGs fed as input (no masking)

# ─────────────────────────────────────────────────────────────────────────────
# V4b hyperparameters
# ─────────────────────────────────────────────────────────────────────────────
LR="${LR:-1e-4}"
ENCODER_LR="${ENCODER_LR:-2e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
BATCH_SIZE="${BATCH_SIZE:-32}"
ACCUM="${ACCUM:-4}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-300}"         # V4b: 150→300 (full cosine over 300 epochs)
EARLY_STOP="${EARLY_STOP:-100}"                   # V4b: 50→100 (more room with 300-epoch budget)
FREEZE_ENCODER="${FREEZE_ENCODER:-true}"
UNFREEZE_EPOCH="${UNFREEZE_EPOCH:-10}"
WARMUP_STEPS="${WARMUP_STEPS:-500}"               # V4b: 100→500 (protect trained weights during LR ramp-in)
RECON_WEIGHT="${RECON_WEIGHT:-0.0}"
HEAD_HIDDEN="${HEAD_HIDDEN:-256}"
HEAD_DROPOUT="${HEAD_DROPOUT:-0.0}"
POOLING="${POOLING:-mean}"
LOSS_TYPE="${LOSS_TYPE:-huber}"                   # V4b: mse→huber (delta=5yr/age_std, corrected in code)
BETA_NOISE="${BETA_NOISE:-0.0}"

# Warmstart: V4 best val/mae checkpoint — weights only, no optimizer state
# Fresh optimizer gives full LR from epoch 0 (vs RESUME_CHECKPOINT which inherits dead LR)
# Default: warmstart from V4 best checkpoint. Override with WARMSTART_WEIGHTS="" to train from scratch.
WARMSTART_WEIGHTS="${WARMSTART_WEIGHTS-${REPO}/outputs/finetune-llama-small/llama-small-ft-v4-b32-uf10-enc2e-5-44744875/checkpoints/epoch=117-val_mae=6.3071.ckpt}"

RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
EVAL_CHECKPOINT="${EVAL_CHECKPOINT:-}"

# ─────────────────────────────────────────────────────────────────────────────
# WandB
# ─────────────────────────────────────────────────────────────────────────────
WANDB_ENTITY="netanelazran11-hebrew-university-of-jerusalem"
WANDB_PROJECT="finetune-llama-small"
WS_TAG=$( [ -n "${WARMSTART_WEIGHTS}" ] && echo "ws" || echo "scratch" )
WANDB_RUN_NAME="llama-small-ft-v4b-huber-ep${FINETUNE_EPOCHS}-wu${WARMUP_STEPS}-${WS_TAG}-${SLURM_JOB_ID}"

OUTROOT="${REPO}/outputs/${WANDB_PROJECT}"
OUTDIR="${OUTROOT}/${WANDB_RUN_NAME}"

mkdir -p "${LOGDIR}" "${OUTDIR}"

echo "============================================================"
echo "METHYLLAMA-SMALL FINE-TUNING V4b (warmstart from V4)"
echo "============================================================"
echo "Job: ${SLURM_JOB_ID} | Host: $(hostname) | Time: $(date)"
echo "Warmstart weights: ${WARMSTART_WEIGHTS}"
echo "Loss: ${LOSS_TYPE} | Pooling: ${POOLING} | beta_noise=${BETA_NOISE}"
echo "epochs=${FINETUNE_EPOCHS} | early_stop=${EARLY_STOP} | warmup=${WARMUP_STEPS} steps"
echo "batch=${BATCH_SIZE}×${ACCUM}=$(( BATCH_SIZE * ACCUM )) eff"
echo "lr=${LR} | encoder_lr=${ENCODER_LR} | unfreeze_epoch=${UNFREEZE_EPOCH}"
echo "Data: ${DATA}"
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
    finetune.beta_noise="${BETA_NOISE}" \
    finetune_epochs="${FINETUNE_EPOCHS}" \
    accumulate_grad_batches="${ACCUM}" \
    gradient_clip_val=1.0 \
    early_stop_patience="${EARLY_STOP}" \
    precision="16-mixed" \
    track_wandb.enabled=true \
    track_wandb.project="${WANDB_PROJECT}" \
    track_wandb.entity="${WANDB_ENTITY}" \
    track_wandb.name="${WANDB_RUN_NAME}" \
    ${WARMSTART_WEIGHTS:+"+warmstart_weights_path='${WARMSTART_WEIGHTS}'"} \
    ${RESUME_CHECKPOINT:+"+resume_checkpoint='${RESUME_CHECKPOINT}'"} \
    ${EVAL_CHECKPOINT:+"+eval_checkpoint='${EVAL_CHECKPOINT}'"}

echo "============================================================"
echo "V4b fine-tuning finished: $(date)"
echo "Checkpoints: ${OUTDIR}/checkpoints/"
echo "============================================================"
