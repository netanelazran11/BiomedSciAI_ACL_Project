#!/bin/bash -l
#SBATCH --job-name=finetune-llama-small-v5
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
# Fine-tuning MethylLlama-Small V5 — CLS pooling (vs V4b mean pooling)
#
# Motivation: WCED pretraining routes all global signal through CLS (pooler_output).
#   Mean pooling (V4b) reads token embeddings that were NOT trained for global tasks.
#   CLS pooling reads pooler_output = Linear(CLS)+Tanh — the exact readout used
#   during pretraining for reconstruction, age supervision, and InfoNCE.
#
# V4b reference (44884146): test/r2=0.919, test/mae=5.01yr @ epoch 79 (still running)
#   - Pooling: mean over all CpG tokens
#
# V5 strategy: scratch from WCED checkpoint + CLS pooling
#   Starting from scratch (not warmstart from V4b) gives a clean comparison.
#
# Changes from V4b:
#   1. POOLING: mean → cls  (pooler_output: Linear(CLS)+Tanh)
#   2. No warmstart (from WCED pretrain checkpoint, not V4b weights)
#      Fresh start isolates the effect of pooling change.
#
# Unchanged from V4b:
#   - Architecture: 256D × 4L × 4H, RoPE, SwiGLU, RMSNorm
#   - Loss: Huber (delta=5yr/age_std)
#   - LR: 1e-4 (head), 2e-5 (encoder after unfreeze)
#   - Batch: 32 × accum 4 = eff batch 128
#   - Epochs: 300, early stop patience: 100
#   - Unfreeze encoder: epoch 10, warmup: 500 steps
#   - Data: outlier-free stratified split (298 samples removed)
#   - All 19,608 CpGs used
# ─────────────────────────────────────────────────────────────────────────────

REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
LOGDIR="${REPO}/logs_llama-wced"

DATA="/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_21k_h5ad/altumage_21k_3way.h5ad"

CHECKPOINT="${CHECKPOINT:-${REPO}/outputs/pretrain-llama-wced/llama-small-all49k-r0.5-w0.0-44450919/checkpoints/epoch=98-val_loss=0.0059.ckpt}"
TOKENIZER_PATH="${REPO}/tokenizer_llama_pretrain49k"

# ─────────────────────────────────────────────────────────────────────────────
# Data settings
# ─────────────────────────────────────────────────────────────────────────────
SUBSET_K="${SUBSET_K:-49156}"
INPUT_RATIO="${INPUT_RATIO:-1.0}"

# ─────────────────────────────────────────────────────────────────────────────
# V5 hyperparameters
# ─────────────────────────────────────────────────────────────────────────────
LR="${LR:-1e-4}"
ENCODER_LR="${ENCODER_LR:-2e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
BATCH_SIZE="${BATCH_SIZE:-32}"
ACCUM="${ACCUM:-4}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-300}"
EARLY_STOP="${EARLY_STOP:-100}"
FREEZE_ENCODER="${FREEZE_ENCODER:-true}"
UNFREEZE_EPOCH="${UNFREEZE_EPOCH:-10}"
WARMUP_STEPS="${WARMUP_STEPS:-500}"
RECON_WEIGHT="${RECON_WEIGHT:-0.0}"
HEAD_HIDDEN="${HEAD_HIDDEN:-256}"
HEAD_DROPOUT="${HEAD_DROPOUT:-0.0}"
POOLING="${POOLING:-cls}"           # V5: cls pooler_output (matches WCED pretraining)
LOSS_TYPE="${LOSS_TYPE:-huber}"
BETA_NOISE="${BETA_NOISE:-0.0}"

WARMSTART_WEIGHTS="${WARMSTART_WEIGHTS:-}"   # no warmstart — clean scratch run
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
EVAL_CHECKPOINT="${EVAL_CHECKPOINT:-}"

# ─────────────────────────────────────────────────────────────────────────────
# WandB
# ─────────────────────────────────────────────────────────────────────────────
WANDB_ENTITY="netanelazran11-hebrew-university-of-jerusalem"
WANDB_PROJECT="finetune-llama-small"
WS_TAG=$( [ -n "${WARMSTART_WEIGHTS}" ] && echo "ws" || echo "scratch" )
WANDB_RUN_NAME="llama-small-ft-v5-cls-huber-ep${FINETUNE_EPOCHS}-wu${WARMUP_STEPS}-${WS_TAG}-${SLURM_JOB_ID}"

OUTROOT="${REPO}/outputs/${WANDB_PROJECT}"
OUTDIR="${OUTROOT}/${WANDB_RUN_NAME}"

mkdir -p "${LOGDIR}" "${OUTDIR}"

echo "============================================================"
echo "METHYLLAMA-SMALL FINE-TUNING V5 (CLS pooling)"
echo "============================================================"
echo "Job: ${SLURM_JOB_ID} | Host: $(hostname) | Time: $(date)"
echo "Pooling: ${POOLING} (CLS pooler_output — matches WCED pretraining)"
echo "Loss: ${LOSS_TYPE} | beta_noise=${BETA_NOISE}"
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
    data_module.max_length=21369 \
    data_module.batch_size="${BATCH_SIZE}" \
    data_module.num_workers=8 \
    data_module.filter_age_outliers=true \
    "data_module.duplicate_pairs_csv='${REPO}/dataset_fingerprint_outputs/duplicate_pairs.csv'" \
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
echo "V5 fine-tuning finished: $(date)"
echo "Checkpoints: ${OUTDIR}/checkpoints/"
echo "============================================================"
