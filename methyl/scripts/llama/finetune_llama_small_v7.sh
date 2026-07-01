#!/bin/bash -l
#SBATCH --job-name=finetune-llama-small-v7
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
# Fine-tuning MethylLlama V7 — from 6L pretrain checkpoint
#
# Pretrain: pretrain_llama_small_6L_contrastive.sh
#   Architecture: 256D × 6L × 4H, FFN=512
#   Training:     6L, InfoNCE w=0.05, genomic RoPE ordering
#
# V7 changes vs V5 (the current best: test/R²=0.905, MedAE=3.65yr):
#   - New pretrain checkpoint (6L instead of 4L)
#   - Architecture auto-inferred from checkpoint by load_wced_llama_checkpoint
#     (hidden_size // 64 fix in finetune_llama.py ensures 4 heads, not 8)
#
# All fine-tuning hyperparameters unchanged from V5 for fair comparison.
# ─────────────────────────────────────────────────────────────────────────────

REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
LOGDIR="${REPO}/logs_llama-wced"

DATA="/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_21k_h5ad/altumage_21k_3way.h5ad"

# ── Set CHECKPOINT to the best epoch from the 6L pretrain run ────────────────
# Pattern: outputs/pretrain-llama-wced/llama-6L-all49k-r0.5-w0.05-genomic-<JOB_ID>/checkpoints/epoch=XX-val_loss=X.XXXX.ckpt
# Override at submit time: CHECKPOINT=/path/to/ckpt sbatch finetune_llama_small_v7.sh
CHECKPOINT="${CHECKPOINT:?ERROR: set CHECKPOINT to the 6L pretrain checkpoint path}"

TOKENIZER_PATH="${REPO}/tokenizer_llama_pretrain49k"

# ─── Data settings (same as V5) ───────────────────────────────────────────────
SUBSET_K="${SUBSET_K:-49156}"
INPUT_RATIO="${INPUT_RATIO:-1.0}"

# ─── Fine-tuning hyperparameters (identical to V5 for fair comparison) ────────
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
POOLING="${POOLING:-cls}"
LOSS_TYPE="${LOSS_TYPE:-huber}"
BETA_NOISE="${BETA_NOISE:-0.0}"

WARMSTART_WEIGHTS="${WARMSTART_WEIGHTS:-}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
EVAL_CHECKPOINT="${EVAL_CHECKPOINT:-}"

# ─── WandB ───────────────────────────────────────────────────────────────────
WANDB_ENTITY="netanelazran11-hebrew-university-of-jerusalem"
WANDB_PROJECT="finetune-llama-small"
WS_TAG=$( [ -n "${WARMSTART_WEIGHTS}" ] && echo "ws" || echo "scratch" )
WANDB_RUN_NAME="llama-small-ft-v7-6L-cls-huber-ep${FINETUNE_EPOCHS}-wu${WARMUP_STEPS}-${WS_TAG}-${SLURM_JOB_ID}"

OUTROOT="${REPO}/outputs/${WANDB_PROJECT}"
OUTDIR="${OUTROOT}/${WANDB_RUN_NAME}"

mkdir -p "${LOGDIR}" "${OUTDIR}"

echo "============================================================"
echo "METHYLLAMA V7 FINE-TUNING (from 6L pretrain)"
echo "============================================================"
echo "Job: ${SLURM_JOB_ID} | Host: $(hostname) | Time: $(date)"
echo "Pretrain ckpt: ${CHECKPOINT}"
echo "Pooling: ${POOLING} | Loss: ${LOSS_TYPE}"
echo "epochs=${FINETUNE_EPOCHS} | early_stop=${EARLY_STOP} | warmup=${WARMUP_STEPS}"
echo "batch=${BATCH_SIZE}×${ACCUM}=$(( BATCH_SIZE * ACCUM )) eff"
echo "lr=${LR} | encoder_lr=${ENCODER_LR} | unfreeze_epoch=${UNFREEZE_EPOCH}"
echo "Data: ${DATA}"
echo "Output: ${OUTDIR}"
echo "============================================================"

# ─── Environment ─────────────────────────────────────────────────────────────
source /etc/profile.d/modules.sh 2>/dev/null || source /usr/share/modules/init/bash 2>/dev/null || true
module purge
module load spack/all
module load cuda/12.3.2-gcc-5bv3kyh

cd "${REPO}"
source bmfm_methyl_env/bin/activate

export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# ─── Fine-tuning ─────────────────────────────────────────────────────────────
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
echo "V7 fine-tuning finished: $(date)"
echo "Checkpoints: ${OUTDIR}/checkpoints/"
echo "============================================================"
