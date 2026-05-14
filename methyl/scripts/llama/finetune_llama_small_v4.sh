#!/bin/bash -l
#SBATCH --job-name=finetune-llama-small-v4
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
# Fine-tuning MethylLlama-Small v4 — fixes the three root causes of v3 regression
#
# Results:
#   V1 (44695049): test/r2=0.862, test/mae=6.81yr  ← baseline
#   V2 (44705936): test/r2=0.823, test/mae=7.50yr  ← worse (CLS pooling, dropout 0.2)
#   V3:            worse than V1  ← 3 root causes (see below)
#
# V3 root causes fixed in v4:
#   1. warmup_steps 500→100: 500 steps = 9 epochs for a fresh head starting from
#      random init. V1 used CosineAnnealingLR with no warmup (immediate full LR).
#      Fresh layers need aggressive LR; warmup is for pretrained models.
#
#   2. unfreeze_epoch 20→10: with 150 epochs budget, encoder only adapted for
#      150-20=130 epochs in v3. Also, cosine LR has decayed significantly by
#      epoch 20, so the encoder gets a much lower effective LR than intended.
#
#   3. beta_noise 0.02→0.0: clean data (0 NaN confirmed) needs no noise.
#      Train-time noise creates a distribution mismatch at inference time.
#
# Kept from v3 (correct):
#   - pooling=mean (CLS hurt in v2)
#   - head_dropout=0.1 (0.2 caused underfitting)
#   - num_attention_heads=4 fix in loader
#   - warmup_steps for scheduler (reduced, not removed)
#   - early_stop patience=50
#   - LambdaLR warmup+cosine scheduler
#
# Exploit H200 VRAM (80GB):
#   - batch_size 8→32: 4× better gradient quality per step, fits easily in 80GB
#   - accum 16→4: same effective batch (128), but 4× faster (real batch=32 vs 8)
#   - encoder_lr 1e-5→2e-5: correct scheduler now works; v1's 5e-5 was with
#     broken scheduler so effective LR was lower — 2e-5 is a principled middle.
# ─────────────────────────────────────────────────────────────────────────────
REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
LOGDIR="${REPO}/logs_llama-wced"

# Stratified split with outliers removed (298 samples with mean_beta >3σ removed)
DATA="/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_finetune_19k_h5ad/finetuning_19608_clean_stratified_no_outliers.h5ad"

CHECKPOINT="${CHECKPOINT:-${REPO}/outputs/pretrain-llama-wced/llama-small-all49k-r0.5-w0.0-44450919/checkpoints/epoch=98-val_loss=0.0059.ckpt}"
TOKENIZER_PATH="${REPO}/tokenizer_llama_pretrain49k"

# ─────────────────────────────────────────────────────────────────────────────
# Data settings
# ─────────────────────────────────────────────────────────────────────────────
SUBSET_K="${SUBSET_K:-49156}"
INPUT_RATIO="${INPUT_RATIO:-1.0}"

# ─────────────────────────────────────────────────────────────────────────────
# Fine-tuning settings — v4
# ─────────────────────────────────────────────────────────────────────────────
LR="${LR:-1e-4}"
ENCODER_LR="${ENCODER_LR:-2e-5}"         # FIX: v1=5e-5 broken sched, v3=1e-5 too low → 2e-5
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
BATCH_SIZE="${BATCH_SIZE:-32}"            # FIX: 8→32 (H200 has 80GB, fits easily)
ACCUM="${ACCUM:-4}"                       # FIX: 16→4  (eff batch stays 128, 4× faster)
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-150}"
EARLY_STOP="${EARLY_STOP:-50}"            # keep from v3
FREEZE_ENCODER="${FREEZE_ENCODER:-true}"
UNFREEZE_EPOCH="${UNFREEZE_EPOCH:-10}"    # FIX: 20→10 (more encoder adaptation time)
WARMUP_STEPS="${WARMUP_STEPS:-100}"       # FIX: 500→100 (~2 epochs, not 9)
RECON_WEIGHT="${RECON_WEIGHT:-0.0}"
HEAD_HIDDEN="${HEAD_HIDDEN:-256}"
HEAD_DROPOUT="${HEAD_DROPOUT:-0.0}"       # no dropout: weight decay (now correctly applied) is sufficient
POOLING="${POOLING:-mean}"                # keep from v3
LOSS_TYPE="${LOSS_TYPE:-mse}"
BETA_NOISE="${BETA_NOISE:-0.0}"           # FIX: 0.02→0.0  (clean data, no train/test mismatch)
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
# Warmstart: load weights from a finetune checkpoint WITHOUT restoring optimizer/epoch.
# Use this instead of RESUME_CHECKPOINT when extending training after cosine LR decay:
#   RESUME_CHECKPOINT → inherits near-zero LR from cosine floor (almost no learning)
#   WARMSTART_WEIGHTS → fresh optimizer (full LR from epoch 0) + trained weights (V4b strategy)
WARMSTART_WEIGHTS="${WARMSTART_WEIGHTS:-}"

# ─────────────────────────────────────────────────────────────────────────────
# WandB
# ─────────────────────────────────────────────────────────────────────────────
WANDB_ENTITY="netanelazran11-hebrew-university-of-jerusalem"
WANDB_PROJECT="finetune-llama-small"
WANDB_RUN_NAME="llama-small-ft-v4-b${BATCH_SIZE}-uf${UNFREEZE_EPOCH}-enc${ENCODER_LR}-${SLURM_JOB_ID}"

OUTROOT="${REPO}/outputs/${WANDB_PROJECT}"
OUTDIR="${OUTROOT}/${WANDB_RUN_NAME}"

mkdir -p "${LOGDIR}" "${OUTDIR}"

echo "============================================================"
echo "METHYLLAMA-SMALL FINE-TUNING v4 (root cause fixes)"
echo "============================================================"
echo "Job: ${SLURM_JOB_ID} | Host: $(hostname) | Time: $(date)"
echo "Checkpoint: ${CHECKPOINT}"
echo "Pooling: ${POOLING} | Loss: ${LOSS_TYPE} | beta_noise=${BETA_NOISE}"
[ -n "${WARMSTART_WEIGHTS}" ] && echo "WARMSTART weights: ${WARMSTART_WEIGHTS}"
[ -n "${RESUME_CHECKPOINT}" ] && echo "RESUME from: ${RESUME_CHECKPOINT}"
echo "batch=${BATCH_SIZE}×${ACCUM}=$(( BATCH_SIZE * ACCUM )) eff | warmup=${WARMUP_STEPS} steps"
echo "encoder_lr=${ENCODER_LR} | unfreeze_epoch=${UNFREEZE_EPOCH} | patience=${EARLY_STOP}"
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
    ${RESUME_CHECKPOINT:+"resume_checkpoint='${RESUME_CHECKPOINT}'"} \
    ${WARMSTART_WEIGHTS:+"warmstart_weights_path='${WARMSTART_WEIGHTS}'"}

echo "============================================================"
echo "v4 fine-tuning finished: $(date)"
echo "Checkpoint: ${OUTDIR}/checkpoints/"
echo "============================================================"
