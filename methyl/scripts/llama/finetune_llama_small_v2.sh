#!/bin/bash -l
#SBATCH --job-name=finetune-llama-small-v2
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
# Fine-tuning MethylLlama-Small v2 — improved hyperparameters
#
# Changes vs v1 (44695049):
#   1. Warmup fixed: LambdaLR warmup+cosine (was CosineAnnealingLR, no warmup)
#   2. Scheduler fixed: encoder pre-registered, base_lr set at unfreeze (not add_param_group)
#   3. Pooling: cls (matches WCED pretraining objective) vs mean
#   4. encoder_lr: 5e-6 (vs 5e-5 — less aggressive backbone adaptation)
#   5. unfreeze_epoch: 30 (vs 10 — longer head warmup before encoder moves)
#   6. head_dropout: 0.2 (vs 0.1 — more regularization for 7k training samples)
#   7. loss_type: huber (vs mse — robust to age outliers, delta=5yr)
#   8. head simplified: 256→128→1 (vs 256→256→128→64→1)
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
# Fine-tuning settings — v2 improvements
# ─────────────────────────────────────────────────────────────────────────────
LR="${LR:-1e-4}"
ENCODER_LR="${ENCODER_LR:-5e-6}"         # 10× lower than v1 (5e-5 → 5e-6)
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
BATCH_SIZE="${BATCH_SIZE:-8}"
ACCUM="${ACCUM:-16}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-150}"
EARLY_STOP="${EARLY_STOP:-30}"
FREEZE_ENCODER="${FREEZE_ENCODER:-true}"
UNFREEZE_EPOCH="${UNFREEZE_EPOCH:-20}"    # later unfreeze: epoch 20 vs 10
WARMUP_STEPS="${WARMUP_STEPS:-500}"
RECON_WEIGHT="${RECON_WEIGHT:-0.0}"
HEAD_HIDDEN="${HEAD_HIDDEN:-256}"
HEAD_DROPOUT="${HEAD_DROPOUT:-0.2}"       # more regularization: 0.2 vs 0.1
POOLING="${POOLING:-cls}"                 # match WCED pretraining: cls vs mean
LOSS_TYPE="${LOSS_TYPE:-huber}"           # robust to age outliers: huber vs mse
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

# ─────────────────────────────────────────────────────────────────────────────
# WandB
# ─────────────────────────────────────────────────────────────────────────────
WANDB_ENTITY="netanelazran11-hebrew-university-of-jerusalem"
WANDB_PROJECT="finetune-llama-small"
WANDB_RUN_NAME="llama-small-ft-v2-${POOLING}-enc${ENCODER_LR}-uf${UNFREEZE_EPOCH}-${SLURM_JOB_ID}"

OUTROOT="${REPO}/outputs/${WANDB_PROJECT}"
OUTDIR="${OUTROOT}/${WANDB_RUN_NAME}"

mkdir -p "${LOGDIR}" "${OUTDIR}"

echo "============================================================"
echo "METHYLLAMA-SMALL FINE-TUNING v2 (improved hyperparams)"
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
echo "v2 fine-tuning finished: $(date)"
echo "Checkpoint: ${OUTDIR}/checkpoints/"
echo "============================================================"
