#!/bin/bash -l
#SBATCH --job-name=finetune-llama-small
#SBATCH --partition=goldfish
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/%x_%j.out
#SBATCH --error=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/%x_%j.err

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Fine-tuning MethylLlama-Small for age prediction.
# Requires a checkpoint from pretrain_llama_small.sh.
# ─────────────────────────────────────────────────────────────────────────────
REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
LOGDIR="${REPO}/logs_llama-wced"

DATA="/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_finetune_49k_h5ad/finetuning_49k.h5ad"

# Pretrained small-model checkpoint (h256_l4, epoch 72, val_loss=0.0061)
CHECKPOINT="${CHECKPOINT:-${REPO}/outputs/pretrain-llama-wced/llama-small-all49k-r0.5-w0.0-44450919/checkpoints/epoch=98-val_loss=0.0059.ckpt}"

# Same tokenizer as pretrain — CpG ID→index mapping must be consistent
TOKENIZER_PATH="${REPO}/tokenizer_llama_pretrain49k"

# ─────────────────────────────────────────────────────────────────────────────
# Data settings — use ALL 49k CpGs, match pretraining input ratio
# ─────────────────────────────────────────────────────────────────────────────
SUBSET_K="${SUBSET_K:-49156}"
INPUT_RATIO="${INPUT_RATIO:-0.5}"    # All valid CpGs go to input (60% NaN → only 19608 valid < 24578 requested)

# Age normalization is computed automatically from the training split by
# MethylationDataset — no need to hardcode mean/std here.

# ─────────────────────────────────────────────────────────────────────────────
# Fine-tuning settings
# ─────────────────────────────────────────────────────────────────────────────
LR="${LR:-1e-4}"
ENCODER_LR="${ENCODER_LR:-5e-5}"       # encoder lr after unfreeze (5x lower than head)
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
BATCH_SIZE="${BATCH_SIZE:-64}"
ACCUM="${ACCUM:-2}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-150}"
EARLY_STOP="${EARLY_STOP:-30}"
FREEZE_ENCODER="${FREEZE_ENCODER:-true}"
UNFREEZE_EPOCH="${UNFREEZE_EPOCH:-10}"   # unfreeze encoder after head warmup
RECON_WEIGHT="${RECON_WEIGHT:-0.0}"      # disabled: input_ratio=1.0 leaves no recon targets
INPUT_RATIO="${INPUT_RATIO:-1.0}"        # use ALL valid CpGs as input
HEAD_HIDDEN="${HEAD_HIDDEN:-512}"        # wider head: 256→512→256→128→1
POOLING="${POOLING:-mean}"               # mean pooling over all tokens (better than CLS-only)
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

# ─────────────────────────────────────────────────────────────────────────────
# WandB
# ─────────────────────────────────────────────────────────────────────────────
WANDB_ENTITY="netanelazran11-hebrew-university-of-jerusalem"
WANDB_PROJECT="finetune-llama-small"
WANDB_RUN_NAME="llama-small-ft-freeze${FREEZE_ENCODER}-${SLURM_JOB_ID}"

OUTROOT="${REPO}/outputs/${WANDB_PROJECT}"
OUTDIR="${OUTROOT}/${WANDB_RUN_NAME}"

mkdir -p "${LOGDIR}" "${OUTDIR}"

echo "============================================================"
echo "METHYLLAMA-SMALL FINE-TUNING (age prediction)"
echo "============================================================"
echo "Job: ${SLURM_JOB_ID} | Host: $(hostname) | Time: $(date)"
echo "Checkpoint: ${CHECKPOINT}"
echo "Strategy: freeze_encoder=${FREEZE_ENCODER}, recon_weight=${RECON_WEIGHT}"
echo "LR=${LR}, batch=${BATCH_SIZE}×${ACCUM}=$(( BATCH_SIZE * ACCUM )) eff"
echo "Age norm: auto-computed from training split"
echo "Output: ${OUTDIR}"
echo "Resume: ${RESUME_CHECKPOINT:-none}"
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
    data_module.max_length=$(python3 -c "print(int(${SUBSET_K} * ${INPUT_RATIO}) + 1)") \
    data_module.batch_size="${BATCH_SIZE}" \
    data_module.num_workers=8 \
    wced_input_ratio="${INPUT_RATIO}" \
    finetune.head_hidden_size="${HEAD_HIDDEN}" \
    finetune.learning_rate="${LR}" \
    finetune.encoder_lr="${ENCODER_LR}" \
    finetune.weight_decay="${WEIGHT_DECAY}" \
    finetune.freeze_encoder="${FREEZE_ENCODER}" \
    finetune.unfreeze_encoder_epoch="${UNFREEZE_EPOCH}" \
    finetune.recon_weight="${RECON_WEIGHT}" \
    finetune.pooling="${POOLING}" \
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
echo "Small model fine-tuning finished: $(date)"
echo "Checkpoint: ${OUTDIR}/checkpoints/"
echo "============================================================"
