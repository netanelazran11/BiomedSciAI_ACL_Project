#!/bin/bash -l
#SBATCH --job-name=pretrain-wced-21k
#SBATCH --partition=goldfish
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=72:00:00

#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# -------------------------
# Paths
# -------------------------
REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
LOGDIR="${REPO}/logs"

# 21k h5ad — upload this file to the cluster before running
DATA="/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_21k_h5ad/altumage_21k_combined.h5ad"

# New tokenizer directory (auto-built from 21k CpGs on first run)
TOKENIZER_PATH="${REPO}/tokenizer_21k"

# -------------------------
# 21k-specific settings
# -------------------------
# Use ALL 21,368 CpGs as vocabulary (decoder reconstructs all of them)
# Set higher than actual CpG count so WCEDCollator uses every CpG
SUBSET_K="${SUBSET_K:-99999}"

# Input ratio: 4000 / 21368 = 0.187
# → each view sees ~4,000 CpGs (same sequence length as 8k pretraining)
# → views are non-overlapping (remaining 81% >> 18.7% needed)
# → decoder still reconstructs ALL 21,368 CpGs from each view
INPUT_RATIO="${INPUT_RATIO:-0.187}"

# Position embeddings: 21368 * 0.187 + 1 (CLS) ≈ 4,000 + 1 = 4,001
# Same as 8k pretraining (which also had 4,000 tokens per view)
# Set slightly higher for safety
MAX_POS_EMB=8002

# Same batch as 8k pretraining (same sequence length, same memory)
BATCH_SIZE="${BATCH_SIZE:-32}"
ACCUMULATE_GRAD="${ACCUMULATE_GRAD:-4}"

# -------------------------
# WCED settings (same as successful 8k run)
# -------------------------
COMBINE_STYLE="${COMBINE_STYLE:-add}"
AGE_WEIGHT="${AGE_WEIGHT:-1.0}"
CONTRASTIVE="${CONTRASTIVE:-true}"
CONTRASTIVE_WEIGHT="${CONTRASTIVE_WEIGHT:-0.1}"
CONTRASTIVE_TEMP="${CONTRASTIVE_TEMP:-0.1}"
NORMALIZE_LOSS="${NORMALIZE_LOSS:-true}"
WCED_DECODER_DROPOUT="0.1"

# -------------------------
# Architecture (same as 8k run)
# -------------------------
HIDDEN_SIZE="${HIDDEN_SIZE:-512}"
NUM_ATTENTION_HEADS="${NUM_ATTENTION_HEADS:-8}"
INTERMEDIATE_SIZE="${INTERMEDIATE_SIZE:-2048}"

# -------------------------
# Training schedule
# -------------------------
PRETRAIN_EPOCHS="300"
EARLY_STOP_PATIENCE="20"

# -------------------------
# W&B
# -------------------------
WANDB_ENTITY="netanelazran11-hebrew-university-of-jerusalem"
WANDB_PROJECT="pretrain-wced-21k"
WANDB_RUN_NAME="wced-21k-contrastive-w${CONTRASTIVE_WEIGHT}-${SLURM_JOB_ID}"

# -------------------------
# Output directory
# -------------------------
OUTROOT="${REPO}/outputs/${WANDB_PROJECT}"
OUTDIR="${OUTROOT}/${WANDB_RUN_NAME}"

mkdir -p "${LOGDIR}"
mkdir -p "${OUTDIR}"

echo "============================================================"
echo "WCED PRETRAINING — 21k CpGs"
echo "============================================================"
echo "Job started:   $(date)"
echo "Host:          $(hostname)"
echo "JobID:         ${SLURM_JOB_ID}"
echo "============================================================"
echo "Data:          ${DATA}"
echo "Tokenizer:     ${TOKENIZER_PATH}  (auto-built if missing)"
echo "CpG vocab:     ALL ~21k CpGs  (SUBSET_K=${SUBSET_K})"
echo "Input/view:    ${INPUT_RATIO} × 21368 ≈ 4000 CpGs per view (same speed as 8k)"
echo "Views:         2 non-overlapping (contrastive=${CONTRASTIVE})"
echo "Max pos emb:   ${MAX_POS_EMB}"
echo "Batch:         ${BATCH_SIZE} × ${ACCUMULATE_GRAD} = $((BATCH_SIZE * ACCUMULATE_GRAD)) effective"
echo "Age weight:    ${AGE_WEIGHT}"
echo "Contrastive:   ${CONTRASTIVE} (weight=${CONTRASTIVE_WEIGHT})"
echo "Model:         hidden=${HIDDEN_SIZE}, heads=${NUM_ATTENTION_HEADS}"
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
# Run WCED Pretraining (21k)
# -------------------------
python -m bmfm_methylation.pretrain_21k \
    data_path="${DATA}" \
    output_directory="${OUTDIR}" \
    tokenizer_path="${TOKENIZER_PATH}" \
    pretraining_mode=wced \
    combine_style="${COMBINE_STYLE}" \
    data_module.subset_k="${SUBSET_K}" \
    data_module.fixed_subset="true" \
    data_module.fixed_subset_seed="42" \
    data_module.max_length=${MAX_POS_EMB} \
    data_module.batch_size=${BATCH_SIZE} \
    data_module.num_workers=4 \
    wced_input_ratio="${INPUT_RATIO}" \
    wced_age_weight="${AGE_WEIGHT}" \
    wced_contrastive="${CONTRASTIVE}" \
    wced_contrastive_weight="${CONTRASTIVE_WEIGHT}" \
    wced_contrastive_temp="${CONTRASTIVE_TEMP}" \
    wced_normalize_loss="${NORMALIZE_LOSS}" \
    wced_decoder_dropout=${WCED_DECODER_DROPOUT} \
    model.hidden_size=${HIDDEN_SIZE} \
    model.num_attention_heads=${NUM_ATTENTION_HEADS} \
    model.intermediate_size=${INTERMEDIATE_SIZE} \
    model.max_position_embeddings=${MAX_POS_EMB} \
    accumulate_grad_batches=${ACCUMULATE_GRAD} \
    early_stop_patience=${EARLY_STOP_PATIENCE} \
    pretrain_epochs=${PRETRAIN_EPOCHS} \
    track_wandb.enabled=true \
    track_wandb.project="${WANDB_PROJECT}" \
    track_wandb.entity="${WANDB_ENTITY}" \
    track_wandb.name="${WANDB_RUN_NAME}"

echo "============================================================"
echo "WCED 21k Pretraining finished: $(date)"
echo "============================================================"
echo "Checkpoint: ${OUTDIR}"
echo "============================================================"
echo "Next steps:"
echo "  1. Check WandB — val/pcc should reach > 0.95"
echo "  2. Fine-tune with frozen encoder + all 21k fixed input"
echo "  3. Compare R² with 8k baseline (0.9327)"
echo "============================================================"
