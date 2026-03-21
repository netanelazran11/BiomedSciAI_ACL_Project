#!/bin/bash -l
#SBATCH --job-name=pretrain-wced-21k-fullpos
#SBATCH --partition=goldfish
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=96:00:00

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

# -------------------------
# Why this script exists
# -------------------------
# The original 21k pretraining (pretrain_wced_21k.sh) used:
#   max_position_embeddings=8002, input_ratio=0.187 → 4000 tokens/view
# This limited fine-tuning input to max 8001 tokens.
#
# This script trains with:
#   max_position_embeddings=21370 → full 21k position table
#   input_ratio=0.5              → 10,684 tokens/view
#   Flash Attention (attention="torch") → O(n) memory, not O(n²)
#
# After this pretraining, fine-tuning can use ALL 21,368 CpGs as input.
# -------------------------

# -------------------------
# 21k-specific settings
# -------------------------
SUBSET_K="${SUBSET_K:-21368}"         # Decoder vocab: all 21,368 CpGs (unchanged)

# Input ratio: 0.5 × 21368 = 10,684 tokens/view
# → 2× more CpGs per view than original 21k pretraining (was 4000)
# → trains position embeddings 0–10683 (vs 0–3999 before)
# → Flash Attention makes this memory-feasible on H200
# → decoder still reconstructs ALL 21,368 CpGs from 10,684 input
INPUT_RATIO="${INPUT_RATIO:-0.5}"

# Position embedding table covers ALL 21,368 CpGs + CLS + SEP
# Fine-tuning can now use up to 21,368 fixed CpGs as input
MAX_POS_EMB=21370

# -------------------------
# Batch size
# -------------------------
# Sequences are 10,684 tokens (2.7× longer than original 4000)
# Flash Attention: O(n) memory → batch_size=16 is safe on H200
# accumulate_grad=8 → effective batch = 128 (same as original)
BATCH_SIZE="${BATCH_SIZE:-16}"
ACCUMULATE_GRAD="${ACCUMULATE_GRAD:-8}"

# -------------------------
# WCED settings (same as successful runs)
# -------------------------
COMBINE_STYLE="${COMBINE_STYLE:-add}"
AGE_WEIGHT="${AGE_WEIGHT:-1.0}"
CONTRASTIVE="${CONTRASTIVE:-true}"
CONTRASTIVE_WEIGHT="${CONTRASTIVE_WEIGHT:-0.1}"
CONTRASTIVE_TEMP="${CONTRASTIVE_TEMP:-0.1}"
NORMALIZE_LOSS="${NORMALIZE_LOSS:-true}"
WCED_DECODER_DROPOUT="0.1"

# -------------------------
# Architecture (same as previous 21k run)
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
WANDB_PROJECT="pretrain-wced-21k-fullpos"
WANDB_RUN_NAME="wced-21k-fullpos-ratio0.5-${SLURM_JOB_ID}"

# -------------------------
# Output directory
# -------------------------
OUTROOT="${REPO}/outputs/${WANDB_PROJECT}"
OUTDIR="${OUTROOT}/${WANDB_RUN_NAME}"

mkdir -p "${LOGDIR}"
mkdir -p "${OUTDIR}"

echo "============================================================"
echo "WCED PRETRAINING — 21k CpGs, FULL POSITION EMBEDDINGS"
echo "============================================================"
echo "Job started:   $(date)"
echo "Host:          $(hostname)"
echo "JobID:         ${SLURM_JOB_ID}"
echo "============================================================"
echo "Key difference from pretrain_wced_21k.sh:"
echo "  max_position_embeddings: 8002  →  21370"
echo "  input_ratio:             0.187 →  0.5"
echo "  tokens/view:             4000  →  10684"
echo "  Flash Attention:         ON    (attention='torch', H200)"
echo "  batch_size:              32    →  16 (longer sequences)"
echo "  effective batch:         128   (same, via accumulate_grad=8)"
echo "============================================================"
echo "Data:          ${DATA}"
echo "Tokenizer:     ${TOKENIZER_PATH}"
echo "CpG vocab:     ALL 21,368 CpGs  (SUBSET_K=${SUBSET_K})"
echo "Input/view:    ${INPUT_RATIO} × 21368 ≈ 10684 CpGs per view"
echo "Max pos emb:   ${MAX_POS_EMB}  (covers all 21k positions)"
echo "Batch:         ${BATCH_SIZE} × ${ACCUMULATE_GRAD} = $((BATCH_SIZE * ACCUMULATE_GRAD)) effective"
echo "Contrastive:   ${CONTRASTIVE} (weight=${CONTRASTIVE_WEIGHT})"
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
# Verify Flash Attention is available
try:
    from torch.nn.functional import scaled_dot_product_attention
    print("Flash Attention (scaled_dot_product_attention): AVAILABLE")
except ImportError:
    print("WARNING: scaled_dot_product_attention not available")
PY

# -------------------------
# Run WCED Pretraining (21k full position)
# -------------------------
python -m bmfm_methylation.pretrain_21k \
    data_path="${DATA}" \
    output_directory="${OUTDIR}" \
    tokenizer_path="${TOKENIZER_PATH}" \
    pretraining_mode=wced \
    fields=methylation_21k \
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
echo "WCED 21k-fullpos Pretraining finished: $(date)"
echo "============================================================"
echo "Checkpoint: ${OUTDIR}"
echo "============================================================"
echo "Next steps:"
echo "  1. Check WandB — val/pcc should reach > 0.95"
echo "  2. Run finetune_frozen_21k_fullpos.sh with all 21k CpGs as input"
echo "     subset_k=21368, max_length=21370, fixed_subset=true"
echo "============================================================"
