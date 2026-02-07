#!/bin/bash -l
#SBATCH --job-name=transformer-scaled-cpg-finetune-8k
#SBATCH --partition=goldfish
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00

#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# -------------------------
# Paths
# -------------------------
REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
LOGDIR="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs"

DATA="/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_8k_h5ad/methylgpt_8k_altumage_combined.h5ad"

# Pretrained checkpoint - USE MULTIPLY MODE CHECKPOINT!
# IMPORTANT: Must match combine_style (multiply checkpoint for multiply finetuning)
CHECKPOINT="${CHECKPOINT:-/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/outputs/pretrain-multiply-bmfm-rna-methylation-8k/multiply-44032612/pretrain/checkpoints/epoch=epoch=2-val_loss=validation/loss=0.0151.ckpt}"

# Combine style: "multiply" (scGPT style) or "add" (scaled CpG)
COMBINE_STYLE="${COMBINE_STYLE:-multiply}"

# Initial CpG scale (only used in "add" mode)
INITIAL_CPG_SCALE="${INITIAL_CPG_SCALE:-0.1}"

# W&B naming
WANDB_ENTITY="netanelazran11-hebrew-university-of-jerusalem"
WANDB_PROJECT="transformer-scaled-cpg-finetune-bmfm-rna-methylation-8k"
WANDB_RUN_NAME="${COMBINE_STYLE}-${SLURM_JOB_ID}"

# Output directory (unique per run)
OUTROOT="${REPO}/outputs/${WANDB_PROJECT}"
OUTDIR="${OUTROOT}/${WANDB_RUN_NAME}"

mkdir -p "${LOGDIR}"
mkdir -p "${OUTDIR}"

echo "============================================================"
echo "TRANSFORMER FINE-TUNING"
echo "============================================================"
echo "Job started: $(date)"
echo "Host: $(hostname)"
echo "JobID: ${SLURM_JOB_ID}"
echo "Node(s): ${SLURM_NODELIST}"
echo "============================================================"
echo "Combine style: ${COMBINE_STYLE}"
if [ "${COMBINE_STYLE}" = "multiply" ]; then
    echo "Architecture: h = CpG_embed * β_value (scGPT style)"
else
    echo "Architecture: h = α * CpG_embed + β_embed (α=${INITIAL_CPG_SCALE})"
fi
echo "============================================================"
echo "W&B project: ${WANDB_PROJECT}"
echo "W&B run:     ${WANDB_RUN_NAME}"
echo "Data:        ${DATA}"
echo "Checkpoint:  ${CHECKPOINT}"
echo "Output dir:  ${OUTDIR}"
echo "============================================================"

# -------------------------
# Modules (CUDA/NVCC)
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

# Perf / stability knobs
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Tensor cores utilization
python - <<'PY'
import torch
torch.set_float32_matmul_precision("medium")
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("matmul_precision:", torch.get_float32_matmul_precision())
PY

# -------------------------
# Step 1: Run Ridge baseline (quick sanity check)
# -------------------------
echo "Running Ridge baseline..."
python scripts/baseline_ridge.py "${DATA}" || true
echo "============================================================"

# -------------------------
# Step 2: Run Transformer Fine-tuning
# -------------------------
echo ""
echo "Starting Transformer fine-tuning..."
echo "  - Combine style: ${COMBINE_STYLE}"
if [ "${COMBINE_STYLE}" = "multiply" ]; then
    echo "  - h = CpG_embed * β_value (scGPT style)"
else
    echo "  - h = α * CpG_embed + β_embed (α starts at ${INITIAL_CPG_SCALE})"
fi
echo ""

python -m bmfm_methylation.finetune_transformer_scaled \
    data_path="${DATA}" \
    "checkpoint_path='${CHECKPOINT}'" \
    output_directory="${OUTDIR}" \
    combine_style="${COMBINE_STYLE}" \
    initial_cpg_scale=${INITIAL_CPG_SCALE} \
    freeze_encoder=true \
    unfreeze_encoder_epoch=5 \
    track_wandb.enabled=true \
    track_wandb.project="${WANDB_PROJECT}" \
    track_wandb.entity="${WANDB_ENTITY}" \
    track_wandb.name="${WANDB_RUN_NAME}"

echo "============================================================"
echo "Job finished: $(date)"
echo "============================================================"
