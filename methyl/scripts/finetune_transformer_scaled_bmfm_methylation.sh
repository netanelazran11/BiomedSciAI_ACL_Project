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

# Pretrained checkpoint (standard pretraining with CpG IDs + beta values)
# This checkpoint has learned representations for both CpG IDs and beta values
CHECKPOINT="${CHECKPOINT:-/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/outputs/pretrain-bmfm-rna-methylation-8k/bmfm-methyl-8k-43987959/pretrain/checkpoints/epoch=epoch=5-val_loss=validation/loss=0.0149.ckpt}"

# Initial CpG scale (0.1 means CpG embeddings start at 10% of original)
INITIAL_CPG_SCALE="${INITIAL_CPG_SCALE:-0.1}"

# W&B naming
WANDB_ENTITY="netanelazran11-hebrew-university-of-jerusalem"
WANDB_PROJECT="transformer-scaled-cpg-finetune-bmfm-rna-methylation-8k"
WANDB_RUN_NAME="scaled-cpg-${INITIAL_CPG_SCALE}-${SLURM_JOB_ID}"

# Output directory (unique per run)
OUTROOT="${REPO}/outputs/${WANDB_PROJECT}"
OUTDIR="${OUTROOT}/${WANDB_RUN_NAME}"

mkdir -p "${LOGDIR}"
mkdir -p "${OUTDIR}"

echo "============================================================"
echo "TRANSFORMER FINE-TUNING WITH SCALED CpG EMBEDDINGS"
echo "============================================================"
echo "Job started: $(date)"
echo "Host: $(hostname)"
echo "JobID: ${SLURM_JOB_ID}"
echo "Node(s): ${SLURM_NODELIST}"
echo "============================================================"
echo "Architecture: h_i = α * CpG_embed + β_embed + pos_embed"
echo "              α is LEARNABLE, initialized to ${INITIAL_CPG_SCALE}"
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
# Step 2: Run Transformer Fine-tuning with Scaled CpG Embeddings
# -------------------------
echo ""
echo "Starting Transformer fine-tuning with scaled CpG embeddings..."
echo "  - CpG scale initialized to: ${INITIAL_CPG_SCALE}"
echo "  - CpG scale is LEARNABLE (model will adjust it)"
echo ""

python -m bmfm_methylation.finetune_transformer_scaled \
    data_path="${DATA}" \
    "checkpoint_path='${CHECKPOINT}'" \
    output_directory="${OUTDIR}" \
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
