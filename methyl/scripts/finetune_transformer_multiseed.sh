#!/bin/bash -l
#SBATCH --job-name=finetune-seed
#SBATCH --partition=goldfish
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00

#SBATCH --output=logs/%x_%j_seed%a.out
#SBATCH --error=logs/%x_%j_seed%a.err

set -euo pipefail

# -------------------------
# Seed from environment or array task ID
# -------------------------
# Can be set via: sbatch --export=SEED=42 or via array task
SEED=${SEED:-${SLURM_ARRAY_TASK_ID:-42}}

# -------------------------
# Paths
# -------------------------
REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
LOGDIR="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs"

DATA="/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_8k_h5ad/methylgpt_8k_altumage_combined.h5ad"

# Pretrained checkpoint (loss=0.0013, 250 epochs)
CHECKPOINT="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/outputs/pretrain-fixed2048-bmfm-rna-methylation/add-fixed2048-44043043/pretrain/checkpoints/epoch=epoch=234-val_loss=validation/loss=0.0013.ckpt"

# W&B naming - include seed in run name
WANDB_ENTITY="netanelazran11-hebrew-university-of-jerusalem"
WANDB_PROJECT="finetune-bmfm-multiseed"
WANDB_RUN_NAME="finetune-seed${SEED}-${SLURM_JOB_ID}"
WANDB_GROUP="multiseed-experiment"

# Output directory (unique per seed)
OUTROOT="${REPO}/outputs/${WANDB_PROJECT}"
OUTDIR="${OUTROOT}/seed${SEED}-${SLURM_JOB_ID}"

mkdir -p "${LOGDIR}"
mkdir -p "${OUTDIR}"

echo "============================================================"
echo "Job started: $(date)"
echo "Host: $(hostname)"
echo "JobID: ${SLURM_JOB_ID}"
echo "Node(s): ${SLURM_NODELIST}"
echo "============================================================"
echo "SEED:        ${SEED}"
echo "MODE:        PRETRAINED (from value-only checkpoint)"
echo "W&B project: ${WANDB_PROJECT}"
echo "W&B group:   ${WANDB_GROUP}"
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
# Run Fine-tuning with specific seed
# -------------------------
python -m bmfm_methylation.finetune_transformer \
    data_path="${DATA}" \
    "checkpoint_path='${CHECKPOINT}'" \
    output_directory="${OUTDIR}" \
    seed.seed_value=${SEED} \
    freeze_encoder=true \
    unfreeze_encoder_epoch=5 \
    track_wandb.enabled=true \
    track_wandb.project="${WANDB_PROJECT}" \
    track_wandb.entity="${WANDB_ENTITY}" \
    track_wandb.name="${WANDB_RUN_NAME}" \
    +track_wandb.group="${WANDB_GROUP}"

echo "============================================================"
echo "Seed ${SEED} - Job finished: $(date)"
echo "============================================================"
