#!/bin/bash -l
#SBATCH --job-name=pretrain-wced-bmfm
#SBATCH --partition=goldfish
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=50:00:00

#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# -------------------------
# Paths
# -------------------------
REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
LOGDIR="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs"

DATA="/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_8k_h5ad/methylgpt_8k_altumage_combined.h5ad"

# Combine style: "add" (standard) or "multiply" (scGPT style)
COMBINE_STYLE="${COMBINE_STYLE:-add}"

# CpG subset settings (FIXED subset - same CpGs for all samples)
SUBSET_K="${SUBSET_K:-2048}"
FIXED_SUBSET="true"
FIXED_SUBSET_SEED="42"

# WCED-specific settings
WCED_DECODER_HIDDEN="[2048,4096]"  # Decoder: 512 -> 2048 -> 4096 -> num_cpg
WCED_DECODER_DROPOUT="0.1"
WCED_USE_POSITIONAL="false"  # Set to "true" for attention-based decoder

# W&B naming
WANDB_ENTITY="netanelazran11-hebrew-university-of-jerusalem"
WANDB_PROJECT="pretrain-wced-fixed${SUBSET_K}-bmfm-rna-methylation"
WANDB_RUN_NAME="wced-${COMBINE_STYLE}-fixed${SUBSET_K}-${SLURM_JOB_ID}"

# Output directory (unique per run to avoid overwriting checkpoints)
OUTROOT="${REPO}/outputs/${WANDB_PROJECT}"
OUTDIR="${OUTROOT}/${WANDB_RUN_NAME}"

mkdir -p "${LOGDIR}"
mkdir -p "${OUTDIR}"

echo "============================================================"
echo "METHYLATION WCED PRETRAINING (FIXED SUBSET)"
echo "============================================================"
echo "Job started: $(date)"
echo "Host: $(hostname)"
echo "JobID: ${SLURM_JOB_ID}"
echo "Node(s): ${SLURM_NODELIST}"
echo "============================================================"
echo "PRETRAINING MODE: WCED (Whole Cell Expression Decoder)"
echo "  - No masking: use ALL beta values"
echo "  - Reconstruct ALL positions from [CLS] token"
echo "  - Creates global bottleneck for cell-level representation"
echo "============================================================"
echo "CpG Subset:  FIXED ${SUBSET_K} CpGs (same for all samples)"
echo "Seed:        ${FIXED_SUBSET_SEED}"
echo "Combine style: ${COMBINE_STYLE}"
if [ "${COMBINE_STYLE}" = "multiply" ]; then
    echo "Architecture: h = CpG_embed * β_value (scGPT style)"
else
    echo "Architecture: h = CpG_embed + β_embed (standard BMFM)"
fi
echo "============================================================"
echo "WCED Decoder:"
echo "  Hidden sizes: ${WCED_DECODER_HIDDEN}"
echo "  Dropout:      ${WCED_DECODER_DROPOUT}"
echo "  Positional:   ${WCED_USE_POSITIONAL}"
echo "============================================================"
echo "W&B project: ${WANDB_PROJECT}"
echo "W&B run:     ${WANDB_RUN_NAME}"
echo "Data:        ${DATA}"
echo "Output dir:  ${OUTDIR}"
echo "============================================================"

# -------------------------
# Modules (CUDA/NVCC)
# -------------------------
# Initialize module system (required for non-login shells)
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
# Run WCED Pretraining
# -------------------------
# Option 1: Use dedicated WCED config file
# python -m bmfm_methylation.pretrain \
#     --config-name=pretrain_wced_config \
#     ...

# Option 2: Use main config with pretraining_mode=wced (more flexible)
python -m bmfm_methylation.pretrain \
    data_path="${DATA}" \
    output_directory="${OUTDIR}" \
    pretraining_mode="wced" \
    combine_style="${COMBINE_STYLE}" \
    data_module.subset_k="${SUBSET_K}" \
    data_module.fixed_subset="${FIXED_SUBSET}" \
    data_module.fixed_subset_seed="${FIXED_SUBSET_SEED}" \
    data_module.max_length=$((SUBSET_K + 2)) \
    wced_decoder_hidden_sizes="${WCED_DECODER_HIDDEN}" \
    wced_decoder_dropout="${WCED_DECODER_DROPOUT}" \
    wced_use_positional_decoder="${WCED_USE_POSITIONAL}" \
    track_wandb.enabled=true \
    track_wandb.project="${WANDB_PROJECT}" \
    track_wandb.entity="${WANDB_ENTITY}" \
    track_wandb.name="${WANDB_RUN_NAME}"

echo "============================================================"
echo "WCED Pretraining finished: $(date)"
echo "============================================================"
echo "Next steps:"
echo "  1. Check WandB for training curves"
echo "  2. Best checkpoint saved to: ${OUTDIR}"
echo "  3. Use checkpoint for fine-tuning with:"
echo "     python -m bmfm_methylation.finetune \\"
echo "         checkpoint_path=${OUTDIR}/best_checkpoint.ckpt \\"
echo "         ..."
echo "============================================================"
