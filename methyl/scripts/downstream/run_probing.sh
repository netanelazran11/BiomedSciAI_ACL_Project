#!/bin/bash -l
#SBATCH --job-name=downstream-probing
#SBATCH --partition=salmon
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"

CHECKPOINT="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/outputs/pretrain-wced-bmfm/wced-contrastive-k8000-w0.1-44206138/pretrain/checkpoints/epoch=epoch=190-val_loss=validation/loss=0.1264.ckpt"

# Use smoking h5ad (has smoking_status labels) for probing
DATA_SMOKING="/sci/labs/benjamin.yakir/netanel.azran/data/smoking/smoking_data.h5ad"
# Use age h5ad (has age labels) for age probing
DATA_AGE="/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_8k_h5ad/methylgpt_8k_altumage_combined.h5ad"

OUTDIR="${REPO}/outputs/downstream/probing"
mkdir -p "${REPO}/logs" "${OUTDIR}"

source /etc/profile.d/modules.sh 2>/dev/null || true
module purge
module load spack/all
module load cuda/12.3.2-gcc-5bv3kyh

cd "${REPO}"
source bmfm_methyl_env/bin/activate
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

echo "============================================================"
echo "TASK C-1 — Data Efficiency (smoking)"
echo "============================================================"
python -m bmfm_methylation.downstream.probing.data_efficiency \
    --checkpoint_path "${CHECKPOINT}" \
    --data_path "${DATA_SMOKING}" \
    --task smoking \
    --output_dir "${OUTDIR}/data_efficiency_smoking" \
    --n_epochs 50 \
    --n_seeds 3

echo "============================================================"
echo "TASK C-2 — Embedding Analysis (smoking)"
echo "============================================================"
python -m bmfm_methylation.downstream.probing.embedding_analysis \
    --checkpoint_path "${CHECKPOINT}" \
    --data_path "${DATA_SMOKING}" \
    --output_dir "${OUTDIR}/embeddings_smoking" \
    --label_cols smoking_status age sex \
    --compare_random_init

echo "Done: $(date)"
