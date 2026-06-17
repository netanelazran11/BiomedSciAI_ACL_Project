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

CHECKPOINT="${REPO}/outputs/pretrain-wced-bmfm/wced-contrastive-k8000-w0.1-44206138/pretrain/checkpoints/epoch=epoch=190-val_loss=validation/loss=0.1264.ckpt"

DATA_AGE="/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_8k_h5ad/methylgpt_8k_altumage_combined.h5ad"
DATA_SMOKING="/sci/labs/benjamin.yakir/netanel.azran/data/smoking_geo/smoking_combined_aligned.h5ad"
DATA_MULTITASK="/sci/labs/benjamin.yakir/netanel.azran/data/smoking_geo/multitask_data.h5ad"

# Fall back to single-source smoking data if combined not available
if [ ! -f "${DATA_SMOKING}" ]; then
    DATA_SMOKING="/sci/labs/benjamin.yakir/netanel.azran/data/smoking_geo/smoking_data_aligned.h5ad"
fi

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
echo "TASK C-1a — Data Efficiency (AGE — primary result)"
echo "WCED was pretrained on age-methylation: expect clear WCED > random"
echo "============================================================"
python -m bmfm_methylation.downstream.probing.data_efficiency \
    --checkpoint_path "${CHECKPOINT}" \
    --data_path "${DATA_AGE}" \
    --task age \
    --output_dir "${OUTDIR}/data_efficiency_age" \
    --n_epochs 50 \
    --n_seeds 3

echo "============================================================"
echo "TASK C-1b — Data Efficiency (SMOKING — cross-domain)"
echo "Negative / contrast result: WCED linear probe ≈ random on smoking"
echo "============================================================"
python -m bmfm_methylation.downstream.probing.data_efficiency \
    --checkpoint_path "${CHECKPOINT}" \
    --data_path "${DATA_SMOKING}" \
    --task smoking \
    --output_dir "${OUTDIR}/data_efficiency_smoking" \
    --n_epochs 50 \
    --n_seeds 3

echo "============================================================"
echo "TASK C-2 — Embedding Analysis (UMAP + linear probes)"
echo "Uses multitask h5ad for age + smoking + sex visualization"
echo "============================================================"
if [ -f "${DATA_MULTITASK}" ]; then
    python -m bmfm_methylation.downstream.probing.embedding_analysis \
        --checkpoint_path "${CHECKPOINT}" \
        --data_path "${DATA_MULTITASK}" \
        --output_dir "${OUTDIR}/embeddings_multitask" \
        --label_cols age smoking_status sex \
        --compare_random_init
else
    echo "Multitask h5ad not found, running embedding analysis on age data only"
    python -m bmfm_methylation.downstream.probing.embedding_analysis \
        --checkpoint_path "${CHECKPOINT}" \
        --data_path "${DATA_AGE}" \
        --output_dir "${OUTDIR}/embeddings_age" \
        --label_cols age sex \
        --compare_random_init
fi

echo "Done: $(date)"
