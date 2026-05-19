#!/bin/bash -l
#SBATCH --job-name=methylgpt-large-ft
#SBATCH --partition=salmon
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/MethylGPT-Thesis/logs/%x_%j.out
#SBATCH --error=/sci/labs/benjamin.yakir/netanel.azran/repos/MethylGPT-Thesis/logs/%x_%j.err

set -euo pipefail

REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/MethylGPT-Thesis"
METHYLGPT="${REPO}/external/MethylGPT"
SCRIPTS="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/scripts/methylgpt"

echo "============================================================"
echo "MethylGPT Large — Fine-tune on 49k dataset"
echo "============================================================"
echo "Job: ${SLURM_JOB_ID} | Host: $(hostname) | Time: $(date)"
echo "============================================================"

source /sci/labs/benjamin.yakir/netanel.azran/venv_torch22/bin/activate

mkdir -p "${REPO}/logs"
mkdir -p "${REPO}/finetune_checkpoints/large_49k"

# Step 1: Convert h5ad to parquet (skip if already done)
PARQUET_DIR="${REPO}/data/finetune_49k_parquet"
if [ ! -f "${PARQUET_DIR}/finetune49k_train.parquet" ]; then
    echo "Converting h5ad to parquet..."
    python "${SCRIPTS}/convert_h5ad_to_parquet.py"
else
    echo "Parquet files already exist, skipping conversion."
fi

# Step 2: Fine-tune
echo "Starting MethylGPT fine-tuning..."
cd "${METHYLGPT}/tutorials/finetuning_age_prediction"

python finetuning_age_main.py \
    --config "${SCRIPTS}/train_methylgpt_large_49k.yml"

echo "============================================================"
echo "Done: $(date)"
echo "============================================================"
