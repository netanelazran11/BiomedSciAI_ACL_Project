#!/usr/bin/env bash
#SBATCH --job-name=pretrain-overlap
#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs/pretrain_overlap_%j.out
#SBATCH --error=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs/pretrain_overlap_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1
# ─────────────────────────────────────────────────────────────────────────────
# Which AltuMAge profiles are inside the pretraining corpus, and in which
# pretraining split?  READ-ONLY: loads two h5ad files, writes three small files
# under outputs/pretrain_overlap/. No model, no training, nothing modified.
# The GPU line is only the partition's admission ticket; the job runs on CPU.
#
# Usage: sbatch scripts/utils/run_pretrain_downstream_overlap.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
PRETRAIN="/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_pretrain_type3_h5ad/methylgpt_pretrain_type3.h5ad"
DOWNSTREAM="/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_21k_h5ad/altumage_21k_3way.h5ad"
TEST_IDS="${REPO}/outputs/kfold_splits/test_ids.npy"
OUTDIR="${REPO}/outputs/pretrain_overlap"

cd "$REPO"
source bmfm_methyl_env/bin/activate
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

[ -f "${PRETRAIN}" ]   || { echo "ERROR: pretraining h5ad not found: ${PRETRAIN}"; exit 1; }
[ -f "${DOWNSTREAM}" ] || { echo "ERROR: downstream h5ad not found: ${DOWNSTREAM}"; exit 1; }
[ -f "${TEST_IDS}" ]   || { echo "ERROR: test ids not found: ${TEST_IDS}"; exit 1; }

echo "============================================================"
echo "Pretraining / downstream overlap | job ${SLURM_JOB_ID} | $(hostname) | $(date)"
echo "data directory of the pretraining matrix (what else lives next to it):"
ls -la "$(dirname "${PRETRAIN}")"
echo "============================================================"

python scripts/utils/pretrain_downstream_overlap.py \
    --pretrain "${PRETRAIN}" --downstream "${DOWNSTREAM}" \
    --test_ids "${TEST_IDS}" --outdir "${OUTDIR}"

echo "DONE: $(date) -> ${OUTDIR}/"
