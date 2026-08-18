#!/usr/bin/env bash
#SBATCH --job-name=elasticnet-age-baseline
#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs/elasticnet_%j.out
#SBATCH --error=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs/elasticnet_%j.err
#SBATCH --time=03:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1

set -euo pipefail

REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
cd "$REPO"
mkdir -p logs
source bmfm_methyl_env/bin/activate
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"

DATA="/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_21k_h5ad/altumage_21k_3way.h5ad"
DUP_CSV="${REPO}/dataset_fingerprint_outputs/duplicate_pairs.csv"
OUTDIR="${REPO}/outputs/baselines/elasticnet/gridsearch-${SLURM_JOB_ID:-local}"

# ── Fail loudly before burning a job allocation on a missing/moved path ──────
[ -f "${DATA}" ]    || { echo "ERROR: h5ad not found: ${DATA}"; exit 1; }
[ -f "${DUP_CSV}" ] || { echo "ERROR: duplicate_pairs_csv not found: ${DUP_CSV}"; exit 1; }

mkdir -p "${OUTDIR}"

echo "============================================================"
echo "ElasticNet Age Baseline — grid search, matched eval protocol"
echo "Data:            ${DATA}"
echo "Duplicate pairs: ${DUP_CSV}"
echo "Out:             ${OUTDIR}"
echo "============================================================"

python scripts/baselines/elasticnet_age.py \
    --h5ad     "${DATA}" \
    --outdir   "${OUTDIR}" \
    --duplicate_pairs_csv "${DUP_CSV}" \
    --filter_age_outliers

echo "============================================================"
echo "Done: $(date)"
echo "Results: ${OUTDIR}/elasticnet_results.json"
echo "============================================================"
