#!/bin/bash -l
#SBATCH --job-name=verify-elasticnet-test-set
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00

#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs/verify_elasticnet_test_set_%j.out
#SBATCH --error=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs/verify_elasticnet_test_set_%j.err

# ─────────────────────────────────────────────────────────────────────────────
# One-off check: does elasticnet_age.py's live re-derived test split match the
# official fixed test_ids.npy exactly (same GSM IDs, not just same count)?
# CPU-only, backed-mode h5ad read (never loads the methylation matrix) -- runs
# as a tiny sbatch job because the login node's memory cap OOM-killed it twice.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
mkdir -p "${REPO}/logs"
cd "${REPO}"

source /etc/profile.d/modules.sh 2>/dev/null || true
module purge 2>/dev/null || true
module load spack/all 2>/dev/null || true
module load cuda/12.3.2-gcc-5bv3kyh 2>/dev/null || true
source bmfm_methyl_env/bin/activate

echo "============================================================"
echo "Verify ElasticNet test-set membership vs. official test_ids.npy"
echo "Job: ${SLURM_JOB_ID}  Host: $(hostname)  Time: $(date)"
echo "============================================================"

python scripts/utils/verify_elasticnet_test_set.py

echo "============================================================"
echo "DONE: $(date)"
echo "============================================================"
