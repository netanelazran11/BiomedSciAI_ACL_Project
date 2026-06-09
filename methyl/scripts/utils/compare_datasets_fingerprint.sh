#!/usr/bin/env bash
#SBATCH --job-name=dataset_fingerprint
#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs/dataset_fingerprint_%j.out
#SBATCH --error=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs/dataset_fingerprint_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=salmon

set -euo pipefail

REPO=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl
cd "$REPO"

mkdir -p logs

source bmfm_methyl_env/bin/activate

OUTDIR="$REPO/dataset_fingerprint_outputs"
mkdir -p "$OUTDIR"

echo "=============================="
echo "Dataset Fingerprint Comparison"
echo "Started: $(date)"
echo "=============================="

python scripts/utils/compare_datasets_fingerprint.py \
    --outdir "$OUTDIR" \
    --splits "valid,test"

echo ""
echo "=============================="
echo "Done: $(date)"
echo "Output → $OUTDIR"
echo "=============================="
