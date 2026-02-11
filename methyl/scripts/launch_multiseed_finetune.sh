#!/bin/bash
# ============================================================
# Launch multiple finetune jobs with different seeds
#
# Usage:
#   ./scripts/launch_multiseed_finetune.sh
#
# This will submit 5 jobs (seeds 40-44) simultaneously
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FINETUNE_SCRIPT="${SCRIPT_DIR}/finetune_transformer_multiseed.sh"

# Seeds to run (matching MethylGPT baseline seeds)
SEEDS=(40 41 42 43 44)

echo "============================================================"
echo "Launching multi-seed finetune experiment"
echo "Seeds: ${SEEDS[*]}"
echo "Script: ${FINETUNE_SCRIPT}"
echo "============================================================"

# Make sure logs directory exists
mkdir -p "${SCRIPT_DIR}/../logs"

# Submit jobs
JOB_IDS=()
for seed in "${SEEDS[@]}"; do
    echo "Submitting seed=${seed}..."
    JOB_ID=$(sbatch --export=SEED=${seed} "${FINETUNE_SCRIPT}" | awk '{print $4}')
    JOB_IDS+=("${JOB_ID}")
    echo "  -> JobID: ${JOB_ID}"
done

echo "============================================================"
echo "All jobs submitted!"
echo "Job IDs: ${JOB_IDS[*]}"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo ""
echo "View logs:"
for i in "${!SEEDS[@]}"; do
    echo "  tail -f logs/finetune-seed_${JOB_IDS[$i]}_seed${SEEDS[$i]}.out"
done
echo ""
echo "W&B group: multiseed-experiment"
echo "============================================================"
