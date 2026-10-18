#!/bin/bash -l
#SBATCH --job-name=pilot-subset
#SBATCH --partition=salmon
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_ablation_infonce/%x_%j.out
#SBATCH --error=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_ablation_infonce/%x_%j.err

# Builds outputs/ablation_infonce/pilot_data/pilot_9k_seed2026.h5ad (9,000 train / 1,000 valid /
# 1,000 test, drawn from the published train / valid / held-out partitions). Read-only on the
# source; refuses to overwrite. CPU work — the GPU line is only the partition's admission ticket.
#   sbatch ablation_infonce/create_pilot_subset.sh

set -euo pipefail
REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
SOURCE="/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_pretrain_type3_h5ad/methylgpt_pretrain_type3.h5ad"
mkdir -p "${REPO}/logs_ablation_infonce"
cd "${REPO}"
source /etc/profile.d/modules.sh 2>/dev/null || true
module purge 2>/dev/null || true
module load spack/all 2>/dev/null || true
source bmfm_methyl_env/bin/activate
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
echo "Git: $(git rev-parse --short HEAD) | Host: $(hostname) | $(date)"
python -m ablation_infonce.create_pilot_subset --source "${SOURCE}" --outdir outputs/ablation_infonce/pilot_data
