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
#   mkdir -p logs_ablation_infonce   # once: Slurm cannot create the --output directory itself
#   sbatch ablation_infonce/create_pilot_subset.sh

set -euo pipefail
REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
SOURCE="/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_pretrain_type3_h5ad/methylgpt_pretrain_type3.h5ad"
mkdir -p "${REPO}/logs_ablation_infonce"
cd "${REPO}"
source /etc/profile.d/modules.sh 2>/dev/null || true
# Cluster modules changed in Oct 2026 (spack/all now needs spack-huji; cuda/12.3.2-gcc-5bv3kyh is gone).
# Nothing here needs spack: Python comes from env.sh and torch 2.10.0+cu128 bundles its CUDA runtime.
module purge 2>/dev/null || true
module load cuda/12.8.1 2>/dev/null || echo "WARN: module cuda/12.8.1 not loaded (torch bundles its CUDA runtime)"
source ablation_infonce/env.sh
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
echo "Git: $(git rev-parse --short HEAD) | Host: $(hostname) | $(date)"
"${PY}" -m ablation_infonce.create_pilot_subset --source "${SOURCE}" --outdir outputs/ablation_infonce/pilot_data
