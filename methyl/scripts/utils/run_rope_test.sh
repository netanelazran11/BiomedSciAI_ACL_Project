#!/bin/bash -l
#SBATCH --job-name=rope-v2-test
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/%x_%j.out
#SBATCH --error=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/%x_%j.err

set -euo pipefail

REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
mkdir -p "${REPO}/logs_llama-wced"
cd "${REPO}"
source bmfm_methyl_env/bin/activate

echo "============================================================"
echo " Level 2 Genomic RoPE Test"
echo " Job: ${SLURM_JOB_ID}  Host: $(hostname)  Time: $(date)"
echo "============================================================"

python tests/test_genomic_rope_v2.py

echo "============================================================"
echo " DONE: $(date)"
echo "============================================================"
