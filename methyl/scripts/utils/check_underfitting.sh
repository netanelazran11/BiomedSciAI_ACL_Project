#!/bin/bash -l
#SBATCH --job-name=check-underfitting
#SBATCH --partition=catfish
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/%x_%j.out
#SBATCH --error=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/%x_%j.err

set -euo pipefail

REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
mkdir -p "${REPO}/logs_llama-wced"
cd "${REPO}"
source bmfm_methyl_env/bin/activate
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"

# ── Pre-computed CLS embeddings (from cls_probing job 44892802) ────────────
CLS_NPY="${REPO}/outputs/repr_analysis/pretrain_cls_169k_44892802/embeddings_cls.npy"

OUTDIR="${REPO}/outputs/diagnostics/underfitting_check_${SLURM_JOB_ID}"
mkdir -p "${OUTDIR}"

echo "============================================================"
echo " MethylLlama — Underfitting / Architecture Scaling Check"
echo " Job : ${SLURM_JOB_ID}  Host: $(hostname)  Time: $(date)"
echo "============================================================"
echo " Embeddings : ${CLS_NPY}"
echo " Outdir     : ${OUTDIR}"
echo "============================================================"

python scripts/utils/check_underfitting.py \
    --embeddings "${CLS_NPY}" \
    --outdir     "${OUTDIR}"

echo ""
echo "============================================================"
echo " DONE: $(date)"
echo " Results → ${OUTDIR}/"
echo "============================================================"
