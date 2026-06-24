#!/bin/bash -l
#SBATCH --job-name=cls-rank-viz
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/%x_%j.out
#SBATCH --error=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/%x_%j.err

set -euo pipefail

REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
mkdir -p "${REPO}/logs_llama-wced"
cd "${REPO}"
source bmfm_methyl_env/bin/activate
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"

# ── Paths ──────────────────────────────────────────────────────────────────
CLS_NPY="${REPO}/outputs/repr_analysis/pretrain_cls_169k_44892802/embeddings_cls.npy"
OUTDIR="${REPO}/outputs/repr_analysis/cls_rank_figures_${SLURM_JOB_ID}"
mkdir -p "${OUTDIR}"

echo "============================================================"
echo " MethylLlama — CLS Rank Visualization"
echo " Job : ${SLURM_JOB_ID}  Host: $(hostname)  Time: $(date)"
echo "============================================================"
echo " Embeddings : ${CLS_NPY}"
echo " Outdir     : ${OUTDIR}"
echo "============================================================"

python scripts/repr_analysis/visualize_cls_rank.py \
    --embeddings "${CLS_NPY}" \
    --outdir     "${OUTDIR}"  \
    --n_pca      100

echo ""
echo "============================================================"
echo " DONE: $(date)"
echo " Figures → ${OUTDIR}/"
echo "   cls_rank_analysis.png — 2-panel combined (A+B)"
echo "   panel_A_scree.png     — singular value spectrum + eff.rank"
echo "   panel_B_cumvar.png    — cumulative variance (90/95/99%)"
echo "============================================================"
