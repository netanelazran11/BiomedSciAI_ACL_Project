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
METADATA="${REPO}/data/pretrain_metadata.csv.gz"
OUTDIR="${REPO}/outputs/repr_analysis/cls_rank_figures_${SLURM_JOB_ID}"
mkdir -p "${OUTDIR}"

echo "============================================================"
echo " MethylLlama — CLS Rank Visualization"
echo " Job : ${SLURM_JOB_ID}  Host: $(hostname)  Time: $(date)"
echo "============================================================"
echo " Embeddings : ${CLS_NPY}"
echo " Metadata   : ${METADATA}"
echo " Outdir     : ${OUTDIR}"
echo "============================================================"

python scripts/repr_analysis/visualize_cls_rank.py \
    --embeddings  "${CLS_NPY}"   \
    --metadata    "${METADATA}"  \
    --outdir      "${OUTDIR}"    \
    --n_pca       100            \
    --n_samples   50000

echo ""
echo "============================================================"
echo " DONE: $(date)"
echo " Figures → ${OUTDIR}/"
echo "   cls_rank_analysis.png    — 4-panel combined figure"
echo "   panel_A_scree.png        — singular value spectrum"
echo "   panel_B_cumvar.png       — cumulative variance"
echo "   panel_C_pca_tissue.png   — PCA coloured by tissue"
echo "   panel_D_pca_age.png      — PCA coloured by age"
echo "============================================================"
