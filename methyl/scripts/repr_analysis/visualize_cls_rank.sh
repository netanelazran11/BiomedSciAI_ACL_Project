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
echo " MethylLlama — CLS Rank Visualization (stratified)"
echo " Job : ${SLURM_JOB_ID}  Host: $(hostname)  Time: $(date)"
echo "============================================================"
echo " Embeddings    : ${CLS_NPY}"
echo " Metadata      : ${METADATA}"
echo " Outdir        : ${OUTDIR}"
echo " Tissue sample : 300 per tissue  (all tissues equally visible)"
echo " Age sample    : 400 per bin × 15 bins  (all ages equally visible)"
echo "============================================================"

python scripts/repr_analysis/visualize_cls_rank.py \
    --embeddings    "${CLS_NPY}"   \
    --metadata      "${METADATA}"  \
    --outdir        "${OUTDIR}"    \
    --n_pca         100            \
    --n_per_tissue  300            \
    --n_per_age_bin 400            \
    --n_age_bins    15

echo ""
echo "============================================================"
echo " DONE: $(date)"
echo " Figures → ${OUTDIR}/"
echo "   cls_rank_analysis.png    — 4-panel combined"
echo "   panel_A_scree.png        — singular value spectrum + eff.rank"
echo "   panel_B_cumvar.png       — cumulative variance (90/95/99%)"
echo "   panel_C_pca_tissue.png   — stratified: 300/tissue (all visible)"
echo "   panel_D_pca_age.png      — stratified: 400/age-bin (all ages)"
echo "============================================================"
