#!/bin/bash -l
#SBATCH --job-name=avg-bio
#SBATCH --partition=salmon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00

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
MEAN_NPY="${REPO}/outputs/repr_analysis/pretrain_cls_169k_44892802/embeddings_mean.npy"
RANDOM_NPY="${REPO}/outputs/repr_analysis/cls_probing_44905909/embeddings_random_cls.npy"
METADATA="${REPO}/outputs/repr_analysis/cls_probing_44931911/metadata.csv"
OUTDIR="${REPO}/outputs/repr_analysis/avg_bio_${SLURM_JOB_ID}"

echo "============================================================"
echo " scIB Avg_bio Evaluation"
echo " Job : ${SLURM_JOB_ID}  Host: $(hostname)  Time: $(date)"
echo "============================================================"
echo " CLS npy    : ${CLS_NPY}"
echo " Mean npy   : ${MEAN_NPY}"
echo " Random npy : ${RANDOM_NPY}"
echo " Metadata   : ${METADATA}"
echo " Outdir     : ${OUTDIR}"
echo "============================================================"

# Validate required files
for f in "${CLS_NPY}" "${MEAN_NPY}" "${METADATA}"; do
    if [ ! -f "${f}" ]; then
        echo "ERROR: required file not found: ${f}"
        exit 1
    fi
done

RANDOM_ARG=""
if [ -f "${RANDOM_NPY}" ]; then
    RANDOM_ARG="--random_npy ${RANDOM_NPY}"
    echo "  Random embeddings found — will include in comparison"
else
    echo "  WARNING: random embeddings not found at ${RANDOM_NPY} — skipping"
fi

python scripts/repr_analysis/avg_bio_eval.py \
    --cls_npy      "${CLS_NPY}"   \
    --mean_npy     "${MEAN_NPY}"  \
    ${RANDOM_ARG}                 \
    --metadata_csv "${METADATA}"  \
    --label_col    tissue         \
    --min_samples  10             \
    --n_pca        50             \
    --leiden_res   0.6            \
    --outdir       "${OUTDIR}"

echo ""
echo "============================================================"
echo " ALL DONE: $(date)"
echo " Outputs → ${OUTDIR}/"
echo "   avg_bio_results.csv         NMI / ARI / ASW / Avg_bio per embedding"
echo "   figures/avg_bio_comparison.png   bar chart"
echo "============================================================"
