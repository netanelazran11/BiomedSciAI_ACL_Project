#!/bin/bash -l
#SBATCH --job-name=view-design-eval
#SBATCH --partition=salmon
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00

#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/%x_%j.out
#SBATCH --error=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/%x_%j.err

# ─────────────────────────────────────────────────────────────────────────────
# Disjoint-view evaluation + multi-seed view consistency on the existing
# checkpoint (inference only, no retraining).
#
# Motivated by scConcept, whose views are disjoint gene panels while ours are
# independent 50% draws that overlap. Tests (a) whether consistency survives a
# strict disjoint partition, and (b) how much the reported numbers move across
# random view draws.
#
# Usage: sbatch scripts/repr_analysis_v7b/run_view_design_eval.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
CKPT="${REPO}/outputs/pretrain-llama-wced/llama-6L-all49k-r0.5-w0.05-genomic-45468861/checkpoints/epoch=85-recon=0.0552-pcc=0.9713.ckpt"
DATA="${REPO}/../../../data/data_methyl_21k_h5ad/altumage_21k_3way.h5ad"
TOKENIZER="${REPO}/tokenizer_llama_pretrain49k"
GENOMIC_RANK="${REPO}/outputs/cpg_genomic_sort/cpg_genomic_rank_finetune.npy"
OUTDIR="${REPO}/figures/v7b_pretrain_cls/view_design"
N_SAMPLES="${N_SAMPLES:-2000}"
N_SEEDS="${N_SEEDS:-5}"

cd "${REPO}"
source /etc/profile.d/modules.sh 2>/dev/null || true
module purge 2>/dev/null || true
module load spack/all 2>/dev/null || true
module load cuda/12.3.2-gcc-5bv3kyh 2>/dev/null || true
source bmfm_methyl_env/bin/activate
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

[ -f "${CKPT}" ]         || { echo "ERROR: checkpoint not found: ${CKPT}"; exit 1; }
[ -f "${DATA}" ]         || { echo "ERROR: h5ad not found: ${DATA}"; exit 1; }
[ -f "${GENOMIC_RANK}" ] || { echo "ERROR: genomic rank not found: ${GENOMIC_RANK}"; exit 1; }

echo "============================================================"
echo "View-design evaluation (overlap vs disjoint, ${N_SEEDS} seeds)"
echo "Job: ${SLURM_JOB_ID}  Host: $(hostname)  Time: $(date)"
echo "============================================================"

python scripts/repr_analysis_v7b/view_design_eval.py \
    --checkpoint "${CKPT}" \
    --data "${DATA}" \
    --tokenizer "${TOKENIZER}" \
    --genomic_rank "${GENOMIC_RANK}" \
    --n_samples "${N_SAMPLES}" \
    --n_seeds "${N_SEEDS}" \
    --outdir "${OUTDIR}"

echo "DONE: $(date)"
