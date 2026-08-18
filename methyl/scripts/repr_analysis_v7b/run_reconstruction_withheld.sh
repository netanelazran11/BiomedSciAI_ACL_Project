#!/bin/bash -l
#SBATCH --job-name=recon-withheld-eval
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
# Canonical withheld-CpG reconstruction eval (Figure 2C/2D + Supp S3).
# Replaces the legacy reconstruction_baselines run, which used the wrong
# checkpoint, input_ratio=1.0 (no withheld CpGs), and no genomic position_ids.
#
# Also prints the pretraining h5ad's total row count + split sizes, to close
# the manuscript's "verify 169,120" todo in the same job.
#
# Usage: sbatch scripts/repr_analysis_v7b/run_reconstruction_withheld.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
CKPT="${REPO}/outputs/pretrain-llama-wced/llama-6L-all49k-r0.5-w0.05-genomic-45468861/checkpoints/epoch=85-recon=0.0552-pcc=0.9713.ckpt"
DATA="/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_pretrain_type3_h5ad/methylgpt_pretrain_type3.h5ad"
TOKENIZER="${REPO}/tokenizer_llama_pretrain49k"
GENOMIC_RANK="${REPO}/outputs/cpg_genomic_sort/cpg_genomic_rank.npy"
OUTDIR="${REPO}/figures/v7b_pretrain_cls/reconstruction_withheld"
SPLIT="${SPLIT:-test}"
MAX_SAMPLES="${MAX_SAMPLES:-5000}"

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
[ -f "${DATA}" ]         || { echo "ERROR: pretrain h5ad not found: ${DATA}"; exit 1; }
[ -f "${GENOMIC_RANK}" ] || { echo "ERROR: genomic rank not found: ${GENOMIC_RANK}"; exit 1; }

echo "============================================================"
echo "Withheld-CpG reconstruction eval (canonical ep85 checkpoint)"
echo "Job: ${SLURM_JOB_ID}  Host: $(hostname)  Time: $(date)"
echo "Split: ${SPLIT}  Max samples: ${MAX_SAMPLES}"
echo "============================================================"

# ── Manuscript verification: total pretraining corpus size + split counts ────
python - <<'PY'
import anndata
a = anndata.read_h5ad(
    "/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_pretrain_type3_h5ad/methylgpt_pretrain_type3.h5ad",
    backed="r",
)
print(f"PRETRAIN H5AD VERIFICATION: n_obs={a.n_obs:,}  n_vars={a.n_vars:,}")
if "split" in a.obs.columns:
    print("  split counts:", a.obs["split"].value_counts().to_dict())
else:
    print("  NOTE: no 'split' column; columns =", list(a.obs.columns))
PY

python scripts/repr_analysis_v7b/reconstruction_withheld_eval.py \
    --checkpoint "${CKPT}" \
    --data "${DATA}" \
    --tokenizer "${TOKENIZER}" \
    --genomic_rank "${GENOMIC_RANK}" \
    --split "${SPLIT}" \
    --max_samples "${MAX_SAMPLES}" \
    --outdir "${OUTDIR}"

echo "DONE: $(date)"
