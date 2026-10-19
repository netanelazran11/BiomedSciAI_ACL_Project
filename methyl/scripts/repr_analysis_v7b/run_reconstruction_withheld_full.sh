#!/bin/bash -l
#SBATCH --job-name=recon-withheld-full
#SBATCH --partition=salmon
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/%x_%j.out
#SBATCH --error=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/%x_%j.err

# ─────────────────────────────────────────────────────────────────────────────
# Withheld-CpG reconstruction on ALL 16,912 held-out pretraining profiles, with Pearson r and the
# observed-vs-predicted histogram over ALL withheld positions (Fig. 2c and Supplementary Fig. S2c).
# Same checkpoint, data, tokenizer, genomic rank, input_ratio and seed as run_reconstruction_withheld.sh;
# writes to a NEW folder, so the existing 5,000-profile results are not touched.
#
# Usage (from methyl/ on the cluster):
#   sbatch scripts/repr_analysis_v7b/run_reconstruction_withheld_full.sh
# Smoke test first (≈10 min): same command with  --export=ALL,SMOKE=1  -> 320 profiles, separate folder.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
CKPT="${REPO}/outputs/pretrain-llama-wced/llama-6L-all49k-r0.5-w0.05-genomic-45468861/checkpoints/epoch=85-recon=0.0552-pcc=0.9713.ckpt"
DATA="/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_pretrain_type3_h5ad/methylgpt_pretrain_type3.h5ad"
TOKENIZER="${REPO}/tokenizer_llama_pretrain49k"
GENOMIC_RANK="${REPO}/outputs/cpg_genomic_sort/cpg_genomic_rank.npy"
OUTDIR="${REPO}/figures/v7b_pretrain_cls/reconstruction_withheld_full"
EXPECTED_N=16912
SMOKE="${SMOKE:-0}"

cd "${REPO}"
mkdir -p "${REPO}/logs_llama-wced"
source ablation_infonce/env.sh                     # sets PY (Python 3.11 + bmfm_methyl_env packages)
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

for f in "${CKPT}" "${DATA}" "${GENOMIC_RANK}"; do
    [ -f "${f}" ] || { echo "ERROR: not found: ${f}"; exit 1; }
done
[ -d "${TOKENIZER}" ] || { echo "ERROR: tokenizer folder not found: ${TOKENIZER}"; exit 1; }
"${PY}" -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('GPU:', torch.cuda.get_device_name(0))"

SCRIPT="scripts/repr_analysis_v7b/reconstruction_withheld_eval_full.py"
if [ "${SMOKE}" = "1" ]; then
    # smoke: first 320 profiles only, own folder, no population check -- verifies the code path, not results
    OUTDIR="${OUTDIR}_smoke"
    "${PY}" - <<PY
import runpy, sys
from bmfm_methylation.shared import data_module as dm
_orig = dm.MethylationDataset.__len__
dm.MethylationDataset.__len__ = lambda self: min(320, _orig(self))
sys.argv = ["${SCRIPT}", "--checkpoint", "${CKPT}", "--data", "${DATA}", "--tokenizer", "${TOKENIZER}",
            "--genomic_rank", "${GENOMIC_RANK}", "--split", "test", "--expected_n", "0", "--outdir", "${OUTDIR}"]
runpy.run_path("${SCRIPT}", run_name="__main__")
PY
    echo "SMOKE DONE: $(date)  -> ${OUTDIR}"
    exit 0
fi

echo "============================================================"
echo "Withheld-CpG reconstruction, ALL held-out profiles (expected ${EXPECTED_N})"
echo "Job: ${SLURM_JOB_ID:-local}  Host: $(hostname)  Time: $(date)"
echo "Output: ${OUTDIR}"
echo "============================================================"

"${PY}" "${SCRIPT}" \
    --checkpoint "${CKPT}" \
    --data "${DATA}" \
    --tokenizer "${TOKENIZER}" \
    --genomic_rank "${GENOMIC_RANK}" \
    --split test \
    --expected_n "${EXPECTED_N}" \
    --outdir "${OUTDIR}"

echo "DONE: $(date)"
