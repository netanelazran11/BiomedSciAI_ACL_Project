#!/bin/bash -l
#SBATCH --job-name=v7b-cpg-context-full
#SBATCH --partition=salmon
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2:00:00

#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/%x_%j.out
#SBATCH --error=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/%x_%j.err

# ─────────────────────────────────────────────────────────────────────────────
# Re-run of extract_contextual_cpg.py over the FULL dataset (~10,988 samples,
# input_ratio=1.0 so all 21,368 CpGs per sample) instead of the previous
# max_samples=512 default. More samples -> less per-CpG averaging noise ->
# expected to sharpen (not weaken) the genomic-locality-decay signal used in
# the Genomic RoPE evidence. Also now saves contextual_cpg_meta.json with the
# actual n_samples used, so this is no longer ambiguous.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
CKPT="${REPO}/outputs/pretrain-llama-wced/llama-6L-all49k-r0.5-w0.05-genomic-45468861/checkpoints/epoch=85-recon=0.0552-pcc=0.9713.ckpt"
DATA="${REPO}/../../../data/data_methyl_21k_h5ad/altumage_21k_3way.h5ad"
TOKENIZER="${REPO}/tokenizer_llama_pretrain49k"
GENOMIC_RANK="${REPO}/outputs/cpg_genomic_sort/cpg_genomic_rank_finetune.npy"
OUTDIR="${REPO}/figures/v7b_cpg_context"

cd "${REPO}"
source /etc/profile.d/modules.sh 2>/dev/null || true
module purge 2>/dev/null || true
module load spack/all 2>/dev/null || true
module load cuda/12.3.2-gcc-5bv3kyh 2>/dev/null || true
source bmfm_methyl_env/bin/activate
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

python scripts/repr_analysis_v7b/extract_contextual_cpg.py \
    --checkpoint "${CKPT}" --data "${DATA}" --tokenizer "${TOKENIZER}" \
    --genomic_rank "${GENOMIC_RANK}" --max_samples 999999 --batch_size 16 \
    --outdir "${OUTDIR}"

echo "DONE: $(date) — contextual_cpg_emb.npy / contextual_cpg_meta.json in ${OUTDIR}/"
echo "Next: rebuild cpg_genomic_locality_decay.png from the refreshed contextual_cpg_emb.npy"
