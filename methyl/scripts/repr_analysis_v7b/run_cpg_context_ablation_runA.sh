#!/bin/bash -l
#SBATCH --job-name=cpg-context-ablationA
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
# Genomic RoPE ablation — contextualized CpG extraction for RUN A (genomic).
# feed_position_ids=true (default): matches how Run A was actually trained.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
CKPT="${REPO}/outputs/pretrain-ablation-rope/ropeA-genomic-5k-45740643/checkpoints/epoch=60-recon=0.0906-pcc=0.9357.ckpt"
DATA="${REPO}/outputs/ablation_rope/pretrain_subset_5000.h5ad"
TOKENIZER="${REPO}/tokenizer_llama_pretrain49k"
GENOMIC_RANK="${REPO}/outputs/cpg_genomic_sort/cpg_genomic_rank.npy"
OUTDIR="${REPO}/outputs/ablation_rope/cpg_context_A"

cd "${REPO}"
source /etc/profile.d/modules.sh 2>/dev/null || true
module purge 2>/dev/null || true
module load spack/all 2>/dev/null || true
module load cuda/12.3.2-gcc-5bv3kyh 2>/dev/null || true
source bmfm_methyl_env/bin/activate
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo "Checkpoint: ${CKPT}"
[ -f "${CKPT}" ] || { echo "ERROR: checkpoint not found"; exit 1; }

python scripts/repr_analysis_v7b/extract_contextual_cpg.py \
    --checkpoint "${CKPT}" \
    --data "${DATA}" \
    --tokenizer "${TOKENIZER}" \
    --genomic_rank "${GENOMIC_RANK}" \
    --feed_position_ids true \
    --max_samples 5000 \
    --batch_size 16 \
    --outdir "${OUTDIR}"

echo "DONE: $(date) -> ${OUTDIR}/"
