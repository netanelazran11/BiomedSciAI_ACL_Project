#!/bin/bash -l
#SBATCH --job-name=pilot-eval
#SBATCH --partition=salmon
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --array=0-3,6
#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_ablation_infonce/%x_%A_%a.out
#SBATCH --error=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_ablation_infonce/%x_%A_%a.err

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Pilot evaluation: the SAME analyses the manuscript reports, run unchanged on each arm.
#   tasks 0–3  pilot arms λ 0 / 0.01 / 0.05 / 0.2 (4–5 = optional seed-43 replicates), checkpoint EVAL_EPOCH
#   task  6    published checkpoint (epoch 85) — pipeline check: must reproduce the paper's numbers
#
# Per task, into outputs/ablation_infonce/pilot_eval/<arm>/ (each step skipped if already done):
#   1. view consistency on the FULL held-out partition (16,912), overlap + disjoint, seed 0
#      (view_consistency_heldout_full.py; also saves the embeddings for hard_negative_eval.py)
#   2. reconstruction of withheld CpGs, 5,000 held-out profiles (reconstruction_withheld_eval.py)
#   3. frozen CLS on AltumAge: geometry + age/tissue/sex/dataset probes (extract_pretrain_cls.py)
# Then, locally after rsync: hard_negative_eval.py per arm (with --per-query-out) and summarize_pilot.py.
#
#   sbatch ablation_infonce/evaluate_pilot.sh
#   EVAL_EPOCH=9 sbatch --array=0,2 ablation_infonce/evaluate_pilot.sh   # an earlier snapshot
# ─────────────────────────────────────────────────────────────────────────────

LAMBDAS=(0.0 0.01 0.05 0.2 0.0 0.05)
SEEDS=(42 42 42 42 43 43)
TASK="${SLURM_ARRAY_TASK_ID:?run as an array job}"
EVAL_EPOCH="${EVAL_EPOCH:-19}"

REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
PRETRAIN_H5AD="/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_pretrain_type3_h5ad/methylgpt_pretrain_type3.h5ad"
ALTUMAGE_H5AD="${REPO}/../../../data/data_methyl_21k_h5ad/altumage_21k_3way.h5ad"
TOKENIZER="${REPO}/tokenizer_llama_pretrain49k"
RANK_PRETRAIN="${REPO}/outputs/cpg_genomic_sort/cpg_genomic_rank.npy"
RANK_FINETUNE="${REPO}/outputs/cpg_genomic_sort/cpg_genomic_rank_finetune.npy"
PILOT_ROOT="${REPO}/outputs/ablation_infonce/pilot"
EVAL_ROOT="${REPO}/outputs/ablation_infonce/pilot_eval"

if [ "${TASK}" = "6" ]; then
    ARM="published_ep85"
    CKPT="${REPO}/outputs/pretrain-llama-wced/llama-6L-all49k-r0.5-w0.05-genomic-45468861/checkpoints/epoch=85-recon=0.0552-pcc=0.9713.ckpt"
else
    ARM="lam${LAMBDAS[$TASK]}_seed${SEEDS[$TASK]}"
    CKPT="${PILOT_ROOT}/${ARM}/checkpoints/analysis/epoch=${EVAL_EPOCH}-analysis.ckpt"
    ARM="${ARM}_ep${EVAL_EPOCH}"
fi
OUT="${EVAL_ROOT}/${ARM}"
mkdir -p "${OUT}" "${REPO}/logs_ablation_infonce"

for f in "${CKPT}" "${PRETRAIN_H5AD}" "${ALTUMAGE_H5AD}" "${RANK_PRETRAIN}" "${RANK_FINETUNE}"; do
    [ -f "$f" ] || { echo "ERROR: not found: $f"; exit 1; }
done

echo "PILOT EVAL task=${TASK} arm=${ARM} | Git $(git -C "${REPO}" rev-parse --short HEAD) | $(hostname) | $(date)"
echo "Checkpoint: ${CKPT}"

source /etc/profile.d/modules.sh 2>/dev/null || true
# Cluster modules changed in Oct 2026 (spack/all now needs spack-huji; cuda/12.3.2-gcc-5bv3kyh is gone).
# Nothing here needs spack: Python comes from env.sh and torch 2.10.0+cu128 bundles its CUDA runtime.
module purge 2>/dev/null || true
module load cuda/12.8.1 2>/dev/null || echo "WARN: module cuda/12.8.1 not loaded (torch bundles its CUDA runtime)"
cd "${REPO}"
source ablation_infonce/env.sh
"${PY}" -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('GPU:', torch.cuda.get_device_name(0), '| torch', torch.__version__, '| CUDA', torch.version.cuda)"
export PYTHONPATH="${REPO}:${REPO}/scripts/repr_analysis_v7b:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
D=scripts/repr_analysis_v7b

# 1. view consistency (held-out, full)
for COND in overlap disjoint; do
    if [ -f "${OUT}/view_consistency/per_run/${COND}_seed0.json" ]; then echo "skip view ${COND}"; continue; fi
    "${PY}" ${D}/view_consistency_heldout_full.py \
        --checkpoint "${CKPT}" --data "${PRETRAIN_H5AD}" --split test \
        --tokenizer "${TOKENIZER}" --genomic_rank "${RANK_PRETRAIN}" \
        --condition "${COND}" --seed 0 --n_samples 0 --outdir "${OUT}/view_consistency"
done

# 2. reconstruction of withheld CpGs
if [ ! -f "${OUT}/reconstruction/reconstruction_withheld_summary.json" ]; then
    "${PY}" ${D}/reconstruction_withheld_eval.py \
        --checkpoint "${CKPT}" --data "${PRETRAIN_H5AD}" --tokenizer "${TOKENIZER}" \
        --genomic_rank "${RANK_PRETRAIN}" --split test --max_samples 5000 --outdir "${OUT}/reconstruction"
else echo "skip reconstruction"; fi

# 3. frozen CLS probes on AltumAge
if [ ! -f "${OUT}/frozen_probes/analysis_summary.json" ]; then
    "${PY}" ${D}/extract_pretrain_cls.py \
        --checkpoint "${CKPT}" --data "${ALTUMAGE_H5AD}" --tokenizer "${TOKENIZER}" \
        --genomic_rank "${RANK_FINETUNE}" --outdir "${OUT}/frozen_probes" --batch_size 32 \
        --age_col age --label_cols tissue_type sex dataset --split_col split
else echo "skip frozen probes"; fi

echo "Done ${ARM}: $(date)"
