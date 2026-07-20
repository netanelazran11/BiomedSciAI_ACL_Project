#!/bin/bash -l
#SBATCH --job-name=v7b-repr-full
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
# Full V7b pretrain (ep85) representation analysis — new-architecture folder.
# Runs: (1) extract CLS + CpG-embedding matrices  →  answers Q1-Q4
#       (2) visualize_cls        (Bonus 1: PCA/UMAP colored by age/tissue)
#       (3) analyze_cpg_embeddings (Bonus 2: genomic locality in CpG embeddings)
#       (4) two_view_consistency (Bonus 3: contrastive-quality check, GPU)
# All outputs → figures/v7b_pretrain_cls/
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
cd "${REPO}"
source bmfm_methyl_env/bin/activate
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

CKPT="${CKPT:-${REPO}/outputs/pretrain-llama-wced/llama-6L-all49k-r0.5-w0.05-genomic-45468861/checkpoints/epoch=85-recon=0.0552-pcc=0.9713.ckpt}"
DATA="${DATA:-${REPO}/../../../data/data_methyl_21k_h5ad/altumage_21k_3way.h5ad}"
TOKENIZER="${REPO}/tokenizer_llama_pretrain49k"
GENOMIC_RANK="${REPO}/outputs/cpg_genomic_sort/cpg_genomic_rank_finetune.npy"
OUTDIR="${REPO}/figures/v7b_pretrain_cls"
D="scripts/repr_analysis_v7b"

echo "=== [1/4] extract CLS + CpG embedding matrices (Q1-Q4) ==="
python ${D}/extract_pretrain_cls.py \
    --checkpoint "${CKPT}" --data "${DATA}" --tokenizer "${TOKENIZER}" \
    --genomic_rank "${GENOMIC_RANK}" --outdir "${OUTDIR}" \
    --age_col age --label_cols tissue_type sex dataset --split_col split

echo "=== [2/4] visualize CLS (Bonus 1: PCA/UMAP) ==="
python ${D}/visualize_cls.py --dir "${OUTDIR}" \
    --color_cols age tissue_type sex dataset

echo "=== [3/4] CpG embedding genomic-locality (Bonus 2) ==="
python ${D}/analyze_cpg_embeddings.py --dir "${OUTDIR}"

echo "=== [4/4] two-view contrastive consistency (Bonus 3) ==="
python ${D}/two_view_consistency.py \
    --checkpoint "${CKPT}" --data "${DATA}" --tokenizer "${TOKENIZER}" \
    --genomic_rank "${GENOMIC_RANK}" --n_samples 2000 --outdir "${OUTDIR}"

echo "============================================================"
echo " FULL ANALYSIS DONE: $(date)"
echo " Outputs → ${OUTDIR}/"
echo "   report.txt / analysis_summary.json        (Q1-Q4)"
echo "   cls_pca_panels.png / cls_umap_panels.png   (Bonus 1)"
echo "   cpg_embedding_analysis.json                (Bonus 2)"
echo "   two_view_consistency.json                  (Bonus 3)"
echo "============================================================"
