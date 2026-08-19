#!/bin/bash -l
#SBATCH --job-name=viewdesign-pilot
#SBATCH --partition=salmon
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00

#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/%x_%j.out
#SBATCH --error=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/%x_%j.err

# ─────────────────────────────────────────────────────────────────────────────
# View-design pretraining pilot (scConcept-motivated, Bahrami et al. 2025).
#
# Four matched conditions on the SAME fixed 5,000-profile subset already used
# for the genomic-RoPE ablation, so runtime and comparability are known:
#
#   COND=baseline   overlapping views, cross-view negatives   (published setup)
#   COND=disjoint   disjoint views,    cross-view negatives
#   COND=sameview   overlapping views, + same-view negatives
#   COND=both       disjoint views,    + same-view negatives  (closest to scConcept)
#
# Everything else -- architecture, data, seed, optimiser, epochs -- is held
# fixed, so any difference is attributable to view construction and the
# negative set. Isolated: writes to its own output tree, touches no existing
# run directory.
#
# Usage:
#   COND=baseline sbatch scripts/llama/pretrain_ablation_viewdesign.sh
#   COND=disjoint sbatch scripts/llama/pretrain_ablation_viewdesign.sh
#   COND=sameview sbatch scripts/llama/pretrain_ablation_viewdesign.sh
#   COND=both     sbatch scripts/llama/pretrain_ablation_viewdesign.sh
#
# Resume after a wall-clock timeout:
#   COND=<cond> RESUME_CHECKPOINT=/path/to/last.ckpt sbatch ...
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

COND="${COND:?ERROR: set COND=baseline|disjoint|sameview|both}"

case "${COND}" in
    baseline) DISJOINT=false; SAMEVIEW=false ;;
    disjoint) DISJOINT=true;  SAMEVIEW=false ;;
    sameview) DISJOINT=false; SAMEVIEW=true  ;;
    both)     DISJOINT=true;  SAMEVIEW=true  ;;
    *) echo "ERROR: unknown COND='${COND}'"; exit 1 ;;
esac

REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
LOGDIR="${REPO}/logs_llama-wced"

# Same 5k subset as the RoPE ablation (created by create_pretrain_subset_h5ad.py)
PRETRAIN_DATA="${REPO}/outputs/ablation_rope/pretrain_subset_5k.h5ad"
PROBE_IDS_CSV="/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_pretrain_type3_h5ad/probe_ids_type3_pretrain.csv"
GENOMIC_RANK_NPY="${REPO}/outputs/cpg_genomic_sort/cpg_genomic_rank.npy"
TOKENIZER_PATH="${REPO}/tokenizer_llama_pretrain49k"

# Architecture — identical to the production model
HIDDEN_SIZE=256; NUM_LAYERS=6; NUM_HEADS=4; INTERMEDIATE_SIZE=512
SUBSET_K=49156; INPUT_RATIO=0.5
CONTRASTIVE_WEIGHT=0.05; CONTRASTIVE_TEMP=0.1
LR=3e-4; WARMUP_STEPS=500; BATCH_SIZE=32; ACCUM=2
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-150}"
EARLY_STOP="${EARLY_STOP:-30}"
SEED="${SEED:-42}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

WANDB_ENTITY="netanelazran11-hebrew-university-of-jerusalem"
WANDB_PROJECT="pretrain-llama-wced"
WANDB_RUN_NAME="viewdesign-${COND}-5k-${SLURM_JOB_ID}"
OUTDIR="${REPO}/outputs/ablation_viewdesign/${WANDB_RUN_NAME}"

mkdir -p "${LOGDIR}" "${OUTDIR}"

[ -f "${PRETRAIN_DATA}" ]  || { echo "ERROR: subset h5ad not found: ${PRETRAIN_DATA}"; exit 1; }
[ -f "${GENOMIC_RANK_NPY}" ] || { echo "ERROR: genomic rank not found"; exit 1; }

echo "============================================================"
echo "VIEW-DESIGN PILOT — condition: ${COND}"
echo "  disjoint_views          = ${DISJOINT}"
echo "  same_view_negatives     = ${SAMEVIEW}"
echo "Job: ${SLURM_JOB_ID} | Host: $(hostname) | Time: $(date)"
echo "Data:   ${PRETRAIN_DATA}"
echo "Output: ${OUTDIR}"
echo "============================================================"

source /etc/profile.d/modules.sh 2>/dev/null || true
module purge 2>/dev/null || true
module load spack/all 2>/dev/null || true
module load cuda/12.3.2-gcc-5bv3kyh 2>/dev/null || true

cd "${REPO}"
source bmfm_methyl_env/bin/activate
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONUNBUFFERED=1

python -m bmfm_methylation.llama.pretrain_llama \
    data_path="${PRETRAIN_DATA}" \
    probe_ids_csv="${PROBE_IDS_CSV}" \
    tokenizer_path="${TOKENIZER_PATH}" \
    output_directory="${OUTDIR}" \
    pretraining_mode=wced \
    data_module.subset_k="${SUBSET_K}" \
    data_module.batch_size="${BATCH_SIZE}" \
    data_module.num_workers=8 \
    data_module.bmfm_style=false \
    model.hidden_size="${HIDDEN_SIZE}" \
    model.num_hidden_layers="${NUM_LAYERS}" \
    model.num_attention_heads="${NUM_HEADS}" \
    model.intermediate_size="${INTERMEDIATE_SIZE}" \
    trainer.learning_rate="${LR}" \
    trainer.warmup_steps="${WARMUP_STEPS}" \
    wced_input_ratio="${INPUT_RATIO}" \
    wced_age_weight=0.0 \
    wced_contrastive=true \
    wced_contrastive_weight="${CONTRASTIVE_WEIGHT}" \
    wced_contrastive_temp="${CONTRASTIVE_TEMP}" \
    +wced_contrastive_same_view_negatives="${SAMEVIEW}" \
    +wced_disjoint_views="${DISJOINT}" \
    wced_normalize_loss=true \
    wced_genomic_rank_path="${GENOMIC_RANK_NPY}" \
    pretrain_epochs="${PRETRAIN_EPOCHS}" \
    accumulate_grad_batches="${ACCUM}" \
    early_stop_patience="${EARLY_STOP}" \
    gradient_clip_val=1.0 \
    precision="16-mixed" \
    seed.seed_value="${SEED}" \
    track_wandb.enabled=true \
    track_wandb.project="${WANDB_PROJECT}" \
    track_wandb.entity="${WANDB_ENTITY}" \
    track_wandb.name="${WANDB_RUN_NAME}" \
    "+track_wandb.group=viewdesign-pilot" \
    ${RESUME_CHECKPOINT:+"resume_checkpoint='${RESUME_CHECKPOINT}'"}

echo "============================================================"
echo "Condition ${COND} finished: $(date)"
echo "Checkpoints: ${OUTDIR}/checkpoints/"
echo "============================================================"
