#!/bin/bash -l
#SBATCH --job-name=pretrain-llama-6L-recononly
#SBATCH --partition=goldfish
#SBATCH --gres=gpu:h200:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --time=168:00:00

#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_ablation_infonce/%x_%j.out
#SBATCH --error=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_ablation_infonce/%x_%j.err

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# InfoNCE ablation — reconstruction-only arm, full corpus.
#
# Identical to scripts/llama/pretrain_llama_small_6L_contrastive.sh (the published
# run: jobs 45402392 → 45468861, checkpoint epoch 85) except:
#   - training module = WCEDLlamaReconOnlyModule: both views encoded, both
#     reconstruction losses averaged, InfoNCE NOT in the loss (monitor only)
#   - wced_contrastive=true is kept: it only makes the collator build view 2
#   - analysis checkpoints at ANALYSIS_EPOCHS; stops after STOP_AFTER_EPOCH
#     (the LR schedule still uses the 300-epoch budget, as in the published run)
#   - own output tree (outputs/ablation_infonce/), own log dir
#
# Usage:
#   SMOKE=1 sbatch --partition=salmon --gres=gpu:l40s:1 --cpus-per-task=16 --mem=128G \
#       --time=02:00:00 ablation_infonce/pretrain_recon_only.sh
#   sbatch ablation_infonce/pretrain_recon_only.sh
#   # continue after the 168 h wall (same output dir, new W&B run, like the published run):
#   RUN_TAG=<first job id> RESUME_CHECKPOINT=<outdir>/checkpoints/last.ckpt \
#       sbatch ablation_infonce/pretrain_recon_only.sh
# ─────────────────────────────────────────────────────────────────────────────

REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
LOGDIR="${REPO}/logs_ablation_infonce"
SMOKE="${SMOKE:-0}"

# ─── Data (same as published run) ────────────────────────────────────────────
DATA_DIR="/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_pretrain_type3_h5ad"
PRETRAIN_DATA="${DATA_DIR}/methylgpt_pretrain_type3.h5ad"
PROBE_IDS_CSV="${DATA_DIR}/probe_ids_type3_pretrain.csv"
GENOMIC_RANK_NPY="${REPO}/outputs/cpg_genomic_sort/cpg_genomic_rank.npy"
TOKENIZER_PATH="${REPO}/tokenizer_llama_pretrain49k"

# ─── Architecture (same) ─────────────────────────────────────────────────────
HIDDEN_SIZE=256
NUM_LAYERS=6
NUM_HEADS=4
INTERMEDIATE_SIZE=512
ROPE_THETA=10000.0
N_SIN_BASIS=48
BASIS_SCALE=2.0

# ─── WCED (same, except the objective) ───────────────────────────────────────
SUBSET_K=49156
INPUT_RATIO=0.5
AGE_WEIGHT=0.0
CONTRASTIVE_TEMP=0.1          # used only by the monitor
NORMALIZE_LOSS=true
DECODER_DROPOUT=0.1
DIAG_CHECK_EVERY=5

# ─── Training (same) ─────────────────────────────────────────────────────────
LR=3e-4
WEIGHT_DECAY=0.01
WARMUP_STEPS=3000
BATCH_SIZE="${BATCH_SIZE:-32}"
ACCUM=2
PRETRAIN_EPOCHS=300           # schedule budget — keep 300 to match the published LR curve
EARLY_STOP=100                # > STOP_AFTER_EPOCH: must not fire before epoch 87. The published run never early-stopped
                              # (job 45402392: patience 60 on val/loss, older code; job 45468861: 30 on val/recon).

# ─── Ablation controls ───────────────────────────────────────────────────────
export ANALYSIS_EPOCHS="${ANALYSIS_EPOCHS:-0,1,2,4,9,19,36,85}"
export STOP_AFTER_EPOCH="${STOP_AFTER_EPOCH:-87}"   # published run's last epoch
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
RUN_TAG="${RUN_TAG:-${SLURM_JOB_ID}}"

LIMITS=()
WANDB_ON=true
if [ "${SMOKE}" = "1" ]; then
    BATCH_SIZE="${BATCH_SIZE_SMOKE:-4}"
    export ANALYSIS_EPOCHS="0"
    export STOP_AFTER_EPOCH="1"
    LIMITS=(limit_train_batches=20 limit_val_batches=5)
    WANDB_ON=false
    RUN_TAG="smoke-${SLURM_JOB_ID}"
    export LIMIT_TEST_BATCHES=5
fi

WANDB_ENTITY="netanelazran11-hebrew-university-of-jerusalem"
WANDB_PROJECT="pretrain-llama-wced"   # same project as the published run, for overlays
WANDB_RUN_NAME="llama-6L-all49k-r${INPUT_RATIO}-RECONONLY-2view-genomic-${SLURM_JOB_ID}"
OUTDIR="${REPO}/outputs/ablation_infonce/recon_only_${RUN_TAG}"

mkdir -p "${LOGDIR}" "${OUTDIR}"

if [ ! -f "${GENOMIC_RANK_NPY}" ]; then
    echo "ERROR: genomic rank file not found: ${GENOMIC_RANK_NPY}"; exit 1
fi
if [ -n "${RESUME_CHECKPOINT}" ] && [ ! -f "${RESUME_CHECKPOINT}" ]; then
    echo "ERROR: resume checkpoint not found: ${RESUME_CHECKPOINT}"; exit 1
fi

echo "============================================================"
echo "INFONCE ABLATION — RECON-ONLY ARM (two-view reconstruction, λ=0)"
echo "Job: ${SLURM_JOB_ID} | Host: $(hostname) | Time: $(date) | SMOKE=${SMOKE}"
echo "Git: $(git -C "${REPO}" rev-parse --short HEAD)  dirty=$(git -C "${REPO}" status --porcelain -- bmfm_methylation ablation_infonce | wc -l)"
echo "Output: ${OUTDIR}"
echo "Analysis epochs: ${ANALYSIS_EPOCHS} | stop after: ${STOP_AFTER_EPOCH} | resume: ${RESUME_CHECKPOINT:-none}"
echo "============================================================"

source /etc/profile.d/modules.sh 2>/dev/null || source /usr/share/modules/init/bash 2>/dev/null || true
module purge
module load spack/all
module load cuda/12.3.2-gcc-5bv3kyh

cd "${REPO}"
source ablation_infonce/env.sh

export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Equivalence test against the unmodified module (CPU, seconds). Abort on failure.
"${PY}" -m ablation_infonce.test_recon_only

"${PY}" -m ablation_infonce.pretrain_recon_only \
    data_path="${PRETRAIN_DATA}" \
    probe_ids_csv="${PROBE_IDS_CSV}" \
    tokenizer_path="${TOKENIZER_PATH}" \
    output_directory="${OUTDIR}" \
    pretraining_mode=wced \
    data_module.subset_k="${SUBSET_K}" \
    data_module.batch_size="${BATCH_SIZE}" \
    data_module.num_workers=14 \
    data_module.bmfm_style=false \
    model.hidden_size="${HIDDEN_SIZE}" \
    model.num_hidden_layers="${NUM_LAYERS}" \
    model.num_attention_heads="${NUM_HEADS}" \
    model.intermediate_size="${INTERMEDIATE_SIZE}" \
    model.rope_theta="${ROPE_THETA}" \
    model.n_sin_basis="${N_SIN_BASIS}" \
    model.basis_scale="${BASIS_SCALE}" \
    trainer.learning_rate="${LR}" \
    trainer.weight_decay="${WEIGHT_DECAY}" \
    trainer.warmup_steps="${WARMUP_STEPS}" \
    wced_input_ratio="${INPUT_RATIO}" \
    wced_age_weight="${AGE_WEIGHT}" \
    wced_contrastive=true \
    wced_contrastive_weight=0.0 \
    wced_contrastive_temp="${CONTRASTIVE_TEMP}" \
    wced_normalize_loss="${NORMALIZE_LOSS}" \
    wced_decoder_dropout="${DECODER_DROPOUT}" \
    wced_genomic_rank_path="${GENOMIC_RANK_NPY}" \
    pretrain_epochs="${PRETRAIN_EPOCHS}" \
    accumulate_grad_batches="${ACCUM}" \
    early_stop_patience="${EARLY_STOP}" \
    gradient_clip_val=1.0 \
    precision="16-mixed" \
    diag_check_every="${DIAG_CHECK_EVERY}" \
    track_wandb.enabled="${WANDB_ON}" \
    track_wandb.project="${WANDB_PROJECT}" \
    track_wandb.entity="${WANDB_ENTITY}" \
    track_wandb.name="${WANDB_RUN_NAME}" \
    ${LIMITS[@]+"${LIMITS[@]}"} \
    ${RESUME_CHECKPOINT:+"resume_checkpoint='${RESUME_CHECKPOINT}'"}

echo "============================================================"
echo "Recon-only arm finished: $(date)"
echo "Checkpoints: ${OUTDIR}/checkpoints/ (analysis/ holds the fixed-epoch snapshots)"
echo "============================================================"
