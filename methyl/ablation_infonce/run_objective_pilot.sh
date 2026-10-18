#!/bin/bash -l
#SBATCH --job-name=pilot-infonce
#SBATCH --partition=goldfish
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --array=0-5
#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_ablation_infonce/%x_%A_%a.out
#SBATCH --error=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_ablation_infonce/%x_%A_%a.err

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# InfoNCE-weight pilot: six arms, one array task each, identical except λ (and seed for 4–5).
#
#   task  λ      seed       question
#   0     0      42         reconstruction only (two views)
#   1     0.01   42         weak contrastive
#   2     0.05   42         published weight
#   3     0.2    42         strong contrastive
#   4     0      43         seed replicate of 0  ─┐ the 0 vs 0.05 gap must exceed
#   5     0.05   43         seed replicate of 2  ─┘ the seed-to-seed gap to count
#
# Every arm runs WCEDLlamaAblationModule (ABLATION_MODULE=sweep): two-view reconstruction
# + λ · InfoNCE; at λ > 0 identical to the published objective (test_recon_only.py, check 7).
# Same architecture, genomic RoPE, optimiser, per-GPU batch 32 (31 InfoNCE negatives, as in
# the published run), data subset, views and data order (per seed).
#
# Pilot budget (not the published one): 1×H200, effective batch 64 (32 × accum 2, as the RoPE
# pilot), 20 epochs over 9,000 profiles ≈ 2,800 optimiser steps, warm-up 200 steps, cosine
# over the 20 epochs. Estimated ~1.7 h/epoch → ~35 h per arm (RoPE pilot throughput).
#
# Usage:
#   SMOKE=1 sbatch --time=01:00:00 ablation_infonce/run_objective_pilot.sh   # all 6 arms, tiny
#   sbatch ablation_infonce/run_objective_pilot.sh
#   sbatch --array=2,5 ablation_infonce/run_objective_pilot.sh               # a subset of arms
#   # resume one arm after the wall:
#   RESUME_CHECKPOINT=<armdir>/checkpoints/last.ckpt sbatch --array=<k> ablation_infonce/run_objective_pilot.sh
# ─────────────────────────────────────────────────────────────────────────────

LAMBDAS=(0.0 0.01 0.05 0.2 0.0 0.05)
SEEDS=(42 42 42 42 43 43)
TASK="${SLURM_ARRAY_TASK_ID:?run as an array job}"
LAM="${LAMBDAS[$TASK]}"
SEED="${SEEDS[$TASK]}"
SMOKE="${SMOKE:-0}"

REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
LOGDIR="${REPO}/logs_ablation_infonce"
PILOT_DATA="${PILOT_DATA:-${REPO}/outputs/ablation_infonce/pilot_data/pilot_9k_seed2026.h5ad}"
PROBE_IDS_CSV="/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_pretrain_type3_h5ad/probe_ids_type3_pretrain.csv"
GENOMIC_RANK_NPY="${REPO}/outputs/cpg_genomic_sort/cpg_genomic_rank.npy"
TOKENIZER_PATH="${REPO}/tokenizer_llama_pretrain49k"

# Pilot budget
BATCH_SIZE=32
ACCUM=2
PRETRAIN_EPOCHS=20
WARMUP_STEPS=200
export ANALYSIS_EPOCHS="${ANALYSIS_EPOCHS:-0,1,4,9,14,19}"
export STOP_AFTER_EPOCH="-1"
export ABLATION_MODULE=sweep
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

LIMITS=()
WANDB_ON=true
ROOT="${REPO}/outputs/ablation_infonce/pilot"
if [ "${SMOKE}" = "1" ]; then
    BATCH_SIZE=4
    export ANALYSIS_EPOCHS="0"
    export STOP_AFTER_EPOCH="1"
    export LIMIT_TEST_BATCHES=5
    LIMITS=(limit_train_batches=20 limit_val_batches=5)
    WANDB_ON=false
    ROOT="${REPO}/outputs/ablation_infonce/pilot_smoke"
fi

ARM="lam${LAM}_seed${SEED}"
OUTDIR="${ROOT}/${ARM}"
mkdir -p "${LOGDIR}" "${OUTDIR}"

[ -f "${PILOT_DATA}" ]       || { echo "ERROR: pilot subset not found: ${PILOT_DATA} (run create_pilot_subset.sh)"; exit 1; }
[ -f "${GENOMIC_RANK_NPY}" ] || { echo "ERROR: genomic rank not found: ${GENOMIC_RANK_NPY}"; exit 1; }
if [ -n "${RESUME_CHECKPOINT}" ] && [ ! -f "${RESUME_CHECKPOINT}" ]; then
    echo "ERROR: resume checkpoint not found: ${RESUME_CHECKPOINT}"; exit 1
fi

echo "============================================================"
echo "INFONCE PILOT  task=${TASK}  arm=${ARM}  λ=${LAM}  seed=${SEED}  SMOKE=${SMOKE}"
echo "Job: ${SLURM_ARRAY_JOB_ID}_${TASK} | Host: $(hostname) | $(date)"
echo "Git: $(git -C "${REPO}" rev-parse --short HEAD)  dirty=$(git -C "${REPO}" status --porcelain -- bmfm_methylation ablation_infonce | wc -l)"
echo "Data: ${PILOT_DATA}"
echo "Output: ${OUTDIR}"
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

"${PY}" -m ablation_infonce.test_recon_only

"${PY}" -m ablation_infonce.pretrain_recon_only \
    data_path="${PILOT_DATA}" \
    probe_ids_csv="${PROBE_IDS_CSV}" \
    tokenizer_path="${TOKENIZER_PATH}" \
    output_directory="${OUTDIR}" \
    pretraining_mode=wced \
    data_module.subset_k=49156 \
    data_module.batch_size="${BATCH_SIZE}" \
    data_module.num_workers=14 \
    data_module.bmfm_style=false \
    model.hidden_size=256 \
    model.num_hidden_layers=6 \
    model.num_attention_heads=4 \
    model.intermediate_size=512 \
    model.rope_theta=10000.0 \
    model.n_sin_basis=48 \
    model.basis_scale=2.0 \
    trainer.learning_rate=3e-4 \
    trainer.weight_decay=0.01 \
    trainer.warmup_steps="${WARMUP_STEPS}" \
    wced_input_ratio=0.5 \
    wced_age_weight=0.0 \
    wced_contrastive=true \
    wced_contrastive_weight="${LAM}" \
    wced_contrastive_temp=0.1 \
    wced_normalize_loss=true \
    wced_decoder_dropout=0.1 \
    wced_genomic_rank_path="${GENOMIC_RANK_NPY}" \
    pretrain_epochs="${PRETRAIN_EPOCHS}" \
    accumulate_grad_batches="${ACCUM}" \
    early_stop_patience=100 \
    gradient_clip_val=1.0 \
    precision="16-mixed" \
    diag_check_every=5 \
    seed.seed_value="${SEED}" \
    track_wandb.enabled="${WANDB_ON}" \
    track_wandb.project="pretrain-llama-infonce-pilot" \
    track_wandb.entity="netanelazran11-hebrew-university-of-jerusalem" \
    track_wandb.name="pilot-${ARM}-${SLURM_ARRAY_JOB_ID}" \
    ${LIMITS[@]+"${LIMITS[@]}"} \
    ${RESUME_CHECKPOINT:+"resume_checkpoint='${RESUME_CHECKPOINT}'"}

echo "Arm ${ARM} finished: $(date) — checkpoints in ${OUTDIR}/checkpoints/analysis/"
