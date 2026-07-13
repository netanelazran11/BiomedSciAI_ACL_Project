#!/bin/bash -l
#SBATCH --job-name=finetune-llama-v7b
#SBATCH --partition=goldfish
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00

#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/%x_%j.out
#SBATCH --error=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/%x_%j.err

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Fine-tuning MethylLlama V7b — ablation of head architecture
#
# Identical to V7 except:
#   HEAD: 256→128→1  (V7: 256→256→128→1)   — simpler, less overfit risk
#   DROPOUT: 0.1     (V7: 0.0)              — regularization for 17k samples
#
# Purpose: determine whether a simpler regularized head improves on V7.
# If V7b > V7 on test/MedAE: head was overfitting with 0 dropout.
# If V7b ≈ V7: head architecture doesn't matter, pretrain quality is the lever.
#
# V5 reference (best baseline): test/R²=0.905, MedAE=3.65yr (4L pretrain, job 44895876)
# ─────────────────────────────────────────────────────────────────────────────

REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
LOGDIR="${REPO}/logs_llama-wced"

DATA="/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_21k_h5ad/altumage_21k_3way.h5ad"

GENOMIC_RANK_FT_NPY="${REPO}/outputs/cpg_genomic_sort/cpg_genomic_rank_finetune.npy"

CHECKPOINT="${CHECKPOINT:?ERROR: set CHECKPOINT to the 6L pretrain checkpoint path}"

TOKENIZER_PATH="${REPO}/tokenizer_llama_pretrain49k"

# ─── Data (same as V7) ────────────────────────────────────────────────────────
SUBSET_K="${SUBSET_K:-49156}"
INPUT_RATIO="${INPUT_RATIO:-1.0}"

# ─── Fine-tuning hyperparameters (same as V7 except head) ────────────────────
LR="${LR:-1e-4}"
ENCODER_LR="${ENCODER_LR:-2e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
BATCH_SIZE="${BATCH_SIZE:-32}"
ACCUM="${ACCUM:-4}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-300}"
EARLY_STOP="${EARLY_STOP:-100}"
FREEZE_ENCODER="${FREEZE_ENCODER:-true}"
UNFREEZE_EPOCH="${UNFREEZE_EPOCH:-10}"
WARMUP_STEPS="${WARMUP_STEPS:-500}"
RECON_WEIGHT="${RECON_WEIGHT:-0.0}"
POOLING="${POOLING:-cls}"
LOSS_TYPE="${LOSS_TYPE:-huber}"
BETA_NOISE="${BETA_NOISE:-0.0}"

# ─── V7b head changes ────────────────────────────────────────────────────────
HEAD_HIDDEN="${HEAD_HIDDEN:-128}"     # 256→128→1 instead of 256→256→128→1
HEAD_DROPOUT="${HEAD_DROPOUT:-0.1}"  # regularization (V7 used 0.0)

WARMSTART_WEIGHTS="${WARMSTART_WEIGHTS:-}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
EVAL_CHECKPOINT="${EVAL_CHECKPOINT:-}"

# ─── WandB ───────────────────────────────────────────────────────────────────
WANDB_ENTITY="netanelazran11-hebrew-university-of-jerusalem"
WANDB_PROJECT="finetune-llama-small"
WS_TAG=$( [ -n "${WARMSTART_WEIGHTS}" ] && echo "ws" || echo "scratch" )
WANDB_RUN_NAME="llama-small-ft-v7b-6L-cls-huber-h${HEAD_HIDDEN}-do${HEAD_DROPOUT}-ep${FINETUNE_EPOCHS}-${WS_TAG}-${SLURM_JOB_ID}"

OUTROOT="${REPO}/outputs/${WANDB_PROJECT}"
OUTDIR="${OUTROOT}/${WANDB_RUN_NAME}"

mkdir -p "${LOGDIR}" "${OUTDIR}"

echo "============================================================"
echo "METHYLLAMA V7b FINE-TUNING (ablation: simpler head + dropout)"
echo "============================================================"
echo "Job: ${SLURM_JOB_ID} | Host: $(hostname) | Time: $(date)"
echo "Pretrain ckpt: ${CHECKPOINT}"
echo "Head: 256→${HEAD_HIDDEN}→1  dropout=${HEAD_DROPOUT}  (V7 was 256→256→128→1  dropout=0.0)"
echo "Pooling: ${POOLING} | Loss: ${LOSS_TYPE}"
echo "epochs=${FINETUNE_EPOCHS} | early_stop=${EARLY_STOP} | warmup=${WARMUP_STEPS}"
echo "batch=${BATCH_SIZE}×${ACCUM}=$(( BATCH_SIZE * ACCUM )) eff"
echo "lr=${LR} | encoder_lr=${ENCODER_LR} | unfreeze_epoch=${UNFREEZE_EPOCH}"
echo "Genomic RoPE (ft): ${GENOMIC_RANK_FT_NPY}"
echo "Data: ${DATA}"
echo "Output: ${OUTDIR}"
echo "============================================================"

source /etc/profile.d/modules.sh 2>/dev/null || source /usr/share/modules/init/bash 2>/dev/null || true
module purge
module load spack/all
module load cuda/12.3.2-gcc-5bv3kyh

cd "${REPO}"
source bmfm_methyl_env/bin/activate

if [ ! -f "${GENOMIC_RANK_FT_NPY}" ]; then
    echo "Genomic rank file not found — generating: ${GENOMIC_RANK_FT_NPY}"
    python scripts/llama/create_finetune_genomic_rank.py
fi

export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

python -m bmfm_methylation.llama.finetune_llama \
    "data_path='${DATA}'" \
    "checkpoint_path='${CHECKPOINT}'" \
    "tokenizer_path='${TOKENIZER_PATH}'" \
    "output_directory='${OUTDIR}'" \
    data_module.subset_k="${SUBSET_K}" \
    data_module.fixed_subset_seed=42 \
    data_module.max_length=21369 \
    data_module.batch_size="${BATCH_SIZE}" \
    data_module.num_workers=8 \
    data_module.filter_age_outliers=true \
    "data_module.duplicate_pairs_csv='${REPO}/dataset_fingerprint_outputs/duplicate_pairs.csv'" \
    wced_input_ratio="${INPUT_RATIO}" \
    wced_genomic_rank_path="${GENOMIC_RANK_FT_NPY}" \
    finetune.head_hidden_size="${HEAD_HIDDEN}" \
    finetune.head_dropout="${HEAD_DROPOUT}" \
    finetune.learning_rate="${LR}" \
    finetune.encoder_lr="${ENCODER_LR}" \
    finetune.weight_decay="${WEIGHT_DECAY}" \
    finetune.warmup_steps="${WARMUP_STEPS}" \
    finetune.freeze_encoder="${FREEZE_ENCODER}" \
    finetune.unfreeze_encoder_epoch="${UNFREEZE_EPOCH}" \
    finetune.recon_weight="${RECON_WEIGHT}" \
    finetune.pooling="${POOLING}" \
    finetune.loss_type="${LOSS_TYPE}" \
    finetune.beta_noise="${BETA_NOISE}" \
    finetune_epochs="${FINETUNE_EPOCHS}" \
    accumulate_grad_batches="${ACCUM}" \
    gradient_clip_val=1.0 \
    early_stop_patience="${EARLY_STOP}" \
    precision="16-mixed" \
    track_wandb.enabled=true \
    track_wandb.project="${WANDB_PROJECT}" \
    track_wandb.entity="${WANDB_ENTITY}" \
    track_wandb.name="${WANDB_RUN_NAME}" \
    ${WARMSTART_WEIGHTS:+"+warmstart_weights_path='${WARMSTART_WEIGHTS}'"} \
    ${RESUME_CHECKPOINT:+"+resume_checkpoint='${RESUME_CHECKPOINT}'"} \
    ${EVAL_CHECKPOINT:+"+eval_checkpoint='${EVAL_CHECKPOINT}'"}

echo "============================================================"
echo "V7b fine-tuning finished: $(date)"
echo "Checkpoints: ${OUTDIR}/checkpoints/"
echo "============================================================"
