#!/bin/bash -l
#SBATCH --job-name=mini-pretrain-check
#SBATCH --partition=goldfish
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/%x_%j.out
#SBATCH --error=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/%x_%j.err

# ─────────────────────────────────────────────────────────────────────────────
# Mini pretrain convergence check — identical pipeline to the real pretrain,
# but limited to 150 batches/epoch for 20 epochs (~2400 samples/epoch).
#
# Goal: verify loss decreases and InfoNCE is > 0 before the big 4-GPU run.
# Expected: val_loss should drop from ~0.25 to <0.10 within 10 epochs.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
LOGDIR="${REPO}/logs_llama-wced"
mkdir -p "${LOGDIR}"

# ─── Paths (identical to real pretrain) ──────────────────────────────────────
DATA_DIR="/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_pretrain_type3_h5ad"
PRETRAIN_DATA="${DATA_DIR}/methylgpt_pretrain_type3.h5ad"
PROBE_IDS_CSV="${DATA_DIR}/probe_ids_type3_pretrain.csv"
GENOMIC_RANK_NPY="${REPO}/outputs/cpg_genomic_sort/cpg_genomic_rank.npy"
TOKENIZER_PATH="${REPO}/tokenizer_llama_pretrain49k"

# ─── Architecture (identical to real pretrain) ───────────────────────────────
HIDDEN_SIZE=256
NUM_LAYERS=6
NUM_HEADS=4
INTERMEDIATE_SIZE=512
ROPE_THETA=10000.0
N_SIN_BASIS=48
BASIS_SCALE=2.0

# ─── WCED (identical to real pretrain) ───────────────────────────────────────
SUBSET_K=49156
INPUT_RATIO=0.5
CONTRASTIVE=true
CONTRASTIVE_WEIGHT=0.05
CONTRASTIVE_TEMP=0.1

# ─── Mini-run settings ───────────────────────────────────────────────────────
BATCH_SIZE=16
LIMIT_TRAIN_BATCHES=150    # 150 × 16 = 2400 samples/epoch
LIMIT_VAL_BATCHES=30       # 30  × 16 = 480  val samples/epoch
PRETRAIN_EPOCHS=20
WARMUP_STEPS=100           # short warmup for mini run
ACCUM=1                    # no accumulation (single GPU, single step)
EARLY_STOP=10
LR=3e-4

OUTDIR="${REPO}/outputs/mini-pretrain-check/mini-${SLURM_JOB_ID}"
mkdir -p "${OUTDIR}"

echo "============================================================"
echo " Mini Pretrain Convergence Check"
echo " Job: ${SLURM_JOB_ID}  Host: $(hostname)  Time: $(date)"
echo " Batches/epoch: ${LIMIT_TRAIN_BATCHES} train, ${LIMIT_VAL_BATCHES} val"
echo " Samples/epoch: $((LIMIT_TRAIN_BATCHES * BATCH_SIZE)) train, $((LIMIT_VAL_BATCHES * BATCH_SIZE)) val"
echo " Epochs: ${PRETRAIN_EPOCHS}  LR: ${LR}  Warmup: ${WARMUP_STEPS}"
echo " Contrastive: ${CONTRASTIVE}  weight=${CONTRASTIVE_WEIGHT}  temp=${CONTRASTIVE_TEMP}"
echo " Genomic RoPE: ${GENOMIC_RANK_NPY}"
echo "============================================================"

# Validate genomic rank exists
if [ ! -f "${GENOMIC_RANK_NPY}" ]; then
    echo "ERROR: genomic rank file not found: ${GENOMIC_RANK_NPY}"
    exit 1
fi

source /etc/profile.d/modules.sh 2>/dev/null || true
module purge
module load spack/all
module load cuda/12.3.2-gcc-5bv3kyh

cd "${REPO}"
source bmfm_methyl_env/bin/activate

export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

python - <<'PY'
import torch
print("torch:", torch.__version__, " | CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

python -m bmfm_methylation.llama.pretrain_llama \
    data_path="${PRETRAIN_DATA}" \
    probe_ids_csv="${PROBE_IDS_CSV}" \
    tokenizer_path="${TOKENIZER_PATH}" \
    output_directory="${OUTDIR}" \
    pretraining_mode=wced \
    data_module.subset_k="${SUBSET_K}" \
    data_module.batch_size="${BATCH_SIZE}" \
    data_module.num_workers=4 \
    data_module.bmfm_style=false \
    model.hidden_size="${HIDDEN_SIZE}" \
    model.num_hidden_layers="${NUM_LAYERS}" \
    model.num_attention_heads="${NUM_HEADS}" \
    model.intermediate_size="${INTERMEDIATE_SIZE}" \
    model.rope_theta="${ROPE_THETA}" \
    model.n_sin_basis="${N_SIN_BASIS}" \
    model.basis_scale="${BASIS_SCALE}" \
    trainer.learning_rate="${LR}" \
    trainer.weight_decay=0.01 \
    trainer.warmup_steps="${WARMUP_STEPS}" \
    wced_input_ratio="${INPUT_RATIO}" \
    wced_age_weight=0.0 \
    wced_contrastive="${CONTRASTIVE}" \
    wced_contrastive_weight="${CONTRASTIVE_WEIGHT}" \
    wced_contrastive_temp="${CONTRASTIVE_TEMP}" \
    wced_normalize_loss=false \
    wced_decoder_dropout=0.1 \
    wced_genomic_rank_path="${GENOMIC_RANK_NPY}" \
    pretrain_epochs="${PRETRAIN_EPOCHS}" \
    accumulate_grad_batches="${ACCUM}" \
    early_stop_patience="${EARLY_STOP}" \
    gradient_clip_val=1.0 \
    precision="16-mixed" \
    limit_train_batches="${LIMIT_TRAIN_BATCHES}" \
    limit_val_batches="${LIMIT_VAL_BATCHES}" \
    track_wandb.enabled=false

echo "============================================================"
echo " Mini pretrain done: $(date)"
echo " Check loss curve above — if val_loss decreased, safe to run full pretrain."
echo "============================================================"
