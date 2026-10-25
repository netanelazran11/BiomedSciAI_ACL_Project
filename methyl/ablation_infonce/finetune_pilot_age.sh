#!/bin/bash -l
#SBATCH --job-name=pilot-ft-age
#SBATCH --partition=goldfish
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=80:00:00
#SBATCH --array=0-2
#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_ablation_infonce/%x_%A_%a.out
#SBATCH --error=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_ablation_infonce/%x_%A_%a.err

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Age fine-tuning of the InfoNCE pilot models (fold 0), three arms that differ ONLY in the starting checkpoint:
#   task 0  pilot λ=0    (reconstruction only)   outputs/ablation_infonce/pilot/lam0.0_seed42/.../epoch=19-analysis.ckpt
#   task 1  pilot λ=0.05 (published objective)   outputs/ablation_infonce/pilot/lam0.05_seed42/.../epoch=19-analysis.ckpt
#   task 2  pilot λ=0.2  (strongest contrastive; weakest frozen probes in the pilot)
#
# Fine-tuning recipe = the paper's fold-0 run (scripts/llama/finetune_llama_small_v7b_kfold.sh, W&B a154lzjy):
# same data, fold files, test set, tokenizer, fine-tune genomic rank, head 256→128→1, dropout 0.1, Huber,
# CLS pooling, lr 1e-4, encoder lr 2e-5, encoder frozen until epoch 10, warm-up 500, batch 32 × accum 4,
# 16-mixed, clip 1.0, seed 42. Changed on purpose (both arms alike): epoch budget 200 (cosine over 200) with
# early stopping on val/mae, patience 40, instead of 300 epochs cut by the 48 h wall.
# Checkpoints: top-3 by val/mae and top-3 by val/medae (built into finetune_llama.py).
# After training: per-subject test predictions (2,149 subjects) from the best val/medae checkpoint (primary,
# as in the paper) and the best val/mae checkpoint (secondary).
#
#   SMOKE=1 sbatch --array=0 --time=01:00:00 ablation_infonce/finetune_pilot_age.sh   # 1 epoch, no W&B
#   sbatch ablation_infonce/finetune_pilot_age.sh
# ─────────────────────────────────────────────────────────────────────────────

ARMS=(lam0.0_seed42 lam0.05_seed42 lam0.2_seed42)
TASK="${SLURM_ARRAY_TASK_ID:?run as an array job}"
ARM="${ARMS[$TASK]}"
SMOKE="${SMOKE:-0}"

REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
DATA="${REPO}/../../../data/data_methyl_21k_h5ad/altumage_21k_3way.h5ad"
KFOLD_DIR="${REPO}/outputs/kfold_splits"
RANK_FT="${REPO}/outputs/cpg_genomic_sort/cpg_genomic_rank_finetune.npy"
TOKENIZER="${REPO}/tokenizer_llama_pretrain49k"
DUP_CSV="${REPO}/dataset_fingerprint_outputs/duplicate_pairs.csv"
PRE_CKPT="${REPO}/outputs/ablation_infonce/pilot/${ARM}/checkpoints/analysis/epoch=19-analysis.ckpt"

FINETUNE_EPOCHS=200
EARLY_STOP=40
UNFREEZE_EPOCH=10
# Batch per step and accumulation; effective batch 32 x 4 = 128 as in the paper. On a 48 GB L40S use
# BATCH_SIZE=16 ACCUM=8 (same effective batch). All arms must use the same values and GPU type.
BATCH_SIZE="${BATCH_SIZE:-32}"
ACCUM="${ACCUM:-4}"
WANDB_ON=true
ROOT="${REPO}/outputs/ablation_infonce/pilot_finetune"
if [ "${SMOKE}" = "1" ]; then
    FINETUNE_EPOCHS=1
    UNFREEZE_EPOCH=0      # unfreeze at once: the smoke must hit the peak (unfrozen) GPU memory
    WANDB_ON=false
    ROOT="${REPO}/outputs/ablation_infonce/pilot_finetune_smoke"
fi
OUTDIR="${ROOT}/${ARM}_fold0"
RUN_NAME="pilot-ft-${ARM}-fold0-ep${FINETUNE_EPOCHS}-${SLURM_ARRAY_JOB_ID}"
mkdir -p "${REPO}/logs_ablation_infonce" "${OUTDIR}"

for f in "${DATA}" "${KFOLD_DIR}/fold_0_train.npy" "${KFOLD_DIR}/fold_0_val.npy" "${KFOLD_DIR}/test_ids.npy" \
         "${RANK_FT}" "${DUP_CSV}" "${PRE_CKPT}"; do
    [ -f "$f" ] || { echo "ERROR: not found: $f"; exit 1; }
done

echo "============================================================"
echo "PILOT AGE FINE-TUNE  task=${TASK}  arm=${ARM}  SMOKE=${SMOKE}  epochs=${FINETUNE_EPOCHS}  early_stop=${EARLY_STOP}  unfreeze=${UNFREEZE_EPOCH}  batch=${BATCH_SIZE}x${ACCUM}"
echo "Job: ${SLURM_ARRAY_JOB_ID}_${TASK} | Host: $(hostname) | $(date)"
echo "Git: $(git -C "${REPO}" rev-parse --short HEAD)  dirty=$(git -C "${REPO}" status --porcelain -- bmfm_methylation ablation_infonce | wc -l)"
echo "Pretrained: ${PRE_CKPT}"
echo "Output: ${OUTDIR}"
echo "============================================================"

source /etc/profile.d/modules.sh 2>/dev/null || true
module purge 2>/dev/null || true
module load cuda/12.8.1 2>/dev/null || echo "WARN: module cuda/12.8.1 not loaded (torch bundles its CUDA runtime)"
cd "${REPO}"
source ablation_infonce/env.sh
"${PY}" -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('GPU:', torch.cuda.get_device_name(0), '| torch', torch.__version__)"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Sample GPU memory during training (the smoke unfreezes at once, so it shows the peak)
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 >> "${OUTDIR}/gpu_mem_mib.txt"; sleep 20; done ) &
SAMPLER_PID=$!

"${PY}" -m bmfm_methylation.llama.finetune_llama \
    "data_path='${DATA}'" \
    "checkpoint_path='${PRE_CKPT}'" \
    "tokenizer_path='${TOKENIZER}'" \
    "output_directory='${OUTDIR}'" \
    data_module.subset_k=49156 \
    data_module.fixed_subset_seed=42 \
    data_module.max_length=21369 \
    data_module.batch_size="${BATCH_SIZE}" \
    data_module.num_workers=8 \
    data_module.filter_age_outliers=true \
    "data_module.duplicate_pairs_csv='${DUP_CSV}'" \
    "+data_module.fold_train_ids_npy='${KFOLD_DIR}/fold_0_train.npy'" \
    "+data_module.fold_val_ids_npy='${KFOLD_DIR}/fold_0_val.npy'" \
    wced_input_ratio=1.0 \
    wced_genomic_rank_path="${RANK_FT}" \
    finetune.head_hidden_size=128 \
    finetune.head_dropout=0.1 \
    finetune.learning_rate=1e-4 \
    finetune.encoder_lr=2e-5 \
    finetune.weight_decay=0.01 \
    finetune.warmup_steps=500 \
    finetune.freeze_encoder=true \
    finetune.unfreeze_encoder_epoch="${UNFREEZE_EPOCH}" \
    finetune.recon_weight=0.0 \
    finetune.pooling=cls \
    finetune.loss_type=huber \
    finetune.beta_noise=0.0 \
    finetune_epochs="${FINETUNE_EPOCHS}" \
    accumulate_grad_batches="${ACCUM}" \
    gradient_clip_val=1.0 \
    early_stop_patience="${EARLY_STOP}" \
    precision="16-mixed" \
    track_wandb.enabled="${WANDB_ON}" \
    track_wandb.project=finetune-llama-small \
    track_wandb.entity=netanelazran11-hebrew-university-of-jerusalem \
    track_wandb.name="${RUN_NAME}" \
    "+track_wandb.group='infonce-pilot-ft'"

kill "${SAMPLER_PID}" 2>/dev/null || true
echo "Training finished: $(date)"
echo "Peak GPU memory used during training (nvidia-smi, every 20 s): $(sort -n "${OUTDIR}/gpu_mem_mib.txt" | tail -1) MiB of $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1) MiB"
ls -la "${OUTDIR}/checkpoints/"

# Best checkpoint by the value in the file name (lowest), for one metric
best_ckpt () {
    "${PY}" - "$1" "${OUTDIR}/checkpoints" <<'PYEOF'
import re, sys, pathlib
metric, d = sys.argv[1], pathlib.Path(sys.argv[2])
c = [(float(re.search(rf"{metric}=([0-9.]+)\.ckpt$", p.name).group(1)), p) for p in d.glob(f"epoch=*-{metric}=*.ckpt")]
print(min(c)[1] if c else "")
PYEOF
}

for METRIC in val_medae val_mae; do
    CK=$(best_ckpt "${METRIC}")
    if [ -z "${CK}" ]; then echo "WARNING: no ${METRIC} checkpoint"; continue; fi
    echo ">>> Test predictions from best ${METRIC}: ${CK}"
    "${PY}" scripts/repr_analysis/extract_kfold_test_predictions.py \
        --fold 0 --checkpoint "${CK}" --data "${DATA}" \
        --tokenizer "${TOKENIZER}" --genomic_rank "${RANK_FT}" \
        --test_ids "${KFOLD_DIR}/test_ids.npy" --duplicate_pairs_csv "${DUP_CSV}" \
        --outdir "${OUTDIR}/predictions_${METRIC}" --model_name "pilot-${ARM}"
done
echo "Arm ${ARM} DONE: $(date)"
