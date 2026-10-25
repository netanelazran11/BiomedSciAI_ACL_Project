#!/bin/bash -l
#SBATCH --job-name=repro-fold0
#SBATCH --partition=goldfish
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_ablation_infonce/%x_%j.out
#SBATCH --error=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_ablation_infonce/%x_%j.err

# Does the upgraded cluster environment (uv CPython 3.11.17 + bmfm_methyl_env, Oct 2026) reproduce the
# published fold-0 test predictions? Re-runs the exact extraction behind the paper's bootstrap
# (scripts/repr_analysis/extract_kfold_test_predictions.py, same checkpoint, same inputs) into a separate
# folder and compares with the stored file outputs/bootstrap_predictions/methyllama/fold_0_predictions.csv.
# Inference only; writes only under outputs/ablation_infonce/repro_fold0/.
#   sbatch ablation_infonce/check_repro_published_fold0.sh

set -euo pipefail
REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
DATA="${REPO}/../../../data/data_methyl_21k_h5ad/altumage_21k_3way.h5ad"
CKPT="${REPO}/outputs/finetune-llama-small/llama-v7b-kfold-fold0-ep300-45586010/checkpoints/epoch=92-val_medae=2.6250.ckpt"
STORED="${REPO}/outputs/bootstrap_predictions/methyllama/fold_0_predictions.csv"
OUT="${REPO}/outputs/ablation_infonce/repro_fold0"
mkdir -p "${REPO}/logs_ablation_infonce" "${OUT}"
cd "${REPO}"

source /etc/profile.d/modules.sh 2>/dev/null || true
module purge 2>/dev/null || true
module load cuda/12.8.1 2>/dev/null || echo "WARN: module cuda/12.8.1 not loaded (torch bundles its CUDA runtime)"
source ablation_infonce/env.sh
"${PY}" -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('GPU:', torch.cuda.get_device_name(0), '| torch', torch.__version__)"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
echo "Git: $(git rev-parse --short HEAD) | Host: $(hostname) | $(date)"

for f in "${DATA}" "${CKPT}" "${STORED}"; do [ -f "$f" ] || { echo "ERROR: not found: $f"; exit 1; }; done

"${PY}" scripts/repr_analysis/extract_kfold_test_predictions.py \
    --fold 0 --checkpoint "${CKPT}" --data "${DATA}" \
    --tokenizer "${REPO}/tokenizer_llama_pretrain49k" \
    --genomic_rank "${REPO}/outputs/cpg_genomic_sort/cpg_genomic_rank_finetune.npy" \
    --test_ids "${REPO}/outputs/kfold_splits/test_ids.npy" \
    --duplicate_pairs_csv "${REPO}/dataset_fingerprint_outputs/duplicate_pairs.csv" \
    --outdir "${OUT}" --model_name MethylLlamaV7b --determinism_check

"${PY}" -m ablation_infonce.compare_predictions --mode repro \
    --a "${STORED}" --b "${OUT}/fold_0_predictions.csv" \
    --label_a "stored (July 2026)" --label_b "new environment" --out "${OUT}/repro_check.json"
echo "REPRO DONE: $(date)"
