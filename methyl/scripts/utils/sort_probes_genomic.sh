#!/bin/bash -l
#SBATCH --job-name=sort-cpg-genomic
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/%x_%j.out
#SBATCH --error=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/%x_%j.err

set -euo pipefail

REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
mkdir -p "${REPO}/logs_llama-wced"
cd "${REPO}"
source bmfm_methyl_env/bin/activate
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"

MANIFEST="/sci/labs/benjamin.yakir/netanel.azran/data/manifests/HM450.hg38.manifest.tsv"
PROBE_IDS="/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_pretrain_type3_h5ad/probe_ids_type3_pretrain.csv"
OUTDIR="${REPO}/outputs/cpg_genomic_sort"

echo "============================================================"
echo " CpG Genomic Sort"
echo " Job : ${SLURM_JOB_ID}  Host: $(hostname)  Time: $(date)"
echo "============================================================"
echo " Manifest  : ${MANIFEST}"
echo " Probe IDs : ${PROBE_IDS}"
echo " Outdir    : ${OUTDIR}"
echo "============================================================"

python scripts/utils/sort_probes_genomic.py \
    --manifest  "${MANIFEST}" \
    --probe_ids "${PROBE_IDS}" \
    --outdir    "${OUTDIR}"

echo ""
echo "============================================================"
echo " DONE: $(date)"
echo " Key output: ${OUTDIR}/cpg_genomic_rank.npy"
echo " Use as: wced_genomic_rank_path=${OUTDIR}/cpg_genomic_rank.npy"
echo "============================================================"
