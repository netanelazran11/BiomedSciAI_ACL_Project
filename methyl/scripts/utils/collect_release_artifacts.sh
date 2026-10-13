#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# collect_release_artifacts.sh — gather, on the cluster, what the public
# MethylLlama release needs and what exists only here.
#
# Copies the small files into a new folder release_bundle_<date>/ and records
# the large checkpoints by path, size and sha256 without copying them.
# Reads only; changes nothing outside the new folder; refuses to overwrite it.
# Runs on the login node in a few minutes (sha256 of ~1 GB of checkpoints).
#
#   cd /sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl
#   git pull && bash scripts/utils/collect_release_artifacts.sh
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail

REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
cd "${REPO}" || exit 1
OUT="${REPO}/release_bundle_$(date +%Y%m%d)"
[ -e "${OUT}" ] && { echo "ERROR: ${OUT} exists -- nothing is overwritten"; exit 1; }
mkdir -p "${OUT}"/{tokenizer,resources,runs,env}
MISSING=()

note() { printf '%s\n' "$*" | tee -a "${OUT}/COLLECT_LOG.txt"; }
need() { [ -e "$1" ] || { MISSING+=("$1"); note "MISSING: $1"; return 1; }; }

note "collected $(date -Is) on $(hostname) from ${REPO}"
note "repo HEAD: $(git -C "${REPO}" rev-parse HEAD) ($(git -C "${REPO}" log -1 --format=%cd --date=short))"

# 1. tokenizer used by pretraining run 45468861 (49,161 entries)
TOK="${REPO}/tokenizer_llama_pretrain49k"
if need "${TOK}"; then
    cp -r "${TOK}/." "${OUT}/tokenizer/"
    note "tokenizer: $(find "${OUT}/tokenizer" -name vocab.txt -exec wc -l {} \; | head -3 | tr '\n' ' ')"
fi

# 2. genomic-rank files (position ids for genomic RoPE)
for f in outputs/cpg_genomic_sort/cpg_genomic_rank.npy outputs/cpg_genomic_sort/cpg_genomic_rank_finetune.npy; do
    need "${REPO}/${f}" && cp "${REPO}/${f}" "${OUT}/resources/"
done
ls "${REPO}/outputs/cpg_genomic_sort/" > "${OUT}/resources/cpg_genomic_sort_listing.txt" 2>/dev/null

# 3. checkpoints behind the manuscript (recorded, not copied)
CKPTS=(
  "pretrain|outputs/pretrain-llama-wced/llama-6L-all49k-r0.5-w0.05-genomic-45468861/checkpoints/epoch=85-recon=0.0552-pcc=0.9713.ckpt"
  "fold0|outputs/finetune-llama-small/llama-v7b-kfold-fold0-ep300-45586010/checkpoints/epoch=92-val_medae=2.6250.ckpt"
  "fold1|outputs/finetune-llama-small/llama-v7b-kfold-fold1-ep300-45586011/checkpoints/epoch=137-val_medae=2.5940.ckpt"
  "fold2|outputs/finetune-llama-small/llama-v7b-kfold-fold2-ep300-45586012/checkpoints/epoch=131-val_medae=2.6875.ckpt"
  "fold3|outputs/finetune-llama-small/llama-v7b-kfold-fold3-ep300-45586013/checkpoints/epoch=137-val_medae=2.7188.ckpt"
  "fold4|outputs/finetune-llama-small/llama-v7b-kfold-fold4-ep300-45586014/checkpoints/epoch=138-val_medae=2.6875.ckpt"
)
printf 'role\tpath\tbytes\tsha256\n' > "${OUT}/checkpoints.tsv"
for e in "${CKPTS[@]}"; do
    role="${e%%|*}"; path="${e#*|}"
    if need "${REPO}/${path}"; then
        printf '%s\t%s\t%s\t%s\n' "${role}" "${path}" "$(stat -c %s "${REPO}/${path}")" \
            "$(sha256sum "${REPO}/${path}" | cut -d' ' -f1)" >> "${OUT}/checkpoints.tsv"
    fi
done
# random-initialisation run (Fig. 4c is drawn from its validation history; checkpoint listed for completeness)
find "${REPO}/outputs" -path "*45683236*" -name "*.ckpt" -printf 'random_init\t%p\t%s\t-\n' 2>/dev/null \
    | sed "s|${REPO}/||" >> "${OUT}/checkpoints.tsv"
note "checkpoints recorded: $(($(wc -l < "${OUT}/checkpoints.tsv") - 1))"

# 4. run configurations (Hydra config/overrides, Lightning hparams) of the six runs
for d in outputs/pretrain-llama-wced/llama-6L-all49k-r0.5-w0.05-genomic-45468861 \
         outputs/finetune-llama-small/llama-v7b-kfold-fold0-ep300-45586010 \
         outputs/finetune-llama-small/llama-v7b-kfold-fold1-ep300-45586011 \
         outputs/finetune-llama-small/llama-v7b-kfold-fold2-ep300-45586012 \
         outputs/finetune-llama-small/llama-v7b-kfold-fold3-ep300-45586013 \
         outputs/finetune-llama-small/llama-v7b-kfold-fold4-ep300-45586014; do
    need "${REPO}/${d}" || continue
    dst="${OUT}/runs/$(basename "$d")"; mkdir -p "${dst}"
    find "${REPO}/${d}" -maxdepth 4 \( -name "config.yaml" -o -name "overrides.yaml" -o -name "hparams.yaml" -o -name "hydra.yaml" \) \
        -not -path "*/wandb/*" -print0 2>/dev/null | while IFS= read -r -d '' f; do
            rel="${f#${REPO}/${d}/}"; mkdir -p "${dst}/$(dirname "${rel}")"; cp "$f" "${dst}/${rel}"
        done
    ls -la "${REPO}/${d}" > "${dst}/LISTING.txt" 2>/dev/null
done

# 5. software environment of the runs
if [ -f "${REPO}/bmfm_methyl_env/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${REPO}/bmfm_methyl_env/bin/activate"
    python --version > "${OUT}/env/python_version.txt" 2>&1
    pip freeze > "${OUT}/env/pip_freeze.txt" 2>/dev/null
    python - > "${OUT}/env/versions.txt" 2>&1 <<'PY'
import importlib, platform
print("platform", platform.platform())
for m in ["torch", "pytorch_lightning", "transformers", "numpy", "scipy", "pandas", "anndata", "scanpy", "h5py", "sklearn", "hydra", "omegaconf"]:
    try:
        mod = importlib.import_module(m); print(m, getattr(mod, "__version__", "?"))
    except Exception as e:
        print(m, "not importable:", type(e).__name__)
try:
    import torch; print("torch.version.cuda", torch.version.cuda)
except Exception:
    pass
PY
    pip show bmfm-targets > "${OUT}/env/bmfm_targets.txt" 2>/dev/null
    deactivate 2>/dev/null || true
else
    MISSING+=("bmfm_methyl_env"); note "MISSING: bmfm_methyl_env"
fi

# 6. SLURM records of the manuscript runs
sacct -j 45468861,45586010,45586011,45586012,45586013,45586014,45683236 -X \
      --format=JobID,JobName%30,Partition,State,ExitCode,Elapsed,Start,End,AllocTRES%60 \
      > "${OUT}/env/sacct_runs.txt" 2>&1

# 7. integrity manifest of everything copied
( cd "${OUT}" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS )
note "bundle size: $(du -sh "${OUT}" | cut -f1)"
if [ ${#MISSING[@]} -gt 0 ]; then
    note "DONE WITH ${#MISSING[@]} MISSING ITEM(S): ${MISSING[*]}"
else
    note "DONE: all items found"
fi
note "bundle: ${OUT}"
