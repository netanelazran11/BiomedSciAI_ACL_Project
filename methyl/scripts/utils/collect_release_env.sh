#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# collect_release_env.sh — second pass for release_bundle_<date>/.
#
# The first pass (collect_release_artifacts.sh) recorded an empty environment:
# it called `python`/`pip`, which in bmfm_methyl_env resolve to the system
# interpreter, and `sacct` without a start date returned no rows. This pass:
#   env_v2/        python3 version, `python3 -m pip freeze`, package versions
#                  (read from package metadata; nothing heavy is imported)
#   env_v2/        sacct records of the seven manuscript jobs (start 2026-06-01)
#   runs_wandb/    W&B config.yaml of each manuscript run (hyper-parameters as logged)
# and rewrites SHA256SUMS last, so it covers every file in the bundle.
# Writes only inside the bundle; refuses if env_v2/ already exists.
#
#   cd /sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl
#   git pull && bash scripts/utils/collect_release_env.sh release_bundle_20261013
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail
shopt -s nullglob

REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
cd "${REPO}" || exit 1
B="${1:?usage: collect_release_env.sh <bundle directory>}"
[ -d "${B}" ] || { echo "ERROR: ${B} not found"; exit 1; }
[ -e "${B}/env_v2" ] && { echo "ERROR: ${B}/env_v2 exists -- nothing is overwritten"; exit 1; }
mkdir -p "${B}/env_v2" "${B}/runs_wandb"

# environment exactly as the run scripts set it up
source /etc/profile.d/modules.sh 2>/dev/null || true
module load spack/all 2>/dev/null || true
module load cuda/12.3.2-gcc-5bv3kyh 2>/dev/null || true
# shellcheck disable=SC1091
source "${REPO}/bmfm_methyl_env/bin/activate"
PY="${REPO}/bmfm_methyl_env/bin/python3"
[ -x "${PY}" ] || PY="$(command -v python3)"

"${PY}" --version > "${B}/env_v2/python_version.txt" 2>&1
echo "interpreter: ${PY} -> $(readlink -f "${PY}")" >> "${B}/env_v2/python_version.txt"
"${PY}" -m pip freeze > "${B}/env_v2/pip_freeze.txt" 2> "${B}/env_v2/pip_freeze.err"
"${PY}" - > "${B}/env_v2/versions.txt" 2>&1 <<'PY'
import importlib.metadata as md, platform, sys
print("python", sys.version.split()[0], "|", platform.platform())
for p in ["torch", "pytorch-lightning", "lightning", "torchmetrics", "transformers", "tokenizers", "numpy", "scipy",
          "pandas", "anndata", "scanpy", "h5py", "scikit-learn", "hydra-core", "omegaconf", "bmfm-targets", "wandb", "einops"]:
    try:
        print(p, md.version(p))
    except md.PackageNotFoundError:
        print(p, "not installed")
for d in md.distributions():
    if (d.metadata["Name"] or "").lower() == "bmfm-targets":
        print("bmfm-targets location", d.locate_file(""))
PY
"${PY}" -m pip show bmfm-targets > "${B}/env_v2/bmfm_targets.txt" 2>&1

sacct -j 45468861,45586010,45586011,45586012,45586013,45586014,45683236 -X -S 2026-06-01 -E now \
      --format=JobID,JobName%34,Partition,State,ExitCode,Elapsed,Start,End,AllocTRES%70 \
      > "${B}/env_v2/sacct_runs.txt" 2>&1

for d in outputs/pretrain-llama-wced/llama-6L-all49k-r0.5-w0.05-genomic-45468861 \
         outputs/finetune-llama-small/llama-v7b-kfold-fold?-ep300-4558601? \
         outputs/finetune-llama-small/llama-v7b-random-init-fold0-ep300-45683236; do
    for c in "${d}"/wandb/*/files/config.yaml; do
        run="$(basename "$(dirname "$(dirname "${c}")")")"
        mkdir -p "${B}/runs_wandb/$(basename "${d}")"
        cp "${c}" "${B}/runs_wandb/$(basename "${d}")/${run}_config.yaml"
    done
done

echo "second pass $(date -Is): env_v2/ and runs_wandb/ added" >> "${B}/COLLECT_LOG.txt"
( cd "${B}" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS )

echo "versions:"; cat "${B}/env_v2/versions.txt"
echo "pip freeze lines: $(wc -l < "${B}/env_v2/pip_freeze.txt")"
echo "sacct:"; cat "${B}/env_v2/sacct_runs.txt"
echo "W&B configs: $(find "${B}/runs_wandb" -name '*_config.yaml' | wc -l)"
echo "DONE: ${B}"
