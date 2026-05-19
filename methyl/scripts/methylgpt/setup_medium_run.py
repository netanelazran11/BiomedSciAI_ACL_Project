"""
Run this once on the cluster to set up the MethylGPT medium model fine-tuning.
It fixes the yml and creates the SLURM script — nothing else changes.

Usage:
    python setup_medium_run.py
"""
import os
import json
from pathlib import Path

DST = Path("/sci/labs/benjamin.yakir/netanel.azran/repos/MethylGPT-Thesis/scripts/finetuning_age_prediction_medium")
MEDIUM_CKPT = "/sci/labs/benjamin.yakir/netanel.azran/repos/MethylGPT/pretrained_models/large/methylGPT-medium/medium-best_model_epoch6.pt"
PROBE_IDS   = "/sci/labs/benjamin.yakir/netanel.azran/MethylGPT/data/altumage_data/finetuning_data_21k/cpg_mapping/probe_ids_type3_21k.csv"
DATA_DIR    = "/sci/labs/benjamin.yakir/netanel.azran/MethylGPT/data/altumage_data/finetuning_data_21k"
VENV        = "/sci/labs/benjamin.yakir/netanel.azran/venv_torch22"

assert DST.exists(), f"Directory not found: {DST}"

# ── 1. Fix args.json ──────────────────────────────────────────────────────────
args_path = DST / "args.json"
with open(args_path) as f:
    args = json.load(f)
args["probe_id_dir"] = PROBE_IDS
args["n_hvg"] = 21368  # match actual data (medium model pretrained on 49k, but we fine-tune on 21k subset)
with open(args_path, "w") as f:
    json.dump(args, f, indent=4)
print(f"args.json: layer_size={args['layer_size']}, n_hvg={args['n_hvg']}, probe_id_dir OK")

# ── 2. Fix yml ────────────────────────────────────────────────────────────────
yml_path = DST / "train_methylgpt_21k_dataset.yml"
txt = yml_path.read_text()
txt = txt.replace(
    "/sci/labs/benjamin.yakir/netanel.azran/MethylGPT/models/base/tiny-best_model_epoch10.pt",
    MEDIUM_CKPT
)
txt = txt.replace(
    "/sci/labs/benjamin.yakir/netanel.azran/MethylGPT/methylgpt-work/finetuning_age_prediction_21k",
    str(DST / "checkpoints")
)
yml_path.write_text(txt)
print(f"yml: pretrained_file -> medium checkpoint")
print(f"yml: weights_save_path -> {DST}/checkpoints")

# ── 3. Create SLURM script ───────────────────────────────────────────────────
(DST / "checkpoints").mkdir(exist_ok=True)
(DST / "logs").mkdir(exist_ok=True)

slurm = f"""#!/bin/bash -l
#SBATCH --job-name=methylgpt-medium-21k
#SBATCH --partition=salmon
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output={DST}/logs/%x_%j.out
#SBATCH --error={DST}/logs/%x_%j.err

set -euo pipefail

source {VENV}/bin/activate

cd {DST}
python finetuning_age_main.py
"""
slurm_path = DST / "run_medium.sbatch"
slurm_path.write_text(slurm)
print(f"SLURM script written: {slurm_path}")

print("\nDone. Submit with:")
print(f"  sbatch {slurm_path}")
