"""
Run this once on the cluster to set up the MethylGPT medium model fine-tuning.
Fixes args.json, yml, main.py (checkpoints + wandb), and creates the SLURM script.

Usage:
    python setup_medium_run.py
"""
import os
import json
import re
from pathlib import Path

DST        = Path("/sci/labs/benjamin.yakir/netanel.azran/repos/MethylGPT-Thesis/scripts/finetuning_age_prediction_medium")
MEDIUM_CKPT= "/sci/labs/benjamin.yakir/netanel.azran/repos/MethylGPT/pretrained_models/large/methylGPT-medium/medium-best_model_epoch6.pt"
PROBE_IDS  = "/sci/labs/benjamin.yakir/netanel.azran/MethylGPT/data/altumage_data/finetuning_data_21k/cpg_mapping/probe_ids_type3_21k.csv"
VENV       = "/sci/labs/benjamin.yakir/netanel.azran/venv_torch22"
WANDB_ENTITY = "netanelazran11-hebrew-university-of-jerusalem"

assert DST.exists(), f"Directory not found: {DST}"

# ── 1. Fix args.json ──────────────────────────────────────────────────────────
args_path = DST / "args.json"
with open(args_path) as f:
    args = json.load(f)
args["probe_id_dir"] = PROBE_IDS
args["n_hvg"] = 21368
args["dataset"] = "finetune_21k_medium"
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
txt = re.sub(r"max_epochs\s*:\s*\d+", "max_epochs: 200", txt)
txt = re.sub(r"wandb\s*:\s*\w+", "wandb: True", txt)
yml_path.write_text(txt)
print(f"yml: pretrained_file -> medium checkpoint, max_epochs=200, wandb=True")

# ── 3. Patch main.py — save_top_k=3 for medae + add mae checkpoint ────────────
main_path = DST / "finetuning_age_main.py"
main_txt = main_path.read_text()

# Change save_top_k=1 -> save_top_k=3
main_txt = main_txt.replace("save_top_k=1,", "save_top_k=3,")

# Add MAE checkpoint callback after the medae one
mae_callback = """
        checkpoint_callback_mae = pl.pytorch.callbacks.ModelCheckpoint(
            dirpath=model_args["weights_save_path"],
            filename=model_args["weights_name"] + "_best_mae",
            monitor="valid_mae",
            mode="min",
            save_top_k=3,
        )"""

# Insert after the closing ) of the medae checkpoint block
main_txt = main_txt.replace(
    "        lr_logger = pl.pytorch.callbacks.LearningRateMonitor()",
    mae_callback + "\n        lr_logger = pl.pytorch.callbacks.LearningRateMonitor()"
)

# Add mae callback to trainer
main_txt = main_txt.replace(
    "callbacks=[lr_logger, checkpoint_callback],",
    "callbacks=[lr_logger, checkpoint_callback, checkpoint_callback_mae],"
)

# Add wandb entity
main_txt = main_txt.replace(
    "WandbLogger(project=model_args[\"project\"],",
    f"WandbLogger(project=model_args[\"project\"], entity=\"{WANDB_ENTITY}\","
)

main_path.write_text(main_txt)
print("main.py: save_top_k=3, MAE checkpoint added, wandb entity set")

# ── 4. Create SLURM script ───────────────────────────────────────────────────
(DST / "checkpoints").mkdir(exist_ok=True)
(DST / "logs").mkdir(exist_ok=True)

slurm = f"""#!/bin/bash -l
#SBATCH --job-name=methylgpt-medium-21k
#SBATCH --partition=goldfish
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output={DST}/logs/%x_%j.out
#SBATCH --error={DST}/logs/%x_%j.err

set -euo pipefail

source {VENV}/bin/activate

cd {DST}
python finetuning_age_main.py
"""
slurm_path = DST / "run_medium.sbatch"
slurm_path.write_text(slurm)
print(f"SLURM: goldfish/H200, 48h time limit, written to {slurm_path}")

print("\nDone. Submit with:")
print(f"  sbatch {slurm_path}")
