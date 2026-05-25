"""
Update MethylGPT medium fine-tuning config for stable training:
  - Lower LR (pretrained: 1e-4 -> 1e-5, head: 1e-3 -> 1e-4)
  - Add weight_decay: 0.01
  - Add early stopping (patience=20 on valid_medae)

Run this on the cluster before resubmitting the job.
Usage:
    python3 update_training_config.py
"""
import re
from pathlib import Path

DST = Path("/sci/labs/benjamin.yakir/netanel.azran/repos/MethylGPT-Thesis/scripts/finetuning_age_prediction_medium")

# ── 1. Update yml ─────────────────────────────────────────────────────────────
yml_path = DST / "train_methylgpt_21k_dataset.yml"
txt = yml_path.read_text()

txt = re.sub(r"pretrained_lr\s*:\s*[\d.e+-]+", "pretrained_lr: 1.0e-5", txt)
txt = re.sub(r"head_lr\s*:\s*[\d.e+-]+",       "head_lr: 1.0e-4",       txt)
txt = re.sub(r"weight_decay\s*:\s*[\d.e+-]+",   "weight_decay: 0.01",    txt)

yml_path.write_text(txt)
print("yml updated:")
print("  pretrained_lr: 1e-5  (was 1e-4)")
print("  head_lr:       1e-4  (was 1e-3)")
print("  weight_decay:  0.01  (was 0)")

# ── 2. Add early stopping to main.py ─────────────────────────────────────────
main_path = DST / "finetuning_age_main.py"
main_txt = main_path.read_text()

if "EarlyStopping" in main_txt:
    print("main.py: EarlyStopping already present, skipping")
else:
    early_stop = """
        early_stop_callback = pl.pytorch.callbacks.EarlyStopping(
            monitor="valid_medae",
            patience=20,
            mode="min",
            verbose=True,
        )"""

    main_txt = main_txt.replace(
        "        lr_logger = pl.pytorch.callbacks.LearningRateMonitor()",
        early_stop + "\n        lr_logger = pl.pytorch.callbacks.LearningRateMonitor()"
    )

    # Add to trainer callbacks
    main_txt = main_txt.replace(
        "callbacks=[lr_logger, checkpoint_callback, checkpoint_callback_mae],",
        "callbacks=[lr_logger, checkpoint_callback, checkpoint_callback_mae, early_stop_callback],"
    )

    main_path.write_text(main_txt)
    print("main.py: EarlyStopping added (patience=20 on valid_medae)")

print("\nDone. Resubmit with:")
print(f"  sbatch {DST}/run_medium.sbatch")
