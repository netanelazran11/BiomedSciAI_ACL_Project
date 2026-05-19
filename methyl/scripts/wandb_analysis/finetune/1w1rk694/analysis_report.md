# Fine-tuning Analysis Report — V4

**Run:** llama-small-ft-v4b-huber-ep300-wu500-44756236
**Run ID:** 1w1rk694
**URL:** https://wandb.ai/netanelazran11-hebrew-university-of-jerusalem/finetune-llama-small/runs/1w1rk694
**State:** finished
**Generated:** 2026-05-17 12:11:40

---

## Final Results

| Metric | Train (final) | Best Val | Test | V1 best | Ridge Baseline |
|--------|--------------|----------|------|---------|----------------|
| MAE (yr)   | 4.01 | 5.62 | 5.55 | 6.81 | 4.49 |
| MedAE (yr) | — | 3.56 | 3.63 | — | — |
| R²         | — | 0.9059 | 0.9044 | 0.862 | 0.94 |

## Performance Summary

- **Best Val MAE:**   5.62 yr  @ epoch 156
- **Best Val MedAE:** 3.56 yr  @ epoch 139
- **Test MAE:**       5.55 yr
- **Test MedAE:**     3.63 yr
- **Test MAE−MedAE gap:** 1.91 yr  ← outlier impact on the mean
- **Test R²:**        0.9044
- **vs Ridge baseline:**  -23.5% improvement in MAE
- **vs V1 (6.81 yr):**    18.6% improvement in MAE

## Training Details

- **Total Epochs:** 257
- **Early Stopping Triggered:** Yes
- **Epochs After Best:** 100

## Overfitting Analysis

- **Train-Val MAE Gap (final):** 1.79 yr
- **Assessment:** Mild (gap 1–2 yr)

## Plots

- `all_metrics_combined.png`  — 2×4 overview of all key metrics
- `loss_curves.png`           — Train vs Validation loss
- `mae_curves.png`            — MAE and MedAE curves side by side
- `mae_medae_gap.png`         — MAE–MedAE gap (outlier impact over training)
- `r2_curves.png`             — R² over training
- `lr_schedule.png`           — 4-group AdamW LR schedule
- `train_val_gap.png`         — Overfitting analysis
- `convergence.png`           — Convergence speed analysis
