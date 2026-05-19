# Fine-tuning Analysis Report — V4

**Run:** llama-small-ft-v4b-huber-ep300-wu500-scratch-44770333
**Run ID:** 8mjxsoez
**URL:** https://wandb.ai/netanelazran11-hebrew-university-of-jerusalem/finetune-llama-small/runs/8mjxsoez
**State:** finished
**Generated:** 2026-05-17 11:57:11

---

## Final Results

| Metric | Train (final) | Best Val | Test | V1 best | Ridge Baseline |
|--------|--------------|----------|------|---------|----------------|
| MAE (yr)   | 4.49 | 5.67 | 5.69 | 6.81 | 4.49 |
| MedAE (yr) | — | 3.65 | 3.75 | — | — |
| R²         | — | 0.9043 | 0.9011 | 0.862 | 0.94 |

## Performance Summary

- **Best Val MAE:**   5.67 yr  @ epoch 149
- **Best Val MedAE:** 3.65 yr  @ epoch 121
- **Test MAE:**       5.69 yr
- **Test MedAE:**     3.75 yr
- **Test MAE−MedAE gap:** 1.94 yr  ← outlier impact on the mean
- **Test R²:**        0.9011
- **vs Ridge baseline:**  -26.7% improvement in MAE
- **vs V1 (6.81 yr):**    16.4% improvement in MAE

## Training Details

- **Total Epochs:** 250
- **Early Stopping Triggered:** Yes
- **Epochs After Best:** 100

## Overfitting Analysis

- **Train-Val MAE Gap (final):** 1.25 yr
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
