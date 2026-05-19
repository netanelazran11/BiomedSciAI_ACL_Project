# Fine-tuning Analysis Report — V4

**Run:** llama-small-ft-v4-b32-uf10-enc2e-5-44744875
**Run ID:** bvt444p3
**URL:** https://wandb.ai/netanelazran11-hebrew-university-of-jerusalem/finetune-llama-small/runs/bvt444p3
**State:** finished
**Generated:** 2026-05-17 12:09:38

---

## Final Results

| Metric | Train (final) | Best Val | Test | V1 best | Ridge Baseline |
|--------|--------------|----------|------|---------|----------------|
| MAE (yr)   | 5.48 | 6.31 | 6.49 | 6.81 | 4.49 |
| MedAE (yr) | — | 4.43 | 4.68 | — | — |
| R²         | — | 0.8860 | 0.8810 | 0.862 | 0.94 |

## Performance Summary

- **Best Val MAE:**   6.31 yr  @ epoch 117
- **Best Val MedAE:** 4.43 yr  @ epoch 88
- **Test MAE:**       6.49 yr
- **Test MedAE:**     4.68 yr
- **Test MAE−MedAE gap:** 1.81 yr  ← outlier impact on the mean
- **Test R²:**        0.8810
- **vs Ridge baseline:**  -44.6% improvement in MAE
- **vs V1 (6.81 yr):**    4.7% improvement in MAE

## Training Details

- **Total Epochs:** 150
- **Early Stopping Triggered:** No
- **Epochs After Best:** 32

## Overfitting Analysis

- **Train-Val MAE Gap (final):** 0.99 yr
- **Assessment:** Healthy (gap < 1 yr)

## Plots

- `all_metrics_combined.png`  — 2×4 overview of all key metrics
- `loss_curves.png`           — Train vs Validation loss
- `mae_curves.png`            — MAE and MedAE curves side by side
- `mae_medae_gap.png`         — MAE–MedAE gap (outlier impact over training)
- `r2_curves.png`             — R² over training
- `lr_schedule.png`           — 4-group AdamW LR schedule
- `train_val_gap.png`         — Overfitting analysis
- `convergence.png`           — Convergence speed analysis
