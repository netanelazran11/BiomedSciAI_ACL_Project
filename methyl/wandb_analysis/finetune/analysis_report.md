# Fine-tuning Analysis Report

**Run:** finetune-fixed2048-44045989
**Run ID:** ycap9s0f
**URL:** https://wandb.ai/netanelazran11-hebrew-university-of-jerusalem/finetune-fixed2048-bmfm-rna-methylation/runs/ycap9s0f
**State:** finished
**Generated:** 2026-02-10 15:13:12

---

## Final Results

| Metric | Train | Validation | Test | Ridge Baseline |
|--------|-------|------------|------|----------------|
| MAE (years) | 2.21 | 4.97 | 4.85 | 4.49 |
| R² | - | 0.9220 | 0.9226 | 0.94 |

## Performance Summary

- **Best Val MAE:** 4.97 years (at epoch 144)
- **Test MAE:** 4.85 years
- **Test R²:** 0.9226
- **vs Ridge Baseline:** -8.1% improvement in MAE

## Training Details

- **Total Epochs:** 245
- **Early Stopping Triggered:** Yes
- **Epochs After Best:** 100

## Overfitting Analysis

- **Train-Val MAE Gap:** 2.91 years
- **Assessment:** Concerning (gap > 2 years)

## Key Observations

1. **Model Architecture:** Direct MLP on raw beta values (8000 -> 512 -> 256 -> 128 -> 1)
   - Bypasses the transformer encoder which washes out per-sample differences
   - Uses LayerNorm, GELU activations, and dropout for regularization

2. **vs Ridge Regression:**
   - Ridge (linear): MAE = 4.49 years, R² = 0.94
   - MLP (nonlinear): MAE = 4.85 years, R² = 0.9226
   - The MLP captures nonlinear interactions between CpG sites

3. **Training Dynamics:**
   - Loss function: MSE (on z-score normalized ages)
   - Optimizer: AdamW with cosine LR decay and warmup
   - The model converged smoothly without instability

## Plots

- `all_metrics_combined.png` - Overview of all metrics (6-panel)
- `loss_curves.png` - Train vs Validation loss
- `mae_curves.png` - MAE curves with Ridge baseline
- `r2_curves.png` - R² over training
- `lr_schedule.png` - Learning rate schedule
- `train_val_gap.png` - Overfitting analysis
- `convergence.png` - Convergence speed analysis
