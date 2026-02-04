# Fine-tuning Analysis Report

**Run:** bmfm-methyl-finetune-43998394
**Run ID:** v4yn0u15
**URL:** https://wandb.ai/netanelazran11-hebrew-university-of-jerusalem/finetune-bmfm-rna-methylation-8k/runs/v4yn0u15
**State:** finished
**Generated:** 2026-02-04 11:39:33

---

## Final Results

| Metric | Train | Validation | Test | Ridge Baseline |
|--------|-------|------------|------|----------------|
| MAE (years) | 2.98 | 3.77 | 3.62 | 4.49 |
| R² | - | 0.9483 | 0.9574 | 0.94 |

## Performance Summary

- **Best Val MAE:** 3.77 years (at epoch 94)
- **Test MAE:** 3.62 years
- **Test R²:** 0.9574
- **vs Ridge Baseline:** 19.4% improvement in MAE

## Training Details

- **Total Epochs:** 100
- **Early Stopping Triggered:** No
- **Epochs After Best:** 5

## Overfitting Analysis

- **Train-Val MAE Gap:** 0.80 years
- **Assessment:** Healthy (gap < 1 year)

## Key Observations

1. **Model Architecture:** Direct MLP on raw beta values (8000 -> 512 -> 256 -> 128 -> 1)
   - Bypasses the transformer encoder which washes out per-sample differences
   - Uses LayerNorm, GELU activations, and dropout for regularization

2. **vs Ridge Regression:**
   - Ridge (linear): MAE = 4.49 years, R² = 0.94
   - MLP (nonlinear): MAE = 3.62 years, R² = 0.9574
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
