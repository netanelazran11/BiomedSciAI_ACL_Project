# Pretraining Analysis Report

**Run:** add-fixed2048-44043043
**Run ID:** hj2v23ae
**URL:** https://wandb.ai/netanelazran11-hebrew-university-of-jerusalem/pretrain-fixed2048-bmfm-rna-methylation/runs/hj2v23ae
**State:** finished
**Generated:** 2026-02-10 15:26:14

---

## Summary

| Metric | Best Value | At Epoch |
|--------|------------|----------|
| Validation Loss | 0.0013 | 234.0 |
| Validation MAE | 0.0216 | 234.0 |
| Validation PCC | 0.9943 | 234.0 |

## Training Stability

**Collapse Detected:** No ✅

Training appeared stable throughout the run.

## Metrics Interpretation

- **Loss (MSE):** Mean Squared Error on masked beta values. Lower is better.
- **MAE:** Mean Absolute Error. For beta values (0-1), MAE < 0.1 is good.
- **PCC:** Pearson Correlation between predictions and ground truth. Higher is better (>0.8 is excellent).

## Plots

- `all_metrics_combined.png` - Overview of all metrics
- `loss_curves.png` - Train vs Validation loss
- `mae_curves.png` - Mean Absolute Error curves
- `pcc_curves.png` - Pearson Correlation curves
- `lr_schedule.png` - Learning rate over training

## Best Checkpoint for Fine-tuning

Based on this analysis, use the checkpoint at **epoch {best_epoch}** with validation loss **{best_loss}**.
