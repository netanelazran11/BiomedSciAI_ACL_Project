#!/usr/bin/env python3
"""
Extract all metrics from BMFM wandb run: ycap9s0f
Metrics: MAE, R², MEDAE, RMSE, PCC (Pearson Correlation Coefficient)
"""

import wandb
import pandas as pd
import numpy as np

print("="*80)
print("EXTRACTING ALL METRICS FROM WANDB RUN")
print("="*80)

# Initialize wandb API
api = wandb.Api(timeout=600)

# Run details
entity = "netanelazran11-hebrew-university-of-jerusalem"
project = "finetune-fixed2048-bmfm-rna-methylation"
run_id = "ycap9s0f"

run_path = f"{entity}/{project}/{run_id}"

print(f"\nFetching run: {run_path}")
print(f"URL: https://wandb.ai/{entity}/{project}/runs/{run_id}")

# Fetch the run
run = api.run(run_path)

print(f"\nRun Name: {run.name}")
print(f"State: {run.state}")
print(f"Created: {run.created_at}")

# Get summary metrics (final values)
summary = run.summary

print("\n" + "="*80)
print("SUMMARY METRICS (Final Test Results)")
print("="*80)

# Extract all available metrics
all_keys = list(summary.keys())
print(f"\nTotal keys in summary: {len(all_keys)}")

# Filter for test metrics
test_keys = [k for k in all_keys if 'test' in k.lower()]

print("\n" + "-"*80)
print("TEST METRICS")
print("-"*80)

metrics_dict = {}

# Standard metrics
metric_names = {
    'test/mae': 'MAE (Mean Absolute Error)',
    'test/r2': 'R² (Coefficient of Determination)',
    'test/medae': 'MEDAE (Median Absolute Error)',
    'test/rmse': 'RMSE (Root Mean Squared Error)',
    'test/pcc': 'PCC (Pearson Correlation Coefficient)',
    'test/spearman': 'Spearman Correlation',
    'test/mse': 'MSE (Mean Squared Error)',
    'test/loss': 'Test Loss'
}

for key, description in metric_names.items():
    if key in summary:
        value = summary[key]
        metrics_dict[key] = value
        print(f"{description:40s}: {value:.4f}")
    else:
        print(f"{description:40s}: NOT AVAILABLE")

# Check for any other test metrics not in our list
print("\n" + "-"*80)
print("ALL TEST-RELATED KEYS IN SUMMARY:")
print("-"*80)
for key in sorted(test_keys):
    if key in summary:
        value = summary[key]
        if isinstance(value, (int, float)):
            print(f"  {key:40s}: {value:.6f}")
        else:
            print(f"  {key:40s}: {value}")

# Also get validation metrics
print("\n" + "-"*80)
print("VALIDATION METRICS")
print("-"*80)

val_keys = [k for k in all_keys if 'val' in k.lower()]
for key in sorted(val_keys):
    if key in summary:
        value = summary[key]
        if isinstance(value, (int, float)):
            print(f"  {key:40s}: {value:.6f}")

# Get training metrics
print("\n" + "-"*80)
print("TRAINING METRICS")
print("-"*80)

train_keys = [k for k in all_keys if 'train' in k.lower()]
for key in sorted(train_keys):
    if key in summary:
        value = summary[key]
        if isinstance(value, (int, float)):
            print(f"  {key:40s}: {value:.6f}")

# Get best epoch info
print("\n" + "="*80)
print("TRAINING DETAILS")
print("="*80)

if 'epoch' in summary:
    print(f"Total Epochs: {summary['epoch']:.0f}")

if '_step' in summary:
    print(f"Total Steps: {summary['_step']}")

# Download full history to find best values
print("\n" + "="*80)
print("DOWNLOADING FULL HISTORY...")
print("="*80)

history = run.history()
print(f"Downloaded {len(history)} history records")

# Find best validation MAE and corresponding epoch
if 'val/mae' in history.columns:
    val_mae = history['val/mae'].dropna()
    if len(val_mae) > 0:
        best_val_mae_idx = val_mae.idxmin()
        best_val_mae = val_mae.loc[best_val_mae_idx]
        best_epoch = history.loc[best_val_mae_idx, 'epoch'] if 'epoch' in history.columns else None

        print(f"\nBest Validation MAE: {best_val_mae:.4f}")
        if best_epoch is not None:
            print(f"Best Epoch: {best_epoch:.0f}")

# Create comprehensive summary
print("\n" + "="*80)
print("COMPREHENSIVE METRICS SUMMARY")
print("="*80)

print("\n┌─────────────────────────────────────────────────────────┐")
print("│                    TEST METRICS                         │")
print("├─────────────────────────────────────────────────────────┤")

if 'test/mae' in summary:
    print(f"│  MAE (years)                    : {summary['test/mae']:8.4f}       │")
if 'test/medae' in summary:
    print(f"│  MEDAE (years)                  : {summary['test/medae']:8.4f}       │")
if 'test/rmse' in summary:
    print(f"│  RMSE (years)                   : {summary['test/rmse']:8.4f}       │")
if 'test/r2' in summary:
    print(f"│  R² Score                       : {summary['test/r2']:8.4f}       │")
if 'test/pcc' in summary:
    print(f"│  PCC (Pearson Correlation)      : {summary['test/pcc']:8.4f}       │")
if 'test/spearman' in summary:
    print(f"│  Spearman Correlation           : {summary['test/spearman']:8.4f}       │")

print("└─────────────────────────────────────────────────────────┘")

# Save to file
print("\n" + "="*80)
print("SAVING METRICS TO FILE")
print("="*80)

# Create DataFrame with all metrics
metrics_data = {
    'metric': [],
    'value': [],
    'description': []
}

# Add test metrics
for key, desc in metric_names.items():
    if key in summary:
        metrics_data['metric'].append(key.replace('test/', ''))
        metrics_data['value'].append(summary[key])
        metrics_data['description'].append(desc)

df_metrics = pd.DataFrame(metrics_data)

output_file = "/Users/netanelazran/Projects/BMFM-RNA_thesis/methyl/wandb_analysis/finetune/all_test_metrics.csv"
df_metrics.to_csv(output_file, index=False)
print(f"\n✓ Saved to: {output_file}")

# Also create a formatted text report
report_file = "/Users/netanelazran/Projects/BMFM-RNA_thesis/methyl/wandb_analysis/finetune/metrics_report.txt"
with open(report_file, 'w') as f:
    f.write("="*80 + "\n")
    f.write("BMFM METHYLATION AGE PREDICTION - ALL METRICS\n")
    f.write("="*80 + "\n")
    f.write(f"\nRun ID: {run_id}\n")
    f.write(f"Run Name: {run.name}\n")
    f.write(f"URL: https://wandb.ai/{entity}/{project}/runs/{run_id}\n")
    f.write(f"State: {run.state}\n")
    f.write("\n" + "="*80 + "\n")
    f.write("TEST METRICS\n")
    f.write("="*80 + "\n\n")

    if 'test/mae' in summary:
        f.write(f"MAE (Mean Absolute Error)          : {summary['test/mae']:.4f} years\n")
    if 'test/medae' in summary:
        f.write(f"MEDAE (Median Absolute Error)      : {summary['test/medae']:.4f} years\n")
    if 'test/rmse' in summary:
        f.write(f"RMSE (Root Mean Squared Error)     : {summary['test/rmse']:.4f} years\n")
    if 'test/r2' in summary:
        f.write(f"R² (Coefficient of Determination)  : {summary['test/r2']:.4f}\n")
    if 'test/pcc' in summary:
        f.write(f"PCC (Pearson Correlation)          : {summary['test/pcc']:.4f}\n")
    if 'test/spearman' in summary:
        f.write(f"Spearman Correlation               : {summary['test/spearman']:.4f}\n")

    f.write("\n" + "="*80 + "\n")
    f.write("VALIDATION METRICS\n")
    f.write("="*80 + "\n\n")

    if 'val/mae' in summary:
        f.write(f"Validation MAE                     : {summary['val/mae']:.4f} years\n")
    if 'val/r2' in summary:
        f.write(f"Validation R²                      : {summary['val/r2']:.4f}\n")

    f.write("\n" + "="*80 + "\n")

print(f"✓ Saved report to: {report_file}")

print("\n" + "="*80)
print("DONE!")
print("="*80)
