#!/usr/bin/env python3
"""
Comprehensive Multi-Seed Analysis for BMFM Methylation Fine-tuning

Analyzes multiple seed runs from wandb project: finetune-bmfm-multiseed
Excludes run: finetune-seed44-44067725

Generates:
- Summary statistics across seeds
- Individual run metrics
- Convergence plots
- Comparison tables
- Comprehensive report

Output: /Users/netanelazran/Projects/BMFM-RNA_thesis/methyl/wandb_analysis/multiply_seeds_wandb/
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import wandb

# Configuration
ENTITY = "netanelazran11-hebrew-university-of-jerusalem"
PROJECT = "finetune-bmfm-multiseed"
EXCLUDE_RUN_NAME = "finetune-seed44-44067725"
OUTPUT_DIR = "/Users/netanelazran/Projects/BMFM-RNA_thesis/methyl/wandb_analysis/multiply_seeds_wandb"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*80)
print("BMFM MULTI-SEED ANALYSIS")
print("="*80)
print(f"Project: {PROJECT}")
print(f"Output: {OUTPUT_DIR}")
print(f"Excluding: {EXCLUDE_RUN_NAME}")

# Initialize API
api = wandb.Api(timeout=600)

# Fetch all runs
print("\n[1/6] Fetching runs...")
all_runs = api.runs(f"{ENTITY}/{PROJECT}")
runs = [r for r in all_runs if EXCLUDE_RUN_NAME not in r.name]

print(f"  Total runs: {len(all_runs)}")
print(f"  Analyzing: {len(runs)}")
print(f"  Excluded: {len(all_runs) - len(runs)}")

# Extract run information
print("\n[2/6] Extracting run information...")
runs_info = []

for run in runs:
    info = {
        'run_id': run.id,
        'run_name': run.name,
        'state': run.state,
        'created_at': run.created_at,
        'url': f"https://wandb.ai/{ENTITY}/{PROJECT}/runs/{run.id}",
    }

    # Extract seed from name
    if 'seed' in run.name:
        try:
            seed = int(run.name.split('seed')[1].split('-')[0])
            info['seed'] = seed
        except:
            info['seed'] = None

    # Get summary metrics
    summary = run.summary
    info.update({
        'test_mae': summary.get('test/mae', np.nan),
        'test_r2': summary.get('test/r2', np.nan),
        'val_mae': summary.get('val/mae', np.nan),
        'val_r2': summary.get('val/r2', np.nan),
        'train_mae': summary.get('train/mae', np.nan),
        'train_loss': summary.get('train/loss_epoch', np.nan),
        'val_loss': summary.get('val/loss', np.nan),
        'final_epoch': summary.get('epoch', np.nan),
    })

    runs_info.append(info)
    print(f"  ✓ {run.name} (seed={info.get('seed', '?')}, state={run.state})")

# Create DataFrame
df_runs = pd.DataFrame(runs_info)
df_runs = df_runs.sort_values('seed') if 'seed' in df_runs.columns else df_runs

# Save runs table
runs_table_path = os.path.join(OUTPUT_DIR, "runs_table.csv")
df_runs.to_csv(runs_table_path, index=False)
print(f"\n  ✓ Saved runs table: runs_table.csv")

# Download history for each run
print("\n[3/6] Downloading training history...")
history_data = {}

for run in runs:
    print(f"  Downloading: {run.name}...")
    try:
        hist = run.history(pandas=True, samples=50000)
        history_data[run.id] = hist
        print(f"    ✓ {len(hist)} records")
    except Exception as e:
        print(f"    ✗ Error: {e}")
        history_data[run.id] = pd.DataFrame()

# Find best validation MAE for each run
print("\n[4/6] Finding best checkpoints...")
best_checkpoints = []

for run_info in runs_info:
    run_id = run_info['run_id']
    hist = history_data.get(run_id, pd.DataFrame())

    if 'val/mae' in hist.columns:
        val_mae = hist['val/mae'].dropna()
        if len(val_mae) > 0:
            best_idx = val_mae.idxmin()
            best_val_mae = val_mae.loc[best_idx]
            best_epoch = hist.loc[best_idx, 'epoch'] if 'epoch' in hist.columns else np.nan

            checkpoint_info = {
                'run_id': run_id,
                'run_name': run_info['run_name'],
                'seed': run_info.get('seed'),
                'best_val_mae': best_val_mae,
                'best_epoch': best_epoch,
                'final_test_mae': run_info['test_mae'],
                'final_test_r2': run_info['test_r2'],
            }

            best_checkpoints.append(checkpoint_info)
            print(f"  ✓ {run_info['run_name']}: Best val MAE = {best_val_mae:.4f} @ epoch {best_epoch:.0f}")

df_best = pd.DataFrame(best_checkpoints)
best_checkpoints_path = os.path.join(OUTPUT_DIR, "best_checkpoints.csv")
df_best.to_csv(best_checkpoints_path, index=False)
print(f"\n  ✓ Saved best checkpoints: best_checkpoints.csv")

# Compute summary statistics
print("\n[5/6] Computing summary statistics...")

# Filter only finished runs for statistics
finished_runs = df_runs[df_runs['state'] == 'finished'].copy()

if len(finished_runs) > 0:
    summary_stats = {
        'metric': [],
        'mean': [],
        'std': [],
        'min': [],
        'max': [],
        'n': []
    }

    metrics = ['test_mae', 'test_r2', 'val_mae', 'val_r2', 'train_mae']
    metric_names = {
        'test_mae': 'Test MAE (years)',
        'test_r2': 'Test R²',
        'val_mae': 'Validation MAE (years)',
        'val_r2': 'Validation R²',
        'train_mae': 'Train MAE (years)'
    }

    for metric in metrics:
        if metric in finished_runs.columns:
            values = finished_runs[metric].dropna()
            if len(values) > 0:
                summary_stats['metric'].append(metric_names.get(metric, metric))
                summary_stats['mean'].append(values.mean())
                summary_stats['std'].append(values.std())
                summary_stats['min'].append(values.min())
                summary_stats['max'].append(values.max())
                summary_stats['n'].append(len(values))

    df_summary = pd.DataFrame(summary_stats)
    summary_path = os.path.join(OUTPUT_DIR, "summary_statistics.csv")
    df_summary.to_csv(summary_path, index=False)

    print(f"\n  Summary Statistics (n={len(finished_runs)} finished runs):")
    print(f"  {'-'*76}")
    for _, row in df_summary.iterrows():
        print(f"  {row['metric']:30s}: {row['mean']:6.4f} ± {row['std']:6.4f}  [{row['min']:6.4f}, {row['max']:6.4f}]")

    print(f"\n  ✓ Saved summary statistics: summary_statistics.csv")

# Create plots
print("\n[6/6] Creating plots...")

# Set style
plt.style.use('default')
sns.set_palette("husl")

# Plot 1: MAE curves for all runs
fig1, ax1 = plt.subplots(figsize=(12, 7))

for run_info in runs_info:
    run_id = run_info['run_id']
    hist = history_data.get(run_id, pd.DataFrame())

    if 'val/mae' in hist.columns:
        val_mae = hist[['epoch', 'val/mae']].dropna()
        if len(val_mae) > 0:
            label = f"Seed {run_info.get('seed', '?')} ({run_info['state']})"
            ax1.plot(val_mae['epoch'], val_mae['val/mae'],
                    label=label, alpha=0.8, linewidth=2)

ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax1.set_ylabel('Validation MAE (years)', fontsize=12, fontweight='bold')
ax1.set_title('Multi-Seed Training: Validation MAE Convergence',
              fontsize=14, fontweight='bold')
ax1.legend(loc='upper right', fontsize=10)
ax1.grid(True, alpha=0.3)

plot1_path = os.path.join(OUTPUT_DIR, "mae_convergence_all_seeds.png")
plt.tight_layout()
plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Created: mae_convergence_all_seeds.png")

# Plot 2: R² curves for all runs
fig2, ax2 = plt.subplots(figsize=(12, 7))

for run_info in runs_info:
    run_id = run_info['run_id']
    hist = history_data.get(run_id, pd.DataFrame())

    if 'val/r2' in hist.columns:
        val_r2 = hist[['epoch', 'val/r2']].dropna()
        if len(val_r2) > 0:
            label = f"Seed {run_info.get('seed', '?')}"
            ax2.plot(val_r2['epoch'], val_r2['val/r2'],
                    label=label, alpha=0.8, linewidth=2)

ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax2.set_ylabel('Validation R²', fontsize=12, fontweight='bold')
ax2.set_title('Multi-Seed Training: Validation R² Convergence',
              fontsize=14, fontweight='bold')
ax2.legend(loc='lower right', fontsize=10)
ax2.grid(True, alpha=0.3)

plot2_path = os.path.join(OUTPUT_DIR, "r2_convergence_all_seeds.png")
plt.tight_layout()
plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Created: r2_convergence_all_seeds.png")

# Plot 3: Final test metrics comparison (if finished runs exist)
if len(finished_runs) > 0:
    fig3, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Test MAE
    ax3a = axes[0]
    seeds = finished_runs['seed'].values if 'seed' in finished_runs.columns else range(len(finished_runs))
    test_mae_vals = finished_runs['test_mae'].values

    bars1 = ax3a.bar(seeds, test_mae_vals, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax3a.axhline(test_mae_vals.mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean = {test_mae_vals.mean():.4f}')
    ax3a.set_xlabel('Seed', fontsize=12, fontweight='bold')
    ax3a.set_ylabel('Test MAE (years)', fontsize=12, fontweight='bold')
    ax3a.set_title('Test MAE by Seed', fontsize=13, fontweight='bold')
    ax3a.legend()
    ax3a.grid(True, alpha=0.3, axis='y')

    # Add values on bars
    for bar, val in zip(bars1, test_mae_vals):
        height = bar.get_height()
        ax3a.text(bar.get_x() + bar.get_width()/2., height,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    # Test R²
    ax3b = axes[1]
    test_r2_vals = finished_runs['test_r2'].values

    bars2 = ax3b.bar(seeds, test_r2_vals, alpha=0.7, color='green',
                    edgecolor='black', linewidth=1.5)
    ax3b.axhline(test_r2_vals.mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean = {test_r2_vals.mean():.4f}')
    ax3b.set_xlabel('Seed', fontsize=12, fontweight='bold')
    ax3b.set_ylabel('Test R²', fontsize=12, fontweight='bold')
    ax3b.set_title('Test R² by Seed', fontsize=13, fontweight='bold')
    ax3b.legend()
    ax3b.grid(True, alpha=0.3, axis='y')

    # Add values on bars
    for bar, val in zip(bars2, test_r2_vals):
        height = bar.get_height()
        ax3b.text(bar.get_x() + bar.get_width()/2., height,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=9)

    plt.suptitle('Final Test Metrics Across Seeds', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()

    plot3_path = os.path.join(OUTPUT_DIR, "test_metrics_comparison.png")
    plt.savefig(plot3_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Created: test_metrics_comparison.png")

# Plot 4: Box plots for metrics
if len(finished_runs) > 0 and len(finished_runs) > 1:
    fig4, axes = plt.subplots(2, 2, figsize=(14, 10))

    metrics_to_plot = [
        ('test_mae', 'Test MAE (years)', axes[0, 0]),
        ('test_r2', 'Test R²', axes[0, 1]),
        ('val_mae', 'Validation MAE (years)', axes[1, 0]),
        ('val_r2', 'Validation R²', axes[1, 1])
    ]

    for metric, title, ax in metrics_to_plot:
        if metric in finished_runs.columns:
            values = finished_runs[metric].dropna()
            if len(values) > 0:
                bp = ax.boxplot([values], labels=['All Seeds'], patch_artist=True,
                               showmeans=True, meanline=True)
                bp['boxes'][0].set_facecolor('lightblue')
                ax.set_ylabel(title, fontsize=11, fontweight='bold')
                ax.set_title(f'{title} Distribution', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y')

                # Add statistics text
                stats_text = f'Mean: {values.mean():.4f}\nStd: {values.std():.4f}\nN: {len(values)}'
                ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                       fontsize=9, verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.suptitle('Multi-Seed Metrics Distribution', fontsize=15, fontweight='bold')
    plt.tight_layout()

    plot4_path = os.path.join(OUTPUT_DIR, "metrics_distribution.png")
    plt.savefig(plot4_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Created: metrics_distribution.png")

# Create comprehensive report
print("\n[7/7] Creating comprehensive report...")

report_path = os.path.join(OUTPUT_DIR, "MULTISEED_ANALYSIS_REPORT.md")

with open(report_path, 'w') as f:
    f.write("# BMFM Multi-Seed Analysis Report\n\n")
    f.write("**Generated:** " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n\n")
    f.write("**Project:** finetune-bmfm-multiseed\n\n")
    f.write("**Excluded Run:** finetune-seed44-44067725\n\n")
    f.write("---\n\n")

    f.write("## Overview\n\n")
    f.write(f"- **Total Runs Analyzed:** {len(runs)}\n")
    f.write(f"- **Finished Runs:** {len(finished_runs)}\n")
    f.write(f"- **Running Runs:** {len(df_runs[df_runs['state'] == 'running'])}\n\n")

    f.write("## Runs Table\n\n")
    f.write("| Seed | Run Name | State | Test MAE | Test R² | URL |\n")
    f.write("|------|----------|-------|----------|---------|-----|\n")
    for _, row in df_runs.iterrows():
        seed = row.get('seed', '?')
        mae = f"{row['test_mae']:.4f}" if pd.notna(row['test_mae']) else 'N/A'
        r2 = f"{row['test_r2']:.4f}" if pd.notna(row['test_r2']) else 'N/A'
        f.write(f"| {seed} | {row['run_name']} | {row['state']} | {mae} | {r2} | [Link]({row['url']}) |\n")

    f.write("\n## Summary Statistics (Finished Runs)\n\n")
    if len(finished_runs) > 0:
        f.write(f"**N = {len(finished_runs)} runs**\n\n")
        f.write("| Metric | Mean | Std | Min | Max |\n")
        f.write("|--------|------|-----|-----|-----|\n")
        for _, row in df_summary.iterrows():
            f.write(f"| {row['metric']} | {row['mean']:.4f} | {row['std']:.4f} | {row['min']:.4f} | {row['max']:.4f} |\n")
    else:
        f.write("*No finished runs yet.*\n\n")

    f.write("\n## Best Checkpoints\n\n")
    f.write("| Seed | Best Val MAE | Best Epoch | Final Test MAE | Final Test R² |\n")
    f.write("|------|--------------|------------|----------------|---------------|\n")
    for _, row in df_best.iterrows():
        seed = row.get('seed', '?')
        best_val = f"{row['best_val_mae']:.4f}" if pd.notna(row['best_val_mae']) else 'N/A'
        epoch = f"{row['best_epoch']:.0f}" if pd.notna(row['best_epoch']) else 'N/A'
        test_mae = f"{row['final_test_mae']:.4f}" if pd.notna(row['final_test_mae']) else 'N/A'
        test_r2 = f"{row['final_test_r2']:.4f}" if pd.notna(row['final_test_r2']) else 'N/A'
        f.write(f"| {seed} | {best_val} | {epoch} | {test_mae} | {test_r2} |\n")

    f.write("\n## Visualizations\n\n")
    f.write("1. `mae_convergence_all_seeds.png` - Validation MAE over epochs\n")
    f.write("2. `r2_convergence_all_seeds.png` - Validation R² over epochs\n")
    f.write("3. `test_metrics_comparison.png` - Final test metrics by seed\n")
    f.write("4. `metrics_distribution.png` - Box plots of metric distributions\n\n")

    f.write("## Files Generated\n\n")
    f.write("- `runs_table.csv` - All run information\n")
    f.write("- `summary_statistics.csv` - Aggregate statistics\n")
    f.write("- `best_checkpoints.csv` - Best validation checkpoints\n")
    f.write("- `MULTISEED_ANALYSIS_REPORT.md` - This report\n\n")

    f.write("---\n\n")
    f.write("*Report generated by analyze_multiseed.py*\n")

print(f"\n  ✓ Created comprehensive report: MULTISEED_ANALYSIS_REPORT.md")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print(f"\nOutput directory: {OUTPUT_DIR}")
print("\nGenerated files:")
print("  - runs_table.csv")
print("  - summary_statistics.csv")
print("  - best_checkpoints.csv")
print("  - mae_convergence_all_seeds.png")
print("  - r2_convergence_all_seeds.png")
print("  - test_metrics_comparison.png")
print("  - metrics_distribution.png")
print("  - MULTISEED_ANALYSIS_REPORT.md")
print("="*80)
