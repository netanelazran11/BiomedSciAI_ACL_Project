#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""W&B Analysis for BMFM Methylation Pretraining Runs.

Downloads history from WandB and creates visualizations to analyze:
- Loss curves (train vs validation)
- MAE curves
- PCC (Pearson Correlation) curves
- Identifies when/if training collapse occurred

Usage:
    python analyze_pretrain_wandb.py

Environment variables:
    WANDB_ENTITY: Your WandB entity (default: netanelazran11-hebrew-university-of-jerusalem)
    WANDB_PROJECT: Project name (default: pretrain-bmfm-rna-methylation-8k)
    WANDB_RUN_ID: Specific run ID to analyze (optional, analyzes latest if not set)
    OUTDIR: Output directory for plots and CSVs
"""

import os
import json
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wandb

# ==============================
# CONFIG (env overrides)
# ==============================
ENTITY = os.getenv("WANDB_ENTITY", "netanelazran11-hebrew-university-of-jerusalem")
PROJECT = os.getenv("WANDB_PROJECT", "pretrain-fixed2048-bmfm-rna-methylation")
RUN_ID = os.getenv("WANDB_RUN_ID", None)  # If None, will list runs and let user choose

DEFAULT_OUTDIR = os.path.join(os.path.dirname(__file__), "..", "wandb_analysis")
OUTDIR = os.path.abspath(os.path.expanduser(os.getenv("OUTDIR", DEFAULT_OUTDIR)))

WANDB_TIMEOUT = int(os.getenv("WANDB_TIMEOUT", "600"))

# Metrics to track
TRAIN_METRICS = [
    "train/loss_epoch",
    "train/loss_step",
    "train/beta_values_mae_epoch",
    "train/beta_values_mae_step",
    "train/beta_values_mse_epoch",
    "train/beta_values_mse_step",
    "train/beta_values_pcc_epoch",
    "train/beta_values_pcc_step",
]

VAL_METRICS = [
    "validation/loss",
    "validation/beta_values_mae",
    "validation/beta_values_mse",
    "validation/beta_values_pcc",
]

LR_METRICS = [
    "lr-AdamW/pg1",
    "lr-AdamW/pg2",
]

# ==============================
# Helpers
# ==============================

def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def fetch_runs(api: wandb.Api, entity: str, project: str) -> List[wandb.apis.public.Run]:
    """Fetch all runs from a project."""
    path = f"{entity}/{project}"
    runs = api.runs(path=path)
    return list(runs)


def download_run_history(run: wandb.apis.public.Run) -> pd.DataFrame:
    """Download full history for a run."""
    print(f"Downloading history for run: {run.name} ({run.id})")

    # Download history with all keys
    try:
        hist = run.history(pandas=True, samples=10000)
    except Exception as e:
        print(f"Error downloading history: {e}")
        hist = pd.DataFrame()

    return hist


def identify_collapse(df: pd.DataFrame, pcc_col: str = "validation/beta_values_pcc",
                      threshold: float = 0.5) -> Tuple[bool, Optional[int]]:
    """
    Identify if training collapsed (PCC dropped significantly).

    Returns:
        (collapsed: bool, collapse_epoch: Optional[int])
    """
    if pcc_col not in df.columns:
        return False, None

    pcc = df[pcc_col].dropna()
    if len(pcc) < 5:
        return False, None

    # Find max PCC and check if it dropped significantly after
    max_pcc = pcc.max()
    max_idx = pcc.idxmax()

    # Check if PCC dropped below threshold after reaching max
    after_max = pcc.loc[max_idx:]
    if len(after_max) > 1:
        final_pcc = after_max.iloc[-1]
        if max_pcc > threshold and final_pcc < threshold:
            # Find when it dropped below threshold
            collapse_idx = after_max[after_max < threshold].first_valid_index()
            if collapse_idx is not None and 'epoch' in df.columns:
                collapse_epoch = df.loc[collapse_idx, 'epoch'] if 'epoch' in df.columns else None
                return True, collapse_epoch

    return False, None


def find_best_checkpoint(df: pd.DataFrame, metric: str = "validation/loss") -> Dict[str, Any]:
    """Find the best checkpoint based on a metric."""
    if metric not in df.columns:
        return {}

    valid_df = df[df[metric].notna()].copy()
    if len(valid_df) == 0:
        return {}

    best_idx = valid_df[metric].idxmin()
    best_row = valid_df.loc[best_idx]

    result = {
        "best_loss": best_row.get(metric),
        "best_epoch": best_row.get("epoch"),
        "best_step": best_row.get("trainer/global_step"),
    }

    # Add other metrics at best point
    for col in ["validation/beta_values_mae", "validation/beta_values_pcc"]:
        if col in best_row:
            result[col.replace("validation/", "best_")] = best_row[col]

    return result


# ==============================
# Plotting
# ==============================

def plot_loss_curves(df: pd.DataFrame, outdir: str, run_name: str) -> str:
    """Plot train vs validation loss over epochs."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Find epoch column
    epoch_col = "epoch" if "epoch" in df.columns else "_step"

    # Plot train loss
    if "train/loss_epoch" in df.columns:
        train_df = df[df["train/loss_epoch"].notna()]
        ax.plot(train_df[epoch_col], train_df["train/loss_epoch"],
                label="Train Loss", alpha=0.8, color="blue")

    # Plot validation loss
    if "validation/loss" in df.columns:
        val_df = df[df["validation/loss"].notna()]
        ax.plot(val_df[epoch_col], val_df["validation/loss"],
                label="Validation Loss", alpha=0.8, color="orange", linewidth=2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title(f"Training Loss Curves - {run_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add best point annotation
    if "validation/loss" in df.columns:
        val_df = df[df["validation/loss"].notna()]
        if len(val_df) > 0:
            best_idx = val_df["validation/loss"].idxmin()
            best_loss = val_df.loc[best_idx, "validation/loss"]
            best_epoch = val_df.loc[best_idx, epoch_col]
            ax.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.5)
            ax.annotate(f'Best: {best_loss:.4f}',
                       xy=(best_epoch, best_loss),
                       xytext=(best_epoch + 1, best_loss * 1.1),
                       fontsize=10, color='green')

    plt.tight_layout()
    path = os.path.join(outdir, "loss_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_mae_curves(df: pd.DataFrame, outdir: str, run_name: str) -> str:
    """Plot MAE over epochs."""
    fig, ax = plt.subplots(figsize=(12, 6))

    epoch_col = "epoch" if "epoch" in df.columns else "_step"

    if "train/beta_values_mae_epoch" in df.columns:
        train_df = df[df["train/beta_values_mae_epoch"].notna()]
        ax.plot(train_df[epoch_col], train_df["train/beta_values_mae_epoch"],
                label="Train MAE", alpha=0.8, color="blue")

    if "validation/beta_values_mae" in df.columns:
        val_df = df[df["validation/beta_values_mae"].notna()]
        ax.plot(val_df[epoch_col], val_df["validation/beta_values_mae"],
                label="Validation MAE", alpha=0.8, color="orange", linewidth=2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE (Mean Absolute Error)")
    ax.set_title(f"MAE Curves - {run_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(outdir, "mae_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_pcc_curves(df: pd.DataFrame, outdir: str, run_name: str) -> str:
    """Plot PCC (Pearson Correlation) over epochs."""
    fig, ax = plt.subplots(figsize=(12, 6))

    epoch_col = "epoch" if "epoch" in df.columns else "_step"

    if "train/beta_values_pcc_epoch" in df.columns:
        train_df = df[df["train/beta_values_pcc_epoch"].notna()]
        ax.plot(train_df[epoch_col], train_df["train/beta_values_pcc_epoch"],
                label="Train PCC", alpha=0.8, color="blue")

    if "validation/beta_values_pcc" in df.columns:
        val_df = df[df["validation/beta_values_pcc"].notna()]
        ax.plot(val_df[epoch_col], val_df["validation/beta_values_pcc"],
                label="Validation PCC", alpha=0.8, color="orange", linewidth=2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("PCC (Pearson Correlation)")
    ax.set_title(f"PCC Curves - {run_name}")
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Collapse Threshold (0.5)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.1)

    # Identify collapse
    collapsed, collapse_epoch = identify_collapse(df)
    if collapsed and collapse_epoch is not None:
        ax.axvline(x=collapse_epoch, color='red', linestyle='-', alpha=0.7)
        ax.annotate(f'Collapse @ epoch {collapse_epoch:.0f}',
                   xy=(collapse_epoch, 0.5),
                   xytext=(collapse_epoch + 2, 0.6),
                   fontsize=10, color='red',
                   arrowprops=dict(arrowstyle='->', color='red'))

    plt.tight_layout()
    path = os.path.join(outdir, "pcc_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_lr_schedule(df: pd.DataFrame, outdir: str, run_name: str) -> str:
    """Plot learning rate schedule."""
    fig, ax = plt.subplots(figsize=(12, 6))

    step_col = "trainer/global_step" if "trainer/global_step" in df.columns else "_step"

    for lr_col in LR_METRICS:
        if lr_col in df.columns:
            lr_df = df[df[lr_col].notna()]
            ax.plot(lr_df[step_col], lr_df[lr_col], label=lr_col, alpha=0.8)

    ax.set_xlabel("Step")
    ax.set_ylabel("Learning Rate")
    ax.set_title(f"Learning Rate Schedule - {run_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(outdir, "lr_schedule.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_all_metrics_combined(df: pd.DataFrame, outdir: str, run_name: str) -> str:
    """Create a combined figure with all key metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    epoch_col = "epoch" if "epoch" in df.columns else "_step"

    # 1. Loss
    ax = axes[0, 0]
    if "train/loss_epoch" in df.columns:
        train_df = df[df["train/loss_epoch"].notna()]
        ax.plot(train_df[epoch_col], train_df["train/loss_epoch"], label="Train", alpha=0.7)
    if "validation/loss" in df.columns:
        val_df = df[df["validation/loss"].notna()]
        ax.plot(val_df[epoch_col], val_df["validation/loss"], label="Valid", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. MAE
    ax = axes[0, 1]
    if "train/beta_values_mae_epoch" in df.columns:
        train_df = df[df["train/beta_values_mae_epoch"].notna()]
        ax.plot(train_df[epoch_col], train_df["train/beta_values_mae_epoch"], label="Train", alpha=0.7)
    if "validation/beta_values_mae" in df.columns:
        val_df = df[df["validation/beta_values_mae"].notna()]
        ax.plot(val_df[epoch_col], val_df["validation/beta_values_mae"], label="Valid", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE")
    ax.set_title("Mean Absolute Error")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. PCC
    ax = axes[1, 0]
    if "train/beta_values_pcc_epoch" in df.columns:
        train_df = df[df["train/beta_values_pcc_epoch"].notna()]
        ax.plot(train_df[epoch_col], train_df["train/beta_values_pcc_epoch"], label="Train", alpha=0.7)
    if "validation/beta_values_pcc" in df.columns:
        val_df = df[df["validation/beta_values_pcc"].notna()]
        ax.plot(val_df[epoch_col], val_df["validation/beta_values_pcc"], label="Valid", linewidth=2)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("PCC")
    ax.set_title("Pearson Correlation")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.1)

    # 4. Learning Rate
    ax = axes[1, 1]
    step_col = "trainer/global_step" if "trainer/global_step" in df.columns else "_step"
    for lr_col in LR_METRICS:
        if lr_col in df.columns:
            lr_df = df[df[lr_col].notna()]
            ax.plot(lr_df[step_col], lr_df[lr_col], label=lr_col.split("/")[-1], alpha=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(f"Pretraining Analysis - {run_name}", fontsize=14, fontweight='bold')
    plt.tight_layout()

    path = os.path.join(outdir, "all_metrics_combined.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


# ==============================
# Analysis Report
# ==============================

def generate_report(df: pd.DataFrame, run: wandb.apis.public.Run, outdir: str) -> str:
    """Generate a markdown analysis report."""

    # Find best checkpoint
    best = find_best_checkpoint(df)

    # Check for collapse
    collapsed, collapse_epoch = identify_collapse(df)

    # Calculate statistics
    report = f"""# Pretraining Analysis Report

**Run:** {run.name}
**Run ID:** {run.id}
**URL:** {run.url}
**State:** {run.state}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Summary

| Metric | Best Value | At Epoch |
|--------|------------|----------|
| Validation Loss | {best.get('best_loss', 'N/A'):.4f if best.get('best_loss') else 'N/A'} | {best.get('best_epoch', 'N/A')} |
| Validation MAE | {best.get('best_beta_values_mae', 'N/A'):.4f if best.get('best_beta_values_mae') else 'N/A'} | {best.get('best_epoch', 'N/A')} |
| Validation PCC | {best.get('best_beta_values_pcc', 'N/A'):.4f if best.get('best_beta_values_pcc') else 'N/A'} | {best.get('best_epoch', 'N/A')} |

## Training Stability

**Collapse Detected:** {'YES ⚠️' if collapsed else 'No ✅'}
"""

    if collapsed:
        report += f"""
**Collapse Epoch:** {collapse_epoch}

### What happened:
The model's PCC (Pearson Correlation) dropped significantly after reaching a peak.
This indicates the model started predicting constant values (mean) instead of learning patterns.

### Recommendation:
Use the checkpoint from epoch {best.get('best_epoch', 'early')} (before collapse) for fine-tuning.
Consider lowering the learning rate for future pretraining runs.
"""
    else:
        report += """
Training appeared stable throughout the run.
"""

    report += """
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

Based on this analysis, use the checkpoint at **epoch {best.get('best_epoch', 'N/A')}** with validation loss **{best.get('best_loss', 'N/A'):.4f if best.get('best_loss') else 'N/A'}**.
"""

    path = os.path.join(outdir, "analysis_report.md")
    with open(path, "w") as f:
        f.write(report)

    return path


# ==============================
# Main
# ==============================

def main() -> int:
    ensure_outdir(OUTDIR)

    print("=" * 80)
    print("BMFM Methylation Pretraining - WandB Analysis")
    print(f"ENTITY:  {ENTITY}")
    print(f"PROJECT: {PROJECT}")
    print(f"OUTDIR:  {OUTDIR}")
    print("=" * 80)

    api = wandb.Api(timeout=WANDB_TIMEOUT)

    # Get runs
    runs = fetch_runs(api, ENTITY, PROJECT)
    print(f"\nFound {len(runs)} runs in project.\n")

    # List runs
    print("Available runs:")
    print("-" * 80)
    for i, r in enumerate(runs[:20]):  # Show first 20
        print(f"{i+1:3d}. [{r.state:10s}] {r.name[:50]:50s} | {r.id}")
    print("-" * 80)

    # Select run
    if RUN_ID:
        run = api.run(f"{ENTITY}/{PROJECT}/{RUN_ID}")
    else:
        try:
            choice = input("\nEnter run number to analyze (or press Enter for most recent): ").strip()
            if choice:
                idx = int(choice) - 1
                run = runs[idx]
            else:
                run = runs[0]  # Most recent
        except (ValueError, IndexError):
            print("Invalid choice, using most recent run.")
            run = runs[0]

    print(f"\nAnalyzing run: {run.name} ({run.id})")

    # Download history
    df = download_run_history(run)

    if df.empty:
        print("No history data found!")
        return 1

    print(f"Downloaded {len(df)} history rows.")
    print(f"Columns: {list(df.columns)[:20]}...")

    # Save raw data
    csv_path = os.path.join(OUTDIR, "history_raw.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved raw history: {csv_path}")

    # Generate plots
    print("\nGenerating plots...")

    plots = []
    plots.append(plot_all_metrics_combined(df, OUTDIR, run.name))
    plots.append(plot_loss_curves(df, OUTDIR, run.name))
    plots.append(plot_mae_curves(df, OUTDIR, run.name))
    plots.append(plot_pcc_curves(df, OUTDIR, run.name))
    plots.append(plot_lr_schedule(df, OUTDIR, run.name))

    for p in plots:
        print(f"  Saved: {p}")

    # Generate report
    report_path = generate_report(df, run, OUTDIR)
    print(f"\nSaved report: {report_path}")

    # Print summary
    best = find_best_checkpoint(df)
    collapsed, collapse_epoch = identify_collapse(df)

    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"Best Validation Loss: {best.get('best_loss', 'N/A')}")
    print(f"Best Epoch: {best.get('best_epoch', 'N/A')}")
    print(f"Best PCC: {best.get('best_beta_values_pcc', 'N/A')}")
    print(f"Collapse Detected: {'YES' if collapsed else 'No'}")
    if collapsed:
        print(f"Collapse Epoch: {collapse_epoch}")
    print("=" * 80)

    print(f"\nAll outputs saved to: {OUTDIR}")
    print("DONE.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
