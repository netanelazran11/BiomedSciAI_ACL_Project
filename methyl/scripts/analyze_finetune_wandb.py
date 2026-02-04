#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""W&B Analysis for BMFM Methylation Fine-tuning (Age Regression).

Downloads history from WandB and creates visualizations to analyze:
- Loss curves (train vs validation)
- MAE curves (in years)
- R² curves
- Learning rate schedule
- Train-val gap (overfitting analysis)
- Convergence analysis

Usage:
    python scripts/analyze_finetune_wandb.py

    # Or specify run directly:
    WANDB_RUN_ID=v4yn0u15 python scripts/analyze_finetune_wandb.py

Environment variables:
    WANDB_ENTITY: Your WandB entity (default: netanelazran11-hebrew-university-of-jerusalem)
    WANDB_PROJECT: Project name (default: finetune-bmfm-rna-methylation-8k)
    WANDB_RUN_ID: Specific run ID to analyze (optional, lists runs if not set)
    OUTDIR: Output directory for plots and CSVs
"""

import os
import json
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb

# ==============================
# CONFIG (env overrides)
# ==============================
ENTITY = os.getenv("WANDB_ENTITY", "netanelazran11-hebrew-university-of-jerusalem")
PROJECT = os.getenv("WANDB_PROJECT", "finetune-bmfm-rna-methylation-8k")
RUN_ID = os.getenv("WANDB_RUN_ID", None)

DEFAULT_OUTDIR = os.path.join(os.path.dirname(__file__), "..", "wandb_analysis", "finetune")
OUTDIR = os.path.abspath(os.path.expanduser(os.getenv("OUTDIR", DEFAULT_OUTDIR)))

WANDB_TIMEOUT = int(os.getenv("WANDB_TIMEOUT", "600"))

# Fine-tuning metrics (from MethylationAgeRegressor)
TRAIN_METRICS = [
    "train/loss_epoch",
    "train/loss_step",
    "train/loss",
    "train/mae",
    "train/mae_epoch",
]

VAL_METRICS = [
    "val/loss",
    "val/loss_epoch",
    "val/mae",
    "val/mae_epoch",
    "val/r2",
]

TEST_METRICS = [
    "test/mae",
    "test/mae_epoch",
    "test/r2",
]

LR_METRICS = [
    "lr-AdamW",
    "lr-AdamW/pg1",
    "lr-AdamW/pg2",
]

# Ridge baseline reference (from baseline_ridge.py results)
RIDGE_BASELINE_MAE = 4.49
RIDGE_BASELINE_R2 = 0.94


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
    try:
        hist = run.history(pandas=True, samples=50000)
    except Exception as e:
        print(f"Error downloading history: {e}")
        hist = pd.DataFrame()
    return hist


def get_epoch_data(df: pd.DataFrame) -> pd.DataFrame:
    """Extract epoch-level data (one row per epoch)."""
    if "epoch" not in df.columns:
        return df

    # Get validation rows (they have val/loss populated)
    val_cols = [c for c in df.columns if c.startswith("val/")]
    if not val_cols:
        return df

    # Group by epoch and aggregate: take the non-NaN value for each metric
    epoch_df = df.groupby("epoch").agg(
        lambda x: x.dropna().iloc[-1] if len(x.dropna()) > 0 else np.nan
    ).reset_index()

    return epoch_df


def find_best_checkpoint(df: pd.DataFrame) -> Dict[str, Any]:
    """Find the best checkpoint based on val/mae."""
    metric = "val/mae"
    if metric not in df.columns:
        # Try alternative names
        for alt in ["val/mae_epoch"]:
            if alt in df.columns:
                metric = alt
                break
        else:
            return {}

    valid_df = df[df[metric].notna()].copy()
    if len(valid_df) == 0:
        return {}

    best_idx = valid_df[metric].idxmin()
    best_row = valid_df.loc[best_idx]

    result = {
        "best_mae": best_row.get(metric),
        "best_epoch": best_row.get("epoch"),
        "best_step": best_row.get("trainer/global_step", best_row.get("_step")),
    }

    # Add other metrics at best point
    for col in ["val/loss", "val/r2", "val/mae_epoch"]:
        if col in best_row and pd.notna(best_row[col]):
            key = col.replace("val/", "best_val_")
            result[key] = best_row[col]

    return result


def detect_early_stopping(df: pd.DataFrame, patience: int = 20) -> Dict[str, Any]:
    """Detect if early stopping was triggered."""
    metric = "val/mae"
    if metric not in df.columns:
        return {"triggered": False}

    val_data = df[df[metric].notna()][metric]
    if len(val_data) == 0:
        return {"triggered": False}

    best_idx = val_data.idxmin()
    best_pos = list(val_data.index).index(best_idx)
    epochs_after_best = len(val_data) - best_pos - 1

    return {
        "triggered": epochs_after_best >= patience,
        "best_at_position": best_pos,
        "epochs_after_best": epochs_after_best,
        "total_epochs": len(val_data),
    }


def compute_train_val_gap(df: pd.DataFrame) -> pd.DataFrame:
    """Compute train-val gap over epochs for overfitting analysis."""
    gap_data = []

    epoch_df = get_epoch_data(df)

    # Try to find matching train/val MAE columns
    train_col = None
    val_col = None
    for tc in ["train/mae", "train/mae_epoch"]:
        if tc in epoch_df.columns:
            train_col = tc
            break
    for vc in ["val/mae", "val/mae_epoch"]:
        if vc in epoch_df.columns:
            val_col = vc
            break

    if train_col is None or val_col is None:
        return pd.DataFrame()

    for _, row in epoch_df.iterrows():
        if pd.notna(row.get(train_col)) and pd.notna(row.get(val_col)):
            gap_data.append({
                "epoch": row.get("epoch", row.name),
                "train_mae": row[train_col],
                "val_mae": row[val_col],
                "gap": row[val_col] - row[train_col],
                "ratio": row[val_col] / row[train_col] if row[train_col] > 0 else np.nan,
            })

    return pd.DataFrame(gap_data)


# ==============================
# Plotting
# ==============================

def plot_loss_curves(df: pd.DataFrame, outdir: str, run_name: str) -> str:
    """Plot train vs validation loss over epochs."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    epoch_col = "epoch" if "epoch" in df.columns else "_step"

    # Left: epoch-level loss
    ax = axes[0]
    for col, label, color in [
        ("train/loss_epoch", "Train Loss (epoch)", "blue"),
        ("train/loss", "Train Loss", "blue"),
        ("val/loss", "Val Loss", "orange"),
        ("val/loss_epoch", "Val Loss (epoch)", "orange"),
    ]:
        if col in df.columns:
            data = df[df[col].notna()]
            if len(data) > 0:
                ax.plot(data[epoch_col], data[col], label=label, alpha=0.8,
                        color=color, linewidth=2 if "val" in col else 1)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title(f"Loss Curves (Epoch)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Best val loss annotation
    for col in ["val/loss", "val/loss_epoch"]:
        if col in df.columns:
            val_df = df[df[col].notna()]
            if len(val_df) > 0:
                best_idx = val_df[col].idxmin()
                best_loss = val_df.loc[best_idx, col]
                best_epoch = val_df.loc[best_idx, epoch_col]
                ax.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.4)
                ax.annotate(f'Best: {best_loss:.4f}\n@ epoch {best_epoch:.0f}',
                           xy=(best_epoch, best_loss),
                           xytext=(best_epoch + 3, best_loss * 1.3),
                           fontsize=9, color='green',
                           arrowprops=dict(arrowstyle='->', color='green', alpha=0.6))
                break

    # Right: step-level loss (more granular)
    ax = axes[1]
    step_col = "trainer/global_step" if "trainer/global_step" in df.columns else "_step"
    for col, label, color in [
        ("train/loss_step", "Train Loss (step)", "blue"),
        ("train/loss", "Train Loss", "blue"),
    ]:
        if col in df.columns:
            data = df[df[col].notna()]
            if len(data) > 0:
                ax.plot(data[step_col], data[col], label=label, alpha=0.3, color=color, linewidth=0.5)
                # Rolling average
                if len(data) > 20:
                    rolling = data[col].rolling(window=20, min_periods=1).mean()
                    ax.plot(data[step_col], rolling, label=f"{label} (MA-20)",
                            color="darkblue", linewidth=1.5, alpha=0.8)
                break

    ax.set_xlabel("Step")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title(f"Training Loss (Step-level)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(f"Loss Analysis - {run_name}", fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(outdir, "loss_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_mae_curves(df: pd.DataFrame, outdir: str, run_name: str) -> str:
    """Plot MAE over epochs with Ridge baseline reference."""
    fig, ax = plt.subplots(figsize=(14, 7))

    epoch_col = "epoch" if "epoch" in df.columns else "_step"

    plotted = False
    for col, label, color, lw in [
        ("train/mae", "Train MAE", "blue", 1.5),
        ("train/mae_epoch", "Train MAE (epoch)", "blue", 1.5),
        ("val/mae", "Val MAE", "orange", 2),
        ("val/mae_epoch", "Val MAE (epoch)", "darkorange", 2),
    ]:
        if col in df.columns:
            data = df[df[col].notna()]
            if len(data) > 0:
                ax.plot(data[epoch_col], data[col], label=label, alpha=0.8,
                        color=color, linewidth=lw)
                plotted = True

    # Ridge baseline reference line
    ax.axhline(y=RIDGE_BASELINE_MAE, color='red', linestyle='--', alpha=0.7,
               linewidth=1.5, label=f'Ridge Baseline ({RIDGE_BASELINE_MAE:.2f} yr)')

    # Mean prediction baseline (predict training mean age)
    ax.axhline(y=19.0, color='gray', linestyle=':', alpha=0.5,
               linewidth=1, label='Mean Prediction (~19 yr)')

    # Best val MAE annotation
    for col in ["val/mae", "val/mae_epoch"]:
        if col in df.columns:
            val_df = df[df[col].notna()]
            if len(val_df) > 0:
                best_idx = val_df[col].idxmin()
                best_mae = val_df.loc[best_idx, col]
                best_epoch = val_df.loc[best_idx, epoch_col]
                ax.annotate(f'Best Val MAE: {best_mae:.2f} yr\n@ epoch {best_epoch:.0f}',
                           xy=(best_epoch, best_mae),
                           xytext=(best_epoch + 5, best_mae + 1),
                           fontsize=10, color='darkorange', fontweight='bold',
                           arrowprops=dict(arrowstyle='->', color='darkorange', lw=1.5))
                break

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("MAE (years)", fontsize=12)
    ax.set_title(f"Mean Absolute Error - {run_name}", fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(outdir, "mae_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_r2_curves(df: pd.DataFrame, outdir: str, run_name: str) -> str:
    """Plot R-squared over epochs."""
    fig, ax = plt.subplots(figsize=(14, 7))

    epoch_col = "epoch" if "epoch" in df.columns else "_step"

    plotted = False
    for col, label, color, lw in [
        ("val/r2", "Val R²", "orange", 2),
        ("test/r2", "Test R²", "green", 2),
    ]:
        if col in df.columns:
            data = df[df[col].notna()]
            if len(data) > 0:
                ax.plot(data[epoch_col], data[col], label=label, alpha=0.8,
                        color=color, linewidth=lw, marker='o', markersize=3)
                plotted = True

    # Ridge baseline R²
    ax.axhline(y=RIDGE_BASELINE_R2, color='red', linestyle='--', alpha=0.7,
               linewidth=1.5, label=f'Ridge Baseline (R²={RIDGE_BASELINE_R2:.2f})')

    # Reference lines
    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.3)
    ax.axhline(y=0.0, color='gray', linestyle=':', alpha=0.3)

    # Best R² annotation
    if "val/r2" in df.columns:
        val_df = df[df["val/r2"].notna()]
        if len(val_df) > 0:
            best_idx = val_df["val/r2"].idxmax()
            best_r2 = val_df.loc[best_idx, "val/r2"]
            best_epoch = val_df.loc[best_idx, epoch_col]
            ax.annotate(f'Best Val R²: {best_r2:.4f}\n@ epoch {best_epoch:.0f}',
                       xy=(best_epoch, best_r2),
                       xytext=(best_epoch - 15, best_r2 - 0.08),
                       fontsize=10, color='darkorange', fontweight='bold',
                       arrowprops=dict(arrowstyle='->', color='darkorange', lw=1.5))

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("R²", fontsize=12)
    ax.set_title(f"R-squared (Coefficient of Determination) - {run_name}", fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    path = os.path.join(outdir, "r2_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_lr_schedule(df: pd.DataFrame, outdir: str, run_name: str) -> str:
    """Plot learning rate schedule."""
    fig, ax = plt.subplots(figsize=(14, 6))

    step_col = "trainer/global_step" if "trainer/global_step" in df.columns else "_step"

    plotted = False
    for lr_col in LR_METRICS:
        if lr_col in df.columns:
            lr_df = df[df[lr_col].notna()]
            if len(lr_df) > 0:
                label = lr_col.split("/")[-1] if "/" in lr_col else lr_col
                ax.plot(lr_df[step_col], lr_df[lr_col], label=label, alpha=0.8)
                plotted = True

    if not plotted:
        # Try to find any lr column
        lr_cols = [c for c in df.columns if "lr" in c.lower()]
        for lr_col in lr_cols:
            lr_df = df[df[lr_col].notna()]
            if len(lr_df) > 0:
                ax.plot(lr_df[step_col], lr_df[lr_col], label=lr_col, alpha=0.8)

    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Learning Rate", fontsize=12)
    ax.set_title(f"Learning Rate Schedule - {run_name}", fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    plt.tight_layout()
    path = os.path.join(outdir, "lr_schedule.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_train_val_gap(df: pd.DataFrame, outdir: str, run_name: str) -> str:
    """Plot train-val gap over time to detect overfitting."""
    gap_df = compute_train_val_gap(df)

    if gap_df.empty:
        print("  No train-val gap data available, skipping plot.")
        return ""

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: train vs val MAE side by side
    ax = axes[0]
    ax.plot(gap_df["epoch"], gap_df["train_mae"], label="Train MAE", color="blue",
            linewidth=1.5, alpha=0.8)
    ax.plot(gap_df["epoch"], gap_df["val_mae"], label="Val MAE", color="orange",
            linewidth=2, alpha=0.8)
    ax.fill_between(gap_df["epoch"], gap_df["train_mae"], gap_df["val_mae"],
                    alpha=0.15, color="red", label="Gap")

    # Ridge baseline
    ax.axhline(y=RIDGE_BASELINE_MAE, color='red', linestyle='--', alpha=0.5,
               linewidth=1, label=f'Ridge ({RIDGE_BASELINE_MAE:.2f})')

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("MAE (years)", fontsize=11)
    ax.set_title("Train vs Validation MAE")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right: gap magnitude
    ax = axes[1]
    ax.plot(gap_df["epoch"], gap_df["gap"], color="red", linewidth=1.5, alpha=0.8)
    ax.fill_between(gap_df["epoch"], 0, gap_df["gap"], alpha=0.2, color="red")
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.axhline(y=1.0, color='orange', linestyle='--', alpha=0.5,
               label="1-year gap (mild)")
    ax.axhline(y=3.0, color='red', linestyle='--', alpha=0.5,
               label="3-year gap (concerning)")

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Val MAE - Train MAE (years)", fontsize=11)
    ax.set_title("Generalization Gap")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Annotate final gap
    if len(gap_df) > 0:
        final = gap_df.iloc[-1]
        ax.annotate(f'Final gap: {final["gap"]:.2f} yr',
                   xy=(final["epoch"], final["gap"]),
                   xytext=(final["epoch"] - 20, final["gap"] + 0.5),
                   fontsize=10, fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color='red'))

    plt.suptitle(f"Overfitting Analysis - {run_name}", fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(outdir, "train_val_gap.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_convergence(df: pd.DataFrame, outdir: str, run_name: str) -> str:
    """Plot convergence analysis: how quickly did the model learn?"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    epoch_col = "epoch" if "epoch" in df.columns else "_step"

    # Left: MAE improvement per epoch
    ax = axes[0]
    mae_col = None
    for col in ["val/mae", "val/mae_epoch"]:
        if col in df.columns:
            mae_col = col
            break

    if mae_col:
        val_df = df[df[mae_col].notna()].copy()
        if len(val_df) > 1:
            epochs = val_df[epoch_col].values
            maes = val_df[mae_col].values

            ax.plot(epochs, maes, color="orange", linewidth=2, label="Val MAE")

            # Mark key milestones
            milestones = [
                (RIDGE_BASELINE_MAE, "Ridge baseline", "red"),
                (5.0, "5-year MAE", "blue"),
                (4.0, "4-year MAE", "green"),
            ]
            for threshold, label, color in milestones:
                ax.axhline(y=threshold, color=color, linestyle='--', alpha=0.4, linewidth=1)
                # Find first epoch below threshold
                below = np.where(maes <= threshold)[0]
                if len(below) > 0:
                    first_epoch = epochs[below[0]]
                    ax.axvline(x=first_epoch, color=color, linestyle=':', alpha=0.3)
                    ax.annotate(f'{label}\n@ epoch {first_epoch:.0f}',
                               xy=(first_epoch, threshold),
                               xytext=(first_epoch + 3, threshold + 0.5),
                               fontsize=8, color=color)

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Val MAE (years)", fontsize=11)
    ax.set_title("Convergence Speed")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Right: relative improvement (% of final MAE achieved)
    ax = axes[1]
    if mae_col:
        val_df = df[df[mae_col].notna()].copy()
        if len(val_df) > 1:
            epochs = val_df[epoch_col].values
            maes = val_df[mae_col].values

            # Starting MAE and final MAE
            start_mae = maes[0]
            final_mae = maes[-1]
            total_improvement = start_mae - final_mae

            if total_improvement > 0:
                pct_done = ((start_mae - maes) / total_improvement) * 100
                pct_done = np.clip(pct_done, 0, 100)
                ax.plot(epochs, pct_done, color="green", linewidth=2)
                ax.fill_between(epochs, 0, pct_done, alpha=0.1, color="green")

                # Mark 90% convergence
                above_90 = np.where(pct_done >= 90)[0]
                if len(above_90) > 0:
                    e90 = epochs[above_90[0]]
                    ax.axvline(x=e90, color='green', linestyle='--', alpha=0.5)
                    ax.annotate(f'90% converged\n@ epoch {e90:.0f}',
                               xy=(e90, 90), xytext=(e90 + 5, 75),
                               fontsize=9, color='green', fontweight='bold',
                               arrowprops=dict(arrowstyle='->', color='green'))

                ax.axhline(y=90, color='green', linestyle=':', alpha=0.3)
                ax.axhline(y=100, color='gray', linestyle=':', alpha=0.3)

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("% of Total Improvement", fontsize=11)
    ax.set_title("Convergence Progress")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 105)

    plt.suptitle(f"Convergence Analysis - {run_name}", fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(outdir, "convergence.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_all_metrics_combined(df: pd.DataFrame, outdir: str, run_name: str) -> str:
    """Create a combined 2x3 figure with all key metrics."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    epoch_col = "epoch" if "epoch" in df.columns else "_step"

    # 1. Loss (train vs val)
    ax = axes[0, 0]
    for col, label, color in [
        ("train/loss_epoch", "Train", "blue"),
        ("train/loss", "Train", "blue"),
        ("val/loss", "Valid", "orange"),
    ]:
        if col in df.columns:
            data = df[df[col].notna()]
            if len(data) > 0:
                ax.plot(data[epoch_col], data[col], label=label,
                        color=color, linewidth=1.5 if "val" in col else 1, alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. MAE
    ax = axes[0, 1]
    for col, label, color in [
        ("train/mae", "Train", "blue"),
        ("val/mae", "Valid", "orange"),
    ]:
        if col in df.columns:
            data = df[df[col].notna()]
            if len(data) > 0:
                ax.plot(data[epoch_col], data[col], label=label,
                        color=color, linewidth=1.5 if "val" in col else 1, alpha=0.8)
    ax.axhline(y=RIDGE_BASELINE_MAE, color='red', linestyle='--', alpha=0.5,
               label=f'Ridge ({RIDGE_BASELINE_MAE})')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE (years)")
    ax.set_title("Mean Absolute Error")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. R²
    ax = axes[0, 2]
    if "val/r2" in df.columns:
        data = df[df["val/r2"].notna()]
        if len(data) > 0:
            ax.plot(data[epoch_col], data["val/r2"], label="Val R²",
                    color="orange", linewidth=2, alpha=0.8)
    ax.axhline(y=RIDGE_BASELINE_R2, color='red', linestyle='--', alpha=0.5,
               label=f'Ridge ({RIDGE_BASELINE_R2})')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("R²")
    ax.set_title("R-squared")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    # 4. Learning Rate
    ax = axes[1, 0]
    step_col = "trainer/global_step" if "trainer/global_step" in df.columns else "_step"
    for lr_col in LR_METRICS:
        if lr_col in df.columns:
            lr_df = df[df[lr_col].notna()]
            if len(lr_df) > 0:
                label = lr_col.split("/")[-1] if "/" in lr_col else lr_col
                ax.plot(lr_df[step_col], lr_df[lr_col], label=label, alpha=0.8)
    if not any(c in df.columns for c in LR_METRICS):
        lr_cols = [c for c in df.columns if "lr" in c.lower()]
        for lr_col in lr_cols[:3]:
            lr_df = df[df[lr_col].notna()]
            if len(lr_df) > 0:
                ax.plot(lr_df[step_col], lr_df[lr_col], label=lr_col, alpha=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Learning Rate")
    ax.set_title("LR Schedule")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 5. Train-Val Gap
    ax = axes[1, 1]
    gap_df = compute_train_val_gap(df)
    if not gap_df.empty:
        ax.plot(gap_df["epoch"], gap_df["gap"], color="red", linewidth=1.5, alpha=0.8)
        ax.fill_between(gap_df["epoch"], 0, gap_df["gap"], alpha=0.15, color="red")
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax.axhline(y=1.0, color='orange', linestyle='--', alpha=0.4, label="1-yr gap")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val MAE - Train MAE (years)")
    ax.set_title("Generalization Gap")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 6. MAE vs Ridge comparison bar chart (final results)
    ax = axes[1, 2]
    best = find_best_checkpoint(df)
    best_mae = best.get("best_mae", np.nan)

    # Get test MAE if available
    test_mae = np.nan
    for col in ["test/mae", "test/mae_epoch"]:
        if col in df.columns:
            test_data = df[df[col].notna()]
            if len(test_data) > 0:
                test_mae = test_data[col].iloc[-1]
                break

    labels = ["Ridge\nBaseline", "MLP\n(Best Val)", "MLP\n(Test)"]
    values = [RIDGE_BASELINE_MAE, best_mae, test_mae]
    colors = ["#e74c3c", "#f39c12", "#27ae60"]

    bars = ax.bar(labels, values, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.1,
                    f'{val:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

    ax.set_ylabel("MAE (years)")
    ax.set_title("Model Comparison")
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle(f"Fine-tuning Analysis Overview - {run_name}", fontsize=14, fontweight='bold')
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
    best = find_best_checkpoint(df)
    early_stop = detect_early_stopping(df)
    gap_df = compute_train_val_gap(df)

    # Get test metrics
    test_mae = None
    test_r2 = None
    for col in ["test/mae", "test/mae_epoch"]:
        if col in df.columns:
            test_data = df[df[col].notna()]
            if len(test_data) > 0:
                test_mae = test_data[col].iloc[-1]
                break
    if "test/r2" in df.columns:
        test_data = df[df["test/r2"].notna()]
        if len(test_data) > 0:
            test_r2 = test_data["test/r2"].iloc[-1]

    # Get final train MAE
    train_mae = None
    for col in ["train/mae", "train/mae_epoch"]:
        if col in df.columns:
            data = df[df[col].notna()]
            if len(data) > 0:
                train_mae = data[col].iloc[-1]
                break

    # Compute final gap
    final_gap = None
    if not gap_df.empty:
        final_gap = gap_df.iloc[-1]["gap"]

    # Format helper
    def fmt(val, decimals=4):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "N/A"
        return f"{val:.{decimals}f}"

    # Ridge improvement
    ridge_improvement = None
    if test_mae is not None:
        ridge_improvement = ((RIDGE_BASELINE_MAE - test_mae) / RIDGE_BASELINE_MAE) * 100

    report = f"""# Fine-tuning Analysis Report

**Run:** {run.name}
**Run ID:** {run.id}
**URL:** {run.url}
**State:** {run.state}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Final Results

| Metric | Train | Validation | Test | Ridge Baseline |
|--------|-------|------------|------|----------------|
| MAE (years) | {fmt(train_mae, 2)} | {fmt(best.get('best_mae'), 2)} | {fmt(test_mae, 2)} | {RIDGE_BASELINE_MAE:.2f} |
| R² | - | {fmt(best.get('best_val_r2'), 4)} | {fmt(test_r2, 4)} | {RIDGE_BASELINE_R2:.2f} |

## Performance Summary

- **Best Val MAE:** {fmt(best.get('best_mae'), 2)} years (at epoch {fmt(best.get('best_epoch'), 0)})
- **Test MAE:** {fmt(test_mae, 2)} years
- **Test R²:** {fmt(test_r2, 4)}
- **vs Ridge Baseline:** {fmt(ridge_improvement, 1)}% improvement in MAE

## Training Details

- **Total Epochs:** {early_stop.get('total_epochs', 'N/A')}
- **Early Stopping Triggered:** {'Yes' if early_stop.get('triggered') else 'No'}
- **Epochs After Best:** {early_stop.get('epochs_after_best', 'N/A')}

## Overfitting Analysis

- **Train-Val MAE Gap:** {fmt(final_gap, 2)} years
- **Assessment:** {"Healthy (gap < 1 year)" if final_gap is not None and final_gap < 1.0 else "Mild (gap 1-2 years)" if final_gap is not None and final_gap < 2.0 else "Concerning (gap > 2 years)" if final_gap is not None else "N/A"}

## Key Observations

1. **Model Architecture:** Direct MLP on raw beta values (8000 -> 512 -> 256 -> 128 -> 1)
   - Bypasses the transformer encoder which washes out per-sample differences
   - Uses LayerNorm, GELU activations, and dropout for regularization

2. **vs Ridge Regression:**
   - Ridge (linear): MAE = {RIDGE_BASELINE_MAE:.2f} years, R² = {RIDGE_BASELINE_R2:.2f}
   - MLP (nonlinear): MAE = {fmt(test_mae, 2)} years, R² = {fmt(test_r2, 4)}
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
    print("BMFM Methylation Fine-tuning - WandB Analysis")
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
    for i, r in enumerate(runs[:20]):
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
                run = runs[0]
        except (ValueError, IndexError):
            print("Invalid choice, using most recent run.")
            run = runs[0]

    print(f"\nAnalyzing run: {run.name} ({run.id})")
    print(f"Run state: {run.state}")

    # Print run config summary
    config = run.config
    print(f"\nRun config summary:")
    for key in ["finetune_epochs", "freeze_encoder", "unfreeze_encoder_epoch",
                 "use_huber_loss", "accumulate_grad_batches"]:
        if key in config:
            print(f"  {key}: {config[key]}")
    if "trainer" in config:
        tc = config["trainer"]
        for key in ["learning_rate", "weight_decay", "warmup_steps"]:
            if key in tc:
                print(f"  trainer.{key}: {tc[key]}")
    if "regression_head" in config:
        rh = config["regression_head"]
        print(f"  regression_head: {rh}")

    # Download history
    df = download_run_history(run)

    if df.empty:
        print("No history data found!")
        return 1

    print(f"Downloaded {len(df)} history rows.")
    print(f"Available columns ({len(df.columns)}):")
    for col in sorted(df.columns):
        non_null = df[col].notna().sum()
        if non_null > 0:
            print(f"  {col:50s}  ({non_null} values)")

    # Save raw data
    csv_path = os.path.join(OUTDIR, "finetune_history_raw.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved raw history: {csv_path}")

    # Generate plots
    print("\nGenerating plots...")

    plots = []
    plots.append(("Combined overview", plot_all_metrics_combined(df, OUTDIR, run.name)))
    plots.append(("Loss curves", plot_loss_curves(df, OUTDIR, run.name)))
    plots.append(("MAE curves", plot_mae_curves(df, OUTDIR, run.name)))
    plots.append(("R² curves", plot_r2_curves(df, OUTDIR, run.name)))
    plots.append(("LR schedule", plot_lr_schedule(df, OUTDIR, run.name)))
    plots.append(("Train-val gap", plot_train_val_gap(df, OUTDIR, run.name)))
    plots.append(("Convergence", plot_convergence(df, OUTDIR, run.name)))

    for name, p in plots:
        if p:
            print(f"  Saved: {p}")
        else:
            print(f"  Skipped: {name} (no data)")

    # Generate report
    report_path = generate_report(df, run, OUTDIR)
    print(f"\nSaved report: {report_path}")

    # Print summary
    best = find_best_checkpoint(df)
    early_stop = detect_early_stopping(df)

    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"Best Val MAE:    {best.get('best_mae', 'N/A')}")
    print(f"Best Epoch:      {best.get('best_epoch', 'N/A')}")
    print(f"Best Val R²:     {best.get('best_val_r2', 'N/A')}")
    print(f"Ridge Baseline:  MAE={RIDGE_BASELINE_MAE}, R²={RIDGE_BASELINE_R2}")

    if best.get('best_mae') is not None:
        improvement = ((RIDGE_BASELINE_MAE - best['best_mae']) / RIDGE_BASELINE_MAE) * 100
        print(f"Improvement:     {improvement:.1f}% over Ridge")

    print(f"Early Stopping:  {'Triggered' if early_stop.get('triggered') else 'Not triggered'}")
    print(f"Total Epochs:    {early_stop.get('total_epochs', 'N/A')}")
    print("=" * 80)

    # Get test results
    for col in ["test/mae", "test/mae_epoch"]:
        if col in df.columns:
            test_data = df[df[col].notna()]
            if len(test_data) > 0:
                print(f"Test MAE:        {test_data[col].iloc[-1]:.2f} years")
                break
    if "test/r2" in df.columns:
        test_data = df[df["test/r2"].notna()]
        if len(test_data) > 0:
            print(f"Test R²:         {test_data['test/r2'].iloc[-1]:.4f}")

    print(f"\nAll outputs saved to: {OUTDIR}")
    print("DONE.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
