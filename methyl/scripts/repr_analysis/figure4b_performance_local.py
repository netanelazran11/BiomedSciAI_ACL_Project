#!/usr/bin/env python3
"""
figure4b_performance_local.py  — Option B (local, no cluster needed)
======================================================================
Generates a 2-panel performance comparison figure:
  Left : Predicted vs Actual age scatter (linear probe on pretrained CLS)
  Right: Predicted vs Actual age scatter (linear probe on fine-tuned CLS)
  + bar chart showing MedAE: linear probe before / linear probe after / full model

Shows the dramatic improvement: R²=0.20 → 0.905, MedAE=16.8 → 3.65 yr
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, median_absolute_error

ROOT     = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "figures" / "figure4"
OUT_DIR  = DATA_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def run_probe(emb, ages, splits, label):
    train = (splits == "train") & ~np.isnan(ages)
    test  = (splits == "test")  & ~np.isnan(ages)
    ridge = Ridge(alpha=1.0)
    ridge.fit(emb[train], ages[train])
    pred  = ridge.predict(emb[test])
    r2    = r2_score(ages[test], pred)
    medae = median_absolute_error(ages[test], pred)
    print(f"[{label}] R²={r2:.3f}  MedAE={medae:.2f} yr  (test n={test.sum()})")
    return ages[test], pred, r2, medae


def scatter_panel(ax, actual, pred, r2, medae, title, color):
    ax.set_facecolor("#F7F7F7")
    ax.grid(True, color="white", linewidth=0.8, zorder=0)
    for sp in ax.spines.values():
        sp.set_linewidth(0.6); sp.set_color("#AAAAAA")

    lim = (0, 105)
    ax.plot(lim, lim, "--", color="#999999", linewidth=1.0, zorder=1)
    ax.scatter(actual, pred, c=color, s=8, alpha=0.55,
               linewidths=0, rasterized=True, zorder=2)

    ax.set_xlim(*lim); ax.set_ylim(*lim)
    ax.set_xlabel("Actual Age (years)", fontsize=9)
    ax.set_ylabel("Predicted Age (years)", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=5)
    ax.text(0.05, 0.93, f"R² = {r2:.2f}\nMedAE = {medae:.1f} yr",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#CCCCCC", alpha=0.85))


def bar_panel(ax, labels, medaes, colors):
    ax.set_facecolor("#F7F7F7")
    ax.grid(True, axis="y", color="white", linewidth=0.8, zorder=0)
    for sp in ax.spines.values():
        sp.set_linewidth(0.6); sp.set_color("#AAAAAA")

    bars = ax.bar(labels, medaes, color=colors, edgecolor="white",
                  linewidth=0.8, zorder=2, width=0.5)
    for bar, val in zip(bars, medaes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f} yr", ha="center", va="bottom", fontsize=9,
                fontweight="bold")

    ax.set_ylabel("Median Absolute Error (years)", fontsize=9)
    ax.set_title("Age Prediction Performance", fontsize=11,
                 fontweight="bold", pad=5)
    ax.set_ylim(0, max(medaes) * 1.25)
    ax.tick_params(axis="x", labelsize=8)


def main():
    print("Loading embeddings and metadata...")
    pre_emb = np.load(DATA_DIR / "embeddings_cls_pretrained.npy").astype(np.float32)
    ft_emb  = np.load(DATA_DIR / "embeddings_cls_finetuned.npy").astype(np.float32)
    meta    = pd.read_csv(DATA_DIR / "aligned_metadata.csv", index_col=0)

    ages   = pd.to_numeric(meta["age"], errors="coerce").values
    splits = meta["split"].values

    actual_pre, pred_pre, r2_pre, medae_pre = run_probe(
        pre_emb, ages, splits, "Pretrained CLS linear probe")
    actual_ft, pred_ft, r2_ft, medae_ft = run_probe(
        ft_emb, ages, splits, "Fine-tuned CLS linear probe")

    # Full fine-tuned model result from training logs (epoch=127)
    medae_full = 3.5625
    r2_full    = 0.905

    print(f"\nFull fine-tuned model: R²={r2_full}  MedAE={medae_full} yr")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 5))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35,
                           left=0.07, right=0.97, top=0.88, bottom=0.14)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    scatter_panel(ax1, actual_pre, pred_pre, r2_pre, medae_pre,
                  "a  |  Pretrained CLS\n(linear probe — before fine-tuning)",
                  color="#4DBBD5")

    scatter_panel(ax2, actual_ft, pred_ft, r2_ft, medae_ft,
                  "b  |  Fine-tuned CLS\n(linear probe — after fine-tuning)",
                  color="#9B59B6")

    bar_panel(ax3,
              labels=["Before FT\n(linear probe)", "After FT\n(linear probe)",
                      "After FT\n(full model)"],
              medaes=[medae_pre, medae_ft, medae_full],
              colors=["#4DBBD5", "#9B59B6", "#E64B35"])

    fig.suptitle("MethylLlama Age Prediction — Before vs After Fine-tuning",
                 fontsize=13, fontweight="bold", y=0.98)

    for ext in ["png", "pdf"]:
        out = OUT_DIR / f"figure4_performance.{ext}"
        fig.savefig(out, dpi=200 if ext == "png" else 72,
                    bbox_inches="tight", facecolor="white")
        print(f"  Saved → {out}")
    plt.close()
    print("Done.")


if __name__ == "__main__":
    main()
