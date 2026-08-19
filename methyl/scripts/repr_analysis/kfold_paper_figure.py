#!/usr/bin/env python3
"""
kfold_paper_figure.py
======================
Publication-style figure: V7b pretrain -> 5-fold finetune -> test benchmark
vs MethylGPT 21k. Pulls the same cached WandB histories as
kfold_full_history_analysis.py (run that first) and renders a clean,
paper-quality 4-panel figure (no run IDs / job numbers shown).

Usage:
  /Users/netanelazran/miniconda3/envs/methylgpt-local/bin/python3 \
      scripts/repr_analysis/kfold_paper_figure.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy import stats

IN_DIR = Path("kfold_full_history_analysis")
OUT_DIR = Path("kfold_paper_figure")
OUT_DIR.mkdir(exist_ok=True)

# ── validated categorical pair (dataviz skill, slots 1 & 8) ──────────────────
BLUE        = "#2a78d6"   # V7b / ours
BLUE_LIGHT  = "#9ec5f4"   # individual folds (sequential step, de-emphasized)
ORANGE      = "#eb6834"   # MethylGPT baseline
INK         = "#0b0b0b"
INK_SEC     = "#52514e"
INK_MUTED   = "#898781"
GRID        = "#e1e0d9"

MethylGPT_BASELINE = {"medae": 3.839, "mae": 5.521, "r2": 0.9044}

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 9,
    "axes.edgecolor": INK_MUTED,
    "axes.linewidth": 0.8,
    "xtick.color": INK_SEC,
    "ytick.color": INK_SEC,
    "text.color": INK,
    "axes.labelcolor": INK,
})


def style_ax(ax, ygrid=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("white")
    if ygrid:
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=0.8)


def panel_label(ax, letter):
    ax.text(-0.14, 1.08, letter, transform=ax.transAxes,
             fontsize=13, fontweight="bold", va="top", ha="left", color=INK)


def main():
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.15], hspace=0.48, wspace=0.42)

    # ── Panel A: pretrain reconstruction loss ───────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    pre = pd.read_csv(IN_DIR / "pretrain_v7b_history.csv")
    ax.plot(pre["epoch"], pre["train/recon_loss"], color=BLUE_LIGHT, linewidth=1.6, label="Train")
    ax.plot(pre["epoch"], pre["validation/loss"], color=BLUE, linewidth=2.0, label="Validation")
    style_ax(ax)
    ax.set_xlabel("Pretraining epoch")
    ax.set_ylabel("Reconstruction loss")
    ax.legend(frameon=False, fontsize=8, loc="upper right", handlelength=1.5)
    panel_label(ax, "A")

    # ── Panel B: pretrain reconstruction PCC ────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(pre["epoch"], pre["train/pcc"], color=BLUE_LIGHT, linewidth=1.6, label="Train")
    ax.plot(pre["epoch"], pre["validation/pcc"], color=BLUE, linewidth=2.0, label="Validation")
    style_ax(ax)
    ax.set_xlabel("Pretraining epoch")
    ax.set_ylabel("Reconstruction PCC")
    ax.legend(frameon=False, fontsize=8, loc="lower right", handlelength=1.5)
    panel_label(ax, "B")

    # ── Panel C: encoder capacity note (small text panel) ───────────────────
    ax = fig.add_subplot(gs[0, 2])
    ax.axis("off")
    ax.text(0, 0.9, "Encoder", fontsize=10, fontweight="bold", color=INK)
    ax.text(0, 0.72, "6 transformer layers · 256d · 4 heads", fontsize=8.5, color=INK_SEC)
    ax.text(0, 0.58, "Genomic RoPE (chromosomal position)", fontsize=8.5, color=INK_SEC)
    ax.text(0, 0.44, "Contrastive WCED pretraining (InfoNCE)", fontsize=8.5, color=INK_SEC)
    ax.text(0, 0.30, "49,156 CpG sites · 50% masking ratio", fontsize=8.5, color=INK_SEC)
    ax.text(0, 0.08, "Best checkpoint: epoch 85\nrecon = 0.0552 · PCC = 0.9713",
            fontsize=8.5, color=INK, style="italic")

    # ── Panel D: 5-fold finetune validation MedAE curves ────────────────────
    ax = fig.add_subplot(gs[1, 0:2])
    ft = pd.read_csv(IN_DIR / "finetune_all_folds_history.csv")
    max_epoch = int(ft["epoch"].max())
    grid = np.arange(0, max_epoch + 1)
    fold_curves = []
    for fold, g in ft.groupby("fold"):
        g = g.sort_values("epoch")
        interp = np.interp(grid, g["epoch"], g["val/medae"])
        fold_curves.append(interp)
        ax.plot(g["epoch"], g["val/medae"], color=BLUE_LIGHT, linewidth=1.0, alpha=0.8, zorder=2)
    mean_curve = np.mean(fold_curves, axis=0)
    ax.plot(grid, mean_curve, color=BLUE, linewidth=2.2, zorder=4, label="V7b (mean of 5 folds)")
    ax.axhline(MethylGPT_BASELINE["medae"], color=ORANGE, linewidth=1.8, linestyle="--", zorder=3)
    ax.text(max_epoch * 0.98, MethylGPT_BASELINE["medae"] + 0.35, "MethylGPT 21k baseline",
            color=ORANGE, fontsize=8.5, ha="right", fontweight="medium")
    ax.plot([], [], color=BLUE_LIGHT, linewidth=1.0, label="Individual folds (n=5)")
    style_ax(ax)
    ax.set_xlabel("Fine-tuning epoch")
    ax.set_ylabel("Validation MedAE (years)")
    ax.set_ylim(0, 12)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right", handlelength=1.5)
    panel_label(ax, "C")

    # ── Panel E: final test benchmark ───────────────────────────────────────
    test = pd.read_csv(IN_DIR / "fold_test_results.csv")
    done = test.dropna(subset=["test_medae"])
    n = len(done)

    ax = fig.add_subplot(gs[1, 2])
    metrics = [("test_medae", "MedAE", MethylGPT_BASELINE["medae"]),
               ("test_mae",   "MAE",   MethylGPT_BASELINE["mae"])]
    x = np.arange(len(metrics))
    width = 0.32

    gpt_vals = [m[2] for m in metrics]
    v7b_means, v7b_cis = [], []
    for key, _, _ in metrics:
        vals = done[key].values
        mean = vals.mean()
        ci = stats.sem(vals) * stats.t.ppf(0.975, n - 1) if n > 1 else 0
        v7b_means.append(mean)
        v7b_cis.append(ci)

    b1 = ax.bar(x - width / 2, gpt_vals, width, color=ORANGE, zorder=3)
    b2 = ax.bar(x + width / 2, v7b_means, width, yerr=v7b_cis, color=BLUE,
                capsize=3, error_kw={"linewidth": 1.2, "ecolor": INK_SEC}, zorder=3)

    for rect, val in zip(b1, gpt_vals):
        ax.text(rect.get_x() + rect.get_width() / 2, val + 0.15, f"{val:.2f}",
                ha="center", fontsize=8, color=INK_SEC)
    for rect, val, ci in zip(b2, v7b_means, v7b_cis):
        ax.text(rect.get_x() + rect.get_width() / 2, val + ci + 0.15, f"{val:.2f}",
                ha="center", fontsize=8, color=INK, fontweight="bold")

    style_ax(ax)
    ax.set_xticks(x)
    ax.set_xticklabels([m[1] for m in metrics])
    ax.set_ylabel("Years")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.set_ylim(0, max(gpt_vals) * 1.35)
    handles = [plt.Rectangle((0, 0), 1, 1, color=ORANGE), plt.Rectangle((0, 0), 1, 1, color=BLUE)]
    ax.legend(handles, ["MethylGPT 21k", f"V7b (n={n} folds)"], frameon=False,
              fontsize=8, loc="upper right", handlelength=1.0)
    panel_label(ax, "D")

    fig.savefig(OUT_DIR / "figure1_kfold_benchmark.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_DIR / "figure1_kfold_benchmark.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved -> {OUT_DIR / 'figure1_kfold_benchmark.png'}")
    print(f"Saved -> {OUT_DIR / 'figure1_kfold_benchmark.pdf'}")

    # ── Standalone R2 panel (separate scale, own small figure) ──────────────
    fig2, ax = plt.subplots(figsize=(3.2, 3.4))
    fig2.patch.set_facecolor("white")
    gpt_r2 = MethylGPT_BASELINE["r2"]
    vals = done["test_r2"].values
    mean_r2 = vals.mean()
    ci_r2 = stats.sem(vals) * stats.t.ppf(0.975, n - 1) if n > 1 else 0
    xs = [0, 1]
    ax.bar(xs[0], gpt_r2, width=0.5, color=ORANGE, zorder=3)
    ax.bar(xs[1], mean_r2, width=0.5, yerr=ci_r2, color=BLUE, capsize=3,
           error_kw={"linewidth": 1.2, "ecolor": INK_SEC}, zorder=3)
    ax.text(xs[0], gpt_r2 + 0.01, f"{gpt_r2:.3f}", ha="center", fontsize=8.5, color=INK_SEC)
    ax.text(xs[1], mean_r2 + ci_r2 + 0.01, f"{mean_r2:.3f}", ha="center", fontsize=8.5,
            color=INK, fontweight="bold")
    style_ax(ax)
    ax.set_xticks(xs)
    ax.set_xticklabels(["MethylGPT\n21k", f"V7b\n(n={n})"])
    ax.set_ylabel("Test R²")
    ax.set_ylim(0.85, 1.0)
    fig2.tight_layout()
    fig2.savefig(OUT_DIR / "figure1_supp_r2.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig2)
    print(f"Saved -> {OUT_DIR / 'figure1_supp_r2.png'}")


if __name__ == "__main__":
    main()
