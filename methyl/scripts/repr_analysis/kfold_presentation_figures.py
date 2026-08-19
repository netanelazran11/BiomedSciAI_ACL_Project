#!/usr/bin/env python3
"""
kfold_presentation_figures.py
===============================
Figures for docs/presentations/MethylLlama_vs_MethylGPT_21k.html, styled to
match that deck's exact color language (llama blue #1a4ab0/#3a6acc, gpt
orange #8a3800/#c07030) rather than the generic dataviz palette, for visual
consistency within the single-document report.

Reads the cached WandB pulls in kfold_full_history_analysis/ (run
kfold_full_history_analysis.py first if that dir is missing).

Usage:
  /Users/netanelazran/miniconda3/envs/methylgpt-local/bin/python3 \
      scripts/repr_analysis/kfold_presentation_figures.py
"""

import base64
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

IN_DIR = Path("kfold_full_history_analysis")
OUT_DIR = Path("kfold_presentation_figures")
OUT_DIR.mkdir(exist_ok=True)

LLAMA_BLUE       = "#1a4ab0"
LLAMA_BLUE_LIGHT = "#a8c8ff"
GPT_ORANGE       = "#8a3800"
GPT_ORANGE_LIGHT = "#ffcc99"
INK   = "#1e2535"
INK_SEC = "#5a6a8a"
GRID  = "#e4e8f4"

MethylGPT_BASELINE = {"medae": 3.839, "mae": 5.521, "r2": 0.9044}
FOLD_TRAIN_RUNS = {0: "a154lzjy", 1: "aas23t9m", 2: "vw5f9nh4", 3: "vqogel6f", 4: "6qt4ma56"}

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 10,
    "axes.edgecolor": "#c8d0e0",
    "axes.linewidth": 0.8,
    "xtick.color": INK_SEC,
    "ytick.color": INK_SEC,
    "text.color": INK,
    "axes.labelcolor": INK,
})


def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("white")
    ax.grid(axis="y", color=GRID, linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=0.8)


def fig_to_base64(fig):
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def main():
    # ── Val loss curves (5 folds) ───────────────────────────────────────────
    loss_dfs = {}
    for fold in FOLD_TRAIN_RUNS:
        loss_dfs[fold] = pd.read_csv(IN_DIR / f"fold{fold}_valloss_history.csv")
    max_epoch = int(max(df["epoch"].max() for df in loss_dfs.values()))
    grid = np.arange(0, max_epoch + 1)
    curves = [np.interp(grid, df["epoch"], df["val/loss"]) for df in loss_dfs.values()]
    mean_loss = np.mean(curves, axis=0)

    # ── Val MedAE curves (5 folds) ──────────────────────────────────────────
    ft = pd.read_csv(IN_DIR / "finetune_all_folds_history.csv")
    medae_curves = []
    for fold, g in ft.groupby("fold"):
        g = g.sort_values("epoch")
        medae_curves.append(np.interp(grid, g["epoch"], g["val/medae"]))
    mean_medae = np.mean(medae_curves, axis=0)

    # ── Test results + CI ────────────────────────────────────────────────────
    test = pd.read_csv(IN_DIR / "fold_test_results.csv").dropna(subset=["test_medae"])
    n = len(test)

    fig = plt.figure(figsize=(13.5, 4.6))
    gs = fig.add_gridspec(1, 3, wspace=0.35)

    # Panel 1: val loss
    ax = fig.add_subplot(gs[0])
    for c in curves:
        ax.plot(grid, c, color=LLAMA_BLUE_LIGHT, linewidth=1.0, alpha=0.85)
    ax.plot(grid, mean_loss, color=LLAMA_BLUE, linewidth=2.2, label="Mean of 5 folds")
    style_ax(ax)
    ax.set_xlabel("Fine-tuning epoch"); ax.set_ylabel("Validation loss (Huber)")
    ax.set_title("Validation loss", fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")

    # Panel 2: val medae
    ax = fig.add_subplot(gs[1])
    for c in medae_curves:
        ax.plot(grid, c, color=LLAMA_BLUE_LIGHT, linewidth=1.0, alpha=0.85)
    ax.plot(grid, mean_medae, color=LLAMA_BLUE, linewidth=2.2, label="Mean of 5 folds")
    ax.axhline(MethylGPT_BASELINE["medae"], color=GPT_ORANGE, linewidth=1.8, linestyle="--")
    ax.text(max_epoch * 0.97, MethylGPT_BASELINE["medae"] + 0.4, "MethylGPT 21k",
            color=GPT_ORANGE, fontsize=8.5, ha="right", fontweight="medium")
    style_ax(ax)
    ax.set_ylim(0, 12)
    ax.set_xlabel("Fine-tuning epoch"); ax.set_ylabel("Validation MedAE (years)")
    ax.set_title("Validation MedAE", fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")

    # Panel 3: test result + CI
    ax = fig.add_subplot(gs[2])
    metrics = [("test_medae", "MedAE", MethylGPT_BASELINE["medae"]),
               ("test_mae", "MAE", MethylGPT_BASELINE["mae"])]
    x = np.arange(len(metrics)); width = 0.32
    gpt_vals = [m[2] for m in metrics]
    means, cis = [], []
    for key, _, _ in metrics:
        vals = test[key].values
        mean = vals.mean()
        ci = stats.sem(vals) * stats.t.ppf(0.975, n - 1) if n > 1 else 0
        means.append(mean); cis.append(ci)
    ax.bar(x - width / 2, gpt_vals, width, color=GPT_ORANGE_LIGHT, edgecolor=GPT_ORANGE, linewidth=1.2, zorder=3)
    ax.bar(x + width / 2, means, width, yerr=cis, color=LLAMA_BLUE_LIGHT, edgecolor=LLAMA_BLUE, linewidth=1.2,
           capsize=4, error_kw={"linewidth": 1.3, "ecolor": INK_SEC}, zorder=3)
    for xi, val in zip(x - width / 2, gpt_vals):
        ax.text(xi, val + 0.15, f"{val:.2f}", ha="center", fontsize=8.5, color=GPT_ORANGE, fontweight="bold")
    for xi, val, ci in zip(x + width / 2, means, cis):
        ax.text(xi, val + ci + 0.15, f"{val:.2f}", ha="center", fontsize=8.5, color=LLAMA_BLUE, fontweight="bold")
    style_ax(ax)
    ax.set_xticks(x); ax.set_xticklabels([m[1] for m in metrics])
    ax.set_ylabel("Years")
    ax.set_ylim(0, max(gpt_vals) * 1.35)
    ax.set_title(f"Test result (n={n} folds, 95% CI)", fontsize=12, fontweight="bold")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=GPT_ORANGE_LIGHT, edgecolor=GPT_ORANGE),
               plt.Rectangle((0, 0), 1, 1, facecolor=LLAMA_BLUE_LIGHT, edgecolor=LLAMA_BLUE)]
    ax.legend(handles, ["MethylGPT 21k", "V7b k-fold"], frameon=False, fontsize=8, loc="upper right")

    fig.suptitle("MethylLlama V7b — 5-fold cross-validation: training curves & final result",
                 fontsize=13, fontweight="bold", y=1.04)

    b64 = fig_to_base64(fig)
    (OUT_DIR / "kfold_curves_and_ci_base64.txt").write_text(b64)
    print(f"Saved base64 ({len(b64)} chars) -> {OUT_DIR / 'kfold_curves_and_ci_base64.txt'}")

    # also save a standalone PNG for direct viewing
    import io
    with open(OUT_DIR / "kfold_curves_and_ci.png", "wb") as f:
        f.write(base64.b64decode(b64))
    print(f"Saved -> {OUT_DIR / 'kfold_curves_and_ci.png'}")

    # print final numbers for the table update
    print("\nFinal numbers for slide 4 table:")
    for key, label, gpt in metrics:
        vals = test[key].values
        mean, ci = vals.mean(), (stats.sem(vals) * stats.t.ppf(0.975, n - 1) if n > 1 else 0)
        print(f"  {label}: {mean:.3f} +/- {ci:.3f}  (GPT {gpt})")
    r2v = test["test_r2"].values
    r2m, r2ci = r2v.mean(), (stats.sem(r2v) * stats.t.ppf(0.975, n - 1) if n > 1 else 0)
    print(f"  R2: {r2m:.4f} +/- {r2ci:.4f}  (GPT {MethylGPT_BASELINE['r2']})")


if __name__ == "__main__":
    main()
