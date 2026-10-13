"""
MOCKUP ONLY -- not referenced by the manuscript.

Question being tested: would a training-curve figure (one line per model across
epochs) be better than the current Figure 3?

What the underlying data actually is, which constrains what the panel may claim:
  * These are VALIDATION curves. Figure 3 reports TEST metrics on the fixed
    2,149-sample cohort. Validation MedAE bottoms out near 2.6 years, test MedAE
    is 3.11 -- different quantity, different samples.
  * MethylLlama's five folds each use their own 1,263-profile validation split,
    so the five lines are not even on a common set as against each other.
  * MethylGPT's validation split is not recorded in its run config, so whether
    its curve is on the same samples as any MethylLlama fold is UNVERIFIED.
  * MethylGPT ran 300 epochs; the MethylLlama folds crashed at 155-165. Reading
    "who is lower at epoch N" across that gap compares a finished run against
    truncated ones.
  * ElasticNetCV has no epochs at all -- coordinate descent with internal CV.
    It cannot appear as a line, and its test number must not be drawn on a
    validation axis.

Sources (pulled from WandB, cached under outputs/wandb_curves/):
  llama_kfold_val.csv  runs a154lzjy aas23t9m vw5f9nh4 vqogel6f 6qt4ma56
  methylgpt_val.csv    run xzrw1qwr
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common_style import COL_ENET, COL_GPT, COL_LLAMA, apply_style, panel_label, save
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

REPO = Path(__file__).resolve().parents[2]
CUR = REPO / "outputs/wandb_curves"
OUT = REPO / "figures/mockups"


def main():
    apply_style()
    OUT.mkdir(parents=True, exist_ok=True)
    lla = pd.read_csv(CUR / "llama_kfold_val.csv")
    gpt = pd.read_csv(CUR / "methylgpt_val.csv")

    fig = plt.figure(figsize=(7.2, 3.0))
    gs = GridSpec(1, 2, figure=fig, wspace=0.32, width_ratios=[1.55, 1.0])

    # ── a: the curves ───────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    for f, g in lla.groupby("fold"):
        g = g.sort_values("epoch")
        ax.plot(g["epoch"], g["val/medae"], color=COL_LLAMA, lw=0.7, alpha=0.45,
                zorder=2, label="MethylLlama (5 partitions)" if f == 0 else None)
        b = g.loc[g["val/medae"].idxmin()]
        ax.scatter(b["epoch"], b["val/medae"], s=11, color=COL_LLAMA, zorder=4,
                   edgecolors="white", linewidths=0.4)
    gpt = gpt.sort_values("epoch")
    ax.plot(gpt["epoch"], gpt["valid_medae"], color=COL_GPT, lw=1.1, zorder=3,
            label="MethylGPT (1 run)")
    bg = gpt.loc[gpt["valid_medae"].idxmin()]
    ax.scatter(bg["epoch"], bg["valid_medae"], s=13, color=COL_GPT, zorder=4,
               edgecolors="white", linewidths=0.4)

    # where the MethylLlama runs stopped -- they did not finish 300 epochs
    stop = lla.groupby("fold")["epoch"].max().max()
    ax.axvline(stop, color="0.55", ls=":", lw=0.9, zorder=1)
    ax.text(stop - 6, 8.6, "MethylLlama runs\nend here (crashed)", ha="right",
            va="top", fontsize=6.0, color="0.4", linespacing=1.25)

    ax.set_xlabel("Fine-tuning epoch")
    ax.set_ylabel("Validation MedAE (years)")
    ax.set_ylim(2.0, 9.0)
    ax.set_xlim(-4, 305)
    ax.legend(frameon=False, fontsize=6.6, loc="upper right", handlelength=1.4)
    ax.text(0.5, -0.30, "filled points: best-checkpoint epoch per run",
            transform=ax.transAxes, ha="center", va="top", fontsize=5.9,
            color="0.45", style="italic")
    panel_label(ax, "a", dx=-0.13)

    # ── b: what Figure 3 actually reports, on the same page for contrast ─────
    axb = fig.add_subplot(gs[0, 1])
    best_l = lla.groupby("fold")["val/medae"].min()
    rows = [("MethylLlama", float(best_l.mean()), 3.112, COL_LLAMA),
            ("MethylGPT",   float(gpt["valid_medae"].min()), 3.701, COL_GPT),
            ("ElasticNetCV", np.nan, 3.437, COL_ENET)]
    y = np.arange(len(rows))[::-1]
    for (name, v, t, c), yy in zip(rows, y):
        if not np.isnan(v):
            axb.scatter(v, yy + 0.16, s=26, facecolors="white", edgecolors=c,
                        linewidths=1.2, zorder=3)
        else:
            axb.text(2.45, yy + 0.16, "no epochs", fontsize=6.0, color="0.5",
                     va="center", style="italic")
        axb.scatter(t, yy - 0.16, s=30, color=c, zorder=3)
        axb.plot([v, t] if not np.isnan(v) else [t, t], [yy + 0.16, yy - 0.16],
                 color="0.75", lw=0.7, zorder=2)
    axb.set_yticks(y); axb.set_yticklabels([r[0] for r in rows], fontsize=6.8)
    axb.set_xlabel("MedAE (years)")
    axb.set_xlim(2.3, 4.1)
    axb.set_ylim(-0.7, len(rows) - 0.3)
    axb.spines["left"].set_visible(False); axb.tick_params(axis="y", length=0)
    axb.scatter([], [], s=26, facecolors="white", edgecolors="0.3", linewidths=1.2,
                label="best validation (panel a)")
    axb.scatter([], [], s=30, color="0.3", label="test, 2,149 samples (Fig. 3)")
    axb.legend(frameon=False, fontsize=6.2, loc="lower right",
               bbox_to_anchor=(1.02, -0.04), handletextpad=0.2)
    axb.text(0.5, -0.30, "the two columns are different cohorts,\nnot a before/after",
             transform=axb.transAxes, ha="center", va="top", fontsize=5.9,
             color="0.45", style="italic", linespacing=1.3)
    panel_label(axb, "b", dx=-0.30)

    save(fig, str(OUT / "mockup_training_curves"))
    print(f"\nMethylLlama best val MedAE per fold: "
          f"{', '.join(f'{v:.3f}' for v in best_l)}  (mean {best_l.mean():.3f})")
    print(f"MethylGPT   best val MedAE: {gpt['valid_medae'].min():.3f} @ epoch "
          f"{bg['epoch']:.0f} of 300")
    print(f"MethylLlama runs stopped at epoch {stop:.0f}")
    print("Test MedAE (Fig. 3): MethylLlama 3.112 | ElasticNetCV 3.437 | MethylGPT 3.701")


if __name__ == "__main__":
    main()
