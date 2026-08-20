"""
Figure 1 -- Study design and the WCED pretraining objective.

Panel structure follows the convention used for contrastive/self-supervised
single-cell models (e.g. scConcept Fig. 1): study design, view construction,
the training objective, and the input encoding.

  a  Study design: unlabelled pretraining corpus -> WCED -> encoder, then the
     two downstream evaluation branches (frozen probes; supervised age).
  b  View construction: two 50% CpG subsets of one profile, with the
     complement of each view serving as its reconstruction target.
  c  The WCED objective: shared encoder, CLS bottleneck, decoder scored only at
     withheld CpGs, and cross-view InfoNCE on projected CLS embeddings.
  d  Dual-field token encoding: CpG identity embedding plus a continuous
     beta-value encoder, with a prepended CLS token.

This is a schematic; every number shown is taken from the verified
configuration and dataset headers (see Methods), not from a placeholder.

Usage:  python scripts/paper_figures/fig1_study_design.py
Output: figures/paper/fig1_study_design_wced.{pdf,png}
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common_style import COL_ACCENT, COL_ENET, COL_GPT, COL_LLAMA, apply_style, panel_label, save
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.gridspec import GridSpec

OUTDIR = Path(__file__).resolve().parents[2] / "figures/paper"

GREY = "#9aa3b0"
LIGHT = "#eef1f6"
WITHHELD = "#d8dde6"


def box(ax, x, y, w, h, text, fc=LIGHT, ec=COL_LLAMA, fs=5.4, lw=0.7, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.02",
                                facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            zorder=3, linespacing=1.35,
            fontweight="bold" if bold else "normal")


def arrow(ax, x1, y1, x2, y2, color="0.35", lw=0.7, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=6, color=color, linewidth=lw,
                                 linestyle=ls, zorder=1,
                                 shrinkA=0, shrinkB=0))


def cpg_strip(ax, x, y, w, h, n, mask, on_color, off_color=WITHHELD):
    """Draw n cells across [x, x+w]; mask[i] True -> on_color."""
    cw = w / n
    for i in range(n):
        ax.add_patch(Rectangle((x + i * cw, y), cw * 0.86, h,
                               facecolor=on_color if mask[i] else off_color,
                               edgecolor="white", linewidth=0.3, zorder=2))


def main():
    apply_style()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(7.2, 4.6))
    gs = GridSpec(2, 3, figure=fig, hspace=0.04, wspace=0.16,
                  width_ratios=[1.0, 1.05, 0.95], height_ratios=[1.0, 1.0])

    # ═══ Panel a: study design ══════════════════════════════════════════════
    axA = fig.add_subplot(gs[0, 0]); axA.set_xlim(0, 1); axA.set_ylim(0, 1); axA.axis("off")
    box(axA, 0.06, 0.845, 0.88, 0.135,
        "Public methylation profiles\n169,120 $\\times$ 49,156 CpGs\nno phenotype labels",
        fc="#eaf0fc", ec=COL_LLAMA)
    arrow(axA, 0.5, 0.845, 0.5, 0.775)
    box(axA, 0.14, 0.635, 0.72, 0.14,
        "WCED\nself-supervised\npretraining", fc="#e2ecff", ec=COL_LLAMA, bold=True)
    arrow(axA, 0.5, 0.635, 0.5, 0.565)
    box(axA, 0.06, 0.435, 0.88, 0.13,
        "MethylLlama encoder\n6 layers $\\cdot$ 256-d $\\cdot$ genomic RoPE",
        fc="#eaf0fc", ec=COL_LLAMA)

    # branch
    arrow(axA, 0.3, 0.435, 0.3, 0.375); arrow(axA, 0.7, 0.435, 0.7, 0.375)
    axA.plot([0.3, 0.7], [0.405, 0.405], color="0.35", lw=0.7, zorder=1)
    box(axA, 0.01, 0.20, 0.46, 0.175,
        "Frozen\nrepresentation\nprobes\n(10,988 profiles)", fc="#f2f4f8", ec=GREY, fs=5.0)
    box(axA, 0.53, 0.20, 0.46, 0.175,
        "Supervised\nage fine-tuning\n5 folds\n(2,149 test)", fc="#f2f4f8", ec=GREY, fs=5.0)
    axA.text(0.5, 0.135, "AltuMAge collection $\\cdot$ 21,368 CpGs",
             ha="center", fontsize=4.8, color="0.35")
    panel_label(axA, "a", dx=0.02, dy=0.97)

    # ═══ Panel b: view construction ════════════════════════════════════════
    axB = fig.add_subplot(gs[0, 1:]); axB.set_xlim(0, 1); axB.set_ylim(0, 1); axB.axis("off")
    n = 24
    rng = np.random.default_rng(3)
    m1 = np.zeros(n, bool); m1[rng.choice(n, n // 2, replace=False)] = True
    m2 = np.zeros(n, bool); m2[rng.choice(n, n // 2, replace=False)] = True

    axB.text(0.5, 0.95, "One measured methylation profile", ha="center",
             fontsize=5.6, color="0.2")
    cpg_strip(axB, 0.16, 0.80, 0.68, 0.075, n, np.ones(n, bool), COL_ACCENT)
    axB.text(0.13, 0.838, "$\\beta$", ha="right", va="center", fontsize=5.4)

    arrow(axB, 0.36, 0.79, 0.30, 0.70); arrow(axB, 0.64, 0.79, 0.70, 0.70)
    axB.text(0.5, 0.745, "two independent 50% subsets", ha="center",
             fontsize=5.0, color="0.35")

    for k, (mask, x0, lab) in enumerate([(m1, 0.05, "View 1"), (m2, 0.53, "View 2")]):
        axB.text(x0 + 0.21, 0.665, lab, ha="center", fontsize=5.6,
                 color=COL_LLAMA, fontweight="bold")
        cpg_strip(axB, x0, 0.555, 0.42, 0.075, n, mask, COL_LLAMA)
        axB.text(x0 + 0.21, 0.495, "encoder input", ha="center", fontsize=4.9,
                 color=COL_LLAMA)
        cpg_strip(axB, x0, 0.345, 0.42, 0.075, n, ~mask, COL_GPT)
        axB.text(x0 + 0.21, 0.285, "reconstruction target\n(withheld from input)",
                 ha="center", fontsize=4.9, color=COL_GPT, linespacing=1.3)

    axB.add_patch(Rectangle((0.03, 0.24), 0.94, 0.47, facecolor="none",
                            edgecolor="0.85", linewidth=0.6, zorder=0))
    axB.text(0.5, 0.16,
             "Loss is evaluated only at measured CpGs absent from that view,\n"
             "so the objective cannot be solved by copying visible values.",
             ha="center", fontsize=4.9, color="0.3", linespacing=1.4)
    panel_label(axB, "b", dx=0.01, dy=0.97)

    # ═══ Panel c: the WCED objective ═══════════════════════════════════════
    axC = fig.add_subplot(gs[1, :2]); axC.set_xlim(0, 1); axC.set_ylim(0, 1); axC.axis("off")
    yv1, yv2 = 0.79, 0.31
    for y, lab in [(yv1, "View 1"), (yv2, "View 2")]:
        box(axC, 0.01, y - 0.055, 0.13, 0.11, lab, fc="#eaf0fc", ec=COL_LLAMA, fs=5.2)
        arrow(axC, 0.14, y, 0.20, y)
        box(axC, 0.20, y - 0.075, 0.17, 0.15, "Encoder", fc="#e2ecff",
            ec=COL_LLAMA, fs=5.4, bold=True)
        arrow(axC, 0.37, y, 0.43, y)
        box(axC, 0.43, y - 0.055, 0.10, 0.11, "CLS", fc=COL_LLAMA, ec=COL_LLAMA, fs=5.2)
        axC.texts[-1].set_color("white")
        arrow(axC, 0.53, y, 0.60, y)
        box(axC, 0.60, y - 0.065, 0.15, 0.13, "Decoder", fc="#fdf0e6", ec=COL_GPT, fs=5.2)
        arrow(axC, 0.75, y, 0.81, y)
        box(axC, 0.81, y - 0.075, 0.18, 0.15,
            "predicted\nwithheld\nCpGs", fc="#fdf0e6", ec=COL_GPT, fs=4.9)

    # shared weights
    axC.annotate("", xy=(0.285, yv1 - 0.075), xytext=(0.285, yv2 + 0.075),
                 arrowprops=dict(arrowstyle="<->", color=GREY, lw=0.7,
                                 linestyle=(0, (2, 2))))
    axC.text(0.265, (yv1 + yv2) / 2, "shared\nweights", fontsize=4.8, color=GREY,
             va="center", ha="right", linespacing=1.3)

    # contrastive link between the two CLS tokens
    axC.annotate("", xy=(0.48, yv1 - 0.055), xytext=(0.48, yv2 + 0.055),
                 arrowprops=dict(arrowstyle="<->", color=COL_ACCENT, lw=1.0))
    box(axC, 0.365, (yv1 + yv2) / 2 - 0.075, 0.23, 0.15,
        "projection head\n$256\\!\\to\\!128\\!\\to\\!128$\nInfoNCE, $\\tau=0.1$",
        fc="#eafaf0", ec=COL_ACCENT, fs=4.9)

    axC.text(0.90, (yv1 + yv2) / 2, "$\\mathcal{L}_{\\mathrm{recon}}$\nat withheld\nCpGs only",
             ha="center", va="center", fontsize=4.9, color=COL_GPT, linespacing=1.35)

    axC.text(0.5, 0.045,
             "$\\mathcal{L} = \\mathcal{L}_{\\mathrm{recon}} + "
             "0.05\\,\\mathcal{L}_{\\mathrm{contrastive}}$",
             ha="center", fontsize=6.2)
    panel_label(axC, "c", dx=0.01, dy=0.96)

    # ═══ Panel d: token encoding ═══════════════════════════════════════════
    axD = fig.add_subplot(gs[1, 2]); axD.set_xlim(0, 1); axD.set_ylim(0, 1); axD.axis("off")
    box(axD, 0.03, 0.83, 0.42, 0.13, "CpG identity\n$c_i$", fc="#eaf0fc",
        ec=COL_LLAMA, fs=5.0)
    box(axD, 0.55, 0.83, 0.42, 0.13, "$\\beta$ value\n$b_i$", fc="#eafaf0",
        ec=COL_ACCENT, fs=5.0)
    arrow(axD, 0.24, 0.83, 0.24, 0.72); arrow(axD, 0.76, 0.83, 0.76, 0.72)
    box(axD, 0.02, 0.56, 0.44, 0.15, "learned\nembedding", fc=LIGHT, ec=GREY, fs=4.8)
    box(axD, 0.54, 0.56, 0.44, 0.15, "sinusoidal\nbasis $\\to$ linear",
        fc=LIGHT, ec=GREY, fs=4.8)
    arrow(axD, 0.24, 0.56, 0.43, 0.45); arrow(axD, 0.76, 0.56, 0.57, 0.45)
    axD.add_patch(plt.Circle((0.5, 0.40), 0.048, facecolor="white",
                             edgecolor="0.35", linewidth=0.7, zorder=3))
    axD.text(0.5, 0.40, "+", ha="center", va="center", fontsize=7, zorder=4)
    arrow(axD, 0.5, 0.352, 0.5, 0.28)
    box(axD, 0.20, 0.15, 0.60, 0.13, "CpG token $x_i$", fc="#e2ecff",
        ec=COL_LLAMA, fs=5.2, bold=True)
    axD.text(0.5, 0.055, "a CLS token is prepended\nto every partial profile",
             ha="center", fontsize=4.8, color="0.35", linespacing=1.35)
    panel_label(axD, "d", dx=0.02, dy=0.96)

    save(fig, str(OUTDIR / "fig1_study_design_wced"))


if __name__ == "__main__":
    main()
