"""
Figure 1 -- MethylLlama model and WCED training overview.

Layout follows the convention established for contrastive/self-supervised
representation models (cf. scConcept Fig. 1): panel (a) is a single
bottom-to-top pipeline drawn with concrete objects rather than abstract
boxes, and panels (b)-(d) are zoom-ins connected to it by grey wedges.

  a  Training pipeline. Profiles -> two 50% CpG subsets -> shared encoder ->
     CLS representations, which feed the two objectives side by side: a
     cross-view similarity matrix optimised by InfoNCE (left) and
     reconstruction of the CpGs withheld from each view (right). Showing both
     objectives at the same level is deliberate: unlike contrastive-only
     frameworks, WCED applies both pressures to the same embedding.
  b  The contrastive objective in embedding space, with the InfoNCE loss.
  c  Internals of one transformer block, repeated six times.
  d  Token construction: CpG identity + beta value, ordered by genomic rank.

The similarity matrix in (a) is real data (two_view_simmatrix.npy), not a
cartoon. Every number shown is traced in fig1_provenance.json.

Usage:  python scripts/paper_figures/fig1_study_design.py
Output: figures/paper/fig1_study_design_wced.{pdf,png}
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common_style import COL_ACCENT, COL_GPT, COL_LLAMA, apply_style, panel_label, save
import matplotlib.pyplot as plt
from matplotlib.patches import (Circle, FancyArrowPatch, FancyBboxPatch,
                                Polygon, Rectangle)
from matplotlib.gridspec import GridSpec

REPO = Path(__file__).resolve().parents[2]
SIM = REPO / "figures/v7b_pretrain_cls/two_view_simmatrix.npy"
OUTDIR = REPO / "figures/paper"

V1 = "#c2557a"      # view 1 (rose)
V2 = "#3a6acc"      # view 2 (blue)
ENC = "#dfe3ea"     # encoder slab
GREY = "#8d95a3"
WEDGE = "#f4f6f9"
DATA = "#f2d9a8"    # data matrix cells


def rbox(ax, x, y, w, h, text="", fc="#eef1f6", ec=COL_LLAMA, fs=7.0,
         lw=0.9, bold=False, tc="black", pad=0.004):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad={pad},rounding_size=0.012",
                                facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, zorder=4, linespacing=1.3, color=tc,
                fontweight="bold" if bold else "normal")


def arrow(ax, x1, y1, x2, y2, color="0.3", lw=0.9, ls="-", style="-|>", ms=7):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=ms, color=color, linewidth=lw,
                                 linestyle=ls, zorder=2, shrinkA=0, shrinkB=0))


def strip(ax, x, y, w, h, n, color, on=None, edge="white"):
    """A row of n cells; `on` is a boolean mask (None = all on)."""
    cw = w / n
    for i in range(n):
        lit = True if on is None else on[i]
        ax.add_patch(Rectangle((x + i * cw, y), cw * 0.82, h,
                               facecolor=color if lit else "#dfe3ea",
                               edgecolor=edge, linewidth=0.25, zorder=3))


def wedge(fig, ax_src, rect, ax_dst):
    """Grey zoom wedge: right edge of `rect` in ax_src -> left edge of ax_dst."""
    x, y, w, h = rect
    inv = fig.transFigure.inverted()
    tr = lambda p: inv.transform(ax_src.transData.transform(p))
    top_s, bot_s = tr((x + w, y + h)), tr((x + w, y))
    d = ax_dst.get_position()
    fig.add_artist(Polygon([top_s, (d.x0, d.y1), (d.x0, d.y0), bot_s],
                           closed=True, facecolor=WEDGE, edgecolor="none",
                           zorder=0))


def dashed_frame(ax):
    ax.add_patch(Rectangle((0.005, 0.005), 0.99, 0.99, transform=ax.transAxes,
                           facecolor="white", edgecolor="0.75", linewidth=0.7,
                           linestyle=(0, (3, 2)), zorder=0))


def main():
    apply_style()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # Height is capped so that, scaled to \linewidth, the figure plus its
    # caption fits inside the 9 in text block. A taller figure sends LaTeX's
    # [H] float placement into an infinite page loop.
    fig = plt.figure(figsize=(7.2, 6.5))
    gs = GridSpec(3, 2, figure=fig, width_ratios=[1.3, 1.0],
                  height_ratios=[1.0, 0.92, 0.86],
                  wspace=0.30, hspace=0.30,
                  left=0.035, right=0.985, top=0.975, bottom=0.02)

    axA = fig.add_subplot(gs[:, 0]); axA.set_xlim(0, 1); axA.set_ylim(0, 1); axA.axis("off")
    axB = fig.add_subplot(gs[0, 1]); axB.set_xlim(0, 1); axB.set_ylim(0, 1); axB.axis("off")
    axC = fig.add_subplot(gs[1, 1]); axC.set_xlim(0, 1); axC.set_ylim(0, 1); axC.axis("off")
    axD = fig.add_subplot(gs[2, 1]); axD.set_xlim(0, 1); axD.set_ylim(0, 1); axD.axis("off")

    # ══════════ Panel a : bottom-to-top pipeline ══════════
    for by, bh in [(0.015, 0.181), (0.198, 0.108), (0.318, 0.214), (0.545, 0.355)]:
        axA.add_patch(FancyBboxPatch((0.005, by), 0.99, bh,
                                     boxstyle="round,pad=0.002,rounding_size=0.010",
                                     facecolor="#fafbfc", edgecolor="#e6eaf0",
                                     linewidth=0.6, zorder=0))

    # (1) corpus, drawn as an actual profiles x CpGs matrix
    mx, mw = 0.29, 0.42
    for r in range(5):
        strip(axA, mx, 0.038 + r * 0.0165, mw, 0.014, 14, DATA)
    axA.text(0.5, 0.101, "Pretraining corpus", ha="center", fontsize=7.2)
    axA.text(0.5, 0.086, "169,120 profiles $\\times$ 49,156 CpGs",
             ha="center", fontsize=6.8, color="0.35")
    axA.text(mx - 0.012, 0.078, "profiles", ha="right", va="center",
             fontsize=6.4, color="0.4", rotation=90)
    axA.text(0.5, 0.020, "CpG sites", ha="center", fontsize=6.4, color="0.4")

    # (2) split into two 50% subsets
    arrow(axA, 0.44, 0.122, 0.28, 0.152)
    arrow(axA, 0.56, 0.122, 0.72, 0.152)
    axA.text(0.5, 0.140, "two independent\n50% CpG subsets", ha="center",
             va="center", fontsize=6.8, color="0.35", linespacing=1.3, zorder=5,
             bbox=dict(facecolor="white", edgecolor="none", pad=1.2))
    rng = np.random.default_rng(3)
    m1 = rng.random(28) < 0.5
    m2 = rng.random(28) < 0.5
    strip(axA, 0.07, 0.158, 0.34, 0.017, 28, V1, on=m1)
    strip(axA, 0.59, 0.158, 0.34, 0.017, 28, V2, on=m2)
    axA.text(0.24, 0.183, "subset 1 $\\cdot$ 24,578 CpGs", ha="center", fontsize=6.8, color=V1,
             fontweight="bold")
    axA.text(0.76, 0.183, "subset 2 $\\cdot$ 24,578 CpGs", ha="center", fontsize=6.8, color=V2,
             fontweight="bold")

    # (3) the two views
    for x0, colr, mask, lab in [(0.05, V1, m1, "View 1"), (0.57, V2, m2, "View 2")]:
        axA.add_patch(Rectangle((x0, 0.205), 0.38, 0.082, facecolor="#fbfcfe",
                                edgecolor="0.8", linewidth=0.7, zorder=2))
        for r in range(3):
            strip(axA, x0 + 0.025, 0.216 + r * 0.022, 0.33, 0.014, 28, colr,
                  on=mask)
        axA.text(x0 + 0.19, 0.294, lab, ha="center", fontsize=7.0, color=colr,
                 fontweight="bold")

    # (4) shared encoder
    arrow(axA, 0.24, 0.287, 0.24, 0.325)
    arrow(axA, 0.76, 0.287, 0.76, 0.325)
    rbox(axA, 0.05, 0.328, 0.90, 0.072,
         "Shared MethylLlama encoder", fc=ENC, ec=GREY, fs=7.6, bold=True)
    axA.text(0.5, 0.340, "6 transformer blocks $\\cdot$ 256-d $\\cdot$ genomic-rank RoPE",
             ha="center", fontsize=6.6, color="0.35")

    # (5) CLS representations
    arrow(axA, 0.24, 0.400, 0.24, 0.432)
    arrow(axA, 0.76, 0.400, 0.76, 0.432)
    for x0, colr in [(0.13, V1), (0.65, V2)]:
        # index runs top-to-bottom (CLS_1 first), matching the usual convention
        rbox(axA, x0, 0.492, 0.22, 0.024, r"$\mathrm{CLS}_1$", fc=colr, ec=colr,
             fs=6.8, tc="white", pad=0.0015)
        axA.text(x0 + 0.11, 0.474, r"$\vdots$", ha="center", va="center",
                 fontsize=7.4, color=colr)
        rbox(axA, x0, 0.437, 0.22, 0.024, r"$\mathrm{CLS}_B$", fc=colr, ec=colr,
             fs=6.8, tc="white", pad=0.0015)
    axA.text(0.5, 0.478, "CLS\nrepresentations", ha="center", va="center",
             fontsize=6.6, color="0.35", linespacing=1.25)

    # (6) the two objectives, side by side, both fed from CLS
    # both CLS columns feed a junction, which then feeds both objectives
    arrow(axA, 0.24, 0.520, 0.47, 0.545, style="-", color="0.45")
    arrow(axA, 0.76, 0.520, 0.53, 0.545, style="-", color="0.45")
    axA.add_patch(Circle((0.50, 0.548), 0.010, facecolor="0.45",
                         edgecolor="none", zorder=4))
    arrow(axA, 0.49, 0.556, 0.20, 0.588, color="0.45")
    arrow(axA, 0.51, 0.556, 0.628, 0.714, color="0.45")

    # --- left objective: cross-view similarity matrix (real data) ---
    simx, simy, simw, simh = 0.045, 0.600, 0.30, 0.163  # square at this panel aspect
    sim = np.load(SIM)
    k = 10
    sel = np.sort(np.random.default_rng(0).choice(sim.shape[0], k, replace=False))
    axA.imshow(sim[np.ix_(sel, sel)], cmap="magma", vmin=0, vmax=1,
               extent=(simx, simx + simw, simy, simy + simh), aspect="auto",
               zorder=3, interpolation="nearest")
    # Outline the diagonal: these are the matched (same-profile) pairs, i.e. the
    # quantity InfoNCE maximises. Previously the reader had to infer which cells
    # carried the claim.
    cw_, ch_ = simw / k, simh / k
    for i in range(k):
        axA.add_patch(Rectangle((simx + i * cw_, simy + simh - (i + 1) * ch_),
                                cw_, ch_, facecolor="none", edgecolor="white",
                                linewidth=1.0, zorder=5))
    axA.add_patch(Rectangle((simx, simy), simw, simh, facecolor="none",
                            edgecolor="0.5", linewidth=0.7, zorder=4))
    # colour brackets identifying the two views on the matrix axes
    axA.plot([simx - 0.012] * 2, [simy, simy + simh], color=V1, lw=2.2,
             solid_capstyle="butt", zorder=4)
    axA.plot([simx, simx + simw], [simy - 0.012] * 2, color=V2, lw=2.2,
             solid_capstyle="butt", zorder=4)
    axA.text(simx - 0.022, simy + simh / 2, "View 1", rotation=90, ha="center",
             va="center", fontsize=6.6, color=V1)
    axA.text(simx + simw / 2, simy - 0.030, "View 2", ha="center",
             fontsize=6.6, color=V2)
    axA.text(simx + simw / 2, simy + simh + 0.014,
             "cross-view similarity matrix\noutlined diagonal = matched views",
             ha="center", va="bottom",
             fontsize=6.8, color="0.25", linespacing=1.25)
    axA.text(simx + simw / 2, simy + simh + 0.060, "InfoNCE", ha="center",
             fontsize=7.4, color=COL_ACCENT, fontweight="bold")

    # --- right objective: decoder reconstructs the withheld CpGs ---
    rbox(axA, 0.62, 0.706, 0.34, 0.038, "Decoder", fc="#fdf0e6", ec=COL_GPT,
         fs=7.2, bold=True)
    arrow(axA, 0.79, 0.706, 0.79, 0.688, color=COL_GPT)
    # predicted vs measured at the SAME withheld positions, so that "scored only
    # where the encoder was blind" is visible rather than left to the caption
    strip(axA, 0.645, 0.668, 0.315, 0.015, 28, COL_GPT, on=~m1)
    axA.text(0.635, 0.6755, "predicted", ha="right", va="center",
             fontsize=6.8, color=COL_GPT)
    strip(axA, 0.645, 0.612, 0.315, 0.015, 28, "#7a8290", on=~m1)
    axA.text(0.635, 0.6195, "measured", ha="right", va="center",
             fontsize=6.8, color="0.35")
    axA.annotate("", xy=(0.80, 0.630), xytext=(0.80, 0.667),
                 arrowprops=dict(arrowstyle="<->", color="0.3", lw=0.9,
                                 mutation_scale=6))
    axA.text(0.818, 0.6485, "compare", ha="left", va="center", fontsize=6.8,
             color="0.3")
    axA.text(0.79, 0.757, "reconstruction of CpGs\nwithheld from each view",
             ha="center", va="bottom", fontsize=6.8, color="0.25",
             linespacing=1.25)

    axA.text(0.5, 0.862,
             "$\\mathcal{L} = \\mathcal{L}_{\\mathrm{recon}} + 0.05\\,"
             "\\mathcal{L}_{\\mathrm{InfoNCE}}$", ha="center", fontsize=8.4)
    panel_label(axA, "a", dx=0.03, dy=0.975)

    # ══════════ Panel b : contrastive objective in embedding space ══════════
    dashed_frame(axB)
    axB.text(0.5, 0.94, "Contrastive objective", ha="center", fontsize=7.4,
             fontweight="bold")
    cx, cy = 0.42, 0.55
    axB.add_patch(Circle((cx, cy), 0.045, facecolor=V1, edgecolor="white",
                         linewidth=0.8, zorder=4))
    axB.text(cx, cy - 0.10, r"$z_i^{(1)}$", ha="center", fontsize=6.8, color=V1)
    pos = (0.80, 0.72)
    axB.add_patch(Circle(pos, 0.045, facecolor=V2, edgecolor="white",
                         linewidth=0.8, zorder=4))
    axB.text(pos[0], pos[1] + 0.09, r"$z_i^{(2)}$ same profile", ha="center",
             fontsize=6.8, color=V2)
    arrow(axB, cx + 0.05, cy + 0.02, pos[0] - 0.05, pos[1] - 0.02,
          color=COL_ACCENT, lw=1.4, style="<|-|>", ms=8)
    axB.text(0.60, 0.70, "pull\ntogether", fontsize=6.6, color=COL_ACCENT,
             ha="center", va="center", linespacing=1.25)
    for nx, ny in [(0.14, 0.76), (0.16, 0.34), (0.72, 0.26)]:
        axB.add_patch(Circle((nx, ny), 0.038, facecolor=V2, alpha=0.45,
                             edgecolor="white", linewidth=0.8, zorder=4))
        dx, dy = nx - cx, ny - cy
        L = (dx ** 2 + dy ** 2) ** 0.5
        arrow(axB, cx + 0.05 * dx / L, cy + 0.05 * dy / L,
              nx - 0.05 * dx / L, ny - 0.05 * dy / L,
              color="#b03030", lw=1.0, style="-|>", ms=7)
    axB.text(0.14, 0.20, "push apart\n(other profiles)", fontsize=6.6,
             color="#b03030", ha="left", va="center", linespacing=1.25)
    panel_label(axB, "b", dx=0.03, dy=0.94)

    # ══════════ Panel c : one transformer block ══════════
    dashed_frame(axC)
    axC.text(0.5, 0.965, "One transformer block", ha="center", fontsize=7.4,
             fontweight="bold")
    # drawn bottom-to-top, so the list is in data-flow order
    blocks = [("RMSNorm", "#fde9c8"),
              ("Multi-head self-attention\n+ genomic-rank RoPE", "#f8d3bc"),
              ("Add  $\\oplus$  residual", "#fde9c8"),
              ("RMSNorm", "#fde9c8"),
              ("SwiGLU feed-forward", "#e2d7f2"),
              ("Add  $\\oplus$  residual", "#fde9c8")]
    y0, hb = 0.145, 0.105
    for i, (lab, fc) in enumerate(blocks):
        rbox(axC, 0.14, y0 + i * (hb + 0.012), 0.62, hb, lab, fc=fc, ec="0.6",
             fs=6.8, lw=0.7)
    axC.annotate("", xy=(0.82, y0), xytext=(0.82, y0 + 6 * (hb + 0.012)),
                 arrowprops=dict(arrowstyle="<->", color="0.45", lw=0.9))
    axC.text(0.855, 0.50, "$\\times\\,6$", fontsize=7.6, va="center", color="0.3")
    arrow(axC, 0.45, 0.845, 0.45, 0.892, color="0.4")
    axC.text(0.45, 0.900, "CLS output = sample representation", ha="center",
             va="bottom", fontsize=6.6, color="0.3")
    axC.text(0.45, 0.075, "token sequence in", ha="center", va="center",
             fontsize=6.6, color="0.4")
    arrow(axC, 0.45, 0.095, 0.45, 0.125, color="0.4")
    panel_label(axC, "c", dx=0.03, dy=0.965)

    # ══════════ Panel d : token construction ══════════
    dashed_frame(axD)
    axD.text(0.5, 0.93, "Token construction", ha="center", fontsize=7.4,
             fontweight="bold")
    toks = ["CLS", "cg…", "cg…", "cg…", "cg…", "cg…"]
    tw = 0.115
    for i, t in enumerate(toks):
        fc = COL_LLAMA if i == 0 else V1
        rbox(axD, 0.10 + i * (tw + 0.012), 0.60, tw, 0.14, t, fc=fc, ec=fc,
             fs=6.6, tc="white")
    axD.annotate("", xy=(0.86, 0.55), xytext=(0.22, 0.55),
                 arrowprops=dict(arrowstyle="-|>", color="0.45", lw=0.9))
    axD.text(0.54, 0.50, "CpGs ordered by genomic rank", ha="center", va="top",
             fontsize=6.6, color="0.35")
    rbox(axD, 0.05, 0.20, 0.36, 0.14, "CpG identity\nembedding", fc="#eaf0fc",
         ec=COL_LLAMA, fs=6.8)
    rbox(axD, 0.58, 0.20, 0.36, 0.14, "$\\beta$-value\nencoder", fc="#eafaf0",
         ec=COL_ACCENT, fs=6.8)
    axD.add_patch(Circle((0.495, 0.395), 0.037, facecolor="white",
                         edgecolor="0.4", linewidth=0.8, zorder=4))
    axD.text(0.495, 0.395, "+", ha="center", va="center", fontsize=8.5, zorder=5)
    arrow(axD, 0.23, 0.34, 0.46, 0.375, color="0.45")
    arrow(axD, 0.76, 0.34, 0.53, 0.375, color="0.45")
    arrow(axD, 0.495, 0.432, 0.495, 0.585, color="0.45")
    axD.text(0.5, 0.10, "each token carries both which CpG and how methylated",
             ha="center", fontsize=6.6, color="0.35", style="italic")
    panel_label(axD, "d", dx=0.03, dy=0.93)

    # ══════════ zoom wedges from (a) into (b), (c), (d) ══════════
    wedge(fig, axA, (0.05, 0.328, 0.90, 0.072), axC)        # encoder    -> c
    wedge(fig, axA, (0.57, 0.205, 0.38, 0.082), axD)        # a view     -> d

    save(fig, str(OUTDIR / "fig1_study_design_wced"))

    prov = {
        "figure_type": "schematic; the similarity matrix in panel a is real data",
        "similarity_matrix_source": str(SIM.relative_to(REPO)),
        "displayed_values": {
            "169,120 profiles x 49,156 CpGs": "canonical pretraining h5ad header "
                "(SLURM job 45888987)",
            "6 transformer blocks, 256-d": "scripts/llama/pretrain_llama_small_6L_contrastive.sh",
            "genomic-rank RoPE": "same script, wced_genomic_rank_path set",
            "lambda = 0.05": "same script (CONTRASTIVE_WEIGHT)",
            "50% subsets": "wced_input_ratio=0.5",
        },
        "accessibility": "view 1 rose / view 2 blue are separable under deuteranopia; "
                          "view membership is additionally carried by position and by "
                          "text labels, never by colour alone",
    }
    (OUTDIR / "fig1_provenance.json").write_text(json.dumps(prov, indent=2))


if __name__ == "__main__":
    main()
