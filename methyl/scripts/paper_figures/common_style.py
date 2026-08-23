"""Shared matplotlib style for all main-text paper figures.

Academic conventions: small sans-serif type, no panel titles that duplicate
the caption, colorblind-safe palette, TrueType fonts embedded in the PDF
(fonttype 42, required by most journals), no internal model codenames.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COL_LLAMA = "#1a4ab0"   # MethylLlama  (blue)
COL_GPT = "#c07030"     # MethylGPT    (orange-brown)
COL_ENET = "#5a5a5a"    # ElasticNet   (grey)
COL_ACCENT = "#2a9a4a"  # highlights   (green)
COL_NEG = "#b03030"     # negative R^2 (red)


def apply_style():
    plt.rcParams.update({
        "font.size": 7.6,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "axes.titlesize": 8.6,
        "axes.labelsize": 7.6,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "legend.fontsize": 7.2,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
        "figure.dpi": 120,
    })


def panel_label(ax, letter, dx=-0.12, dy=1.05):
    ax.text(dx, dy, letter, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="bottom", ha="right")


def save(fig, outstem):
    fig.savefig(f"{outstem}.pdf", bbox_inches="tight")
    fig.savefig(f"{outstem}.png", bbox_inches="tight", dpi=300)
    print(f"Saved -> {outstem}.pdf / .png")


# Minimum on-figure font size. All figures are drawn at 7.2 in wide and print
# at Nature Communications' 180 mm two-column width (~98% scale), so a font
# specified here is essentially the printed size. Nothing may go below this.
MIN_FONT_PT = 6.8
