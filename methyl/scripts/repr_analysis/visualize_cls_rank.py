"""
visualize_cls_rank.py
=====================
Publication-quality visualization of the MethylLlama CLS embedding space.

2-panel figure:
  A — Scree plot          : singular value spectrum, effective rank marked
  B — Cumulative variance : % variance explained, 90/95/99% thresholds

PCA is fit on ALL 169k embeddings for accurate statistics.

Usage:
  python scripts/repr_analysis/visualize_cls_rank.py \
      --embeddings  outputs/repr_analysis/pretrain_cls_169k_44892802/embeddings_cls.npy \
      --outdir      outputs/repr_analysis/cls_rank_figures
"""

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from sklearn.decomposition import PCA

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 150,
})


def effective_rank(s: np.ndarray) -> float:
    s = s[s > 1e-10]
    p = s / s.sum()
    return float(np.exp(-np.sum(p * np.log(p + 1e-12))))


def plot_scree(ax, singular_values, eff_rank, n_show=80):
    sv = singular_values[:n_show]
    var_pct = (sv ** 2) / (singular_values ** 2).sum() * 100

    colors = ["#3a6acc" if i + 1 <= eff_rank else "#c8d8f0" for i in range(n_show)]
    ax.bar(range(1, n_show + 1), var_pct, color=colors, width=0.85)

    ax.axvline(eff_rank, color="#c03040", lw=2.5, ls="--", zorder=5)
    ax.annotate(
        f"Effective rank = {eff_rank:.0f}\n({eff_rank / len(singular_values) * 100:.0f}% of dims active)",
        xy=(eff_rank, var_pct[int(eff_rank) - 1]),
        xytext=(eff_rank + 8, var_pct[int(eff_rank) - 1] + 0.5),
        fontsize=10, color="#c03040", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#c03040", lw=1.5),
    )

    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance explained (%)")
    ax.set_title("A  —  Singular Value Spectrum", fontweight="bold", loc="left")
    ax.set_xlim(0, n_show + 1)

    blue_patch = mpatches.Patch(color="#3a6acc", label=f"Active dims (≤ eff. rank {eff_rank:.0f})")
    gray_patch  = mpatches.Patch(color="#c8d8f0", label="Remaining dims")
    ax.legend(handles=[blue_patch, gray_patch], loc="upper right")


def plot_cumvar(ax, singular_values, eff_rank):
    var = singular_values ** 2
    cumvar = np.cumsum(var) / var.sum() * 100
    n_show = min(120, len(cumvar))
    xs = range(1, n_show + 1)

    ax.plot(xs, cumvar[:n_show], color="#3a6acc", lw=2.5)
    ax.fill_between(xs, cumvar[:n_show], alpha=0.12, color="#3a6acc")

    thresholds = [(90, "#c03040"), (95, "#c07030"), (99, "#2a8040")]
    for pct, col in thresholds:
        idx = int(np.searchsorted(cumvar, pct))
        if idx < n_show:
            ax.axhline(pct, color=col, lw=1.4, ls="--", alpha=0.85)
            ax.axvline(idx + 1, color=col, lw=1.4, ls=":", alpha=0.85)
            ax.text(idx + 2, pct + 0.8, f"PC {idx + 1} → {pct}%",
                    fontsize=9, color=col, fontweight="bold")

    er_var = float(cumvar[int(eff_rank) - 1])
    ax.scatter([eff_rank], [er_var], color="#6040b0", s=100, zorder=6,
               label=f"Eff. rank {eff_rank:.0f} → {er_var:.1f}%")
    ax.axvline(eff_rank, color="#6040b0", lw=2, ls="--", alpha=0.7)

    ax.set_xlabel("Number of PCA components")
    ax.set_ylabel("Cumulative variance explained (%)")
    ax.set_title("B  —  Cumulative Variance Explained", fontweight="bold", loc="left")
    ax.set_ylim(0, 103)
    ax.set_xlim(1, n_show)
    ax.legend(loc="lower right")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--outdir",     default="outputs/repr_analysis/cls_rank_figures")
    parser.add_argument("--n_pca",      type=int, default=100)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    log.info(f"Loading {args.embeddings}")
    emb = np.load(args.embeddings).astype(np.float32)
    N, D = emb.shape
    log.info(f"  Shape: {N:,} × {D}")

    log.info(f"Fitting PCA (n_components={args.n_pca}) on all {N:,} samples...")
    n_comp = min(args.n_pca, N, D)
    pca = PCA(n_components=n_comp, random_state=42)
    pca.fit(emb)

    sv = np.sqrt(pca.explained_variance_ * (pca.n_samples_ - 1))
    eff_rank_val = effective_rank(sv)

    log.info(f"  Effective rank : {eff_rank_val:.1f}")
    log.info(f"  Top-1 PC       : {pca.explained_variance_ratio_[0]*100:.1f}%")
    log.info(f"  Top-3 PCs      : {pca.explained_variance_ratio_[:3].sum()*100:.1f}%")
    log.info(f"  Top-10 PCs     : {pca.explained_variance_ratio_[:10].sum()*100:.1f}%")

    # ── Combined 2-panel figure ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f"MethylLlama Pretrained CLS Space — {N:,} samples · {D}D bottleneck\n"
        f"Effective rank: {eff_rank_val:.0f} / {D}  ·  "
        f"Top-10 PCs: {pca.explained_variance_ratio_[:10].sum()*100:.1f}% variance",
        fontsize=13, fontweight="bold",
    )

    plot_scree(axes[0], sv, eff_rank_val)
    plot_cumvar(axes[1], sv, eff_rank_val)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    combined = outdir / "cls_rank_analysis.png"
    fig.savefig(combined, dpi=200, bbox_inches="tight")
    log.info(f"Saved combined: {combined}")
    plt.close(fig)

    # ── Individual panels ─────────────────────────────────────────────────────
    for name, fn, size in [
        ("panel_A_scree",  lambda ax: plot_scree(ax, sv, eff_rank_val),  (8, 5.5)),
        ("panel_B_cumvar", lambda ax: plot_cumvar(ax, sv, eff_rank_val), (8, 5.5)),
    ]:
        fig_s, ax_s = plt.subplots(figsize=size)
        fn(ax_s)
        fig_s.tight_layout()
        p = outdir / f"{name}.png"
        fig_s.savefig(p, dpi=200, bbox_inches="tight")
        plt.close(fig_s)
        log.info(f"Saved: {p}")

    print()
    print("=" * 58)
    print("CLS RANK SUMMARY")
    print("=" * 58)
    print(f"  Total samples   : {N:,}")
    print(f"  CLS dimensions  : {D}")
    print(f"  Effective rank  : {eff_rank_val:.1f}  ({eff_rank_val/D*100:.1f}% of {D}D utilized)")
    for k in [1, 3, 5, 10, 20, 50]:
        if k <= n_comp:
            cv = pca.explained_variance_ratio_[:k].sum() * 100
            print(f"  Top-{k:2d} PCs     : {cv:.1f}%")
    print(f"\n  Output: {outdir}/")
    print("=" * 58)


if __name__ == "__main__":
    main()
