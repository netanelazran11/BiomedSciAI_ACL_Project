"""
visualize_cls_rank.py
=====================
Visual proof that the pretrained CLS space is healthy and biologically
structured. Generates a 4-panel figure:

  Panel A — Scree plot        : singular values, effective rank marked
  Panel B — Cumulative var    : % variance explained, key thresholds annotated
  Panel C — PCA PC1 vs PC2    : colored by tissue type
  Panel D — PCA PC1 vs PC2    : colored by biological age

Usage:
  python scripts/repr_analysis/visualize_cls_rank.py \
      --embeddings  outputs/repr_analysis/pretrain_cls_169k_44892802/embeddings_cls.npy \
      --metadata    data/pretrain_metadata.csv.gz \
      --outdir      outputs/repr_analysis/cls_rank_figures
"""

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

STYLE = {
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
}
plt.rcParams.update(STYLE)


# ── helpers ───────────────────────────────────────────────────────────────────

def effective_rank(s: np.ndarray) -> float:
    s = s[s > 1e-10]
    p = s / s.sum()
    return float(np.exp(-np.sum(p * np.log(p + 1e-12))))


# ── panels ────────────────────────────────────────────────────────────────────

def plot_scree(ax, singular_values, eff_rank):
    """Panel A: singular value spectrum with effective rank marker."""
    n = min(80, len(singular_values))
    sv = singular_values[:n]
    var = sv ** 2
    var_pct = var / var.sum() * 100  # relative to shown components

    ax.bar(range(1, n + 1), var_pct, color="#3a6acc", alpha=0.75, width=0.8)
    ax.axvline(eff_rank, color="#c03040", lw=2, ls="--",
               label=f"Effective rank = {eff_rank:.0f}")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance explained (%)")
    ax.set_title("A — Singular Value Spectrum", fontweight="bold", loc="left")
    ax.set_xlim(0, n + 1)
    ax.legend(fontsize=10)
    ax.annotate(f"eff. rank\n= {eff_rank:.0f}",
                xy=(eff_rank, var_pct[int(eff_rank) - 1]),
                xytext=(eff_rank + 6, var_pct[int(eff_rank) - 1] + 0.5),
                fontsize=9, color="#c03040",
                arrowprops=dict(arrowstyle="->", color="#c03040", lw=1.2))


def plot_cumvar(ax, singular_values, eff_rank):
    """Panel B: cumulative variance explained with threshold lines."""
    var = singular_values ** 2
    cumvar = np.cumsum(var) / var.sum() * 100
    n_show = min(100, len(cumvar))

    ax.plot(range(1, n_show + 1), cumvar[:n_show],
            color="#3a6acc", lw=2)
    ax.fill_between(range(1, n_show + 1), cumvar[:n_show],
                    alpha=0.12, color="#3a6acc")

    # Key threshold lines
    for pct, col, lbl in [(90, "#c03040", "90%"), (95, "#c07030", "95%"), (99, "#408040", "99%")]:
        idx = np.searchsorted(cumvar, pct)
        if idx < n_show:
            ax.axhline(pct, color=col, lw=1.2, ls="--", alpha=0.7)
            ax.axvline(idx + 1, color=col, lw=1.2, ls=":", alpha=0.7)
            ax.annotate(f"PC{idx+1} → {lbl}", xy=(idx + 1, pct),
                        xytext=(idx + 4, pct - 3),
                        fontsize=8, color=col)

    # Effective rank marker
    er_var = float(cumvar[int(eff_rank) - 1])
    ax.axvline(eff_rank, color="#6040b0", lw=2, ls="--",
               label=f"Eff. rank {eff_rank:.0f} → {er_var:.1f}% var")
    ax.scatter([eff_rank], [er_var], color="#6040b0", s=60, zorder=5)

    ax.set_xlabel("Number of PCA components")
    ax.set_ylabel("Cumulative variance explained (%)")
    ax.set_title("B — Cumulative Variance Explained", fontweight="bold", loc="left")
    ax.set_ylim(0, 102)
    ax.set_xlim(1, n_show)
    ax.legend(fontsize=10)


def plot_pca_tissue(ax, pca_coords, tissue_labels, n_tissues=20):
    """Panel C: PC1 vs PC2 coloured by tissue type."""
    mask = tissue_labels != "unknown"
    coords = pca_coords[mask]
    labels = tissue_labels[mask]

    # Keep top n_tissues by count
    counts = pd.Series(labels).value_counts()
    top = counts.index[:n_tissues].tolist()
    keep = np.isin(labels, top)
    coords = coords[keep]
    labels = labels[keep]

    cmap = plt.get_cmap("tab20")
    unique = sorted(set(labels))
    color_map = {t: cmap(i / max(len(unique) - 1, 1)) for i, t in enumerate(unique)}

    for tissue in unique:
        m = labels == tissue
        ax.scatter(coords[m, 0], coords[m, 1],
                   c=[color_map[tissue]], s=2, alpha=0.4,
                   label=tissue, rasterized=True)

    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title("C — PCA coloured by Tissue", fontweight="bold", loc="left")
    legend = ax.legend(fontsize=6, markerscale=4, ncol=2,
                       loc="upper right", framealpha=0.6)
    for lh in legend.legend_handles:
        lh.set_alpha(1.0)


def plot_pca_age(ax, pca_coords, age_values):
    """Panel D: PC1 vs PC2 coloured by biological age."""
    valid = ~np.isnan(age_values)
    coords = pca_coords[valid]
    ages = age_values[valid]

    sc = ax.scatter(coords[:, 0], coords[:, 1],
                    c=ages, cmap="plasma", s=2, alpha=0.4,
                    vmin=np.percentile(ages, 2),
                    vmax=np.percentile(ages, 98),
                    rasterized=True)
    cbar = plt.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
    cbar.set_label("Age (years)", fontsize=10)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title("D — PCA coloured by Age", fontweight="bold", loc="left")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", required=True,
                        help="Path to embeddings_cls.npy [N, 256]")
    parser.add_argument("--metadata",   default=None,
                        help="pretrain_metadata.csv.gz with columns: GSM_ID, tissue, age, sex")
    parser.add_argument("--outdir",     default="outputs/repr_analysis/cls_rank_figures")
    parser.add_argument("--n_pca",      type=int, default=100,
                        help="Number of PCA components to compute")
    parser.add_argument("--n_samples",  type=int, default=50000,
                        help="Subsample for PCA scatter (speed)")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ── Load embeddings ───────────────────────────────────────────────────────
    log.info(f"Loading embeddings from {args.embeddings}")
    emb = np.load(args.embeddings).astype(np.float32)
    N, D = emb.shape
    log.info(f"  Shape: {N} × {D}")

    # ── SVD ──────────────────────────────────────────────────────────────────
    log.info("Running PCA...")
    n_comp = min(args.n_pca, N, D)
    pca = PCA(n_components=n_comp, random_state=42)

    # Use subsample for PCA fit if large
    if N > args.n_samples:
        idx = np.random.RandomState(42).choice(N, args.n_samples, replace=False)
        pca.fit(emb[idx])
    else:
        pca.fit(emb)

    singular_values = np.sqrt(pca.explained_variance_ * (pca.n_samples_in_ - 1))
    eff_rank = effective_rank(singular_values)
    log.info(f"  Effective rank: {eff_rank:.1f}")

    # Project all embeddings for scatter
    idx_scatter = np.random.RandomState(0).choice(N, min(args.n_samples, N), replace=False)
    pca_coords = pca.transform(emb[idx_scatter])

    # ── Load metadata (optional) ──────────────────────────────────────────────
    tissue_labels = np.array(["unknown"] * len(idx_scatter))
    age_values    = np.full(len(idx_scatter), np.nan)

    if args.metadata and Path(args.metadata).exists():
        log.info(f"Loading metadata from {args.metadata}")
        meta = pd.read_csv(args.metadata)
        log.info(f"  Metadata shape: {meta.shape}, columns: {list(meta.columns[:8])}")

        # Try common column names
        tissue_col = next((c for c in ["tissue", "tissue_type", "Tissue", "cell_type"] if c in meta.columns), None)
        age_col    = next((c for c in ["age", "Age", "age_years"] if c in meta.columns), None)
        id_col     = next((c for c in ["GSM_ID", "sample_id", "id", "ID"] if c in meta.columns), None)

        log.info(f"  Using: tissue={tissue_col}, age={age_col}, id={id_col}")

        if tissue_col and id_col:
            tissue_dict = dict(zip(meta[id_col].astype(str), meta[tissue_col].fillna("unknown").astype(str)))
            # We don't have sample IDs in .npy — use positional order
            # Try to load the .npy alongside (metadata assumed row-aligned)
            all_tissues = [tissue_dict.get(str(i), "unknown") for i in range(N)]
            tissue_labels = np.array(all_tissues)[idx_scatter]

        if age_col and id_col:
            age_dict = dict(zip(meta[id_col].astype(str), pd.to_numeric(meta[age_col], errors="coerce")))
            all_ages = [age_dict.get(str(i), np.nan) for i in range(N)]
            age_values = np.array(all_ages, dtype=float)[idx_scatter]

        # Fallback: if metadata has no ID column, assume row-aligned
        if tissue_col and not id_col:
            log.info("  No ID column found — assuming row-aligned metadata")
            if len(meta) == N:
                tissue_labels = meta[tissue_col].fillna("unknown").values[idx_scatter]
            if age_col and len(meta) == N:
                age_values = pd.to_numeric(meta[age_col], errors="coerce").values[idx_scatter]

        log.info(f"  Tissue labels: {len(set(tissue_labels))} unique values")
        log.info(f"  Age: {np.nansum(~np.isnan(age_values))} non-NaN")
    else:
        log.info("No metadata — panels C and D will show unlabelled scatter")

    # ── Build figure ──────────────────────────────────────────────────────────
    log.info("Building figure...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f"MethylLlama CLS Space — {N:,} samples × {D}D\n"
        f"Effective rank: {eff_rank:.0f} / {D}  ·  "
        f"Top-3 PCs: {pca.explained_variance_ratio_[:3].sum()*100:.1f}% variance",
        fontsize=14, fontweight="bold", y=1.01
    )

    plot_scree(axes[0, 0], singular_values, eff_rank)
    plot_cumvar(axes[0, 1], singular_values, eff_rank)
    plot_pca_tissue(axes[1, 0], pca_coords, tissue_labels)
    plot_pca_age(axes[1, 1], pca_coords, age_values)

    plt.tight_layout()
    out_path = outdir / "cls_rank_analysis.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    log.info(f"Saved: {out_path}")

    # Also save individual panels for thesis
    for idx_ax, (row, col, name) in enumerate([
        (0, 0, "panel_A_scree"),
        (0, 1, "panel_B_cumvar"),
        (1, 0, "panel_C_pca_tissue"),
        (1, 1, "panel_D_pca_age"),
    ]):
        fig_single, ax_single = plt.subplots(figsize=(7, 5))
        # Re-draw each panel individually
        if name == "panel_A_scree":
            plot_scree(ax_single, singular_values, eff_rank)
        elif name == "panel_B_cumvar":
            plot_cumvar(ax_single, singular_values, eff_rank)
        elif name == "panel_C_pca_tissue":
            plot_pca_tissue(ax_single, pca_coords, tissue_labels)
        elif name == "panel_D_pca_age":
            plot_pca_age(ax_single, pca_coords, age_values)
        p = outdir / f"{name}.png"
        fig_single.tight_layout()
        fig_single.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig_single)
        log.info(f"Saved: {p}")

    plt.close(fig)

    # ── Print summary ─────────────────────────────────────────────────────────
    print()
    print("=" * 55)
    print("CLS RANK SUMMARY")
    print("=" * 55)
    print(f"  Samples       : {N:,}")
    print(f"  Dimensions    : {D}")
    print(f"  Effective rank: {eff_rank:.1f}  ({eff_rank/D*100:.1f}% of {D}D used)")
    for k in [1, 3, 5, 10, 20, 50]:
        if k <= n_comp:
            cv = pca.explained_variance_ratio_[:k].sum() * 100
            print(f"  Top-{k:2d} PCs    : {cv:.1f}% variance")
    print(f"\n  Figures saved to: {outdir}/")
    print("=" * 55)


if __name__ == "__main__":
    main()
