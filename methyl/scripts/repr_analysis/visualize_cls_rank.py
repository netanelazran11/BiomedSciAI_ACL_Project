"""
visualize_cls_rank.py
=====================
Visualize the structure of the pretrained MethylLlama CLS embedding space.

4-panel figure:
  A — Scree plot          : singular value spectrum, effective rank marked
  B — Cumulative variance : % variance explained, 90/95/99% thresholds
  C — PCA tissue          : STRATIFIED sample — equal N per tissue so rare
                            tissues (prostate, salivary gland) are visible
  D — PCA age             : STRATIFIED by age bins so young/old equally shown

Key design choice:
  SVD/PCA is fit on ALL embeddings (accurate statistics).
  Scatter plots use stratified subsampling (visually balanced, not blood-dominated).

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
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
})


# ── math helpers ──────────────────────────────────────────────────────────────

def effective_rank(s: np.ndarray) -> float:
    s = s[s > 1e-10]
    p = s / s.sum()
    return float(np.exp(-np.sum(p * np.log(p + 1e-12))))


def stratified_sample_by_tissue(indices, tissue_arr, n_per_tissue=300, seed=42):
    """Return indices sampled equally across tissue types."""
    rng = np.random.RandomState(seed)
    selected = []
    tissues = tissue_arr[indices]
    unique = [t for t in np.unique(tissues) if t not in ("unknown", "nan", "")]
    for t in unique:
        mask = tissues == t
        pool = indices[mask]
        n = min(n_per_tissue, len(pool))
        selected.append(rng.choice(pool, n, replace=False))
    return np.concatenate(selected)


def stratified_sample_by_age(indices, age_arr, n_per_bin=400, n_bins=15, seed=42):
    """Return indices sampled equally across age bins."""
    rng = np.random.RandomState(seed)
    ages = age_arr[indices]
    valid = ~np.isnan(ages)
    idx_valid = indices[valid]
    ages_valid = ages[valid]

    bins = np.linspace(np.nanpercentile(ages_valid, 1),
                       np.nanpercentile(ages_valid, 99), n_bins + 1)
    bin_ids = np.digitize(ages_valid, bins)

    selected = []
    for b in range(1, n_bins + 1):
        pool = idx_valid[bin_ids == b]
        if len(pool) == 0:
            continue
        n = min(n_per_bin, len(pool))
        selected.append(rng.choice(pool, n, replace=False))
    return np.concatenate(selected)


# ── panels ────────────────────────────────────────────────────────────────────

def plot_scree(ax, singular_values, eff_rank, n_show=80):
    sv = singular_values[:n_show]
    var_pct = (sv ** 2) / (singular_values ** 2).sum() * 100

    colors = ["#3a6acc" if i + 1 <= eff_rank else "#c8d8f0"
              for i in range(n_show)]
    ax.bar(range(1, n_show + 1), var_pct, color=colors, width=0.85)

    ax.axvline(eff_rank, color="#c03040", lw=2.5, ls="--", zorder=5)
    ax.annotate(
        f"Effective rank = {eff_rank:.0f}\n({eff_rank/len(singular_values)*100:.0f}% of dims active)",
        xy=(eff_rank, var_pct[int(eff_rank) - 1]),
        xytext=(eff_rank + 7, var_pct[int(eff_rank) - 1] + 0.6),
        fontsize=9, color="#c03040", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#c03040", lw=1.4),
    )

    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance explained (%)")
    ax.set_title("A  —  Singular Value Spectrum", fontweight="bold", loc="left")
    ax.set_xlim(0, n_show + 1)

    blue_patch = mpatches.Patch(color="#3a6acc", label=f"Active dims (≤ eff. rank {eff_rank:.0f})")
    gray_patch  = mpatches.Patch(color="#c8d8f0", label="Remaining dims")
    ax.legend(handles=[blue_patch, gray_patch], fontsize=9, loc="upper right")


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
            ax.axhline(pct, color=col, lw=1.3, ls="--", alpha=0.8)
            ax.axvline(idx + 1, color=col, lw=1.3, ls=":", alpha=0.8)
            ax.text(idx + 2, pct + 0.8, f"PC {idx+1} → {pct}%",
                    fontsize=8, color=col, fontweight="bold")

    er_var = float(cumvar[int(eff_rank) - 1])
    ax.scatter([eff_rank], [er_var], color="#6040b0", s=80, zorder=6,
               label=f"Eff. rank {eff_rank:.0f} → {er_var:.1f}%")
    ax.axvline(eff_rank, color="#6040b0", lw=2, ls="--", alpha=0.7)

    ax.set_xlabel("Number of PCA components")
    ax.set_ylabel("Cumulative variance explained (%)")
    ax.set_title("B  —  Cumulative Variance Explained", fontweight="bold", loc="left")
    ax.set_ylim(0, 103)
    ax.set_xlim(1, n_show)
    ax.legend(fontsize=10, loc="lower right")


def plot_pca_tissue(ax, pca_coords, tissue_arr, title_prefix="C"):
    unique_tissues = sorted([t for t in np.unique(tissue_arr)
                             if t not in ("unknown", "nan", "")])
    n_tissues = len(unique_tissues)

    # Color palette — tab20 + tab20b for up to 40 tissues
    cmap1 = plt.get_cmap("tab20").colors
    cmap2 = plt.get_cmap("tab20b").colors
    palette = list(cmap1) + list(cmap2)
    color_map = {t: palette[i % len(palette)] for i, t in enumerate(unique_tissues)}

    for tissue in unique_tissues:
        m = tissue_arr == tissue
        ax.scatter(pca_coords[m, 0], pca_coords[m, 1],
                   c=[color_map[tissue]], s=8, alpha=0.55,
                   label=tissue, rasterized=True, linewidths=0)

    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title(f"{title_prefix}  —  PCA by Tissue  "
                 f"({n_tissues} types · stratified {len(pca_coords):,} samples)",
                 fontweight="bold", loc="left")

    legend = ax.legend(
        fontsize=6.5, markerscale=3, ncol=2,
        loc="upper right", framealpha=0.7,
        handlelength=1.5, columnspacing=0.8,
    )
    for lh in legend.legend_handles:
        lh.set_alpha(1.0)
        lh.set_sizes([20])


def plot_pca_age(ax, pca_coords, age_arr, title_prefix="D"):
    valid = ~np.isnan(age_arr)
    sc = ax.scatter(
        pca_coords[valid, 0], pca_coords[valid, 1],
        c=age_arr[valid], cmap="plasma", s=8, alpha=0.55,
        vmin=np.nanpercentile(age_arr, 2),
        vmax=np.nanpercentile(age_arr, 98),
        rasterized=True, linewidths=0,
    )
    cbar = plt.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
    cbar.set_label("Age (years)", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    n_valid = valid.sum()
    ax.set_title(
        f"{title_prefix}  —  PCA by Age  "
        f"(age-stratified · {n_valid:,} samples)",
        fontweight="bold", loc="left",
    )


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings",    required=True)
    parser.add_argument("--metadata",      default=None,
                        help="pretrain_metadata.csv.gz — columns: tissue, age (row-aligned to embeddings)")
    parser.add_argument("--outdir",        default="outputs/repr_analysis/cls_rank_figures")
    parser.add_argument("--n_pca",         type=int, default=100)
    parser.add_argument("--n_per_tissue",  type=int, default=300,
                        help="Samples per tissue for Panel C (stratified)")
    parser.add_argument("--n_per_age_bin", type=int, default=400,
                        help="Samples per age bin for Panel D (stratified)")
    parser.add_argument("--n_age_bins",    type=int, default=15)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ── Load embeddings ───────────────────────────────────────────────────────
    log.info(f"Loading {args.embeddings}")
    emb = np.load(args.embeddings).astype(np.float32)
    N, D = emb.shape
    log.info(f"  Shape: {N:,} × {D}")

    # ── Fit PCA on ALL data (accurate statistics) ─────────────────────────────
    log.info(f"Fitting PCA (n_components={args.n_pca}) on all {N:,} samples...")
    n_comp = min(args.n_pca, N, D)
    pca = PCA(n_components=n_comp, random_state=42)
    pca.fit(emb)

    sv = np.sqrt(pca.explained_variance_ * (pca.n_samples_in_ - 1))
    eff_rank_val = effective_rank(sv)
    log.info(f"  Effective rank : {eff_rank_val:.1f}")
    log.info(f"  Top-1 PC       : {pca.explained_variance_ratio_[0]*100:.1f}%")
    log.info(f"  Top-3 PCs      : {pca.explained_variance_ratio_[:3].sum()*100:.1f}%")
    log.info(f"  Top-10 PCs     : {pca.explained_variance_ratio_[:10].sum()*100:.1f}%")

    # ── Load metadata ─────────────────────────────────────────────────────────
    all_idx = np.arange(N)
    tissue_all = np.array(["unknown"] * N)
    age_all    = np.full(N, np.nan)
    has_meta   = False

    if args.metadata and Path(args.metadata).exists():
        log.info(f"Loading metadata: {args.metadata}")
        meta = pd.read_csv(args.metadata)
        log.info(f"  Shape: {meta.shape}  cols: {list(meta.columns[:10])}")

        tissue_col = next((c for c in ["tissue", "tissue_type", "Tissue", "cell_type"] if c in meta.columns), None)
        age_col    = next((c for c in ["age", "Age", "age_years"] if c in meta.columns), None)

        if len(meta) == N:
            log.info("  Row-aligned metadata detected")
            if tissue_col:
                tissue_all = meta[tissue_col].fillna("unknown").astype(str).values
                has_meta = True
                n_tissues = len(np.unique(tissue_all[tissue_all != "unknown"]))
                log.info(f"  Tissue types: {n_tissues}")
            if age_col:
                age_all = pd.to_numeric(meta[age_col], errors="coerce").values
                log.info(f"  Age non-NaN: {(~np.isnan(age_all)).sum():,}")
        else:
            log.warning(f"  Metadata rows ({len(meta)}) ≠ embeddings rows ({N}) — skipping labels")
    else:
        log.info("No metadata — panels C and D will show unlabelled scatter")

    # ── Stratified sampling for scatter plots ─────────────────────────────────
    if has_meta:
        log.info(f"Stratified tissue sample: {args.n_per_tissue} per tissue...")
        idx_tissue = stratified_sample_by_tissue(all_idx, tissue_all,
                                                  n_per_tissue=args.n_per_tissue)
        log.info(f"  → {len(idx_tissue):,} samples ({len(np.unique(tissue_all[idx_tissue]))} tissues)")

        log.info(f"Stratified age sample: {args.n_per_age_bin} per bin × {args.n_age_bins} bins...")
        idx_age = stratified_sample_by_age(all_idx, age_all,
                                            n_per_bin=args.n_per_age_bin,
                                            n_bins=args.n_age_bins)
        log.info(f"  → {len(idx_age):,} samples")
    else:
        # No metadata: random subsample
        rng = np.random.RandomState(42)
        idx_tissue = rng.choice(N, min(8000, N), replace=False)
        idx_age    = idx_tissue

    # ── Project stratified subsets ────────────────────────────────────────────
    log.info("Projecting stratified samples into PCA space...")
    pca_tissue = pca.transform(emb[idx_tissue])
    pca_age    = pca.transform(emb[idx_age])

    # ── Build figure ──────────────────────────────────────────────────────────
    log.info("Building combined 4-panel figure...")
    fig, axes = plt.subplots(2, 2, figsize=(17, 13))
    fig.suptitle(
        f"MethylLlama CLS Space Analysis — {N:,} samples · {D}D bottleneck\n"
        f"Effective rank: {eff_rank_val:.0f} / {D}  ·  "
        f"Top-3 PCs: {pca.explained_variance_ratio_[:3].sum()*100:.1f}% var  ·  "
        f"Panels C–D: biologically stratified subsamples",
        fontsize=12, fontweight="bold", y=1.01,
    )

    plot_scree(axes[0, 0], sv, eff_rank_val)
    plot_cumvar(axes[0, 1], sv, eff_rank_val)
    plot_pca_tissue(axes[1, 0], pca_tissue, tissue_all[idx_tissue])
    plot_pca_age(axes[1, 1], pca_age, age_all[idx_age])

    plt.tight_layout(h_pad=3, w_pad=3)
    combined = outdir / "cls_rank_analysis.png"
    fig.savefig(combined, dpi=150, bbox_inches="tight")
    log.info(f"Saved combined: {combined}")
    plt.close(fig)

    # ── Save individual panels ────────────────────────────────────────────────
    panels = [
        ("panel_A_scree",      lambda ax: plot_scree(ax, sv, eff_rank_val),     (8, 5)),
        ("panel_B_cumvar",     lambda ax: plot_cumvar(ax, sv, eff_rank_val),     (8, 5)),
        ("panel_C_pca_tissue", lambda ax: plot_pca_tissue(ax, pca_tissue, tissue_all[idx_tissue]), (9, 7)),
        ("panel_D_pca_age",    lambda ax: plot_pca_age(ax, pca_age, age_all[idx_age]),             (8, 6)),
    ]
    for name, fn, size in panels:
        fig_s, ax_s = plt.subplots(figsize=size)
        fn(ax_s)
        fig_s.tight_layout()
        p = outdir / f"{name}.png"
        fig_s.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig_s)
        log.info(f"Saved: {p}")

    # ── Summary ───────────────────────────────────────────────────────────────
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
    if has_meta:
        print(f"\n  Tissue scatter  : {len(idx_tissue):,} samples "
              f"({args.n_per_tissue}/tissue, stratified)")
        print(f"  Age scatter     : {len(idx_age):,} samples "
              f"({args.n_per_age_bin}/bin × {args.n_age_bins} bins, stratified)")
    print(f"\n  Output: {outdir}/")
    print("=" * 58)


if __name__ == "__main__":
    main()
