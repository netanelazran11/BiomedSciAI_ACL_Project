"""
Fig 3d-f equivalent — RAW methylation vs MODEL (CLS) embedding organization.

MethylGPT's headline "representation quality" figure: show that the model's
learned embedding organizes samples by biology (tissue/sex) and removes batch
structure, compared to a UMAP of the RAW methylation values.

Left column  = raw methylation (β-values) UMAP.
Right column = model CLS embedding UMAP (reuses cls_umap_coords.npy if present).
Rows = colored by tissue_type, dataset (batch), gender.

Also quantifies the improvement with a kNN label-purity score (silhouette-like):
for each coloring, the fraction of a sample's k nearest neighbors sharing its
label — raw vs model. Higher model purity on tissue/sex + LOWER on dataset =
"biology up, batch down" = the MethylGPT claim, but quantified (they only eyeball).

Run on cluster (needs the h5ad for raw β-values):
  python scripts/repr_analysis_v7b/raw_vs_model_umap.py \
     --h5ad <altumage_21k_3way.h5ad> --dir figures/v7b_pretrain_cls
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--h5ad", required=True, help="labeled h5ad with raw beta values")
    p.add_argument("--dir", default="figures/v7b_pretrain_cls",
                   help="dir with metadata.csv + cls embeddings/coords")
    p.add_argument("--max_samples", type=int, default=10988)
    p.add_argument("--knn", type=int, default=30)
    p.add_argument("--top_n", type=int, default=15)
    return p.parse_args()


def umap2d(X):
    Xs = StandardScaler().fit_transform(X)
    p = PCA(n_components=min(50, X.shape[1])).fit_transform(Xs)
    try:
        import umap
        return umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=0).fit_transform(p)
    except Exception as e:
        print(f"[umap unavailable: {e}] falling back to PCA-2D")
        return p[:, :2]


def knn_purity(coords, labels, k):
    labels = pd.Series(labels).astype(str).values
    nn = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, idx = nn.kneighbors(coords)
    idx = idx[:, 1:]
    same = np.array([(labels[idx[i]] == labels[i]).mean() for i in range(len(labels))])
    return float(same.mean())


def scatter(ax, xy, labels, title, top_n, continuous=False):
    if continuous:
        v = pd.to_numeric(labels, errors="coerce").values
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=v, s=4, cmap="viridis", alpha=0.7, lw=0)
        plt.colorbar(sc, ax=ax, fraction=0.046)
    else:
        lab = pd.Series(labels).astype(str).values
        top = pd.Series(lab).value_counts().head(top_n).index.tolist()
        pal = cm.get_cmap("tab20", max(len(top), 1))
        other = ~np.isin(lab, top)
        if other.any():
            ax.scatter(xy[other, 0], xy[other, 1], s=3, c="#dddddd", alpha=0.4, lw=0)
        for k, c in enumerate(top):
            m = lab == c
            ax.scatter(xy[m, 0], xy[m, 1], s=4, color=pal(k), alpha=0.8, lw=0, label=str(c)[:18])
        ax.legend(markerscale=2, fontsize=6, loc="center left",
                  bbox_to_anchor=(1.0, 0.5), framealpha=0.8)
    ax.set_title(title, fontsize=11, weight="bold")
    ax.set_xticks([]); ax.set_yticks([])


def main():
    a = parse_args()
    d = Path(a.dir)
    meta = pd.read_csv(d / "metadata.csv")
    n = min(a.max_samples, len(meta))
    meta = meta.iloc[:n]

    # ── RAW methylation matrix from h5ad (obs order must match metadata) ──
    import anndata as ad
    adata = ad.read_h5ad(a.h5ad)
    # align to metadata sample order
    if "sample_id" in meta.columns:
        adata = adata[meta["sample_id"].values]
    X_raw = adata.X
    X_raw = np.asarray(X_raw.todense()) if hasattr(X_raw, "todense") else np.asarray(X_raw)
    X_raw = np.nan_to_num(X_raw, nan=0.5)[:n]
    print(f"Raw methylation matrix: {X_raw.shape}")

    # ── model CLS coords (reuse if present, else compute from embeddings) ──
    cls_coords_f = d / "cls_umap_coords.npy"
    if cls_coords_f.exists():
        cls_xy = np.load(cls_coords_f)[:n]
        print("Reusing cls_umap_coords.npy")
    else:
        cls_xy = umap2d(np.load(d / "embeddings_cls.npy")[:n].astype(np.float64))

    raw_xy = umap2d(X_raw)
    np.save(d / "raw_umap_coords.npy", raw_xy)

    # ── figure: rows=tissue/dataset/gender, cols=raw/model ──
    colorings = [("tissue_type", "tissue", False),
                 ("dataset", "batch/dataset", False),
                 ("gender", "sex", False)]
    colorings = [(c, lbl, cont) for c, lbl, cont in colorings if c in meta.columns]
    fig, axes = plt.subplots(len(colorings), 2, figsize=(15, 5 * len(colorings)))
    if len(colorings) == 1:
        axes = axes[None, :]
    for r, (col, lbl, cont) in enumerate(colorings):
        scatter(axes[r, 0], raw_xy, meta[col], f"RAW methylation — {lbl}", a.top_n, cont)
        scatter(axes[r, 1], cls_xy, meta[col], f"MODEL CLS — {lbl}", a.top_n, cont)
    fig.suptitle("Raw methylation vs model embedding (MethylGPT Fig 3d-f equivalent)",
                 fontsize=14, weight="bold", y=1.005)
    fig.tight_layout()
    out = d / "pub" / "raw_vs_model_umap.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"saved {out}")

    # ── quantify: kNN label purity raw vs model ──
    print("\n=== kNN label purity (fraction of neighbors sharing label) ===")
    print(f"{'label':>12} {'raw':>8} {'model':>8} {'change':>10}")
    rows = []
    for col, lbl, _ in colorings:
        pr = knn_purity(raw_xy, meta[col], a.knn)
        pm = knn_purity(cls_xy, meta[col], a.knn)
        arrow = "up (good)" if pm > pr else "down"
        if lbl.startswith("batch"):
            arrow = "down (good)" if pm < pr else "up (bad)"
        print(f"{lbl:>12} {pr:>8.3f} {pm:>8.3f}  {arrow}")
        rows.append({"label": lbl, "raw_purity": round(pr, 3), "model_purity": round(pm, 3)})
    pd.DataFrame(rows).to_csv(d / "raw_vs_model_purity.csv", index=False)
    print(f"\nInterpretation: tissue/sex purity should RISE (model organizes biology);")
    print(f"batch/dataset purity should FALL (model removes batch effect).")
    print(f"saved {d/'raw_vs_model_purity.csv'}")


if __name__ == "__main__":
    main()
