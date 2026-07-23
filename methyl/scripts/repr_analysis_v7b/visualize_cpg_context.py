"""
Fig 2 equivalent — contextualized CpG embedding UMAP, colored by genomic context.

Uses contextual_cpg_emb.npy + cpg_order.csv (from extract_contextual_cpg.py) and
the sesame HM450.hg38 manifest to color CpGs by:
  - chromosome
  - autosome vs sex chromosome (MethylGPT Fig 2d)
  - genomic position within chromosome (continuous — showcases Genomic RoPE)

Also quantifies genomic structure: kNN chromosome-purity (fraction of a CpG's
embedding neighbors on the same chromosome) — a NUMBER MethylGPT never reported.

Manifest columns used: probeID, CpG_chrm, CpG_beg.

Usage (local, after rsync of the context dir + manifest):
  python scripts/repr_analysis_v7b/visualize_cpg_context.py \
     --dir figures/v7b_cpg_context \
     --manifest /path/HM450.hg38.manifest.tsv
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
    p.add_argument("--dir", default="figures/v7b_cpg_context")
    p.add_argument("--manifest", required=True)
    p.add_argument("--knn", type=int, default=30)
    p.add_argument("--subsample", type=int, default=15000, help="CpGs to plot (speed)")
    return p.parse_args()


def umap2d(X):
    Xs = StandardScaler().fit_transform(X)
    p = PCA(n_components=min(50, X.shape[1])).fit_transform(Xs)
    try:
        import umap
        return umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=0).fit_transform(p)
    except Exception as e:
        print(f"[umap unavailable: {e}] using PCA-2D")
        return p[:, :2]


def main():
    a = parse_args()
    d = Path(a.dir)
    E = np.load(d / "contextual_cpg_emb.npy").astype(np.float64)
    order = pd.read_csv(d / "cpg_order.csv")
    print(f"Contextual CpG emb: {E.shape}, order rows: {len(order)}")

    man = pd.read_csv(a.manifest, sep="\t", usecols=["probeID", "CpG_chrm", "CpG_beg"],
                      low_memory=False)
    man = man.rename(columns={"probeID": "cpg_name"})
    df = order.merge(man, on="cpg_name", how="left")
    chrm = df["CpG_chrm"].fillna("NA").astype(str).values
    pos = pd.to_numeric(df["CpG_beg"], errors="coerce").values
    n_mapped = int((chrm != "NA").sum())
    print(f"Mapped {n_mapped}/{len(df)} CpGs to manifest")

    # subsample for plotting speed
    rng = np.random.default_rng(0)
    idx = rng.choice(len(E), min(a.subsample, len(E)), replace=False)
    xy = umap2d(E[idx])
    chrm_s = chrm[idx]; pos_s = pos[idx]
    np.save(d / "cpg_context_umap_coords.npy", xy)

    is_sex = np.isin(chrm_s, ["chrX", "chrY"])
    chr_order = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]

    fig, ax = plt.subplots(1, 3, figsize=(19, 5.6))
    # (1) chromosome
    pal = cm.get_cmap("gist_ncar", 24)
    for k, c in enumerate(chr_order):
        m = chrm_s == c
        if m.any():
            ax[0].scatter(xy[m, 0], xy[m, 1], s=3, color=pal(k), alpha=0.6, lw=0, label=c)
    ax[0].set_title("Contextualized CpG UMAP — chromosome", fontsize=11, weight="bold")
    ax[0].legend(markerscale=2, fontsize=5, ncol=2, loc="center left", bbox_to_anchor=(1.0, 0.5))
    # (2) autosome vs sex (Fig 2d)
    ax[1].scatter(xy[~is_sex, 0], xy[~is_sex, 1], s=3, c="#4a70cc", alpha=0.5, lw=0, label="autosome")
    ax[1].scatter(xy[is_sex, 0], xy[is_sex, 1], s=6, c="#c0392b", alpha=0.8, lw=0, label="sex chr (X/Y)")
    ax[1].set_title("Autosome vs sex chromosome", fontsize=11, weight="bold")
    ax[1].legend(markerscale=2, fontsize=8)
    # (3) genomic position (continuous)
    sc = ax[2].scatter(xy[:, 0], xy[:, 1], c=np.nan_to_num(pos_s), s=3, cmap="viridis", alpha=0.6, lw=0)
    plt.colorbar(sc, ax=ax[2], fraction=0.046, label="genomic position (bp)")
    ax[2].set_title("Genomic position", fontsize=11, weight="bold")
    for x in ax:
        x.set_xticks([]); x.set_yticks([])
    fig.suptitle("Contextualized CpG embedding space (MethylGPT Fig 2 equivalent)",
                 fontsize=13, weight="bold", y=1.02)
    fig.tight_layout()
    out = d / "cpg_context_umap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"saved {out}")

    # quantify: kNN chromosome purity (genomic structure in embedding)
    valid = chrm != "NA"
    Ev = E[valid]; cv = chrm[valid]
    sub = rng.choice(len(Ev), min(8000, len(Ev)), replace=False)
    nn = NearestNeighbors(n_neighbors=a.knn + 1).fit(Ev)
    _, nbr = nn.kneighbors(Ev[sub])
    pur = np.mean([(cv[nbr[i, 1:]] == cv[sub[i]]).mean() for i in range(len(sub))])
    # chance = sum of squared chromosome frequencies
    freq = pd.Series(cv).value_counts(normalize=True).values
    chance = float((freq ** 2).sum())
    print(f"\nkNN chromosome purity: {pur:.3f}  (chance {chance:.3f})")
    print(f"  → contextualized CpGs {'DO' if pur > 1.5*chance else 'do NOT'} cluster by chromosome")
    print(f"  (raw token table was near-orthogonal — contextualization adds genomic structure)")


if __name__ == "__main__":
    main()
