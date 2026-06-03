#!/usr/bin/env python3
"""
figure4_umap_local.py
=====================
UMAP version of figure4 — run locally after syncing embeddings from cluster.

Requires:
  figures/figure4/embeddings_cls_pretrained.npy   (10358, 256)
  figures/figure4/embeddings_cls_finetuned.npy    (10358, 256)
  figures/figure4/aligned_metadata.csv

Usage:
  python scripts/repr_analysis/figure4_umap_local.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import umap
from sklearn.preprocessing import StandardScaler

ROOT     = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "figures" / "figure4"
OUT_DIR  = DATA_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

UMAP_PARAMS = dict(n_components=2, n_neighbors=15, min_dist=0.1,
                   metric="cosine", random_state=42, low_memory=False)

TISSUE_COLORS = {
    "Whole Blood":          "#E64B35",
    "Brain":                "#4DBBD5",
    "Other":                "#AAAAAA",
    "Cells":                "#9B59B6",
    "Breast":               "#F39B7F",
    "Lung":                 "#91D1C2",
    "Colon":                "#C0392B",
    "Liver":                "#3C5488",
    "Prostate":             "#7E6148",
    "Skin":                 "#8491B4",
    "Testis":               "#27AE60",
    "Ovary":                "#FF69B4",
    "Stomach":              "#E67E22",
    "Muscle":               "#00A087",
    "Kidney":               "#F4A460",
    "Esophagus":            "#808000",
    "Pancreas":             "#F1C40F",
    "Adipose":              "#B09C85",
    "Bladder":              "#FA8072",
    "Uterus":               "#C39BD3",
    "Cervix":               "#DDA0DD",
    "Thyroid":              "#5B2C6F",
    "Adrenal Gland":        "#D35400",
    "Nerve":                "#F7DC6F",
    "Small Intestine":      "#7DCEA0",
    "Heart":                "#922B21",
    "Minor Salivary Gland": "#708090",
    "Artery":               "#FF4500",
    "Pituitary":            "#98FB98",
    "Fallopian Tube":       "#FF91A4",
    "Spleen":               "#6B8E23",
    "Vagina":               "#FFDAB9",
    "Blood":                "#E64B35",
}


def _style_ax(ax):
    ax.set_facecolor("#F7F7F7")
    ax.grid(True, color="white", linewidth=0.8, alpha=1.0, zorder=0)
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)
        sp.set_color("#AAAAAA")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)


def scatter_age(ax, coords, ages, title):
    _style_ax(ax)
    valid = ~np.isnan(ages)
    sc = ax.scatter(coords[valid, 0], coords[valid, 1],
                    c=ages[valid], cmap="coolwarm", vmin=0, vmax=100,
                    s=9, alpha=0.70, linewidths=0, rasterized=True, zorder=2)
    if (~valid).sum():
        ax.scatter(coords[~valid, 0], coords[~valid, 1],
                   c="#CCCCCC", s=4, alpha=0.25, linewidths=0, rasterized=True, zorder=1)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=5)
    ax.set_xlabel("UMAP 1", fontsize=9)
    ax.set_ylabel("UMAP 2", fontsize=9)
    return sc


def scatter_tissue(ax, coords, tissue_labels, title):
    _style_ax(ax)
    cats = [t for t in dict.fromkeys(tissue_labels)
            if str(t) not in ("unknown", "nan", "None")]
    for cat in cats:
        mask  = np.array([str(t) == str(cat) for t in tissue_labels])
        color = TISSUE_COLORS.get(cat, "#AAAAAA")
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=color, s=9, alpha=0.70, linewidths=0, rasterized=True, zorder=2)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=5)
    ax.set_xlabel("UMAP 1", fontsize=9)
    ax.set_ylabel("UMAP 2", fontsize=9)
    handles = [mpatches.Patch(color=TISSUE_COLORS.get(c, "#AAAAAA"), label=c)
               for c in cats if c in TISSUE_COLORS]
    if handles:
        ncol = 1 if len(handles) <= 12 else 2
        ax.legend(handles=handles, fontsize=6.5, loc="lower right",
                  framealpha=0.75, ncol=ncol, handlelength=1.2,
                  borderpad=0.4, labelspacing=0.25, edgecolor="#CCCCCC")


def run_umap(emb: np.ndarray, label: str) -> np.ndarray:
    print(f"[{label}] StandardScaler + UMAP {emb.shape} → 2D  (n_neighbors=15, cosine) ...")
    X = StandardScaler().fit_transform(emb)
    reducer = umap.UMAP(**UMAP_PARAMS)
    coords  = reducer.fit_transform(X).astype(np.float32)
    print(f"  done → {coords.shape}")
    return coords


def make_figure(pre_coords, ft_coords, ages, tissue_labels, dpi=200):
    fig = plt.figure(figsize=(14, 11))
    fig.patch.set_facecolor("white")

    rows = [
        ("Pretrained CLS  (before fine-tuning)", pre_coords),
        ("Fine-tuned CLS  (after fine-tuning)",  ft_coords),
    ]

    top_m, bot_m, gap = 0.96, 0.08, 0.07
    row_h = (top_m - bot_m - gap) / 2
    cbar_ax = fig.add_axes([0.47, bot_m, 0.015, top_m - bot_m])
    age_sc  = None

    for r, (row_title, coords) in enumerate(rows):
        top    = top_m - r * (row_h + gap)
        bottom = top - row_h
        inner  = top - 0.025
        h      = inner - bottom

        fig.text(0.5, top + 0.005, row_title,
                 ha="center", va="bottom", fontsize=12, fontweight="bold",
                 color="#1E3A5F")
        fig.add_artist(plt.Line2D([0.04, 0.96], [top, top],
                                  transform=fig.transFigure,
                                  color="#1E3A5F", linewidth=1.0, alpha=0.4))

        ax_age = fig.add_axes([0.05, bottom, 0.39, h])
        letter = chr(ord('a') + r * 2)
        sc = scatter_age(ax_age, coords, ages, f"{letter}  |  by Age (years)")
        if r == 0:
            age_sc = sc

        ax_tis = fig.add_axes([0.52, bottom, 0.44, h])
        letter = chr(ord('a') + r * 2 + 1)
        scatter_tissue(ax_tis, coords, tissue_labels, f"{letter}  |  by Tissue")

    if age_sc is not None:
        cbar = fig.colorbar(age_sc, cax=cbar_ax, ticks=[0, 25, 50, 75, 100])
        cbar.set_label("Age (years)", fontsize=9)
        cbar.ax.tick_params(labelsize=8)
        cbar.ax.set_yticklabels(["0", "25", "50", "75", "100"])

    for ext in ["png", "pdf"]:
        out = OUT_DIR / f"figure4_umap.{ext}"
        fig.savefig(out, dpi=dpi if ext == "png" else 72,
                    bbox_inches="tight", facecolor="white")
        print(f"  Saved → {out}")
    plt.close()


def main():
    pre_path = DATA_DIR / "embeddings_cls_pretrained.npy"
    ft_path  = DATA_DIR / "embeddings_cls_finetuned.npy"
    meta_path = DATA_DIR / "aligned_metadata.csv"

    if not pre_path.exists() or not ft_path.exists():
        print("ERROR: embedding files not found. Sync from cluster first:")
        print()
        print("  rsync -av netanel.azran@moriah:/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/outputs/repr_analysis/cls_probing_44905909/embeddings_cls.npy \\")
        print(f"    {DATA_DIR}/embeddings_cls_pretrained.npy")
        print()
        print("  rsync -av netanel.azran@moriah:/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/outputs/repr_analysis/finetune_extract_44944545/embeddings_cls.npy \\")
        print(f"    {DATA_DIR}/embeddings_cls_finetuned.npy")
        return

    print("Loading embeddings...")
    pre_emb = np.load(pre_path).astype(np.float32)
    ft_emb  = np.load(ft_path).astype(np.float32)
    meta    = pd.read_csv(meta_path, index_col=0)

    # Both are already row-aligned (same order, saved by align_by_sample_id)
    assert pre_emb.shape[0] == ft_emb.shape[0] == len(meta), \
        f"Row count mismatch: pre={pre_emb.shape[0]}, ft={ft_emb.shape[0]}, meta={len(meta)}"

    ages          = pd.to_numeric(meta["age"], errors="coerce").values
    tissue_labels = meta["tissue"].fillna("unknown").tolist()

    # Cache UMAP coords so re-running is fast
    pre_umap_path = DATA_DIR / "pretrained_umap_coords.npy"
    ft_umap_path  = DATA_DIR / "finetuned_umap_coords.npy"

    if pre_umap_path.exists():
        print("Loading cached pretrained UMAP coords...")
        pre_coords = np.load(pre_umap_path)
    else:
        pre_coords = run_umap(pre_emb, "Pretrained")
        np.save(pre_umap_path, pre_coords)

    if ft_umap_path.exists():
        print("Loading cached fine-tuned UMAP coords...")
        ft_coords = np.load(ft_umap_path)
    else:
        ft_coords = run_umap(ft_emb, "Fine-tuned")
        np.save(ft_umap_path, ft_coords)

    print("\nGenerating figure...")
    make_figure(pre_coords, ft_coords, ages, tissue_labels, dpi=200)
    print("\nDone.")


if __name__ == "__main__":
    main()
