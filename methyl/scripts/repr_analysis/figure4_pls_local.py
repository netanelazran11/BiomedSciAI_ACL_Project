#!/usr/bin/env python3
"""
figure4_pls_local.py
====================
PLS (Partial Least Squares) version of figure4.

PCA finds directions of maximum VARIANCE — dominated by tissue/batch.
PLS finds directions of maximum COVARIANCE with age — shows age gradient clearly.

This gives a dramatic before vs after fine-tuning comparison:
  Before FT: moderate age gradient (pretrained model encoded some age)
  After FT:  much stronger age gradient (model optimized for age)
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler

ROOT     = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "figures" / "figure4"
OUT_DIR  = DATA_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

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


def remove_tissue_mean(emb: np.ndarray, tissue_labels: list) -> np.ndarray:
    """Subtract per-tissue mean to isolate within-tissue variation."""
    emb_res = emb.copy()
    tissues = np.array(tissue_labels)
    for t in np.unique(tissues):
        mask = tissues == t
        if mask.sum() >= 2:
            emb_res[mask] -= emb[mask].mean(axis=0)
    return emb_res


def run_pls(emb: np.ndarray, ages: np.ndarray, tissue_labels: list, label: str):
    valid = ~np.isnan(ages)
    print(f"[{label}] tissue-residualized PLS {emb.shape} → 2D  ({valid.sum()} samples with age)")
    # Remove tissue mean so age signal is not buried under tissue variation
    emb_res = remove_tissue_mean(emb, tissue_labels)
    X = StandardScaler().fit_transform(emb_res)
    pls = PLSRegression(n_components=2, scale=False)
    pls.fit(X[valid], ages[valid])
    coords = pls.transform(X).astype(np.float32)
    r1 = np.corrcoef(coords[valid, 0], ages[valid])[0, 1]
    r2 = np.corrcoef(coords[valid, 1], ages[valid])[0, 1]
    print(f"  PLS1 r={r1:.3f}  PLS2 r={r2:.3f}")
    return coords, r1, r2


def _style_ax(ax):
    ax.set_facecolor("#F7F7F7")
    ax.grid(True, color="white", linewidth=0.8, alpha=1.0, zorder=0)
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)
        sp.set_color("#AAAAAA")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)


def scatter_age(ax, coords, ages, r1, r2, title):
    _style_ax(ax)
    valid = ~np.isnan(ages)
    sc = ax.scatter(coords[valid, 0], coords[valid, 1],
                    c=ages[valid], cmap="coolwarm", vmin=0, vmax=100,
                    s=9, alpha=0.70, linewidths=0, rasterized=True, zorder=2)
    if (~valid).sum():
        ax.scatter(coords[~valid, 0], coords[~valid, 1],
                   c="#CCCCCC", s=4, alpha=0.25, linewidths=0, rasterized=True, zorder=1)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=5)
    ax.set_xlabel(f"PLS1  (r={r1:.2f} with age)", fontsize=9)
    ax.set_ylabel(f"PLS2  (r={r2:.2f} with age)", fontsize=9)
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
    ax.set_xlabel("PLS1", fontsize=9)
    ax.set_ylabel("PLS2", fontsize=9)
    handles = [mpatches.Patch(color=TISSUE_COLORS.get(c, "#AAAAAA"), label=c)
               for c in cats if c in TISSUE_COLORS]
    if handles:
        ncol = 1 if len(handles) <= 12 else 2
        ax.legend(handles=handles, fontsize=6.5, loc="lower right",
                  framealpha=0.75, ncol=ncol, handlelength=1.2,
                  borderpad=0.4, labelspacing=0.25, edgecolor="#CCCCCC")


def make_figure(pre_coords, pre_r1, pre_r2,
                ft_coords,  ft_r1,  ft_r2,
                ages, tissue_labels, dpi=200):

    fig = plt.figure(figsize=(14, 11))
    fig.patch.set_facecolor("white")

    rows = [
        ("Pretrained CLS  (before fine-tuning)", pre_coords, pre_r1, pre_r2),
        ("Fine-tuned CLS  (after fine-tuning)",  ft_coords,  ft_r1,  ft_r2),
    ]

    top_m, bot_m, gap = 0.96, 0.08, 0.07
    row_h = (top_m - bot_m - gap) / 2
    cbar_ax = fig.add_axes([0.47, bot_m, 0.015, top_m - bot_m])
    age_sc  = None

    for r, (row_title, coords, r1, r2) in enumerate(rows):
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
        sc = scatter_age(ax_age, coords, ages, r1, r2,
                         f"{letter}  |  by Age (years)")
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
        out = OUT_DIR / f"figure4_pls.{ext}"
        fig.savefig(out, dpi=dpi if ext == "png" else 72,
                    bbox_inches="tight", facecolor="white")
        print(f"  Saved → {out}")
    plt.close()


def main():
    pre_path  = DATA_DIR / "embeddings_cls_pretrained.npy"
    ft_path   = DATA_DIR / "embeddings_cls_finetuned.npy"
    meta_path = DATA_DIR / "aligned_metadata.csv"

    print("Loading embeddings...")
    pre_emb = np.load(pre_path).astype(np.float32)
    ft_emb  = np.load(ft_path).astype(np.float32)
    meta    = pd.read_csv(meta_path, index_col=0)

    ages          = pd.to_numeric(meta["age"], errors="coerce").values
    tissue_labels = meta["tissue"].fillna("unknown").tolist()

    pre_coords, pre_r1, pre_r2 = run_pls(pre_emb, ages, tissue_labels, "Pretrained")
    ft_coords,  ft_r1,  ft_r2  = run_pls(ft_emb,  ages, tissue_labels, "Fine-tuned")

    print("\nGenerating figure...")
    make_figure(pre_coords, pre_r1, pre_r2,
                ft_coords,  ft_r1,  ft_r2,
                ages, tissue_labels, dpi=200)
    print("Done.")


if __name__ == "__main__":
    main()
