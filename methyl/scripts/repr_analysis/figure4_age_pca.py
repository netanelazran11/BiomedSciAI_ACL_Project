#!/usr/bin/env python3
"""
figure4_age_pca.py
==================
PCA visualization of CLS embedding space colored by age — before and after fine-tuning.

Data flow (explicit):
  BEFORE FINE-TUNING:
    Model    : WCED pretrained checkpoint (frozen encoder)
    Data     : 19k finetune h5ad (same samples used for finetune)
    Embedding: --pretrained_npy  (pre-computed, from cls_probing run)

  AFTER FINE-TUNING:
    Model    : MethylationAgeRegressorLlama checkpoint (encoder updated from epoch 10)
    Data     : Same 19k finetune h5ad
    Embedding: --finetuned_npy   (pre-computed, from finetune extract run)

Both embedding files must be row-aligned to --metadata_csv.

Panels (2 rows × 2 cols):
  a: Pretrained CLS — by age (continuous)
  b: Pretrained CLS — by tissue
  c: Fine-tuned CLS  — by age (continuous)
  d: Fine-tuned CLS  — by tissue

Usage:
  python scripts/repr_analysis/figure4_age_pca.py \\
      --pretrained_npy  outputs/repr_analysis/cls_probing_44905909/embeddings_cls.npy \\
      --finetuned_npy   outputs/repr_analysis/finetune_extract_JOBID/embeddings_cls.npy \\
      --metadata_csv    outputs/repr_analysis/cls_probing_44905909/metadata.csv \\
      --ext_metadata    data/pretrain_metadata.csv.gz \\
      --outdir          outputs/repr_analysis/figure4
"""

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Tissue palette (same as figure3)
# ─────────────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_npy",  required=True,
                   help="CLS embeddings from WCED pretrained model [N, 256]")
    p.add_argument("--finetuned_npy",   required=True,
                   help="CLS embeddings from fine-tuned model [N, 256]")
    p.add_argument("--metadata_csv",    required=True,
                   help="metadata.csv aligned to pretrained_npy (index = sample IDs)")
    p.add_argument("--ext_metadata",    default=None,
                   help="External metadata CSV.gz for tissue labels")
    p.add_argument("--ext_id_col",      default="GSM_ID")
    p.add_argument("--age_col",         default="age")
    p.add_argument("--outdir",          default="outputs/repr_analysis/figure4")
    p.add_argument("--n_components",    type=int, default=2)
    p.add_argument("--dpi",             type=int, default=200)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────
def load_inputs(args):
    # ── Metadata ─────────────────────────────────────────────────────────────
    log.info(f"Loading metadata: {args.metadata_csv}")
    meta = pd.read_csv(args.metadata_csv, index_col=0)
    log.info(f"  {meta.shape}  columns: {list(meta.columns)}")

    # Join tissue/sex from external metadata if needed
    if args.ext_metadata and Path(args.ext_metadata).exists():
        log.info(f"  Joining external metadata: {args.ext_metadata}")
        ext = pd.read_csv(args.ext_metadata)
        ext = ext.drop_duplicates(subset=args.ext_id_col).set_index(args.ext_id_col)
        join_cols = [c for c in ["tissue", "sex"] if c in ext.columns]
        meta = meta.join(ext[join_cols], how="left")
        for col in join_cols:
            n = meta[col].notna().sum()
            log.info(f"    {col}: {n:,}/{len(meta):,} matched")

    # ── Age labels ────────────────────────────────────────────────────────────
    if args.age_col not in meta.columns:
        raise ValueError(f"age column '{args.age_col}' not in metadata: {list(meta.columns)}")
    ages = pd.to_numeric(meta[args.age_col], errors="coerce").values
    n_valid = (~np.isnan(ages)).sum()
    log.info(f"  Age labels: {n_valid:,}/{len(ages):,} valid")

    # ── Pretrained embeddings (WCED pretrained model on finetune data) ────────
    log.info(f"\nLoading pretrained CLS: {args.pretrained_npy}")
    pre_emb = np.load(args.pretrained_npy).astype(np.float32)
    log.info(f"  Shape: {pre_emb.shape}")
    if pre_emb.shape[0] != len(meta):
        raise ValueError(
            f"pretrained_npy rows ({pre_emb.shape[0]}) ≠ metadata rows ({len(meta)})\n"
            f"Ensure pretrained_npy was extracted on the SAME data as metadata_csv."
        )

    # ── Fine-tuned embeddings (fine-tuned model on same finetune data) ────────
    log.info(f"Loading fine-tuned CLS: {args.finetuned_npy}")
    ft_emb = np.load(args.finetuned_npy).astype(np.float32)
    log.info(f"  Shape: {ft_emb.shape}")
    if ft_emb.shape[0] != len(meta):
        raise ValueError(
            f"finetuned_npy rows ({ft_emb.shape[0]}) ≠ metadata rows ({len(meta)})\n"
            f"Ensure finetuned_npy was extracted on the SAME data and in the SAME ORDER."
        )

    log.info(f"\nSanity check — same samples, same order:")
    log.info(f"  pretrained shape : {pre_emb.shape}")
    log.info(f"  fine-tuned shape : {ft_emb.shape}")
    log.info(f"  metadata rows    : {len(meta)}")

    return pre_emb, ft_emb, meta, ages


# ─────────────────────────────────────────────────────────────────────────────
# PCA
# ─────────────────────────────────────────────────────────────────────────────
def run_pca(emb: np.ndarray, name: str, n_components: int = 2) -> np.ndarray:
    log.info(f"[{name}] PCA {emb.shape} → {n_components}D ...")
    X = StandardScaler().fit_transform(emb)
    pca = PCA(n_components=n_components, random_state=42)
    coords = pca.fit_transform(X).astype(np.float32)
    var = pca.explained_variance_ratio_
    log.info(f"  PC1={var[0]*100:.1f}%  PC2={var[1]*100:.1f}%  "
             f"total={sum(var)*100:.1f}%")
    return coords, var


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────
def _scatter_age(ax, coords, ages, var, title, panel_letter):
    valid = ~np.isnan(ages)
    sc = ax.scatter(coords[valid, 0], coords[valid, 1],
                    c=ages[valid], cmap="plasma",
                    s=3, alpha=0.5, linewidths=0, rasterized=True)
    if invalid := (~valid).sum():
        ax.scatter(coords[~valid, 0], coords[~valid, 1],
                   c="#CCCCCC", s=2, alpha=0.3, linewidths=0, rasterized=True)
    ax.set_title(f"{panel_letter}  |  {title} — by Age", fontsize=11, fontweight="bold", pad=5)
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}%)", fontsize=9)
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}%)", fontsize=9)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    return sc


def _scatter_tissue(ax, coords, tissue_labels, title, panel_letter):
    cats = [t for t in dict.fromkeys(tissue_labels) if t not in ("unknown", "nan", "None")]
    for cat in cats:
        mask = np.array([t == cat for t in tissue_labels])
        color = TISSUE_COLORS.get(cat, "#AAAAAA")
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=color, s=3, alpha=0.5, linewidths=0, rasterized=True)
    ax.set_title(f"{panel_letter}  |  {title} — by Tissue", fontsize=11, fontweight="bold", pad=5)
    ax.set_xlabel("PC1", fontsize=9)
    ax.set_ylabel("PC2", fontsize=9)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    handles = [mpatches.Patch(color=TISSUE_COLORS.get(c, "#AAAAAA"), label=c)
               for c in cats if c in TISSUE_COLORS]
    if handles:
        ncol = 1 if len(handles) <= 12 else 2
        ax.legend(handles=handles, fontsize=6.5, loc="lower right",
                  framealpha=0.6, ncol=ncol, handlelength=1.2,
                  borderpad=0.4, labelspacing=0.25)


# ─────────────────────────────────────────────────────────────────────────────
# Main figure
# ─────────────────────────────────────────────────────────────────────────────
def plot_figure4(pre_coords, pre_var, ft_coords, ft_var,
                 ages, tissue_labels, outdir: Path, dpi: int):
    fig_dir = outdir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(14, 11))
    fig.patch.set_facecolor("white")

    # Row labels
    row_info = [
        ("Pretrained CLS  (before fine-tuning)",  pre_coords, pre_var),
        ("Fine-tuned CLS  (after fine-tuning)",   ft_coords,  ft_var),
    ]

    top_margin    = 0.96
    bottom_margin = 0.08
    row_gap       = 0.08
    n_rows        = 2
    usable        = top_margin - bottom_margin
    row_h         = (usable - row_gap * (n_rows - 1)) / n_rows

    # Colorbar axis (shared for age panels)
    cbar_ax = fig.add_axes([0.47, bottom_margin, 0.015, usable])

    age_sc = None
    for r, (row_title, coords, var) in enumerate(row_info):
        top    = top_margin - r * (row_h + row_gap)
        bottom = top - row_h
        inner_top = top - 0.025

        fig.text(0.5, top + 0.005, row_title,
                 ha="center", va="bottom", fontsize=12, fontweight="bold",
                 color="#1E3A5F")
        fig.add_artist(plt.Line2D([0.04, 0.96], [top, top],
                                  transform=fig.transFigure,
                                  color="#1E3A5F", linewidth=1.0, alpha=0.4))

        panel_h = inner_top - bottom

        # Left: age
        ax_age = fig.add_axes([0.05, bottom, 0.40, panel_h])
        letter_age = chr(ord('a') + r * 2)
        sc = _scatter_age(ax_age, coords, ages, var, row_title.split("(")[0].strip(), letter_age)
        if r == 0:
            age_sc = sc  # use first row's scatter for shared colorbar

        # Right: tissue
        ax_tis = fig.add_axes([0.52, bottom, 0.44, panel_h])
        letter_tis = chr(ord('a') + r * 2 + 1)
        _scatter_tissue(ax_tis, coords, tissue_labels, row_title.split("(")[0].strip(), letter_tis)

    # Shared age colorbar
    if age_sc is not None:
        cbar = fig.colorbar(age_sc, cax=cbar_ax)
        cbar.set_label("Age (years)", fontsize=9)
        cbar.ax.tick_params(labelsize=8)

    for ext in ["png", "pdf"]:
        out = fig_dir / f"figure4_age_pca.{ext}"
        fig.savefig(out, dpi=dpi if ext == "png" else 72,
                    bbox_inches="tight", facecolor="white")
        log.info(f"  Saved → {out}")
    plt.close()

    # Individual panels
    _save_individual_panels(pre_coords, pre_var, ft_coords, ft_var,
                            ages, tissue_labels, fig_dir, dpi)


def _save_individual_panels(pre_c, pre_v, ft_c, ft_v, ages, tissue_l, fig_dir, dpi):
    configs = [
        ("pretrained_age",    pre_c, pre_v, "age"),
        ("pretrained_tissue", pre_c, pre_v, "tissue"),
        ("finetuned_age",     ft_c,  ft_v,  "age"),
        ("finetuned_tissue",  ft_c,  ft_v,  "tissue"),
    ]
    tissue_labels = tissue_l
    for fname, coords, var, mode in configs:
        fig, ax = plt.subplots(figsize=(7, 6))
        if mode == "age":
            sc = _scatter_age(ax, coords, ages, var, fname.replace("_", " ").title(), "")
            plt.colorbar(sc, ax=ax, label="Age (years)")
        else:
            _scatter_tissue(ax, coords, tissue_labels, fname.replace("_", " ").title(), "")
        plt.tight_layout()
        fig.savefig(fig_dir / f"{fname}.png", dpi=dpi, bbox_inches="tight")
        plt.close()
        log.info(f"  Saved panel → {fig_dir / fname}.png")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args   = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info(" Figure 4: CLS Embedding Space — Before vs After Fine-tuning")
    log.info("=" * 60)
    log.info(f" Pretrained embeddings : {args.pretrained_npy}")
    log.info(f" Fine-tuned embeddings : {args.finetuned_npy}")
    log.info(f" Metadata              : {args.metadata_csv}")
    log.info("=" * 60)

    # 1. Load
    pre_emb, ft_emb, meta, ages = load_inputs(args)

    tissue_labels = (meta["tissue"].fillna("unknown").tolist()
                     if "tissue" in meta.columns else ["unknown"] * len(meta))

    # 2. PCA — separately for each embedding
    log.info("\n[2/3] Running PCA ...")
    pre_coords, pre_var = run_pca(pre_emb, "Pretrained", args.n_components)
    ft_coords,  ft_var  = run_pca(ft_emb,  "Fine-tuned", args.n_components)

    # Save coords
    np.save(outdir / "pretrained_pca_coords.npy", pre_coords)
    np.save(outdir / "finetuned_pca_coords.npy",  ft_coords)
    meta.to_csv(outdir / "metadata.csv")

    # 3. Plot
    log.info("\n[3/3] Generating figure ...")
    plot_figure4(pre_coords, pre_var, ft_coords, ft_var,
                 ages, tissue_labels, outdir, dpi=args.dpi)

    log.info("\n" + "=" * 60)
    log.info(f" DONE → {outdir}/figures/figure4_age_pca.png")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
