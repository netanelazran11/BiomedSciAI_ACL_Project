"""
Figure 4 -- Biological information content and cross-study limitations of the
frozen pretrained representation.

Panels:
  A  Frozen linear age probe: CLS vs mean-pooled token representation
     (R^2 and MedAE side by side).
  B  Tissue and dataset/study predictability (balanced accuracy vs chance):
     the representation carries biology AND study-specific structure.
  C  Leave-one-dataset-out age transfer, frozen representation: R^2 per held-
     out study (n >= 100 and age SD >= 10 years), including negative values.
  D  Cohort context for the same studies: sample size and age SD, y-aligned
     with panel C.

Sources (verified artifacts only):
  figures/v7b_pretrain_cls/analysis_summary.json
  figures/v7b_pretrain_cls/lodo_age_probe.csv

Usage:  python scripts/paper_figures/fig4_representation_transfer.py
Output: figures/paper/fig4_representation_cross_study_transfer.{pdf,png}
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common_style import (COL_ACCENT, COL_LLAMA, COL_NEG, apply_style,
                          panel_label, save)
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

REPO = Path(__file__).resolve().parents[2]
SUMMARY = REPO / "figures/v7b_pretrain_cls/analysis_summary.json"
LODO = REPO / "figures/v7b_pretrain_cls/lodo_age_probe.csv"
OUTDIR = REPO / "figures/paper"

MIN_N = 100
MIN_AGE_SD = 10.0


def main():
    apply_style()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    s = json.loads(SUMMARY.read_text())
    lodo = pd.read_csv(LODO)

    keep = lodo[(lodo["n"] >= MIN_N) & (lodo["age_std"] >= MIN_AGE_SD)].copy()
    keep = keep.dropna(subset=["R2_LODO"]).sort_values("R2_LODO")
    n_keep = len(keep)
    print(f"LODO panel: {n_keep} studies pass n>={MIN_N}, age SD>={MIN_AGE_SD} "
          f"({int((keep['R2_LODO'] < 0).sum())} negative R2)")

    fig = plt.figure(figsize=(7.2, 5.6))
    gs = GridSpec(2, 2, figure=fig, hspace=0.5, wspace=0.45,
                  height_ratios=[0.85, 1.25], width_ratios=[1.0, 1.0])

    # ── Panel A: frozen age probe, CLS vs mean pooling ───────────────────────
    gsA = gs[0, 0].subgridspec(1, 2, wspace=0.9)
    probes = [("CLS", s["age_probe_cls"], COL_LLAMA),
              ("Mean-pooled", s["age_probe_mean"], "0.55")]
    axA0 = None
    for i, (key, lab, better_high) in enumerate([
            ("linear_ridge_r2", "Age probe $R^2$", True),
            ("linear_ridge_medae", "Age probe MedAE (years)", False)]):
        ax = fig.add_subplot(gsA[0, i])
        for j, (name, d, colr) in enumerate(probes):
            ax.bar(j, d[key], width=0.55, color=colr)
            ax.text(j, d[key], f"{d[key]:.2f}" if key.endswith("medae") else f"{d[key]:.3f}",
                    ha="center", va="bottom", fontsize=6)
        ax.set_xticks(range(len(probes)))
        ax.set_xticklabels([p[0] for p in probes], rotation=20, ha="right")
        ax.set_ylabel(lab)
        if key.endswith("r2"):
            ax.set_ylim(0, 1.0)
        if i == 0:
            axA0 = ax
    panel_label(axA0, "a", dx=-0.5)

    # ── Panel B: tissue vs dataset predictability ────────────────────────────
    axB = fig.add_subplot(gs[0, 1])
    cp = s["class_probes_cls"]
    items = [("Tissue\n(37 classes)", cp["tissue_type"], COL_ACCENT),
             ("Dataset/study\n(85 classes)", cp["dataset"], "0.45")]
    for j, (name, d, colr) in enumerate(items):
        axB.bar(j, d["balanced_acc"], width=0.55, color=colr)
        axB.text(j, d["balanced_acc"], f"{d['balanced_acc']:.3f}",
                 ha="center", va="bottom", fontsize=6)
        axB.hlines(d["chance"], j - 0.33, j + 0.33, color=COL_NEG, lw=1.1)
        axB.text(j + 0.36, d["chance"], "chance", color=COL_NEG,
                 fontsize=5.5, va="center")
    axB.set_xticks(range(len(items)))
    axB.set_xticklabels([it[0] for it in items])
    axB.set_ylabel("Balanced accuracy")
    axB.set_ylim(0, 1.08)
    axB.set_xlim(-0.55, 1.95)
    panel_label(axB, "b", dx=-0.18)

    # ── Panel C: LODO R^2 per held-out study ─────────────────────────────────
    axC = fig.add_subplot(gs[1, 0])
    y = np.arange(n_keep)
    colors = [COL_NEG if v < 0 else COL_LLAMA for v in keep["R2_LODO"]]
    axC.barh(y, keep["R2_LODO"], color=colors, height=0.65)
    axC.axvline(0, color="0.3", lw=0.7)
    axC.set_yticks(y)
    axC.set_yticklabels(keep["dataset"], fontsize=5.5)
    axC.set_xlabel("Held-out-study age probe $R^2$")
    # clip extreme negative for readability; annotate true value inside the bar
    xmin = -1.2
    axC.set_xlim(xmin, 1.0)
    for yi, v in zip(y, keep["R2_LODO"]):
        if v < xmin:
            axC.text(xmin + 0.04, yi, f"$\\leftarrow$ {v:.1f}", fontsize=5.5,
                     va="center", ha="left", color="white", fontweight="bold")
    panel_label(axC, "c", dx=-0.42)

    # ── Panel D: cohort context, y-aligned with C ────────────────────────────
    gsD = gs[1, 1].subgridspec(1, 2, wspace=0.35)
    axD1 = fig.add_subplot(gsD[0, 0], sharey=axC)
    axD1.scatter(keep["n"], y, s=12, color="0.35")
    axD1.set_xscale("log")
    axD1.set_xticks([100, 200, 400])
    axD1.set_xticklabels(["100", "200", "400"])
    axD1.minorticks_off()
    axD1.set_xlabel("Samples (log)")
    plt.setp(axD1.get_yticklabels(), visible=False)
    axD1.tick_params(axis="y", length=0)
    panel_label(axD1, "d", dx=-0.12)

    axD2 = fig.add_subplot(gsD[0, 1], sharey=axC)
    axD2.scatter(keep["age_std"], y, s=12, color="0.35")
    axD2.set_xlabel("Age SD (years)")
    plt.setp(axD2.get_yticklabels(), visible=False)
    axD2.tick_params(axis="y", length=0)
    axD2.set_ylim(-0.7, n_keep - 0.3)

    save(fig, str(OUTDIR / "fig4_representation_cross_study_transfer"))

    prov = {
        "panel_A_B": str(SUMMARY.relative_to(REPO)),
        "panel_C_D": str(LODO.relative_to(REPO)),
        "lodo_filter": f"n>={MIN_N} and age_std>={MIN_AGE_SD}",
        "n_studies_shown": int(n_keep),
        "n_negative_r2": int((keep["R2_LODO"] < 0).sum()),
    }
    (OUTDIR / "fig4_provenance.json").write_text(json.dumps(prov, indent=2))


if __name__ == "__main__":
    main()
