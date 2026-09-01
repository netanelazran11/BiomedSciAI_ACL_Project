"""
Figure 4 -- Biological information content and cross-study limits of the frozen
pretrained representation.

Panels:
  A  Frozen linear age probe: CLS vs mean-pooled token representation.
  B  Tissue vs dataset/study predictability against chance -- the representation
     carries biology AND study-specific structure.
  C  Leave-one-dataset-out age transfer plotted against the held-out cohort's
     age spread, with marker area proportional to cohort size. This replaces a
     ranked bar chart because the scientific question is not "which studies
     fail" but "what distinguishes them" -- and the answer, shown directly, is
     that neither age spread (Spearman rho=+0.28, p=0.40) nor cohort size
     (rho=+0.26, p=0.43) explains failure. Transfer is heterogeneous for
     reasons not captured by these cohort descriptors, which is consistent with
     the strong study-specific structure quantified in panel B.

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
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from common_style import (COL_ACCENT, COL_LLAMA, COL_NEG, apply_style,
                          panel_label, save)
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

REPO = Path(__file__).resolve().parents[2]
SUMMARY = REPO / "figures/v7b_pretrain_cls/analysis_summary.json"
LODO = REPO / "figures/v7b_pretrain_cls/lodo_age_probe.csv"
REF = REPO / "figures/v7b_pretrain_cls/lodo_reference.json"
OUTDIR = REPO / "figures/paper"

MIN_N = 100
MIN_AGE_SD = 10.0
YFLOOR = -1.3          # display floor; more-negative values are clamped + labelled


def main():
    apply_style()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    s = json.loads(SUMMARY.read_text())
    lodo = pd.read_csv(LODO).dropna(subset=["R2_LODO"])

    keep = lodo[(lodo["n"] >= MIN_N) & (lodo["age_std"] >= MIN_AGE_SD)].copy()
    n_neg = int((keep["R2_LODO"] < 0).sum())
    rho_sd, p_sd = spearmanr(keep["age_std"], keep["R2_LODO"])
    rho_n, p_n = spearmanr(keep["n"], keep["R2_LODO"])
    print(f"LODO: {len(keep)} studies pass (n>={MIN_N}, age SD>={MIN_AGE_SD}); "
          f"{n_neg} with negative R2. Median R2 = {keep['R2_LODO'].median():.3f}")
    print(f"  R2 vs age SD: rho={rho_sd:+.3f} p={p_sd:.3f} | "
          f"R2 vs n: rho={rho_n:+.3f} p={p_n:.3f}")

    fig = plt.figure(figsize=(7.2, 5.2))
    gs = GridSpec(2, 2, figure=fig, hspace=0.5, wspace=0.42,
                  height_ratios=[0.85, 1.2])

    # ── Panel A: frozen age probe, CLS vs mean pooling ───────────────────────
    gsA = gs[0, 0].subgridspec(1, 2, wspace=0.95)
    probes = [("CLS", s["age_probe_cls"], COL_LLAMA),
              ("Mean-\npooled", s["age_probe_mean"], "0.55")]
    axA0 = None
    for i, (key, lab, fmt) in enumerate([
            ("linear_ridge_r2", "Age probe $R^2$", "{:.3f}"),
            ("linear_ridge_medae", "Age probe MedAE (yr)", "{:.2f}")]):
        ax = fig.add_subplot(gsA[0, i])
        for j, (name, d, colr) in enumerate(probes):
            ax.bar(j, d[key], width=0.55, color=colr)
            ax.text(j, d[key], fmt.format(d[key]), ha="center", va="bottom",
                    fontsize=6.8)
        ax.set_xticks(range(len(probes)))
        ax.set_xticklabels([p[0] for p in probes], fontsize=6.8)
        ax.set_ylabel(lab)
        ax.set_xlim(-0.6, 1.6)
        if key.endswith("r2"):
            ax.set_ylim(0, 1.05)
        else:
            ax.set_ylim(0, max(p[1][key] for p in probes) * 1.25)
        if i == 0:
            axA0 = ax
    panel_label(axA0, "a", dx=-0.5)

    # ── Panel B: tissue vs dataset predictability ────────────────────────────
    axB = fig.add_subplot(gs[0, 1])
    cp = s["class_probes_cls"]
    items = [("Tissue\n(%d classes)" % cp["tissue_type"]["n_classes"],
              cp["tissue_type"], COL_ACCENT),
             ("Dataset / study\n(%d classes)" % cp["dataset"]["n_classes"],
              cp["dataset"], "0.45")]
    for j, (name, d, colr) in enumerate(items):
        axB.bar(j, d["balanced_acc"], width=0.5, color=colr, zorder=2)
        axB.text(j, d["balanced_acc"] + 0.02, f"{d['balanced_acc']:.3f}",
                 ha="center", va="bottom", fontsize=7.2)
        # chance marker: dark grey with a distinct dash, not red -- red is
        # reserved in this figure for negative transfer (panel c), and a
        # red/green pairing is hard to separate under deuteranopia
        axB.hlines(d["chance"], j - 0.3, j + 0.3, color="0.15", lw=1.3,
                   linestyle=(0, (3, 1.5)), zorder=3)
        axB.text(j, d["chance"] + 0.035, f"chance {d['chance']:.3f}",
                 color="0.15", fontsize=6.5, va="bottom", ha="center")
    axB.set_xticks(range(len(items)))
    axB.set_xticklabels([it[0] for it in items], fontsize=6.8)
    axB.set_ylabel("Balanced accuracy")
    axB.text(0.5, -0.30, "point estimates, fixed split", transform=axB.transAxes,
             ha="center", va="top", fontsize=5.8, color="0.45", style="italic")
    axB.set_ylim(0, 1.12)
    axB.set_xlim(-0.55, 1.55)
    panel_label(axB, "b", dx=-0.18)

    # ── Panel C: LODO transfer vs cohort age spread ──────────────────────────
    axC = fig.add_subplot(gs[1, :])
    # A scatter against age SD invited the reader to see a trend that the data
    # do not support (rho=+0.28, P=0.40, n=11) and buried the identity of the
    # failing cohorts. An ordered forest names every study and makes the spread
    # itself the message; the descriptor correlations move to Supplementary.
    ref = json.loads(REF.read_text())["in_distribution_random_80_20"]
    ordered = keep.sort_values("R2_LODO", ascending=True).reset_index(drop=True)
    y = np.arange(len(ordered))
    r2 = ordered["R2_LODO"].to_numpy()
    r2_plot = np.maximum(r2, YFLOOR)          # clipped values are annotated below
    colors = [COL_NEG if v < 0 else COL_LLAMA for v in r2]

    axC.axvspan(YFLOOR - 0.30, 0, color=COL_NEG, alpha=0.05, zorder=0)
    axC.axvline(0, color="0.3", lw=0.9, zorder=1)
    axC.axvline(ref["r2"], color=COL_ACCENT, ls="--", lw=1.0, zorder=1)
    axC.text(ref["r2"] - 0.02, 1.4,
             f"random-split\nreference {ref['r2']:.3f}", ha="right", va="center",
             fontsize=6.2, color=COL_ACCENT, linespacing=1.25)

    axC.hlines(y, YFLOOR - 0.30, r2_plot, color="0.75", lw=0.7, zorder=2)
    axC.scatter(r2_plot, y, s=26, c=colors, zorder=3, edgecolors="white", linewidths=0.5)
    for i, row in ordered.iterrows():
        if row["R2_LODO"] < YFLOOR:      # never clip silently
            axC.annotate(f"{row['R2_LODO']:.1f}", (YFLOOR, i), xytext=(6, 0),
                         textcoords="offset points", ha="left", va="center",
                         fontsize=6.2, color=COL_NEG, fontweight="bold")
        axC.text(1.045, i, f"{int(row['n'])}", ha="right", va="center",
                 fontsize=6.2, color="0.35", transform=axC.get_yaxis_transform())
    axC.text(1.045, len(ordered) - 0.1, "n", ha="right", va="center", fontsize=6.2,
             color="0.35", style="italic", transform=axC.get_yaxis_transform())

    axC.set_yticks(y); axC.set_yticklabels(ordered["dataset"], fontsize=6.2)
    axC.set_xlabel("Held-out-study age probe $R^2$")
    axC.set_xlim(YFLOOR - 0.30, 1.02)
    axC.set_ylim(-0.8, len(ordered) - 0.2)
    axC.spines["left"].set_visible(False)
    axC.tick_params(axis="y", length=0)
    axC.text(0.012, 0.985, "transfer fails ($R^2 < 0$)", transform=axC.transAxes,
             ha="left", va="top", fontsize=6.4, color=COL_NEG)
    axC.text(0.5, -0.235,
             f"axis truncated at {YFLOOR:g}; values beyond it are printed. "
             f"Median $R^2$ = {keep['R2_LODO'].median():.3f} across {len(keep)} studies.",
             transform=axC.transAxes, ha="center", va="top", fontsize=5.9,
             color="0.45", style="italic")
    panel_label(axC, "c", dx=-0.07)

    save(fig, str(OUTDIR / "fig4_representation_cross_study_transfer"))

    prov = {
        "panel_A_B": str(SUMMARY.relative_to(REPO)),
        "panel_C": str(LODO.relative_to(REPO)),
        "lodo_filter": f"n>={MIN_N} and age_std>={MIN_AGE_SD}",
        "n_studies_shown": int(len(keep)),
        "n_negative_r2": n_neg,
        "median_r2": round(float(keep["R2_LODO"].median()), 4),
        "spearman_r2_vs_age_sd": {"rho": round(float(rho_sd), 4), "p": round(float(p_sd), 4)},
        "spearman_r2_vs_n": {"rho": round(float(rho_n), 4), "p": round(float(p_n), 4)},
        "display_floor": YFLOOR,
        "panel_C_style": "ordered forest; every study named, clipped values printed",
        "in_distribution_reference": "figures/v7b_pretrain_cls/lodo_reference.json",
        "note": "studies below the display floor are annotated with their true R2; "
                 "neither cohort descriptor significantly predicts transfer success",
    }
    (OUTDIR / "fig4_provenance.json").write_text(json.dumps(prov, indent=2))


if __name__ == "__main__":
    main()
