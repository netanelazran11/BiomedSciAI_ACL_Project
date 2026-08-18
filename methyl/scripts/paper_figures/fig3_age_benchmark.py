"""
Figure 3 -- Age prediction and paired MethylGPT comparison (headline figure).

Panels:
  A  Five-fold test stability (MethylLlama): per-fold MedAE/MAE/R2 dots + mean line.
  B  Ensemble predicted vs chronological age, hexbin, MethylLlama and MethylGPT.
  C  Model-level comparison dot plot: MethylLlama vs ElasticNet vs MethylGPT.
  D  Paired bootstrap gaps (MethylGPT - MethylLlama; positive = MethylLlama
     better), forest plot: D1 in years (MedAE, MAE), D2 in R^2 units.

All numbers are read from verified artifacts -- nothing hardcoded except axis
cosmetics:
  kfold_full_history_analysis/fold_test_results.csv
  outputs/bootstrap_predictions/paired_per_subject_predictions.csv
  outputs/bootstrap_predictions/paired_bootstrap_summary.json
  outputs/baselines/elasticnet/elasticnetcv-45888710/elasticnet_results.json

Usage:  python scripts/paper_figures/fig3_age_benchmark.py
Output: figures/paper/fig3_age_benchmark_paired_comparison.{pdf,png}
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common_style import (COL_ENET, COL_GPT, COL_LLAMA, apply_style,
                          panel_label, save)
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

REPO = Path(__file__).resolve().parents[2]
FOLDS_CSV = REPO / "kfold_full_history_analysis/fold_test_results.csv"
PAIRED_CSV = REPO / "outputs/bootstrap_predictions/paired_per_subject_predictions.csv"
BOOT_JSON = REPO / "outputs/bootstrap_predictions/paired_bootstrap_summary.json"
ENET_JSON = REPO / "outputs/baselines/elasticnet/elasticnetcv-45888710/elasticnet_results.json"
OUTDIR = REPO / "figures/paper"


def main():
    apply_style()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    folds = pd.read_csv(FOLDS_CSV)
    assert len(folds) == 5 and (folds["status"] == "done").all()
    paired = pd.read_csv(PAIRED_CSV)
    assert len(paired) == 2149
    boot = json.loads(BOOT_JSON.read_text())
    enet = json.loads(ENET_JSON.read_text())
    assert enet["test"]["n"] == 2149

    pt = boot["point_estimates"]
    llama = pt["MethylLlamaV7b"]          # internal key name only; not shown in figure
    gpt = pt["MethylGPT"]
    et = enet["test"]

    fig = plt.figure(figsize=(7.2, 6.0))
    gs = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35,
                  height_ratios=[1.0, 1.15])

    # ── Panel A: five-fold stability ─────────────────────────────────────────
    gsA = gs[0, 0].subgridspec(1, 3, wspace=1.05)
    metrics = [("test_medae", "MedAE (years)"),
               ("test_mae", "MAE (years)"),
               ("test_r2", "$R^2$")]
    axA0 = None
    for i, (col, lab) in enumerate(metrics):
        ax = fig.add_subplot(gsA[0, i])
        vals = folds[col].values
        jitter = np.linspace(-0.12, 0.12, len(vals))
        ax.scatter(jitter, vals, s=14, color=COL_LLAMA, zorder=3,
                   edgecolors="white", linewidths=0.4)
        ax.hlines(vals.mean(), -0.3, 0.3, color=COL_LLAMA, lw=1.2, zorder=2)
        ax.set_xlim(-0.5, 0.5)
        ax.set_xticks([])
        ax.set_ylabel(lab)
        pad = (vals.max() - vals.min()) * 1.5 + 1e-3
        ax.set_ylim(vals.mean() - pad, vals.mean() + pad)
        if i == 0:
            axA0 = ax
    panel_label(axA0, "a", dx=-0.55)

    # ── Panel B: ensemble predictions, hexbin per model ──────────────────────
    gsB = gs[0, 1].subgridspec(1, 2, wspace=0.35)
    lims = (-5, 110)
    for i, (col, name, cmap) in enumerate([
            ("predicted_age_llama", "MethylLlama", "Blues"),
            ("predicted_age_gpt", "MethylGPT", "Oranges")]):
        ax = fig.add_subplot(gsB[0, i])
        ax.hexbin(paired["true_age"], paired[col], gridsize=45,
                  cmap=cmap, mincnt=1, linewidths=0.1, extent=(*lims, *lims))
        ax.plot(lims, lims, color="0.35", lw=0.7, ls="--", zorder=3)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel("Chronological age (years)")
        if i == 0:
            ax.set_ylabel("Predicted age (years)")
            panel_label(ax, "b", dx=-0.35)
        ax.text(0.04, 0.96, name, transform=ax.transAxes, va="top", fontsize=7)

    # ── Panel C: three-model comparison dot plot ─────────────────────────────
    gsC = gs[1, 0].subgridspec(1, 3, wspace=0.75)
    models = [("MethylLlama", COL_LLAMA,
               dict(medae=llama["medae"], mae=llama["mae"], r2=llama["r2"])),
              ("ElasticNet", COL_ENET,
               dict(medae=et["medae"], mae=et["mae"], r2=et["r2"])),
              ("MethylGPT", COL_GPT,
               dict(medae=gpt["medae"], mae=gpt["mae"], r2=gpt["r2"]))]
    axC0 = None
    for i, (key, lab, better) in enumerate([
            ("medae", "MedAE (years)", "lower"),
            ("mae", "MAE (years)", "lower"),
            ("r2", "$R^2$", "higher")]):
        ax = fig.add_subplot(gsC[0, i])
        vals_all = [m[2][key] for m in models]
        span = max(vals_all) - min(vals_all)
        for j, (name, colr, m) in enumerate(models):
            ax.scatter(m[key], j, s=22, color=colr, zorder=3)
            ax.text(m[key], j - 0.28, f"{m[key]:.2f}" if key != "r2" else f"{m[key]:.3f}",
                    ha="center", va="top", fontsize=6, color=colr)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels([m[0] for m in models] if i == 0 else [])
        ax.invert_yaxis()
        ax.set_xlabel(lab)
        ax.set_xlim(min(vals_all) - span * 0.35, max(vals_all) + span * 0.35)
        ax.set_ylim(len(models) - 0.5, -0.8)
        if i == 0:
            axC0 = ax
    panel_label(axC0, "c", dx=-0.75)

    # ── Panel D: paired bootstrap forest (D1 years, D2 R^2) ─────────────────
    ci = boot["paired_bootstrap_gap_gpt_minus_llama"]
    gsD = gs[1, 1].subgridspec(1, 2, wspace=0.55, width_ratios=[1.4, 1.0])

    axD1 = fig.add_subplot(gsD[0, 0])
    rows = [("MedAE", ci["medae"]), ("MAE", ci["mae"])]
    for j, (lab, c) in enumerate(rows):
        y = len(rows) - 1 - j
        axD1.errorbar(c["point_estimate"], y,
                      xerr=[[c["point_estimate"] - c["ci_95_low"]],
                            [c["ci_95_high"] - c["point_estimate"]]],
                      fmt="o", ms=4.5, color=COL_LLAMA, capsize=2.5, lw=1.0)
    axD1.axvline(0, color="0.4", lw=0.7, ls="--")
    axD1.set_yticks(range(len(rows)))
    axD1.set_yticklabels([r[0] for r in reversed(rows)])
    axD1.set_xlabel("Error reduction (years)")
    axD1.set_xlim(-0.12, 1.05)
    axD1.set_ylim(-0.7, len(rows) - 0.3)
    panel_label(axD1, "d", dx=-0.45)

    axD2 = fig.add_subplot(gsD[0, 1])
    c = ci["r2"]
    # stored as GPT - Llama (negative); flip so positive = MethylLlama better
    pe, lo, hi = -c["point_estimate"], -c["ci_95_high"], -c["ci_95_low"]
    axD2.errorbar(pe, 0, xerr=[[pe - lo], [hi - pe]],
                  fmt="o", ms=4.5, color=COL_LLAMA, capsize=2.5, lw=1.0)
    axD2.axvline(0, color="0.4", lw=0.7, ls="--")
    axD2.set_yticks([0]); axD2.set_yticklabels(["$\\Delta R^2$"])
    axD2.set_xlabel("$R^2$ increase")
    axD2.set_xticks([0, 0.02, 0.04])
    axD2.set_xlim(-0.004, 0.042)
    axD2.set_ylim(-0.7, 0.7)

    save(fig, str(OUTDIR / "fig3_age_benchmark_paired_comparison"))

    # provenance sidecar so every number in the figure is traceable
    prov = {
        "panel_A": str(FOLDS_CSV.relative_to(REPO)),
        "panel_B": str(PAIRED_CSV.relative_to(REPO)),
        "panel_C": {"MethylLlama_MethylGPT": str(BOOT_JSON.relative_to(REPO)),
                     "ElasticNet": str(ENET_JSON.relative_to(REPO))},
        "panel_D": str(BOOT_JSON.relative_to(REPO)),
        "n_test_subjects": 2149,
    }
    (OUTDIR / "fig3_provenance.json").write_text(json.dumps(prov, indent=2))


if __name__ == "__main__":
    main()
