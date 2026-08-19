"""
Figure 3 -- Age prediction and paired MethylGPT comparison (headline figure).

Panels:
  A  Five-fold test stability (MethylLlama): per-fold MedAE/MAE/R2 + mean line.
  B  Ensemble predicted vs chronological age, both models, per-subject points
     (n=2,149) with the identity line -- points rather than a density map so
     individual subjects and both model colours stay legible.
  C  Model-level comparison: MethylLlama vs ElasticNet vs MethylGPT.
  D  Paired bootstrap sampling distributions of the between-model difference
     (10,000 subject resamples), with the 95% interval shaded and zero marked.
     Showing the distribution rather than an interval alone makes the
     separation from zero directly visible.

The bootstrap in D is RECOMPUTED locally from the per-subject predictions with
the same seed, then asserted against the stored summary -- the panel therefore
cannot drift from the reported confidence intervals.

Sources:
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
CACHE = OUTDIR / "fig3_bootstrap_draws.npz"

N_BOOT = 10_000
SEED = 0


def metrics(true, pred):
    err = np.abs(pred - true)
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    return np.median(err), err.mean(), 1.0 - ss_res / ss_tot


def bootstrap_draws(true, p_llama, p_gpt):
    """Paired subject bootstrap; positive gap = MethylLlama better."""
    if CACHE.exists():
        d = np.load(CACHE)
        if int(d["n_boot"]) == N_BOOT and int(d["seed"]) == SEED:
            print(f"Using cached bootstrap draws ({CACHE.name})")
            return {k: d[k] for k in ["medae", "mae", "r2"]}
    print(f"Recomputing {N_BOOT} paired bootstrap resamples ...")
    rng = np.random.default_rng(SEED)
    n = len(true)
    out = {"medae": np.empty(N_BOOT), "mae": np.empty(N_BOOT), "r2": np.empty(N_BOOT)}
    for b in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        t = true[idx]
        ml = metrics(t, p_llama[idx])
        mg = metrics(t, p_gpt[idx])
        out["medae"][b] = mg[0] - ml[0]     # error reduction (years)
        out["mae"][b] = mg[1] - ml[1]       # error reduction (years)
        out["r2"][b] = ml[2] - mg[2]        # R2 gain
    np.savez_compressed(CACHE, n_boot=N_BOOT, seed=SEED, **out)
    return out


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
    llama, gpt = pt["MethylLlamaV7b"], pt["MethylGPT"]
    et = enet["test"]

    true = paired["true_age"].to_numpy(float)
    p_ml = paired["predicted_age_llama"].to_numpy(float)
    p_gp = paired["predicted_age_gpt"].to_numpy(float)
    draws = bootstrap_draws(true, p_ml, p_gp)

    # the recomputed distribution must reproduce the published intervals
    ci_ref = boot["paired_bootstrap_gap_gpt_minus_llama"]
    for key, flip in [("medae", 1), ("mae", 1), ("r2", -1)]:
        lo, hi = np.percentile(draws[key], [2.5, 97.5])
        ref_lo = flip * ci_ref[key]["ci_95_low"]
        ref_hi = flip * ci_ref[key]["ci_95_high"]
        ref_lo, ref_hi = min(ref_lo, ref_hi), max(ref_lo, ref_hi)
        assert abs(lo - ref_lo) < 0.02 * max(1e-6, abs(ref_lo)) + 5e-4, (key, lo, ref_lo)
        assert abs(hi - ref_hi) < 0.02 * max(1e-6, abs(ref_hi)) + 5e-4, (key, hi, ref_hi)
    print("Recomputed bootstrap CIs match the stored summary.")

    fig = plt.figure(figsize=(7.2, 6.0))
    gs = GridSpec(2, 2, figure=fig, hspace=0.5, wspace=0.35,
                  height_ratios=[1.0, 1.1])

    # ── Panel A: five-fold stability ─────────────────────────────────────────
    gsA = gs[0, 0].subgridspec(1, 3, wspace=1.05)
    axA0 = None
    for i, (col, lab) in enumerate([("test_medae", "MedAE (years)"),
                                     ("test_mae", "MAE (years)"),
                                     ("test_r2", "$R^2$")]):
        ax = fig.add_subplot(gsA[0, i])
        vals = folds[col].values
        ax.scatter(np.linspace(-0.12, 0.12, len(vals)), vals, s=14,
                   color=COL_LLAMA, zorder=3, edgecolors="white", linewidths=0.4)
        ax.hlines(vals.mean(), -0.3, 0.3, color=COL_LLAMA, lw=1.2, zorder=2)
        ax.set_xlim(-0.5, 0.5); ax.set_xticks([])
        ax.set_ylabel(lab)
        pad = (vals.max() - vals.min()) * 1.5 + 1e-3
        ax.set_ylim(vals.mean() - pad, vals.mean() + pad)
        if i == 0:
            axA0 = ax
    panel_label(axA0, "a", dx=-0.55)

    # ── Panel B: per-subject predictions, both models ────────────────────────
    gsB = gs[0, 1].subgridspec(1, 2, wspace=0.3)
    lims = (-3, 108)
    for i, (col, name, colr, m) in enumerate([
            ("predicted_age_llama", "MethylLlama", COL_LLAMA, llama),
            ("predicted_age_gpt", "MethylGPT", COL_GPT, gpt)]):
        ax = fig.add_subplot(gsB[0, i])
        ax.plot(lims, lims, color="0.35", lw=0.7, ls="--", zorder=1)
        ax.scatter(paired["true_age"], paired[col], s=2.5, color=colr,
                   alpha=0.35, linewidths=0, zorder=2, rasterized=True)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xticks([0, 50, 100]); ax.set_yticks([0, 50, 100])
        ax.set_xlabel("Chronological age (yr)")
        ax.text(0.04, 0.97, name, transform=ax.transAxes, va="top",
                fontsize=6.5, color=colr, fontweight="bold")
        ax.text(0.04, 0.88, f"MedAE {m['medae']:.2f} yr",
                transform=ax.transAxes, va="top", fontsize=5.5, color="0.3")
        if i == 0:
            ax.set_ylabel("Predicted age (yr)")
            panel_label(ax, "b", dx=-0.32)
        else:
            ax.set_yticklabels([])

    # ── Panel C: three-model comparison ──────────────────────────────────────
    gsC = gs[1, 0].subgridspec(1, 3, wspace=0.75)
    models = [("MethylLlama", COL_LLAMA, llama),
              ("ElasticNet", COL_ENET, dict(medae=et["medae"], mae=et["mae"], r2=et["r2"])),
              ("MethylGPT", COL_GPT, gpt)]
    axC0 = None
    for i, (key, lab) in enumerate([("medae", "MedAE (years)"),
                                     ("mae", "MAE (years)"),
                                     ("r2", "$R^2$")]):
        ax = fig.add_subplot(gsC[0, i])
        vals_all = [m[2][key] for m in models]
        span = max(vals_all) - min(vals_all)
        for j, (name, colr, m) in enumerate(models):
            ax.scatter(m[key], j, s=22, color=colr, zorder=3)
            ax.text(m[key], j - 0.3, f"{m[key]:.2f}" if key != "r2" else f"{m[key]:.3f}",
                    ha="center", va="top", fontsize=6, color=colr)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels([m[0] for m in models] if i == 0 else [])
        ax.invert_yaxis()
        ax.set_xlabel(lab)
        ax.set_xlim(min(vals_all) - span * 0.35, max(vals_all) + span * 0.35)
        ax.set_ylim(len(models) - 0.5, -0.85)
        if i == 0:
            axC0 = ax
    panel_label(axC0, "c", dx=-0.75)

    # ── Panel D: bootstrap sampling distributions ────────────────────────────
    gsD = gs[1, 1].subgridspec(3, 1, hspace=0.75)
    specs = [("medae", "$\\Delta$ MedAE (years)", COL_LLAMA),
             ("mae", "$\\Delta$ MAE (years)", COL_LLAMA),
             ("r2", "$\\Delta R^2$", COL_LLAMA)]
    axD0 = None
    for i, (key, lab, colr) in enumerate(specs):
        ax = fig.add_subplot(gsD[i, 0])
        d = draws[key]
        lo, hi = np.percentile(d, [2.5, 97.5])
        cnt, edges = np.histogram(d, bins=70, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.fill_between(centers, 0, cnt, color=colr, alpha=0.25, linewidth=0)
        inside = (centers >= lo) & (centers <= hi)
        ax.fill_between(centers[inside], 0, cnt[inside], color=colr, alpha=0.75,
                        linewidth=0)
        ax.plot(centers, cnt, color=colr, lw=0.8)
        ax.axvline(0, color="#b03030", lw=0.9, ls="--", zorder=4)
        ax.set_yticks([])
        ax.set_ylabel("")
        ax.set_xlabel(lab, labelpad=1.5)
        xmin = min(-0.03 * (hi - lo), d.min())
        ax.set_xlim(xmin, d.max() + 0.05 * (hi - lo))
        ax.text(0.03, 0.95, f"95% CI [{lo:.3f}, {hi:.3f}]", transform=ax.transAxes,
                ha="left", va="top", fontsize=5.5, color="0.25")
        if i == 0:
            axD0 = ax
    panel_label(axD0, "d", dx=-0.1, dy=1.15)
    axD0.set_title("Paired bootstrap: MethylLlama advantage\n"
                   "(10,000 subject resamples; 0 = no difference)",
                   fontsize=6.5, pad=4)

    save(fig, str(OUTDIR / "fig3_age_benchmark_paired_comparison"))

    prov = {
        "panel_A": str(FOLDS_CSV.relative_to(REPO)),
        "panel_B": str(PAIRED_CSV.relative_to(REPO)),
        "panel_C": {"MethylLlama_MethylGPT": str(BOOT_JSON.relative_to(REPO)),
                     "ElasticNet": str(ENET_JSON.relative_to(REPO))},
        "panel_D": {"recomputed_from": str(PAIRED_CSV.relative_to(REPO)),
                     "n_boot": N_BOOT, "seed": SEED,
                     "asserted_against": str(BOOT_JSON.relative_to(REPO))},
        "sign_convention": "positive = MethylLlama better",
        "n_test_subjects": 2149,
    }
    (OUTDIR / "fig3_provenance.json").write_text(json.dumps(prov, indent=2))


if __name__ == "__main__":
    main()
