"""
Figure 2 -- WCED representation consistency and held-out reconstruction.

Panels:
  A  Cross-view CLS similarity matrix (subsampled) -- bright diagonal shows each
     view retrieves its own partner profile; marginal violins give the full
     matched vs unmatched distributions on the same similarity scale.
     (Similarity-matrix presentation follows the convention used for
     contrastive cross-view models, e.g. scConcept Fig. 1a.)
  B  Retrieval@k curve vs chance -- top-1 is one point on this curve, so the
     curve is shown rather than a single bar.
  C  Observed vs predicted beta at withheld CpGs (hexbin).
  D  Reconstruction vs controls under identical masks.

Panels C/D render only if the canonical withheld-reconstruction artifact
exists (figures/v7b_pretrain_cls/reconstruction_withheld/). Never falls back
to the legacy reconstruction_baselines results.

Panel A/B statistics are RECOMPUTED from the raw similarity matrix and
asserted against two_view_consistency.json.

Usage:  python scripts/paper_figures/fig2_wced_consistency.py
Output: figures/paper/fig2_wced_consistency_reconstruction.{pdf,png}
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common_style import (COL_ENET, COL_GPT, COL_LLAMA, apply_style,
                          panel_label, save)
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

REPO = Path(__file__).resolve().parents[2]
SIM = REPO / "figures/v7b_pretrain_cls/two_view_simmatrix.npy"
JSON = REPO / "figures/v7b_pretrain_cls/two_view_consistency.json"
RECON_DIR = REPO / "figures/v7b_pretrain_cls/reconstruction_withheld"
OUTDIR = REPO / "figures/paper"

N_HEATMAP = 40          # profiles shown in the similarity-matrix panel
KS = [1, 2, 3, 5, 10, 20, 50, 100]


def main():
    apply_style()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    sim = np.load(SIM).astype(np.float64)
    ref = json.loads(JSON.read_text())
    n = sim.shape[0]
    assert sim.shape == (n, n) and n == ref["n_samples"]

    pos = np.diag(sim)
    off = sim[~np.eye(n, dtype=bool)]

    # rank of the correct partner for each query (0 = best)
    ranks = (sim > pos[:, None]).sum(axis=1)
    top1 = float((ranks == 0).mean())
    at_k = [float((ranks < k).mean()) for k in KS]

    assert abs(pos.mean() - ref["pos_cos"]) < 5e-4
    assert abs(off.mean() - ref["neg_cos"]) < 5e-4
    assert abs(top1 - ref["retrieval_at1"]) < 0.02
    print(f"Recomputed: pos={pos.mean():.4f} neg={off.mean():.4f} top1={top1:.3f}")
    print(f"Retrieval@k: " + ", ".join(f"{k}:{v:.3f}" for k, v in zip(KS, at_k)))

    have_recon = (RECON_DIR / "reconstruction_withheld_summary.json").exists()
    if have_recon:
        rsum = json.loads((RECON_DIR / "reconstruction_withheld_summary.json").read_text())
        scat = np.load(RECON_DIR / "scatter_sample.npz")
        fig = plt.figure(figsize=(7.2, 5.6))
        gs = GridSpec(2, 2, figure=fig, hspace=0.55, wspace=0.42,
                      width_ratios=[1.25, 1.0])
    else:
        print("NOTE: reconstruction artifact missing -- rendering A/B only.")
        fig = plt.figure(figsize=(7.2, 2.8))
        gs = GridSpec(1, 2, figure=fig, wspace=0.42, width_ratios=[1.25, 1.0])

    # ── Panel A: similarity matrix + marginal distributions ──────────────────
    gsA = gs[0, 0].subgridspec(1, 2, wspace=0.45, width_ratios=[3.2, 0.85])
    axA = fig.add_subplot(gsA[0, 0])
    rng = np.random.default_rng(0)
    sel = np.sort(rng.choice(n, size=N_HEATMAP, replace=False))
    sub = sim[np.ix_(sel, sel)]
    im = axA.imshow(sub, cmap="magma", vmin=0.0, vmax=1.0,
                    interpolation="nearest", aspect="equal")
    axA.set_xlabel("View 2 (profile)")
    axA.set_ylabel("View 1 (profile)")
    axA.set_xticks([0, N_HEATMAP - 1]); axA.set_xticklabels(["1", str(N_HEATMAP)])
    axA.set_yticks([0, N_HEATMAP - 1]); axA.set_yticklabels(["1", str(N_HEATMAP)])
    cb = fig.colorbar(im, ax=axA, location="bottom", fraction=0.055, pad=0.28,
                      aspect=28)
    cb.set_label("cosine similarity", fontsize=6.8)
    cb.ax.tick_params(labelsize=6.8)
    panel_label(axA, "a", dx=-0.16)

    axAv = fig.add_subplot(gsA[0, 1])
    off_s = rng.choice(off, size=40_000, replace=False)
    parts = axAv.violinplot([off_s, pos], positions=[0, 1], widths=0.9,
                            showextrema=False, showmedians=True)
    for pc, c in zip(parts["bodies"], ["0.6", COL_LLAMA]):
        pc.set_facecolor(c); pc.set_alpha(0.9); pc.set_edgecolor("none")
    parts["cmedians"].set_color("black"); parts["cmedians"].set_linewidth(0.8)
    axAv.set_ylim(0.0, 1.0)
    axAv.set_xlim(-0.7, 1.7)
    axAv.set_xticks([0, 1])
    axAv.set_xticklabels(["unmatched", "matched"], rotation=90, fontsize=7.0)
    axAv.set_ylabel("cosine similarity", fontsize=6.8)
    axAv.tick_params(axis="y", labelsize=6.8)
    axAv.text(0.42, off.mean(), f"{off.mean():.3f}", ha="left", va="center",
              fontsize=7.0)
    axAv.text(1.42, pos.mean(), f"{pos.mean():.3f}", ha="left", va="center",
              fontsize=7.0, color=COL_LLAMA, fontweight="bold")

    # ── Panel B: retrieval@k curve ───────────────────────────────────────────
    axB = fig.add_subplot(gs[0, 1])
    axB.plot(KS, at_k, "o-", color=COL_LLAMA, ms=3.5, lw=1.2, zorder=3,
             label="MethylLlama CLS")
    axB.plot(KS, [k / n for k in KS], "--", color="#b03030", lw=1.0,
             label="chance")
    axB.set_xscale("log")
    axB.set_xticks(KS)
    axB.set_xticklabels([str(k) for k in KS])
    axB.minorticks_off()
    axB.set_xlabel("k (candidates considered, of %d profiles)" % n)
    axB.set_ylabel("Correct partner within top-k")
    axB.set_ylim(0, 1.02)
    axB.annotate(f"top-1 = {top1*100:.1f}%", xy=(1, top1),
                 xytext=(1.6, top1 - 0.22), fontsize=6.8, color=COL_LLAMA,
                 arrowprops=dict(arrowstyle="-", color=COL_LLAMA, lw=0.6))
    axB.legend(frameon=False, loc="lower right")
    panel_label(axB, "b", dx=-0.2)

    # ── Panels C/D: withheld reconstruction ──────────────────────────────────
    if have_recon:
        axC = fig.add_subplot(gs[1, 0])
        obs = scat["observed"].astype(np.float32)
        prd = scat["predicted"].astype(np.float32)
        hb = axC.hexbin(obs, prd, gridsize=60, cmap="Blues", mincnt=1,
                        bins="log", extent=(0, 1, 0, 1), linewidths=0.1)
        axC.plot([0, 1], [0, 1], color="0.35", lw=0.7, ls="--")
        axC.text(0.04, 0.95, f"$r$ = {rsum['pearson_withheld_obs_vs_pred']:.3f}",
                 transform=axC.transAxes, va="top", fontsize=7)
        axC.set_xlabel(r"Observed $\beta$ (withheld CpGs)")
        axC.set_ylabel(r"Predicted $\beta$")
        axC.set_xlim(0, 1); axC.set_ylim(0, 1)
        cb2 = fig.colorbar(hb, ax=axC, fraction=0.046, pad=0.03)
        cb2.set_label("positions (log)", fontsize=6.8)
        cb2.ax.tick_params(labelsize=6.8)
        panel_label(axC, "c", dx=-0.14)

        axD = fig.add_subplot(gs[1, 1])
        conds = [("Model\n(real CLS)", "raw_mse_model", COL_LLAMA),
                 ("Per-CpG\nmean", "raw_mse_b_mean", COL_ENET),
                 ("Shuffled\nCLS", "raw_mse_b_shuffle", COL_GPT),
                 ("Random\nCLS", "raw_mse_b_random", "0.3")]
        for j, (lab, key, colr) in enumerate(conds):
            d = rsum[key]
            axD.bar(j, d["mean"], width=0.6, color=colr, zorder=2)
            # p10-p90 spread across profiles: bars alone hide the distribution
            axD.vlines(j, d["p10"], d["p90"], color="0.15", lw=0.9, zorder=3)
            axD.text(j, d["p90"] + 0.002, f"{d['mean']:.4f}", ha="center",
                     va="bottom", fontsize=7.0)
        axD.set_xticks(range(len(conds)))
        axD.set_xticklabels([c[0] for c in conds], fontsize=6.8)
        axD.set_ylabel("MSE at withheld CpGs")
        axD.set_ylim(0, None)
        panel_label(axD, "d", dx=-0.2)

    save(fig, str(OUTDIR / "fig2_wced_consistency_reconstruction"))

    prov = {
        "panel_A_B": [str(SIM.relative_to(REPO)), str(JSON.relative_to(REPO))],
        "matrix_derived": {"pos_cos": round(float(pos.mean()), 4),
                            "neg_cos": round(float(off.mean()), 4),
                            "retrieval_at1": round(top1, 4)},
        "retrieval_at_k": {str(k): round(v, 4) for k, v in zip(KS, at_k)},
        "json_summary_jul20_run": {"pos_cos": ref["pos_cos"],
                                     "neg_cos": ref["neg_cos"],
                                     "retrieval_at1": ref["retrieval_at1"]},
        "retrieval_note": "json and matrix are independent random view draws; "
                           "figure and manuscript use the matrix-derived value",
        "heatmap_subsample": N_HEATMAP,
        "panel_C_D": (str(RECON_DIR.relative_to(REPO)) if have_recon
                       else "PENDING"),
        "n_two_view_samples": int(n),
    }
    (OUTDIR / "fig2_provenance.json").write_text(json.dumps(prov, indent=2))


if __name__ == "__main__":
    main()
