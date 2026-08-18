"""
Figure 2 -- WCED representation consistency and held-out reconstruction.

Panels:
  A  Cosine similarity of CLS embeddings: matched two-view pairs of the same
     profile vs unmatched pairs across profiles (densities + means).
  B  Cross-view top-1 retrieval vs chance.
  C  Observed vs predicted beta at withheld CpGs (hexbin).      [needs artifact]
  D  Reconstruction vs controls under identical masks.          [needs artifact]

Panels C/D are drawn only if the canonical withheld-reconstruction artifact
exists (figures/v7b_pretrain_cls/reconstruction_withheld/, produced by
run_reconstruction_withheld.sh); otherwise the figure is rendered as A/B only
and a note is printed. Never falls back to the legacy reconstruction_baselines
results (wrong checkpoint, wrong objective, no genomic positions).

Panel A/B numbers are RECOMPUTED from the raw similarity matrix and asserted
against two_view_consistency.json -- so the figure cannot silently drift from
the published summary statistics.

Sources:
  figures/v7b_pretrain_cls/two_view_simmatrix.npy   (2000x2000, v1 x v2 cosine)
  figures/v7b_pretrain_cls/two_view_consistency.json
  figures/v7b_pretrain_cls/reconstruction_withheld/ (optional, panels C/D)

Usage:  python scripts/paper_figures/fig2_wced_consistency.py
Output: figures/paper/fig2_wced_consistency_reconstruction.{pdf,png}
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
SIM = REPO / "figures/v7b_pretrain_cls/two_view_simmatrix.npy"
JSON = REPO / "figures/v7b_pretrain_cls/two_view_consistency.json"
RECON_DIR = REPO / "figures/v7b_pretrain_cls/reconstruction_withheld"
OUTDIR = REPO / "figures/paper"


def main():
    apply_style()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    sim = np.load(SIM).astype(np.float64)
    ref = json.loads(JSON.read_text())
    n = sim.shape[0]
    assert sim.shape == (n, n) and n == ref["n_samples"]

    pos = np.diag(sim)
    off = sim[~np.eye(n, dtype=bool)]
    top1 = float((sim.argmax(axis=1) == np.arange(n)).mean())

    # Means must reproduce the published summary stats; retrieval may differ
    # slightly because the Jul-20 json and the Aug-04 matrix come from two
    # independent random view draws of the same checkpoint (same sample subset,
    # different 50% CpG views). The figure reports the matrix-derived value so
    # panel and underlying data are self-consistent; both are recorded in the
    # provenance sidecar. If the discrepancy exceeds 2pp something is wrong.
    assert abs(pos.mean() - ref["pos_cos"]) < 5e-4, (pos.mean(), ref["pos_cos"])
    assert abs(off.mean() - ref["neg_cos"]) < 5e-4, (off.mean(), ref["neg_cos"])
    assert abs(top1 - ref["retrieval_at1"]) < 0.02, (top1, ref["retrieval_at1"])
    if abs(top1 - ref["retrieval_at1"]) > 1e-6:
        print(f"NOTE: matrix-derived top-1 retrieval {top1:.3f} differs from "
              f"json {ref['retrieval_at1']:.3f} (independent view draws). "
              f"Figure shows the matrix value; align the manuscript text to it.")
    print(f"Recomputed from matrix: pos={pos.mean():.4f} neg={off.mean():.4f} "
          f"top1={top1:.3f}")

    have_recon = (RECON_DIR / "reconstruction_withheld_summary.json").exists()
    if have_recon:
        rsum = json.loads((RECON_DIR / "reconstruction_withheld_summary.json").read_text())
        scat = np.load(RECON_DIR / "scatter_sample.npz")
        fig = plt.figure(figsize=(7.2, 5.4))
        gs = GridSpec(2, 2, figure=fig, hspace=0.5, wspace=0.4,
                      width_ratios=[1.35, 1.0])
    else:
        print("NOTE: reconstruction_withheld artifact not found -- rendering "
              "panels A/B only (run run_reconstruction_withheld.sh first for C/D).")
        fig = plt.figure(figsize=(7.2, 2.7))
        gs = GridSpec(1, 2, figure=fig, wspace=0.4, width_ratios=[1.35, 1.0])

    # ── Panel A: matched vs unmatched similarity densities ───────────────────
    axA = fig.add_subplot(gs[0, 0])
    bins = np.linspace(-0.2, 1.0, 121)
    rng = np.random.default_rng(0)
    off_sample = rng.choice(off, size=min(len(off), 200_000), replace=False)
    axA.hist(off_sample, bins=bins, density=True, alpha=0.6, color="0.6",
             label=f"unmatched pairs (mean {off.mean():.3f})")
    axA.hist(pos, bins=bins, density=True, alpha=0.75, color=COL_LLAMA,
             label=f"matched views, same profile (mean {pos.mean():.3f})")
    axA.axvline(off.mean(), color="0.35", lw=0.8, ls="--")
    axA.axvline(pos.mean(), color=COL_LLAMA, lw=0.8, ls="--")
    axA.set_xlabel("Cosine similarity of CLS embeddings")
    axA.set_ylabel("Density")
    axA.legend(frameon=False, loc="upper left")
    panel_label(axA, "a")

    # ── Panel B: cross-view retrieval vs chance ──────────────────────────────
    axB = fig.add_subplot(gs[0, 1])
    chance = 1.0 / n
    axB.bar([0], [top1], width=0.5, color=COL_LLAMA)
    axB.text(0, top1, f"{top1*100:.1f}%", ha="center", va="bottom", fontsize=6.5)
    axB.axhline(chance, color="#b03030", lw=1.0, ls="--")
    axB.text(0.42, chance + 0.015, f"chance ({chance*100:.2f}%)",
             color="#b03030", fontsize=6)
    axB.set_xticks([0])
    axB.set_xticklabels(["Top-1\ncross-view retrieval"])
    axB.set_ylabel("Retrieval accuracy")
    axB.set_ylim(0, 1.0)
    axB.set_xlim(-0.6, 0.9)
    panel_label(axB, "b", dx=-0.25)

    # ── Panels C/D: withheld reconstruction (only from canonical artifact) ───
    if have_recon:
        axC = fig.add_subplot(gs[1, 0])
        obs = scat["observed"].astype(np.float32)
        prd = scat["predicted"].astype(np.float32)
        axC.hexbin(obs, prd, gridsize=60, cmap="Blues", mincnt=1,
                   bins="log", extent=(0, 1, 0, 1), linewidths=0.1)
        axC.plot([0, 1], [0, 1], color="0.35", lw=0.7, ls="--")
        r = rsum["pearson_withheld_obs_vs_pred"]
        axC.text(0.04, 0.95, f"$r$ = {r:.3f}", transform=axC.transAxes,
                 va="top", fontsize=7)
        axC.set_xlabel("Observed beta (withheld CpGs)")
        axC.set_ylabel("Predicted beta")
        axC.set_xlim(0, 1); axC.set_ylim(0, 1)
        panel_label(axC, "c")

        axD = fig.add_subplot(gs[1, 1])
        conds = [("Model\n(real CLS)", rsum["raw_mse_model"]["mean"], COL_LLAMA),
                 ("Per-CpG\nmean", rsum["raw_mse_b_mean"]["mean"], COL_ENET),
                 ("Shuffled\nCLS", rsum["raw_mse_b_shuffle"]["mean"], COL_GPT),
                 ("Random\nCLS", rsum["raw_mse_b_random"]["mean"], "0.3")]
        for j, (lab, v, colr) in enumerate(conds):
            axD.bar(j, v, width=0.6, color=colr)
            axD.text(j, v, f"{v:.4f}", ha="center", va="bottom", fontsize=5.5)
        axD.set_xticks(range(len(conds)))
        axD.set_xticklabels([c[0] for c in conds], fontsize=6)
        axD.set_ylabel("MSE at withheld CpGs")
        panel_label(axD, "d", dx=-0.25)

    save(fig, str(OUTDIR / "fig2_wced_consistency_reconstruction"))

    prov = {
        "panel_A_B": [str(SIM.relative_to(REPO)), str(JSON.relative_to(REPO))],
        "panel_A_B_recomputed_and_asserted": True,
        "matrix_derived": {"pos_cos": round(float(pos.mean()), 4),
                            "neg_cos": round(float(off.mean()), 4),
                            "retrieval_at1": round(top1, 4)},
        "json_summary_jul20_run": {"pos_cos": ref["pos_cos"],
                                     "neg_cos": ref["neg_cos"],
                                     "retrieval_at1": ref["retrieval_at1"]},
        "retrieval_note": "json and matrix are independent random view draws of "
                           "the same checkpoint; figure + manuscript should use "
                           "the matrix-derived value",
        "panel_C_D": (str(RECON_DIR.relative_to(REPO)) if have_recon
                       else "PENDING run_reconstruction_withheld.sh"),
        "n_two_view_samples": int(n),
    }
    (OUTDIR / "fig2_provenance.json").write_text(json.dumps(prov, indent=2))


if __name__ == "__main__":
    main()
