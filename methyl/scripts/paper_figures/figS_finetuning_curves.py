"""Supplementary figure - validation trajectories for two fine-tuning runs.

The shared 0-163 epoch window is shown explicitly. MethylGPT's later epochs
remain visible as a dashed continuation because its best saved validation
MedAE occurs after the shared window. The validation-cohort match is recorded
as a user-verified external audit because it cannot be recovered from the
MethylGPT run metadata alone.
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common_style import COL_GPT, COL_LLAMA, apply_style, panel_label, save
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

REPO = Path(__file__).resolve().parents[2]
CUR = REPO / "outputs/wandb_curves"
OUT = REPO / "figures/paper"

# This fixed fold is used because the user reports having verified that its
# validation cohort matches the MethylGPT run. The run metadata itself does
# not preserve enough identifiers to independently reproduce that audit here.
LLAMA_FOLD = 1
LLAMA_FILE = "llama_fold1_val_full.csv"


def main():
    apply_style()
    OUT.mkdir(parents=True, exist_ok=True)

    llama = pd.read_csv(CUR / LLAMA_FILE).sort_values("epoch")
    gpt = pd.read_csv(CUR / "methylgpt_val_full.csv").sort_values("epoch")
    common_end = int(min(llama["epoch"].max(), gpt["epoch"].max()))
    gpt_common = gpt[gpt["epoch"] <= common_end]
    gpt_later = gpt[gpt["epoch"] >= common_end]

    fig = plt.figure(figsize=(7.2, 3.25))
    grid = GridSpec(1, 2, figure=fig, wspace=0.30)
    specifications = [
        ("val/medae", "valid_medae", "Validation MedAE (years)", "a", (2.3, 5.3)),
        ("val/mae", "valid_mae", "Validation MAE (years)", "b", (4.0, 6.6)),
    ]
    metrics = {}

    for j, (llama_column, gpt_column, ylabel, letter, zoom_ylim) in enumerate(specifications):
        ax = fig.add_subplot(grid[0, j])
        ax.axvspan(0, common_end, color="#e9eef8", alpha=0.65, zorder=0)
        ax.plot(llama["epoch"], llama[llama_column], color=COL_LLAMA, lw=1.0,
                zorder=3, label="MethylLlama")
        ax.plot(gpt_common["epoch"], gpt_common[gpt_column], color=COL_GPT, lw=1.0,
                zorder=3, label="MethylGPT")
        ax.plot(gpt_later["epoch"], gpt_later[gpt_column], color=COL_GPT, lw=0.9,
                ls="--", alpha=0.72, zorder=2, label="MethylGPT after shared window")

        llama_best = llama.loc[llama[llama_column].idxmin()]
        gpt_common_best = gpt_common.loc[gpt_common[gpt_column].idxmin()]
        gpt_full_best = gpt.loc[gpt[gpt_column].idxmin()]
        ax.scatter([llama_best["epoch"]], [llama_best[llama_column]], s=21,
                   color=COL_LLAMA, zorder=5, edgecolors="white", linewidths=0.5)
        ax.scatter([gpt_common_best["epoch"]], [gpt_common_best[gpt_column]], s=21,
                   color=COL_GPT, zorder=5, edgecolors="white", linewidths=0.5)
        ax.scatter([gpt_full_best["epoch"]], [gpt_full_best[gpt_column]], s=24,
                   facecolors="white", edgecolors=COL_GPT, linewidths=1.0, zorder=5)

        ax.axvline(common_end, color="0.35", ls=":", lw=0.9, zorder=1)
        ax.set_xlim(0, int(gpt["epoch"].max()))
        ax.set_xlabel("Fine-tuning epoch")
        ax.set_ylabel(ylabel)
        ax.tick_params(labelsize=6.8)
        panel_label(ax, letter, dx=-0.13)

        inset = inset_axes(ax, width="49%", height="43%", loc="center right",
                          borderpad=0.9)
        inset.plot(llama["epoch"], llama[llama_column], color=COL_LLAMA, lw=0.8)
        inset.plot(gpt_common["epoch"], gpt_common[gpt_column], color=COL_GPT, lw=0.8)
        inset.plot(gpt_later["epoch"], gpt_later[gpt_column], color=COL_GPT,
                   lw=0.75, ls="--", alpha=0.72)
        inset.scatter([llama_best["epoch"]], [llama_best[llama_column]], s=12,
                      color=COL_LLAMA, zorder=4)
        inset.scatter([gpt_common_best["epoch"]], [gpt_common_best[gpt_column]],
                      s=12, color=COL_GPT, zorder=4)
        inset.scatter([gpt_full_best["epoch"]], [gpt_full_best[gpt_column]], s=15,
                      facecolors="white", edgecolors=COL_GPT, linewidths=0.8, zorder=4)
        inset.axvline(common_end, color="0.35", ls=":", lw=0.7)
        inset.set_xlim(20, int(gpt["epoch"].max()))
        inset.set_ylim(*zoom_ylim)
        inset.set_title("plateau detail", fontsize=6.8, pad=2)
        inset.tick_params(labelsize=6.8, length=2)
        for spine in inset.spines.values():
            spine.set_linewidth(0.7)

        metrics[ylabel] = {
            "methyllama_best": float(llama_best[llama_column]),
            "methyllama_epoch": int(llama_best["epoch"]),
            "methylgpt_best_common_window": float(gpt_common_best[gpt_column]),
            "methylgpt_common_epoch": int(gpt_common_best["epoch"]),
            "methylgpt_best_full_run": float(gpt_full_best[gpt_column]),
            "methylgpt_full_epoch": int(gpt_full_best["epoch"]),
        }

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=6.8,
               loc="lower center", bbox_to_anchor=(0.5, -0.025), ncol=3,
               handlelength=1.7, columnspacing=1.4)
    fig.subplots_adjust(bottom=0.20)

    save(fig, str(OUT / "figS_finetuning_curves"))

    medae = metrics["Validation MedAE (years)"]
    mae = metrics["Validation MAE (years)"]
    caption = (
        "\\textbf{Validation trajectories during chronological-age fine-tuning.} "
        "\\textbf{a}, Validation median absolute error and \\textbf{b}, mean absolute error "
        f"for MethylLlama fold {LLAMA_FOLD} and the MethylGPT run. The shaded region marks "
        f"the shared training window (epochs 0--{common_end}); MethylGPT's later epochs are "
        "shown as a dashed continuation rather than omitted. Filled markers identify the "
        "minimum of each run within the shared window, and the open orange marker identifies "
        "MethylGPT's minimum over its full 300-epoch run. Within the shared window, minimum "
        f"MedAE was {medae['methyllama_best']:.3f} years for MethylLlama and "
        f"{medae['methylgpt_best_common_window']:.3f} years for MethylGPT; the latter reached "
        f"{medae['methylgpt_best_full_run']:.3f} years at epoch "
        f"{medae['methylgpt_full_epoch']}. Corresponding minimum MAE values were "
        f"{mae['methyllama_best']:.3f}, {mae['methylgpt_best_common_window']:.3f}, and "
        f"{mae['methylgpt_best_full_run']:.3f} years. The validation cohorts were verified "
        "separately as matched; this figure describes optimization trajectories, whereas the "
        "primary accuracy comparison uses paired predictions on the 2,149-sample test cohort."
    )
    (OUT / "figS_finetuning_curves_caption.txt").write_text(caption + "\n")
    (OUT / "figS_finetuning_curves_provenance.json").write_text(json.dumps({
        "methyllama_run": {
            "fold": LLAMA_FOLD,
            "wandb": "aas23t9m",
            "source": f"outputs/wandb_curves/{LLAMA_FILE}",
            "selection": "fixed fold matched to the validation cohort audited by the user; "
                         "not selected from the plotted metric in this script",
        },
        "methylgpt_run": {
            "wandb": "xzrw1qwr",
            "source": "outputs/wandb_curves/methylgpt_val_full.csv",
        },
        "validation_cohort_comparability": {
            "status": "reported by user as verified identical",
            "limitation": "MethylGPT run metadata does not retain identifiers needed to "
                          "reproduce that audit from these curve artifacts alone",
        },
        "shared_epoch_window": [0, common_end],
        "methylgpt_full_epoch_window": [0, int(gpt["epoch"].max())],
        "metrics": metrics,
        "display_note": "the later MethylGPT trajectory is shown because its full-run optimum "
                        "occurs after the shared window",
        "why_not_loss": "the implementations use differently scaled loss definitions; "
                        "MAE and MedAE are comparable and expressed in years",
    }, indent=2))


if __name__ == "__main__":
    main()
