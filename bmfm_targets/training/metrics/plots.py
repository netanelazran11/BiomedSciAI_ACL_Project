"""
Plots that are generated in the course of training.

Generally based on the collected labels and predictions of a validation run.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb


def make_predictions_gt_density_plot(
    predictions_df: pd.DataFrame,
    predicted_label="predicted_expressions",
    gt_label="label_expressions",
    kind="hist",
    include_x_y_line=True,
):
    sb.set_context("paper")
    g = sb.jointplot(
        data=predictions_df,
        y=predicted_label,
        x=gt_label,
        height=8,
        kind=kind,
        color="blue",
        marginal_kws={"bins": 40, "fill": True},
    )
    if include_x_y_line:
        x_min, x_max = (
            predictions_df[predicted_label].min(),
            predictions_df[predicted_label].max(),
        )
        y_min, y_max = predictions_df[gt_label].min(), predictions_df[gt_label].max()

        min_val = min(x_min, y_min)
        max_val = max(x_max, y_max)

        g.ax_joint.plot(
            [min_val, max_val],
            [min_val, max_val],
            color="gray",
            linestyle="--",
            linewidth=1,
        )

    g.set_axis_labels(gt_label, predicted_label, fontsize=12)
    plt.subplots_adjust(top=0.9)
    return plt.gcf()


def make_top20_deg_perturbation_plot(
    predictions_top20: pd.DataFrame,
    mean_control: pd.Series,
    double_boxplot=False,
    title="",
):
    predictions_top20 = predictions_top20.rename(columns={"input_genes": "gene"})
    predictions_top20 = predictions_top20.set_index("gene", append=True)[
        ["predicted_expressions", "label_expressions"]
    ]

    # index names must match for `.sub` broadcast to work
    mean_control.index.name = "gene"
    top20_genes_delta = predictions_top20.sub(mean_control, axis=0).reset_index("gene")

    sb.set_context("talk")
    plt.figure(figsize=(18, 6))
    if double_boxplot:
        melted_df = pd.melt(
            top20_genes_delta,
            id_vars=["gene"],
            value_vars=["predicted_expressions", "label_expressions"],
            var_name="expression_type",
            value_name="expression_value",
        )
        sb.boxplot(
            data=melted_df, x="gene", y="expression_value", hue="expression_type"
        )
    else:
        sb.boxplot(
            data=top20_genes_delta, x="gene", y="label_expressions", color="lightblue"
        )

    means = top20_genes_delta.groupby("gene")["predicted_expressions"].mean()
    plt.plot(means.index, means.values, "o", color="red", label="Predicted Mean")
    plt.legend(loc="best")
    plt.axhline(0, ls="--", color="green")
    plt.suptitle(title)
    plt.xticks(rotation=90)
    plt.tight_layout()
    return plt.gcf()
