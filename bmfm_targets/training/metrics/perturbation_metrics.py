"""
Functions for calculating perturbation-specific metrics.

Perturbation metrics are different because they involve various levels of aggregation
across samples.

Typically the predictions for a given perturbation are averaged into one group prediction
and compared with the ground-truth prediction.

Also, the predictions are compared against the average, unperturbed expression.
"""

import pandas as pd
from scanpy import AnnData


def get_aggregated_perturbation_metrics(
    grouped_df: pd.DataFrame, mean_expressions: pd.DataFrame, perturbation_group: str
):
    this_group = grouped_df.merge(
        mean_expressions, how="outer", left_index=True, right_index=True
    ).fillna(0)
    this_group = this_group.drop(columns=["is_perturbed"])
    this_group_delta = this_group.subtract(this_group["Control"], axis=0)
    this_group_delta.pop("Control")

    corrs = this_group.corr(method="pearson")
    corrs_delta = this_group_delta.corr(method="pearson")

    agg_pcc = corrs.loc["predicted_expressions", perturbation_group]
    delta_agg_pcc = corrs_delta.loc["predicted_expressions", perturbation_group]
    baseline_agg_pcc = corrs.loc["Control", perturbation_group]

    return {
        "agg_pcc": agg_pcc,
        "delta_agg_pcc": delta_agg_pcc,
        "baseline_agg_pcc": baseline_agg_pcc,
    }


def get_group_average_expressions(preds_df: pd.DataFrame, perturbed_id: int = 6):
    # identify perturbed genes per sample and add column for grouping
    preds_df["perturbed_gene"] = preds_df.apply(
        lambda row: row["input_genes"] if row["is_perturbed"] == perturbed_id else None,
        axis=1,
    )
    perturbed_sets = preds_df.groupby("sample_id")["perturbed_gene"].apply(
        lambda x: "_".join(sorted(frozenset(i for i in x.dropna())))
    )
    preds_df["perturbed_set"] = (
        preds_df.reset_index()["sample_id"]
        .map(perturbed_sets)
        .to_frame()
        .set_index(preds_df.index)
    )
    preds_df.pop("perturbed_gene")

    # can't use .mean() because we must calculate
    # sum / n_counts (not n nonzero occurrences )
    sample_counts = (
        preds_df.assign(sample_id_x=preds_df.index)
        .groupby("perturbed_set")["sample_id_x"]
        .nunique()
    )
    grouped = preds_df.groupby(["perturbed_set", "input_genes"]).sum(numeric_only=True)
    sample_counts_aligned = grouped.index.get_level_values("perturbed_set").map(
        sample_counts
    )
    grouped = grouped.div(sample_counts_aligned, axis=0)
    return grouped


def get_mean_expressions(group_means_ad: AnnData):
    mean_expressions = group_means_ad.to_df().T
    all_zero = (mean_expressions == 0).all(axis=1)
    mean_expressions = mean_expressions[~all_zero]
    return mean_expressions
