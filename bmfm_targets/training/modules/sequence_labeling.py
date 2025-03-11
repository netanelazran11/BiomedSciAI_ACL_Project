import logging

import clearml
import pandas as pd
from matplotlib import pyplot as plt

from ..metrics import loss_handling, perturbation_metrics, plots
from .base import BaseTrainingModule

logger = logging.getLogger(__name__)
DEFAULT_LABEL_SMOOTHING = 0.01


class SequenceLabelingTrainingModule(BaseTrainingModule):
    DEFAULT_METRICS = {"pcc"}
    MODELING_STRATEGY = "sequence_labeling"

    def update_metrics(self, labels, outputs, split):
        # Needs to be updated for auc calculation
        predictions = {}
        gt_labels = {}

        predictions = loss_handling.calculate_predictions(
            self.loss_tasks, outputs.logits
        )
        for output_key in predictions:
            gt_labels[output_key] = labels[output_key].view(
                predictions[output_key].shape[0], -1
            )

            keep_index = gt_labels[output_key] != -100

            assert (
                keep_index.shape == predictions[output_key].shape
            ), f"{keep_index.shape} != {predictions[output_key].shape}"

            predictions[output_key] = predictions[output_key][keep_index]

            assert (
                keep_index.shape == gt_labels[output_key].shape
            ), f"{keep_index.shape} != {gt_labels[output_key].shape}"

            gt_labels[output_key] = gt_labels[output_key][keep_index].to(
                predictions[output_key].dtype
            )

        return self.split_metrics(split)(predictions, gt_labels)

    def _shared_test_val_on_end(self, split: str):
        super()._shared_test_val_on_end(split)
        preds_df = self.prediction_df["label_expressions"]
        self.log_perturbation_specific_metrics(split, preds_df)

    def log_perturbation_specific_metrics(self, split, predictions_df):
        group_means = self.kwargs.get("group_means")
        if group_means is None:
            logger.warning(
                "No group_means found in kwargs cannot do perturbation metrics"
            )
            return
        perturbed_id = self.tokenizer.get_field_vocab("perturbations")["1"]
        group_expressions = perturbation_metrics.get_group_average_expressions(
            predictions_df, perturbed_id
        )
        mean_expressions = perturbation_metrics.get_mean_expressions(
            group_means.processed_data
        )
        top20 = self.kwargs.get("top20_de")
        perturbations = group_expressions.reset_index().perturbed_set.unique()
        agg_metrics_list = []
        agg_metrics_top20_list = []
        for pert in perturbations:
            agg_metrics = perturbation_metrics.get_aggregated_perturbation_metrics(
                group_expressions.loc[pert], mean_expressions, pert
            )
            if len(perturbations) <= 10:
                self.log_aggregate_perturbation_metrics(split, agg_metrics, pert, "")
            agg_metrics["pert"] = pert
            agg_metrics_list.append(agg_metrics)

            if top20:
                # it is possible that some highly variable genes are missing
                # mostly due to small batches and short max_length values
                valid_indices = [
                    g for g in top20[pert] if g in group_expressions.loc[pert].index
                ]
                agg_metrics_top20 = (
                    perturbation_metrics.get_aggregated_perturbation_metrics(
                        group_expressions.loc[pert].loc[valid_indices],
                        mean_expressions.loc[top20[pert]],
                        pert,
                    )
                )
                if len(perturbations) <= 10:
                    self.log_aggregate_perturbation_metrics(
                        split,
                        agg_metrics_top20,
                        pert,
                        "top20_de_",
                    )
                    self.create_and_log_top20_deg_perturbation_plot(
                        split, predictions_df, mean_expressions, top20, pert
                    )
                agg_metrics_top20["pert"] = pert
                agg_metrics_top20_list.append(agg_metrics_top20)

        agg_metrics_df = pd.DataFrame().from_records(agg_metrics_list)
        mean_agg_metrics = agg_metrics_df.drop("pert", axis=1).mean()
        mean_row = pd.DataFrame(
            {"pert": ["mean"], **mean_agg_metrics.to_dict()},
            index=[len(mean_agg_metrics)],
        )
        agg_metrics_df_with_mean = pd.concat(
            [agg_metrics_df, mean_row], ignore_index=True
        )

        self.log_table(
            split, "aggregation metrics per perturbation", agg_metrics_df_with_mean
        )
        self.log_mean_aggregate_perturbation_metrics(agg_metrics_df, split, "")

        agg_metrics_top20_df = pd.DataFrame().from_records(agg_metrics_top20_list)
        mean_agg_metrics_top20 = agg_metrics_top20_df.drop("pert", axis=1).mean()
        mean_row = pd.DataFrame(
            {"pert": ["mean"], **mean_agg_metrics_top20.to_dict()},
            index=[len(agg_metrics_top20_df)],
        )
        agg_metrics_top20_df_with_mean = pd.concat(
            [agg_metrics_top20_df, mean_row], ignore_index=True
        )

        self.log_table(
            split,
            "aggregation metrics top20 differentially expressed genes per perturbation",
            agg_metrics_top20_df_with_mean,
        )
        self.log_mean_aggregate_perturbation_metrics(
            agg_metrics_top20_df, split, "top20_de_"
        )

    def log_mean_aggregate_perturbation_metrics(
        self, agg_metrics_df, split, identifier
    ):
        mean_agg_metrics = agg_metrics_df.drop("pert", axis=1).mean()
        for metric_name, mean_value in mean_agg_metrics.items():
            self.logger.experiment.add_scalar(
                f"{split}/mean_{identifier}{metric_name}", mean_value, self.global_step
            )

    def log_aggregate_perturbation_metrics(self, split, agg_metrics, pert, identifier):
        for k, v in agg_metrics.items():
            self.logger.experiment.add_scalar(
                f"{split}/{pert}_{identifier}{k}", v, self.global_step
            )

    def create_and_log_top20_deg_perturbation_plot(
        self, split, preds_df, mean_expressions, top20, pert
    ):
        top20_df = preds_df[preds_df.input_genes.isin(top20[pert])]
        mean_control = mean_expressions["Control"].loc[top20[pert]]
        fig = plots.make_top20_deg_perturbation_plot(top20_df, mean_control)
        cl = clearml.Logger.current_logger()
        if cl:
            cl.report_matplotlib_figure(
                title=f"{pert} - Difference from Control for Top 20 DEG",
                series=split,
                figure=fig,
                iteration=self.global_step,
            )
        plt.close(fig)
