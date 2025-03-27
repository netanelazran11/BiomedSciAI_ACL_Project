import logging
import pathlib
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scanpy as sc
import scib.metrics.metrics as scm
import transformers
from clearml.logger import Logger
from lightning_utilities.core.rank_zero import rank_zero_only
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities import types as pl_types

from bmfm_targets.datasets.datasets_utils import random_subsampling
from bmfm_targets.training.masking import MaskingStrategy

logger = logging.getLogger(__name__)


class SavePretrainedModelCallback(pl.Callback):
    def __init__(
        self,
        save_dir: pathlib.Path,
        tokenizer: transformers.PreTrainedTokenizer,
        epoch_period: int | None = 1,
        step_period: int | None = None,
    ):
        self.epoch_period = epoch_period
        self.step_period = step_period
        self.save_dir = save_dir
        self.tokenizer = tokenizer

    @rank_zero_only
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module,
        outputs: pl_types.STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ):
        step = trainer.global_step
        if self.step_period is not None and step % self.step_period == 0:
            step_save_dir = self.save_dir / f"step_{step}"
            pl_module.save_transformer(step_save_dir, self.tokenizer)

    @rank_zero_only
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module):
        epoch = trainer.current_epoch
        if self.epoch_period is not None and epoch % self.epoch_period == 0:
            epoch_save_dir = self.save_dir / f"epoch_{trainer.current_epoch}"
            pl_module.save_transformer(epoch_save_dir, self.tokenizer)


class InitialCheckpoint(ModelCheckpoint):
    def __init__(self, dirpath, filename="initial.ckpt"):
        super().__init__(dirpath=dirpath, filename=filename, save_top_k=0)
        self.filename = filename

    def on_train_start(self, trainer, pl_module):
        logger.info("saving initial embedding")
        trainer.save_checkpoint(self.dirpath + "/" + self.filename)


class TokenErrorUpdateCallback(pl.Callback):
    """
    Callback for updating token errors.

    Added automatically when DataModule initiated with `TokenProbabilityMaskingStrategy`.
    Requires the token level errors to be calculated and saved to the trainer's
    `token_level_errors` attribute.
    """

    def __init__(self, error_column_name="gene_err", n_bins=100) -> None:
        self.error_column_name = error_column_name
        self.n_bins = n_bins
        super().__init__()

    def on_validation_end(self, trainer, pl_module):
        # Get token errors from the LightningModule
        if "genes" not in pl_module.token_level_errors:
            logger.warning(
                "No gene level errors available to update masking. "
                "No adaptive masking will take place."
            )
            return
        errors = pl_module.token_level_errors["genes"]
        if errors is not None:
            # Compute token probabilities
            token_probs = self.calculate_token_probs(errors)
            masking_strategy = self.get_masking_strategy(trainer)
            # Update masking strategy in the DataModule

            if hasattr(masking_strategy, "update_token_masking_probs"):
                masking_strategy.update_token_masking_probs(token_probs)
            else:
                raise AttributeError(
                    "DataModule does not have valid masking_strategy attribute."
                    " This callback should only be added with a valid masking strategy."
                )

    def get_masking_strategy(self, trainer: pl.Trainer) -> MaskingStrategy | None:
        """
        Load the masking_strategy object from the datamodule or dataloader.

        Depending on how Trainer.fit() is called, there will be either a datamodule
        or dataloaders. The masking_strategy object is shared between them, but where
        it is stored needs to be deduced.

        Args:
        ----
            trainer (pl.Trainer): the lightning trainer

        Raises:
        ------
            ValueError: if there are no valid dataloaders at all. This would only happen
              if this function is called outside the fit/test loop.

        Returns:
        -------
            MaskingStrategy | None: the masking strategy or None if there is no masking
              strategy defined

        """
        if getattr(trainer, "datamodule", None):
            return getattr(trainer.datamodule, "masking_strategy", None)
        if getattr(trainer, "train_dataloader", None):
            collator = trainer.train_dataloader.collate_fn
        elif getattr(trainer, "test_dataloader", None):
            collator = trainer.test_dataloader.collate_fn
        if hasattr(collator, "masker"):
            return getattr(collator.masker, "masking_strategy", None)
        else:
            raise ValueError("No data module or dataloaders found")

    def calculate_token_probs(self, errors: pd.DataFrame) -> dict[str, float]:
        """
        Calculate token masking probabilities based on token error dataframe.

        This makes use of the `error_column_name` attribute to choose which error
        definition to use to calculate masking probabilities. It transforms the errors
        using a quantile transform and shifts the values from 1/n_bins to 1 so that
        nothing has zero probability.

        Args:
        ----
          errors (pd.DataFrame): the token_level error dataframe as produced, eg, by
            `get_gene_level_expression_error`.

        Returns:
        -------
            dict[str,float]: tokens and masking probabilities. The probabilities do not
              need to be valid probabilities, they will be rescaled by the masking
              function.

        """
        error_to_use = errors[self.error_column_name]
        token_probs = pd.cut(error_to_use, bins=self.n_bins, labels=False)
        # we don't want any zeros and we want normalized to 1
        token_probs = (token_probs + 1) / (self.n_bins + 1)

        return token_probs.to_dict()


class BatchIntegrationCallback(pl.Callback):
    def __init__(self, batch_column_name=None):
        super().__init__()
        self.batch_column_name = batch_column_name

    def on_predict_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self.cl = Logger.current_logger()
        if self.cl:
            self.execute_batch_integration(trainer)

    def execute_batch_integration(self, trainer):
        adata_emb = self.get_adata_with_embeddings(trainer)
        self.batch_column_name = self.verify_batch_column_name(trainer)
        self.report_batch_integartion_to_clearml(adata_emb)

    def verify_batch_column_name(self, trainer):
        self.target_column_name = trainer.datamodule.label_columns[0].label_column_name
        if self.batch_column_name is None:
            return self.target_column_name
        else:
            return self.batch_column_name

    def get_adata_with_embeddings(self, trainer):
        batch_preds = trainer.predict_loop.predictions

        def _join_batches(k):
            return np.concatenate([d[k] for d in batch_preds], axis=0)

        predictions = {k: _join_batches(k) for k in batch_preds[0].keys()}
        adata_orig = trainer.datamodule.predict_dataset.processed_data
        adata_emb = self.add_embed_to_obsm(adata_orig, predictions)
        return adata_emb

    def report_batch_integartion_to_clearml(self, adata_emb):
        batch_int_df = self.generate_table_batch_integration(adata_emb)
        self.cl.report_table(
            title="Batch Integration",
            series="Batch Integration",
            table_plot=batch_int_df.T,
        )
        self.cl.report_single_value(
            name="Average Bio",
            value=float(batch_int_df.loc[:, "Avg_bio"]),
        )
        self.cl.report_single_value(
            name="Average Batch",
            value=float(batch_int_df.loc[:, "Avg_batch"]),
        )

        fig = self.generate_fig_batch_integration(adata_emb)
        self.cl.report_matplotlib_figure(
            title="UMAP Visualization",
            series="umap_plot",
            figure=fig,
            report_image=True,
        )
        plt.close(fig)

    def generate_fig_batch_integration(self, adata_emb):
        target_col = self.target_column_name
        batch_col = self.batch_column_name

        sampling_adata_emb = random_subsampling(
            adata=adata_emb,
            n_samples=min((10000, adata_emb.obs.shape[0])),
            shuffle=False,
        )
        sc.pp.neighbors(sampling_adata_emb, use_rep="X_emb")
        sc.tl.umap(sampling_adata_emb)
        sampling_adata_emb.obs[batch_col] = sampling_adata_emb.obs[batch_col].astype(
            "category"
        )
        fig, axs = plt.subplots(2, 1, figsize=(15, 15))
        colors = [
            target_col,
            batch_col,
        ]
        titles = [
            f"Targets embeddings: {target_col} ",
            f"Batch embeddings: {batch_col}",
        ]

        # Plot each UMAP separately
        for i, ax in enumerate(axs):
            sc.pl.umap(
                sampling_adata_emb,
                color=colors[i],
                frameon=False,
                title=titles[i],
                ax=ax,
                show=False,
            )
        plt.tight_layout()
        return fig

    def generate_table_batch_integration(self, adata_emb):
        batch_col = self.batch_column_name
        label_col = self.target_column_name
        sc.pp.neighbors(adata_emb, use_rep="X_emb")
        sc.tl.umap(adata_emb)
        batch_int = scm(
            adata_emb,
            adata_int=adata_emb,
            batch_key=f"{batch_col}",
            label_key=f"{label_col}",
            embed="X_emb",
            isolated_labels_asw_=False,
            silhouette_=True,
            hvg_score_=False,
            graph_conn_=True,
            pcr_=False,
            isolated_labels_f1_=False,
            trajectory_=False,
            nmi_=True,
            ari_=True,
            cell_cycle_=False,
            kBET_=False,
            ilisi_=False,
            clisi_=False,
        )
        batch_int_dict = batch_int[0].to_dict()

        batch_int_dict["avg_bio"] = np.mean(
            [
                batch_int_dict["NMI_cluster/label"],
                batch_int_dict["ARI_cluster/label"],
                batch_int_dict["ASW_label"],
            ]
        )

        batch_int_dict["avg_batch"] = np.mean(
            [
                batch_int_dict["graph_conn"],
                batch_int_dict["ASW_label/batch"],
            ]
        )
        batch_int_dict = {k: v for k, v in batch_int_dict.items() if not np.isnan(v)}
        batch_int_df = pd.DataFrame(
            {k.capitalize(): [np.round(v, 2)] for k, v in batch_int_dict.items()}
        )
        batch_int_df = batch_int_df.rename(
            columns={
                "Nmi_cluster/label": f"NMI_cluster_by_{label_col}_(bio)",
                "Ari_cluster/label": f"ARI_cluster_by{label_col}_(bio)",
                "Asw_label": f"ASW_by_{label_col}_(bio)",
                "Graph_conn": f"graph_conn_by_{batch_col}_(batch)",
                "Asw_label/batch": f"ASW_by_{batch_col}_(batch)",
            }
        )
        return batch_int_df

    def add_embed_to_obsm(self, adata, results):
        adata_emb = adata.copy()
        embeddings = results["embeddings"]

        adata_cell_names = adata_emb.obs_names.values
        dict_cell_names = results["cell_names"]
        name_to_index = {name: idx for idx, name in enumerate(dict_cell_names)}

        aligned_embeddings = np.array(
            [embeddings[name_to_index[name]] for name in adata_cell_names]
        )
        adata_emb.obsm["X_emb"] = aligned_embeddings

        return adata_emb
