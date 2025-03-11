import logging
import pathlib
from typing import Any

import pandas as pd
import pytorch_lightning as pl
import transformers
from lightning_utilities.core.rank_zero import rank_zero_only
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities import types as pl_types

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
