import logging
from collections.abc import Mapping
from itertools import islice
from typing import Any

import pytorch_lightning as pl
import torch
from litdata import StreamingDataLoader, StreamingDataset

from bmfm_targets.training.data_module import DataModule

from .serialization import deserialize_dataloader_states, serialize_dataloader_states

logger = logging.getLogger(__name__)


class StreamingDataModule(DataModule):
    """
    PyTorch Lightning DataModule for scRNA datasets with option to save dataloader state.

    Attributes
    ----------
        dataset_kwargs: Keyword arguments for dataset
        tokenizer (MultiFieldTokenizer): Tokenizer to use
        fields (list[FieldInfo]): List of FieldInfo objects
        batch_size (int, optional): Batch size. Defaults to 32.
        num_workers (int, optional): Number of workers for DataLoader. Defaults to 1.
        max_length (int, optional): Maximum length of input sequences. Defaults to 512.
        padding (PaddingStrategy, optional): Padding strategy. Defaults to True. Available options: PaddingStrategy.MAX_LENGTH, PaddingStrategy.LONGEST.
        truncation (TruncationStrategy, optional): Truncation strategy. Defaults to TruncationStrategy.ONLY_FIRST. Available options: TruncationStrategy.ONLY_FIRST, TruncationStrategy.ONLY_SECOND, TruncationStrategy.LONGEST_FIRST, TruncationStrategy.LONGEST_SECOND.
        save_state (bool): flag to save state of the dataloader into checkpoint
        checkpoint (str: None): File name of checkpoint to laod the state from
    """

    state_entry_name = "datamodule_state"
    DATASET_FACTORY: type[StreamingDataset] = None

    def __init__(
        self,
        *args,
        save_state: bool = False,
        checkpoint: str | None = None,
        limit_dataset_batches: int | Mapping[str, int] | None = None,
        **kwargs,
    ):
        if "limit_dataset_samples" in kwargs:
            raise ValueError(
                "Streaming data module does not support limit_dataset_samples, use limit_dataset_batches instead."
            )

        super().__init__(*args, **kwargs)
        self.save_state = save_state
        self.checkpoint = checkpoint
        self.requires_distributed = True
        self.limit_dataset_batches = limit_dataset_batches

    def prepare_data(self) -> None:
        return

    def _prepare_dataset_kwargs(self):
        final_dataset_kwargs = {}
        if self.dataset_kwargs:
            final_dataset_kwargs = {**self.dataset_kwargs}
        if self.label_columns:
            final_dataset_kwargs["label_columns"] = [
                label.label_column_name
                for label in self.label_columns
                if not label.is_regression_label
            ]
            final_dataset_kwargs["regression_label_columns"] = [
                label.label_column_name
                for label in self.label_columns
                if label.is_regression_label
            ]
        return final_dataset_kwargs

    def setup(self, stage: str = None) -> None:
        self.dataset_kwargs = self._prepare_dataset_kwargs()
        if any(field.vocab_update_strategy == "dynamic" for field in self.fields):
            raise NotImplementedError(
                "Dynamic vocabulary updates is not supported for streaming dataset."
            )

        if stage == "fit" or stage is None:
            # stage predict should be removed here.
            # This is a temporary fix for how the current predict dataloaders are implemented
            self.train_dataset = self.DATASET_FACTORY(
                **self.dataset_kwargs,
                split="train",
                shuffle=self.shuffle,
            )
            self.dev_dataset = self.DATASET_FACTORY(
                **self.dataset_kwargs,
                split="dev",
                shuffle=self.shuffle,
            )
        if stage == "validate":
            self.dev_dataset = self.DATASET_FACTORY(
                **self.dataset_kwargs,
                split="dev",
                shuffle=self.shuffle,
            )
        if stage == "test":
            self.test_dataset = self.DATASET_FACTORY(
                **self.dataset_kwargs,
                split="test",
                shuffle=self.shuffle,
            )

    def get_dataloader_state_from_checkpoint(self):
        if not self.checkpoint:
            return None
        state = torch.load(self.checkpoint, weights_only=False)
        state = deserialize_dataloader_states(state[self.state_entry_name])
        return state

    def get_trainer_callbacks(self) -> list:
        callbacks = super().get_trainer_callbacks()
        if self.save_state:
            callback = SaveStateCallback(self)
            callbacks.append(callback)
        return callbacks

    def get_train_dataloader_state(self) -> dict:
        return self._train_dataloader.state_dict()

    def train_dataloader(self) -> StreamingDataLoader:
        """
        Returns a DataLoader for training.

        Returns
        -------
            DataLoader: DataLoader for training
        """
        self._train_dataloader = StreamingDataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
        state = self.get_dataloader_state_from_checkpoint()
        if state:
            self._train_dataloader.load_state_dict(state)
        return self._train_dataloader

    def val_dataloader(self) -> StreamingDataLoader:
        """
        Returns a DataLoader for validation.

        Returns
        -------
            DataLoader: DataLoader for validation
        """
        dataloader = StreamingDataLoader(
            self.dev_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            shuffle=True,
        )
        if self._dataset_batch_limit("dev"):
            return LimitBatchDataloader(dataloader, self._dataset_batch_limit("dev"))
        else:
            return dataloader

    def test_dataloader(self) -> StreamingDataLoader:
        """
        Returns a DataLoader for testing.

        Returns
        -------
            DataLoader: DataLoader for testing
        """
        return StreamingDataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def _dataset_batch_limit(self, dataset_split):
        if isinstance(self.limit_dataset_batches, Mapping):
            return self.limit_dataset_batches.get(dataset_split, None)
        return self.limit_dataset_batches


class SaveStateCallback(pl.Callback):
    def __init__(self, datamodule: StreamingDataModule) -> None:
        super().__init__()
        self.datamodule = datamodule

    def on_save_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: dict[str, Any],
    ) -> None:
        state = self.datamodule.get_train_dataloader_state()
        state = serialize_dataloader_states(state, pl_module.device)
        checkpoint[StreamingDataModule.state_entry_name] = state


class LimitBatchDataloader:
    """Similar to itertools batched iterator but splits an iterator into a chain of iterators."""

    def __init__(self, base_dataloader, limit_batches):
        self.base_dataloader = base_dataloader
        self.limit_batches = limit_batches

    def __iter__(self):
        return iter(islice(self.base_dataloader, self.limit_batches))
