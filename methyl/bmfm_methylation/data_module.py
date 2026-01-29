"""
Methylation Data Module - PyTorch Lightning DataModule for methylation data

This module follows the BMFM DataModule pattern for loading methylation data
from h5ad files and preparing it for training.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Union

import numpy as np
import pytorch_lightning as pl
import scanpy as sc
import torch
from torch.utils.data import DataLoader, Dataset
from transformers.tokenization_utils_base import PaddingStrategy, TruncationStrategy

from bmfm_targets.config import FieldInfo, LabelColumnInfo
from bmfm_targets.tokenization import MultiFieldCollator, MultiFieldTokenizer
from bmfm_targets.tokenization.multifield_instance import MultiFieldInstance
from bmfm_targets.training.masking import MaskingStrategy

logger = logging.getLogger(__name__)


class MethylationDataset(Dataset):
    """
    Dataset for methylation data from h5ad files.

    Attributes:
        adata: AnnData object containing methylation data
        split: Optional split name to filter (train/valid/test)
        age_column: Column name for age labels
        split_column: Column name for split information
    """

    def __init__(
        self,
        h5ad_path: str,
        split: Optional[str] = None,
        age_column: str = "age",
        split_column: str = "split",
        normalize_age: bool = True,
    ):
        """
        Args:
            h5ad_path: Path to h5ad file
            split: Optional split to filter (train/valid/test)
            age_column: Column name for age in obs
            split_column: Column name for split in obs
            normalize_age: Whether to normalize age values
        """
        self.h5ad_path = h5ad_path
        self.split = split
        self.age_column = age_column
        self.split_column = split_column
        self.normalize_age = normalize_age

        # Load data
        self.adata = sc.read_h5ad(h5ad_path)

        # Filter by split if specified
        if split is not None and split_column in self.adata.obs.columns:
            mask = self.adata.obs[split_column] == split
            self.adata = self.adata[mask].copy()

        # Get CpG site names
        self.cpg_sites = list(self.adata.var_names)
        self.num_cpg_sites = len(self.cpg_sites)

        # Get age values
        if age_column in self.adata.obs.columns:
            self.ages = self.adata.obs[age_column].values.astype(np.float32)
            self.has_ages = True
        else:
            self.ages = np.zeros(len(self.adata), dtype=np.float32)
            self.has_ages = False

        # Compute normalization statistics
        if self.has_ages and normalize_age:
            self.age_mean = float(np.mean(self.ages))
            self.age_std = float(np.std(self.ages))
            if self.age_std == 0:
                self.age_std = 1.0
        else:
            self.age_mean = 0.0
            self.age_std = 1.0

        logger.info(f"Loaded {len(self.adata)} samples with {self.num_cpg_sites} CpG sites")
        if split:
            logger.info(f"Split: {split}")

    def __len__(self) -> int:
        return len(self.adata)

    def __getitem__(self, idx: int) -> MultiFieldInstance:
        """
        Get a sample as a MultiFieldInstance.

        Returns:
            MultiFieldInstance with cpg_sites (names) and beta_values (continuous)
        """
        # Get beta values
        beta_values = self.adata.X[idx]
        if hasattr(beta_values, 'toarray'):
            beta_values = beta_values.toarray().flatten()
        beta_values = beta_values.astype(np.float32)

        # Get age (normalized if requested)
        age = self.ages[idx]
        if self.normalize_age:
            age = (age - self.age_mean) / self.age_std

        # Create MultiFieldInstance with data and metadata
        # Labels go in metadata, not data (data is for model inputs)
        mfi = MultiFieldInstance(
            data={
                "cpg_sites": self.cpg_sites,  # CpG site names/indices
                "beta_values": beta_values.tolist(),  # Continuous values
            },
            metadata={
                "labels": float(age),
                "cell_name": str(idx),  # Sample identifier
            }
        )

        return mfi


class MethylationDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for methylation data.

    Follows the BMFM DataModule pattern.
    """

    def __init__(
        self,
        tokenizer: MultiFieldTokenizer,
        fields: List[FieldInfo],
        label_columns: Optional[List[LabelColumnInfo]] = None,
        h5ad_path: Optional[str] = None,
        train_split: str = "train",
        val_split: str = "valid",
        test_split: str = "test",
        age_column: str = "age",
        split_column: str = "split",
        batch_size: int = 32,
        num_workers: int = 4,
        max_length: int = 8002,  # 8000 CpG + [CLS] + [SEP]
        padding: Union[PaddingStrategy, str, bool] = "max_length",
        truncation: Union[TruncationStrategy, bool] = True,
        pad_to_multiple_of: int = 2,  # 8002 is divisible by 2
        mlm: bool = False,
        change_ratio: float = 0.15,
        mask_ratio: float = 0.8,
        switch_ratio: float = 0.1,
        masking_strategy: Optional[MaskingStrategy] = None,
        normalize_age: bool = True,
        collation_strategy: Literal[
            "language_modeling",
            "sequence_classification",
        ] = "sequence_classification",
    ):
        """
        Args:
            tokenizer: MultiFieldTokenizer for methylation data
            fields: List of FieldInfo configurations
            label_columns: Optional label column configurations
            h5ad_path: Path to h5ad file
            train_split: Name of training split
            val_split: Name of validation split
            test_split: Name of test split
            age_column: Column name for age in obs
            split_column: Column name for split in obs
            batch_size: Batch size for DataLoader
            num_workers: Number of workers for DataLoader
            max_length: Maximum sequence length
            padding: Padding strategy
            truncation: Truncation strategy
            pad_to_multiple_of: Pad to multiple of this value
            mlm: Whether to use masked language modeling
            change_ratio: Ratio of tokens to mask (for MLM)
            mask_ratio: Ratio of masked tokens to replace with [MASK]
            switch_ratio: Ratio of masked tokens to replace with random token
            masking_strategy: Custom masking strategy
            normalize_age: Whether to normalize age values
            collation_strategy: Collation strategy
        """
        super().__init__()

        self.tokenizer = tokenizer
        self.fields = fields
        self.label_columns = label_columns
        self.h5ad_path = h5ad_path
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.age_column = age_column
        self.split_column = split_column
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.pad_to_multiple_of = pad_to_multiple_of
        self.mlm = mlm
        self.change_ratio = change_ratio
        self.mask_ratio = mask_ratio
        self.switch_ratio = switch_ratio
        self.masking_strategy = masking_strategy
        self.normalize_age = normalize_age
        self.collation_strategy = collation_strategy

        # Will be set during setup
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.collator = None

    def setup(self, stage: Optional[str] = None):
        """Set up datasets for each stage."""
        if self.h5ad_path is None:
            raise ValueError("h5ad_path must be provided")

        # Create datasets
        if stage == "fit" or stage is None:
            self.train_dataset = MethylationDataset(
                h5ad_path=self.h5ad_path,
                split=self.train_split,
                age_column=self.age_column,
                split_column=self.split_column,
                normalize_age=self.normalize_age,
            )
            self.val_dataset = MethylationDataset(
                h5ad_path=self.h5ad_path,
                split=self.val_split,
                age_column=self.age_column,
                split_column=self.split_column,
                normalize_age=self.normalize_age,
            )
            # Share normalization stats from training
            self.val_dataset.age_mean = self.train_dataset.age_mean
            self.val_dataset.age_std = self.train_dataset.age_std

        if stage == "test" or stage is None:
            self.test_dataset = MethylationDataset(
                h5ad_path=self.h5ad_path,
                split=self.test_split,
                age_column=self.age_column,
                split_column=self.split_column,
                normalize_age=self.normalize_age,
            )
            if self.train_dataset is not None:
                self.test_dataset.age_mean = self.train_dataset.age_mean
                self.test_dataset.age_std = self.train_dataset.age_std

        # Create masker if MLM is enabled
        masker = None
        if self.mlm:
            from bmfm_targets.training.masking import Masker
            masker = Masker(
                tokenizer=self.tokenizer,
                change_ratio=self.change_ratio,
                mask_ratio=self.mask_ratio,
                switch_ratio=self.switch_ratio,
            )

        # Create collator
        self.collator = MultiFieldCollator(
            tokenizer=self.tokenizer,
            fields=self.fields,
            max_length=self.max_length,
            padding=self.padding,
            truncation=self.truncation,
            pad_to_multiple_of=self.pad_to_multiple_of,
            masker=masker,
            collation_strategy=self.collation_strategy,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collator,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collator,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collator,
            pin_memory=True,
        )

    @property
    def age_mean(self) -> float:
        if self.train_dataset is not None:
            return self.train_dataset.age_mean
        return 0.0

    @property
    def age_std(self) -> float:
        if self.train_dataset is not None:
            return self.train_dataset.age_std
        return 1.0
