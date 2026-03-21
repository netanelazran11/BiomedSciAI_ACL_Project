"""
MethylationDataModule21k — drop-in replacement for MethylationDataModule
that handles datasets without a pre-defined 'valid' split.

The only difference from data_module.py:
  If the requested val_split (default "valid") contains 0 samples,
  we automatically carve out 10% of the training set as validation
  (stratified by age bins so the age distribution is preserved).

Everything else — collators, tokenizers, WCEDCollator — is identical.
"""

import logging
import numpy as np
from typing import Optional

from bmfm_targets.tokenization.multifield_instance import MultiFieldInstance
from bmfm_methylation.data_module import MethylationDataModule, MethylationDataset

logger = logging.getLogger(__name__)


class MethylationDataModule21k(MethylationDataModule):
    """
    Same as MethylationDataModule, but auto-creates a validation split
    when the h5ad file has no 'valid' label in the split column.

    Used for datasets like altumage_21k_combined.h5ad which only have
    'train' and 'test' splits.
    """

    def setup(self, stage: Optional[str] = None):
        # Run the original setup (builds train/val/test datasets + collator)
        super().setup(stage)

        # If validation ended up empty, auto-split from train
        if (
            stage in ("fit", None)
            and self.val_dataset is not None
            and len(self.val_dataset) == 0
        ):
            logger.warning(
                f"No '{self.val_split}' split found in data. "
                f"Auto-creating validation set from 10 %% of training data "
                f"(stratified by age, seed=42)."
            )
            self._auto_split_val(val_frac=0.10, seed=42)

    # ------------------------------------------------------------------
    def _auto_split_val(self, val_frac: float = 0.10, seed: int = 42):
        """
        Move val_frac of training samples into self.val_dataset.

        Stratification: samples are binned into 10-year age groups and
        the fraction is taken from each bin, so the age distribution
        in validation mirrors training.
        """
        train_adata = self.train_dataset.adata
        ages = train_adata.obs[self.train_dataset.age_column].values.astype(float)
        n_total = len(ages)

        # 10-year age bins for stratification
        age_bins = (np.clip(ages, 0, 110) // 10).astype(int)

        rng = np.random.default_rng(seed)
        val_indices = []

        for bin_id in np.unique(age_bins):
            bin_mask = age_bins == bin_id
            bin_idx = np.where(bin_mask)[0]
            n_val = max(1, int(len(bin_idx) * val_frac))
            chosen = rng.choice(bin_idx, size=n_val, replace=False)
            val_indices.extend(chosen.tolist())

        val_indices = sorted(val_indices)
        train_indices = [i for i in range(n_total) if i not in set(val_indices)]

        # Build sub-AnnData objects
        val_adata   = train_adata[val_indices].copy()
        new_train_adata = train_adata[train_indices].copy()

        # ── Rebuild train_dataset ──────────────────────────────────────
        new_train = _AnnDataDataset(
            adata=new_train_adata,
            age_column=self.train_dataset.age_column,
            normalize_age=self.train_dataset.normalize_age,
        )
        # ── Rebuild val_dataset ────────────────────────────────────────
        new_val = _AnnDataDataset(
            adata=val_adata,
            age_column=self.train_dataset.age_column,
            normalize_age=self.train_dataset.normalize_age,
        )

        # Share age normalisation from full training split
        new_train.age_mean = self.train_dataset.age_mean
        new_train.age_std  = self.train_dataset.age_std
        new_val.age_mean   = self.train_dataset.age_mean
        new_val.age_std    = self.train_dataset.age_std

        self.train_dataset = new_train
        self.val_dataset   = new_val

        logger.info(
            f"Auto-split: train={len(self.train_dataset)}, "
            f"val={len(self.val_dataset)} "
            f"(removed from {n_total} original train samples)"
        )


# ---------------------------------------------------------------------------
# Helper: wrap an already-loaded AnnData as a MethylationDataset-like object
# without re-reading from disk.
# ---------------------------------------------------------------------------


class _AnnDataDataset:
    """Lightweight dataset backed by an in-memory AnnData slice."""

    def __init__(self, adata, age_column: str = "age", normalize_age: bool = True):
        self.adata = adata
        self.age_column = age_column
        self.normalize_age = normalize_age
        self.cpg_sites = list(adata.var_names)
        self.num_cpg_sites = len(self.cpg_sites)

        if age_column in adata.obs.columns:
            self.ages = adata.obs[age_column].values.astype(np.float32)
            self.has_ages = True
        else:
            self.ages = np.zeros(len(adata), dtype=np.float32)
            self.has_ages = False

        if self.has_ages and normalize_age:
            self.age_mean = float(np.mean(self.ages))
            self.age_std  = float(np.std(self.ages))
            if self.age_std == 0:
                self.age_std = 1.0
        else:
            self.age_mean = 0.0
            self.age_std  = 1.0

    def __len__(self) -> int:
        return len(self.adata)

    def __getitem__(self, idx: int) -> MultiFieldInstance:
        beta_values = self.adata.X[idx]
        if hasattr(beta_values, "toarray"):
            beta_values = beta_values.toarray().flatten()
        beta_values = beta_values.astype(np.float32)
        valid_mask = np.isfinite(beta_values)

        age = self.ages[idx]
        if self.normalize_age:
            age = (age - self.age_mean) / self.age_std

        return MultiFieldInstance(
            data={
                "beta_values": beta_values.tolist(),
                "valid_mask": valid_mask.tolist(),
            },
            metadata={
                "labels": float(age),
                "cell_name": str(idx),
            },
        )
