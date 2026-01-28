"""
BMFM Methylation Encoder Package
================================

A wrapper around the original BMFM-RNA SCBertModel architecture
for DNA methylation age prediction.

This package uses the ORIGINAL bmfm_targets code (SCBertModel, SCBertConfig)
with configuration adapted for methylation data:
- CpG site IDs (discrete tokens)
- Beta values (continuous values 0-1)

Components:
    - create_methylation_config: Creates SCBertConfig for methylation data
    - MethylationEncoder: Wrapper around original SCBertModel
    - MethylationAgeModel: Complete model with regression head
    - MethylationDataset: Dataset for h5ad files
    - MethylationAgeLightningModule: PyTorch Lightning training module
    - train_methylation_model: High-level training function

Example:
    >>> from bmfm_methylation import create_methylation_config, MethylationAgeModel
    >>> config = create_methylation_config(num_cpg_sites=8000)
    >>> model = MethylationAgeModel(config)

    >>> # Training
    >>> from bmfm_methylation import train_methylation_model
    >>> trainer, module = train_methylation_model(
    ...     data_path="methylation.h5ad",
    ...     train_split="train",
    ...     val_split="valid"
    ... )
"""

# Configuration - wraps original bmfm_targets.config.SCBertConfig
from .config import create_methylation_config, BMFMConfig, SCBertConfig, FieldInfo

# Model - wraps original bmfm_targets SCBertModel
from .model import MethylationEncoder, MethylationAgeModel

# Dataset - standalone data loading for h5ad
from .dataset import MethylationDataset, create_data_loaders

# Optional Lightning components
try:
    from .lightning_module import MethylationAgeLightningModule
    from .trainer import create_trainer, train_methylation_model
    _HAS_LIGHTNING = True
except ImportError:
    MethylationAgeLightningModule = None
    create_trainer = None
    train_methylation_model = None
    _HAS_LIGHTNING = False


__all__ = [
    # Config (wraps bmfm_targets)
    "create_methylation_config",
    "BMFMConfig",  # Alias for create_methylation_config
    "SCBertConfig",  # Re-export for type hints
    "FieldInfo",

    # Model (wraps bmfm_targets SCBertModel)
    "MethylationEncoder",
    "MethylationAgeModel",

    # Dataset (standalone)
    "MethylationDataset",
    "create_data_loaders",

    # Lightning (optional)
    "MethylationAgeLightningModule",
    "create_trainer",
    "train_methylation_model",
]

__version__ = "0.1.0"
