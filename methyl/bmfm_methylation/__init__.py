"""
BMFM Methylation Encoder Package
================================

A complete wrapper around the original BMFM-RNA SCBertModel architecture
for DNA methylation age prediction, following the PhD student's recommendations:

1. Tokenizer: Creates proper tokenizer for CpG sites
2. Pretraining: MLM pretraining to learn methylation patterns
3. Fine-tuning: Age regression on pretrained model

Training Pipeline:
    Step 1: Create tokenizer from h5ad data
    Step 2: Pretrain with MLM (learn methylation patterns)
    Step 3: Fine-tune for age prediction

Quick Start:
    # Pretraining
    python -m bmfm_methylation.pretrain data_path=methylation.h5ad

    # Fine-tuning
    python -m bmfm_methylation.finetune data_path=methylation.h5ad checkpoint_path=pretrain/best.ckpt

    # Or train from scratch (skip pretraining)
    python -m bmfm_methylation.finetune data_path=methylation.h5ad checkpoint_path=null

Components:
    - Tokenizer: create_methylation_multifield_tokenizer, create_tokenizer_from_h5ad
    - Config: create_methylation_config
    - Data: MethylationDataset, MethylationDataModule
    - Model: MethylationAgeRegressor (uses SCBertModel encoder)
    - Training: pretrain.py (MLM), finetune.py (age regression)
"""

# Configuration - wraps original bmfm_targets.config.SCBertConfig
from .config import create_methylation_config, BMFMConfig

# Tokenizer - creates MultiFieldTokenizer for methylation
from .tokenizer import (
    create_methylation_tokenizer,
    create_methylation_multifield_tokenizer,
    create_tokenizer_from_h5ad,
    create_indexed_tokenizer,
    extract_cpg_sites_from_h5ad,
)

# Data Module - loads h5ad and prepares for training
from .data_module import MethylationDataset, MethylationDataModule

# Re-export from bmfm_targets
try:
    from bmfm_targets.config import SCBertConfig, FieldInfo
    from bmfm_targets.tokenization import MultiFieldTokenizer
    from bmfm_targets.models.predictive.scbert.modeling_scbert import (
        SCBertModel,
        SCBertForMaskedLM,
    )
    _HAS_BMFM = True
except ImportError:
    SCBertConfig = None
    FieldInfo = None
    MultiFieldTokenizer = None
    SCBertModel = None
    SCBertForMaskedLM = None
    _HAS_BMFM = False


__all__ = [
    # Config
    "create_methylation_config",
    "BMFMConfig",
    "SCBertConfig",
    "FieldInfo",

    # Tokenizer
    "create_methylation_tokenizer",
    "create_methylation_multifield_tokenizer",
    "create_tokenizer_from_h5ad",
    "create_indexed_tokenizer",
    "extract_cpg_sites_from_h5ad",
    "MultiFieldTokenizer",

    # Data
    "MethylationDataset",
    "MethylationDataModule",

    # Models (from bmfm_targets)
    "SCBertModel",
    "SCBertForMaskedLM",
]

__version__ = "0.2.0"
