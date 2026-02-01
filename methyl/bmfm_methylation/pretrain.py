#!/usr/bin/env python3
"""
Pretraining script for Methylation Model using MLM

This script pretrains the BMFM SCBertModel on methylation data using
Masked Language Modeling (MLM) to learn methylation patterns.

Usage:
    python -m bmfm_methylation.pretrain \
        data_path=/path/to/methylation.h5ad \
        output_directory=./outputs

After pretraining, use finetune.py to fine-tune for age prediction.
"""

import logging
import os
import sys
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
import torch.serialization
from omegaconf import DictConfig, OmegaConf

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Fix for PyTorch 2.6+ checkpoint loading (weights_only=True by default)
# Add all config classes that might be in checkpoints
from bmfm_targets.config.model_config import SCBertConfig
from bmfm_targets.config.training_config import TrainerConfig, TrainingTaskConfig
from bmfm_targets.config import FieldInfo, LabelColumnInfo
torch.serialization.add_safe_globals([
    SCBertConfig,
    TrainerConfig,
    TrainingTaskConfig,
    FieldInfo,
    LabelColumnInfo,
])

from bmfm_methylation.tokenizer import (
    create_indexed_tokenizer,
    extract_cpg_sites_from_h5ad,
    create_methylation_multifield_tokenizer,
)
from bmfm_methylation.data_module import MethylationDataModule

# Import BMFM training modules
from bmfm_targets.training.modules.masked_language_modeling import MLMTrainingModule
from bmfm_targets.config import TrainerConfig

logger = logging.getLogger(__name__)


def setup_tokenizer(cfg: DictConfig):
    """Create or load tokenizer."""
    tokenizer_path = Path(cfg.tokenizer_path)

    if tokenizer_path.exists() and (tokenizer_path / "tokenizers").exists():
        logger.info(f"Loading existing tokenizer from {tokenizer_path}")
        from bmfm_targets.tokenization import MultiFieldTokenizer
        tokenizer = MultiFieldTokenizer.from_pretrained(str(tokenizer_path))
    else:
        logger.info(f"Creating new tokenizer from {cfg.data_path}")
        # Extract CpG sites from h5ad
        cpg_sites = extract_cpg_sites_from_h5ad(cfg.data_path)
        tokenizer = create_methylation_multifield_tokenizer(
            cpg_sites=cpg_sites,
            output_dir=str(tokenizer_path),
        )
        logger.info(f"Tokenizer saved to {tokenizer_path}")

    return tokenizer


def setup_wandb(cfg: DictConfig):
    """Setup WandB logging if enabled."""
    if hasattr(cfg, 'track_wandb') and cfg.track_wandb.get('enabled', False):
        try:
            import wandb
            from pytorch_lightning.loggers import WandbLogger

            wandb_logger = WandbLogger(
                project=cfg.track_wandb.get('project', 'methylation-pretrain'),
                entity=cfg.track_wandb.get('entity'),
                name=cfg.track_wandb.get('name', 'methylation_mlm'),
                save_dir=cfg.output_directory,
            )
            return wandb_logger
        except ImportError:
            logger.warning("WandB not installed, using TensorBoard")

    from pytorch_lightning.loggers import TensorBoardLogger
    return TensorBoardLogger(cfg.output_directory, name="pretrain")


@hydra.main(
    config_path="configs",
    config_name="pretrain_config",
    version_base="1.2"
)
def main(cfg: DictConfig):
    """Main pretraining function."""
    # Print config
    logger.info("=" * 70)
    logger.info("METHYLATION PRETRAINING (MLM)")
    logger.info("=" * 70)
    logger.info(f"\nConfiguration:\n{OmegaConf.to_yaml(cfg)}")

    # Set seed
    if hasattr(cfg, 'seed') and cfg.seed:
        pl.seed_everything(cfg.seed.seed_value, workers=True)

    # Setup output directory
    output_dir = Path(cfg.output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup tokenizer
    tokenizer = setup_tokenizer(cfg)

    # Instantiate fields from config and convert to actual FieldInfo dataclass instances
    from bmfm_targets.config import FieldInfo
    fields = []
    for field_cfg in cfg.fields:
        # Convert OmegaConf to dict, remove _target_, and create FieldInfo
        field_dict = OmegaConf.to_container(field_cfg)
        field_dict.pop('_target_', None)
        fields.append(FieldInfo(**field_dict))

    # Setup data module
    data_module = MethylationDataModule(
        tokenizer=tokenizer,
        fields=fields,
        h5ad_path=cfg.data_path,
        train_split="train",
        val_split="valid",
        test_split="test",
        batch_size=cfg.data_module.batch_size,
        num_workers=cfg.data_module.num_workers,
        max_length=cfg.data_module.max_length,
        mlm=True,  # Enable MLM for pretraining
        change_ratio=cfg.data_module.change_ratio,
        mask_ratio=cfg.data_module.mask_ratio,
        switch_ratio=cfg.data_module.switch_ratio,
        collation_strategy="language_modeling",
    )
    data_module.setup()

    # Setup model config
    # Hydra returns a partial when _partial_: true, so we need to call it with fields
    model_config_partial = hydra.utils.instantiate(cfg.model)
    model_config = model_config_partial(fields=fields)

    # Setup trainer config for MLMTrainingModule
    # Convert losses from OmegaConf to list of dicts
    losses = OmegaConf.to_container(cfg.trainer.losses) if hasattr(cfg.trainer, 'losses') else [{"name": "mse", "field_name": "beta_values"}]

    # Convert metrics from OmegaConf to list of dicts
    metrics = None
    if hasattr(cfg.trainer, 'metrics') and cfg.trainer.metrics:
        metrics = OmegaConf.to_container(cfg.trainer.metrics)
        logger.info(f"Metrics configured: {metrics}")

    # Get batch prediction behavior
    batch_prediction_behavior = None
    if hasattr(cfg.trainer, 'batch_prediction_behavior'):
        batch_prediction_behavior = cfg.trainer.batch_prediction_behavior
        logger.info(f"Batch prediction behavior: {batch_prediction_behavior}")

    trainer_config = TrainerConfig(
        learning_rate=cfg.trainer.learning_rate,
        weight_decay=cfg.trainer.weight_decay,
        warmup_steps=cfg.trainer.warmup_steps,
        lr_decay_steps=cfg.trainer.lr_decay_steps,
        betas=tuple(cfg.trainer.betas),
        epsilon=cfg.trainer.epsilon,
        losses=losses,
        metrics=metrics,
        batch_prediction_behavior=batch_prediction_behavior,
    )

    # Create MLMTrainingModule (proper LightningModule wrapper for SCBertForMaskedLM)
    model = MLMTrainingModule(
        model_config=model_config,
        trainer_config=trainer_config,
        tokenizer=tokenizer,
    )

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup trainer
    wandb_logger = setup_wandb(cfg)

    # Early stopping patience: with val_check_interval=0.25 (4 checks/epoch),
    # patience=20 means ~5 epochs without improvement before stopping
    early_stop_patience = cfg.get("early_stop_patience", 20)
    logger.info(f"Early stopping patience: {early_stop_patience} validation checks")

    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=output_dir / "pretrain" / "checkpoints",
            filename="epoch={epoch}-val_loss={validation/loss:.4f}",
            monitor="validation/loss",
            mode="min",
            save_top_k=3,
            save_last=True,
        ),
        pl.callbacks.EarlyStopping(
            monitor="validation/loss",
            patience=early_stop_patience,
            mode="min",
            verbose=True,  # Log when early stopping is triggered
        ),
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
    ]

    trainer = pl.Trainer(
        max_epochs=cfg.pretrain_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=cfg.task[0].precision if isinstance(cfg.task, list) else "16-mixed",
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        logger=wandb_logger,
        callbacks=callbacks,
        default_root_dir=str(output_dir / "pretrain"),
        log_every_n_steps=10,
    )

    # Train
    logger.info("Starting pretraining...")
    trainer.fit(model, data_module)

    # Save best checkpoint path
    best_ckpt = trainer.checkpoint_callback.best_model_path
    logger.info(f"\nTraining complete!")
    logger.info(f"Best checkpoint: {best_ckpt}")

    # Run test evaluation with best checkpoint
    logger.info("=" * 70)
    logger.info("RUNNING TEST EVALUATION")
    logger.info("=" * 70)
    if best_ckpt:
        test_results = trainer.test(model, data_module, ckpt_path=best_ckpt)
        logger.info(f"\nTest Results:")
        for result in test_results:
            for key, value in result.items():
                logger.info(f"  {key}: {value:.6f}")
    else:
        logger.warning("No best checkpoint found, running test with current model weights")
        test_results = trainer.test(model, data_module)

    logger.info("=" * 70)
    logger.info("PRETRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Best checkpoint: {best_ckpt}")
    logger.info(f"\nNext step: Fine-tune for age prediction:")
    logger.info(f"  python -m bmfm_methylation.finetune \\")
    logger.info(f"      data_path={cfg.data_path} \\")
    logger.info(f"      checkpoint_path={best_ckpt}")

    return best_ckpt


if __name__ == "__main__":
    main()
