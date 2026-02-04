#!/usr/bin/env python3
"""
Value-Only Transformer Pretraining for Methylation Data

Pretrains an SCBertModel using Masked Value Modeling but with ONLY
β-value embeddings + position embeddings (no CpG ID embeddings).

The standard pretraining uses:
    h_i = CpG_ID_embed(s_i) + β_value_embed(β_i) + pos_embed(i)

This script uses:
    h_i = β_value_embed(β_i) + pos_embed(i)

The CpG ID field is still present in the config (for checkpoint compatibility)
but its contribution is zeroed out via a monkey-patched embeddings forward pass.
The masking task remains the same: predict masked β-values (regression).

After pretraining, fine-tune with finetune_transformer.py.

Usage:
    python -m bmfm_methylation.pretrain_transformer_valueonly \
        data_path=/path/to/methylation.h5ad \
        output_directory=./outputs
"""

# =============================================================================
# CRITICAL: This patch MUST be BEFORE any other imports!
# PyTorch 2.6 changed default weights_only=True which breaks Lightning checkpoints
# =============================================================================
import torch
import torch.serialization

_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load
torch.serialization.load = _patched_torch_load
# =============================================================================

import logging
import os
import sys
from pathlib import Path
from typing import Optional

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from bmfm_methylation.tokenizer import (
    extract_cpg_sites_from_h5ad,
    create_methylation_multifield_tokenizer,
)
from bmfm_methylation.data_module import MethylationDataModule

# Import BMFM training modules
from bmfm_targets.training.modules.masked_language_modeling import MLMTrainingModule
from bmfm_targets.config import TrainerConfig

logger = logging.getLogger(__name__)


def patch_embeddings_valueonly(scbert_model):
    """
    Monkey-patch the SCEmbeddingsLayer forward pass to skip CpG ID embeddings.

    Instead of summing field 0 (CpG IDs) + field 1 (β-values), only use field 1.
    The position embeddings, LayerNorm, and dropout are applied as normal.

    Also freezes the CpG ID embedding parameters to save memory.
    """
    embeddings = scbert_model.embeddings

    # Freeze CpG ID embedding (field 0) -- not used, save memory
    cpg_embed = embeddings.cpg_sites_embeddings
    for param in cpg_embed.parameters():
        param.requires_grad = False

    # Store reference to the original forward for potential restoration
    embeddings._original_forward = embeddings.forward

    def valueonly_forward(
        input_ids,
        position_ids=None,
        inputs_embeds=None,
    ):
        """Value-only embedding: skip CpG ID field, only use β-values + position."""
        if inputs_embeds is not None:
            return inputs_embeds

        # Only compute β-value embeddings (field index 1)
        # Skip field 0 (CpG ID embeddings) entirely
        beta_embeds = embeddings.beta_values_embeddings(
            input_ids[:, 1, :].float()
        )

        seq_length = input_ids.size(2)
        updated_embeddings = beta_embeds

        # Add position embeddings (same as original)
        if embeddings.position_embedding_type is not None:
            if position_ids is None:
                position_ids = embeddings.position_ids[:, :seq_length]
            position_embeds = embeddings.position_embeddings(position_ids)
            updated_embeddings = updated_embeddings + position_embeds

        # LayerNorm + dropout (same as original)
        updated_embeddings = embeddings.LayerNorm(updated_embeddings)
        updated_embeddings = embeddings.dropout(updated_embeddings)
        return updated_embeddings

    # Replace the forward method
    embeddings.forward = valueonly_forward
    logger.info("Patched SCEmbeddingsLayer: value-only mode (CpG ID embeddings skipped)")

    # Log parameter counts
    total = sum(p.numel() for p in scbert_model.parameters())
    frozen = sum(p.numel() for p in cpg_embed.parameters())
    trainable = sum(p.numel() for p in scbert_model.parameters() if p.requires_grad)
    logger.info(f"Total params: {total:,}, CpG embed frozen: {frozen:,}, "
                f"Trainable: {trainable:,}")


def setup_tokenizer(cfg: DictConfig):
    """Create or load tokenizer."""
    tokenizer_path = Path(cfg.tokenizer_path)

    if tokenizer_path.exists() and (tokenizer_path / "tokenizers").exists():
        logger.info(f"Loading existing tokenizer from {tokenizer_path}")
        from bmfm_targets.tokenization import MultiFieldTokenizer
        tokenizer = MultiFieldTokenizer.from_pretrained(str(tokenizer_path))
    else:
        logger.info(f"Creating new tokenizer from {cfg.data_path}")
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
                project=cfg.track_wandb.get('project', 'methylation-pretrain-valueonly'),
                entity=cfg.track_wandb.get('entity'),
                name=cfg.track_wandb.get('name', 'methylation_valueonly_pretrain'),
                save_dir=cfg.output_directory,
            )
            return wandb_logger
        except ImportError:
            logger.warning("WandB not installed, using TensorBoard")

    from pytorch_lightning.loggers import TensorBoardLogger
    return TensorBoardLogger(cfg.output_directory, name="pretrain_valueonly")


@hydra.main(
    config_path="configs",
    config_name="pretrain_config",
    version_base="1.2"
)
def main(cfg: DictConfig):
    """Main value-only pretraining function."""
    logger.info("=" * 70)
    logger.info("VALUE-ONLY TRANSFORMER PRETRAINING (Masked Value Modeling)")
    logger.info("=" * 70)
    logger.info("Architecture: β_value_embed + pos_embed (NO CpG ID embed)")
    logger.info(f"\nConfiguration:\n{OmegaConf.to_yaml(cfg)}")

    # Set seed
    if hasattr(cfg, 'seed') and cfg.seed:
        pl.seed_everything(cfg.seed.seed_value, workers=True)

    # Setup output directory
    output_dir = Path(cfg.output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup tokenizer
    tokenizer = setup_tokenizer(cfg)

    # Instantiate fields
    from bmfm_targets.config import FieldInfo
    fields = []
    for field_cfg in cfg.fields:
        field_dict = OmegaConf.to_container(field_cfg)
        field_dict.pop('_target_', None)
        fields.append(FieldInfo(**field_dict))

    # Setup data module (MLM enabled for pretraining)
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
        mlm=True,
        change_ratio=cfg.data_module.change_ratio,
        mask_ratio=cfg.data_module.mask_ratio,
        switch_ratio=cfg.data_module.switch_ratio,
        collation_strategy="language_modeling",
    )
    data_module.setup()

    # Setup model config
    model_config_partial = hydra.utils.instantiate(cfg.model)
    model_config = model_config_partial(fields=fields)

    # Setup trainer config
    losses = OmegaConf.to_container(cfg.trainer.losses) if hasattr(cfg.trainer, 'losses') else [{"name": "mse", "field_name": "beta_values"}]

    metrics = None
    if hasattr(cfg.trainer, 'metrics') and cfg.trainer.metrics:
        metrics = OmegaConf.to_container(cfg.trainer.metrics)

    batch_prediction_behavior = None
    if hasattr(cfg.trainer, 'batch_prediction_behavior'):
        batch_prediction_behavior = cfg.trainer.batch_prediction_behavior

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

    # Create MLMTrainingModule (same as standard pretraining)
    model = MLMTrainingModule(
        model_config=model_config,
        trainer_config=trainer_config,
        tokenizer=tokenizer,
    )

    # =========================================================================
    # KEY DIFFERENCE: Patch the embeddings layer to skip CpG ID embeddings
    # =========================================================================
    # The model structure is: MLMTrainingModule.model (SCBertForMaskedLM)
    #   -> .scbert (SCBertModel) -> .embeddings (SCEmbeddingsLayer)
    patch_embeddings_valueonly(model.model.scbert)

    logger.info(f"Total model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Trainable parameters: "
                f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Setup trainer
    wandb_logger = setup_wandb(cfg)

    early_stop_patience = cfg.get("early_stop_patience", 20)
    logger.info(f"Early stopping patience: {early_stop_patience} validation checks")

    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=output_dir / "pretrain_valueonly" / "checkpoints",
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
            verbose=True,
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
        default_root_dir=str(output_dir / "pretrain_valueonly"),
        log_every_n_steps=10,
    )

    # Train
    logger.info("Starting value-only pretraining...")
    trainer.fit(model, data_module)

    # Save best checkpoint path
    best_ckpt = trainer.checkpoint_callback.best_model_path
    logger.info(f"\nPretraining complete!")
    logger.info(f"Best checkpoint: {best_ckpt}")

    # Test evaluation
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
        logger.warning("No best checkpoint found, running test with current weights")
        test_results = trainer.test(model, data_module)

    logger.info("=" * 70)
    logger.info("VALUE-ONLY PRETRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Best checkpoint: {best_ckpt}")
    logger.info(f"\nNext step: Fine-tune for age prediction:")
    logger.info(f"  python -m bmfm_methylation.finetune_transformer \\")
    logger.info(f"      data_path={cfg.data_path} \\")
    logger.info(f"      checkpoint_path={best_ckpt}")

    return best_ckpt


if __name__ == "__main__":
    main()
