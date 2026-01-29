#!/usr/bin/env python3
"""
Fine-tuning script for Methylation Age Prediction

This script fine-tunes a pretrained BMFM SCBertModel for age prediction
from methylation data.

Usage:
    python -m bmfm_methylation.finetune \
        data_path=/path/to/methylation.h5ad \
        checkpoint_path=/path/to/pretrained.ckpt \
        output_directory=./outputs

Or without pretraining (train from scratch):
    python -m bmfm_methylation.finetune \
        data_path=/path/to/methylation.h5ad \
        checkpoint_path=null \
        output_directory=./outputs
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

import hydra
import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from bmfm_methylation.tokenizer import (
    extract_cpg_sites_from_h5ad,
    create_methylation_multifield_tokenizer,
)
from bmfm_methylation.data_module import MethylationDataModule

logger = logging.getLogger(__name__)


class MethylationAgeRegressor(pl.LightningModule):
    """
    Lightning module for methylation age regression.

    Uses pretrained SCBertModel encoder with a regression head.
    """

    def __init__(
        self,
        encoder,
        hidden_size: int = 512,
        head_hidden_size: int = 256,
        head_dropout: float = 0.1,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        max_steps: int = 10000,
        age_mean: float = 0.0,
        age_std: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['encoder'])

        self.encoder = encoder
        self.age_mean = age_mean
        self.age_std = age_std

        # Regression head
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_size, head_hidden_size),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden_size, head_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden_size // 2, 1),
        )

        # Loss
        self.loss_fn = nn.MSELoss()

        # Metrics
        self.train_mae = pl.metrics.MeanAbsoluteError() if hasattr(pl, 'metrics') else None
        self.val_mae = pl.metrics.MeanAbsoluteError() if hasattr(pl, 'metrics') else None
        self.test_mae = pl.metrics.MeanAbsoluteError() if hasattr(pl, 'metrics') else None

    def forward(self, input_ids, attention_mask=None):
        # Get encoder outputs
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Use pooler output (CLS token representation)
        pooled = outputs.pooler_output

        # Predict age
        age_pred = self.regression_head(pooled)

        return age_pred

    def _shared_step(self, batch, stage: str):
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
        labels = batch["labels"].float().view(-1, 1)

        # Forward pass
        predictions = self(input_ids, attention_mask)

        # Loss (on normalized values)
        loss = self.loss_fn(predictions, labels)

        # Denormalize for metrics
        preds_denorm = predictions * self.age_std + self.age_mean
        labels_denorm = labels * self.age_std + self.age_mean

        # Compute MAE
        mae = torch.abs(preds_denorm - labels_denorm).mean()

        return loss, mae, preds_denorm, labels_denorm

    def training_step(self, batch, batch_idx):
        loss, mae, _, _ = self._shared_step(batch, "train")

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/mae", mae, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, mae, preds, labels = self._shared_step(batch, "val")

        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        self.log("val/mae", mae, on_epoch=True, prog_bar=True)

        # Compute R2
        ss_res = torch.sum((labels - preds) ** 2)
        ss_tot = torch.sum((labels - labels.mean()) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-8)
        self.log("val/r2", r2, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        loss, mae, preds, labels = self._shared_step(batch, "test")

        self.log("test/mae", mae, on_epoch=True)

        # Compute R2
        ss_res = torch.sum((labels - preds) ** 2)
        ss_tot = torch.sum((labels - labels.mean()) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-8)
        self.log("test/r2", r2, on_epoch=True)

        return loss

    def configure_optimizers(self):
        # Separate weight decay
        no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]

        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters()
                           if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.named_parameters()
                           if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        # Learning rate scheduler with warmup
        def lr_lambda(current_step):
            if current_step < self.hparams.warmup_steps:
                return float(current_step) / float(max(1, self.hparams.warmup_steps))
            progress = float(current_step - self.hparams.warmup_steps) / \
                       float(max(1, self.hparams.max_steps - self.hparams.warmup_steps))
            return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(3.14159 * progress)).item()))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


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
                project=cfg.track_wandb.get('project', 'methylation-age'),
                entity=cfg.track_wandb.get('entity'),
                name=cfg.track_wandb.get('name', 'methylation_age'),
                save_dir=cfg.output_directory,
            )
            return wandb_logger
        except ImportError:
            logger.warning("WandB not installed, using TensorBoard")

    from pytorch_lightning.loggers import TensorBoardLogger
    return TensorBoardLogger(cfg.output_directory, name="finetune")


@hydra.main(
    config_path="configs",
    config_name="finetune_config",
    version_base="1.2"
)
def main(cfg: DictConfig):
    """Main fine-tuning function."""
    # Print config
    logger.info("=" * 70)
    logger.info("METHYLATION AGE FINE-TUNING")
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

    # Instantiate fields from config
    fields = hydra.utils.instantiate(cfg.fields)

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
        mlm=False,  # Disable MLM for fine-tuning
        collation_strategy="sequence_classification",
    )
    data_module.setup()

    # Setup model
    model_config = hydra.utils.instantiate(cfg.model, fields=fields)

    # Load pretrained encoder or create new one
    from bmfm_targets.models.predictive.scbert.modeling_scbert import SCBertModel

    if cfg.checkpoint_path and cfg.checkpoint_path != "null":
        logger.info(f"Loading pretrained checkpoint: {cfg.checkpoint_path}")
        # Load from checkpoint
        from bmfm_targets.models.predictive.scbert.modeling_scbert import SCBertForMaskedLM
        pretrained = SCBertForMaskedLM.from_pretrained(cfg.checkpoint_path)
        encoder = pretrained.scbert
    else:
        logger.info("Training from scratch (no pretraining)")
        encoder = SCBertModel(model_config)

    # Create regression model
    model = MethylationAgeRegressor(
        encoder=encoder,
        hidden_size=model_config.hidden_size,
        head_hidden_size=cfg.regression_head.hidden_size,
        head_dropout=cfg.regression_head.dropout,
        learning_rate=cfg.trainer.learning_rate,
        weight_decay=cfg.trainer.weight_decay,
        warmup_steps=cfg.trainer.warmup_steps,
        max_steps=cfg.finetune_epochs * len(data_module.train_dataset) // cfg.data_module.batch_size,
        age_mean=data_module.age_mean,
        age_std=data_module.age_std,
    )

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup trainer
    wandb_logger = setup_wandb(cfg)

    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=output_dir / "finetune_age" / "checkpoints",
            filename="epoch={epoch}-val_mae={val/mae:.4f}",
            monitor="val/mae",
            mode="min",
            save_top_k=3,
            save_last=True,
        ),
        pl.callbacks.EarlyStopping(
            monitor="val/mae",
            patience=cfg.early_stopping.patience,
            mode="min",
        ),
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
    ]

    trainer = pl.Trainer(
        max_epochs=cfg.finetune_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed",
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        logger=wandb_logger,
        callbacks=callbacks,
        default_root_dir=str(output_dir / "finetune_age"),
        log_every_n_steps=10,
    )

    # Train
    logger.info("Starting fine-tuning...")
    trainer.fit(model, data_module)

    # Test
    logger.info("Running test evaluation...")
    trainer.test(model, data_module)

    # Save best checkpoint path
    best_ckpt = trainer.checkpoint_callback.best_model_path
    logger.info(f"\nFine-tuning complete!")
    logger.info(f"Best checkpoint: {best_ckpt}")
    logger.info(f"Best val/mae: {trainer.checkpoint_callback.best_model_score:.4f}")

    return best_ckpt


if __name__ == "__main__":
    main()
