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

# =============================================================================
# CRITICAL: This patch MUST be BEFORE any other imports!
# PyTorch 2.6 changed default weights_only=True which breaks Lightning checkpoints
# We monkey-patch torch.load BEFORE pytorch_lightning imports it
# =============================================================================
import torch
import torch.serialization

_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load
torch.serialization.load = _patched_torch_load  # Patch both locations
# =============================================================================

import logging
import os
import sys
from pathlib import Path
from typing import Optional

import hydra
import pytorch_lightning as pl
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
torch.load = _patched_torch_load
# =============================================================================

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

    Uses the pretrained BMFM SCBert encoder to produce per-token representations
    from the multi-field input (CpG IDs + beta values), then mean-pools and
    feeds through an MLP head for age prediction.

    Pipeline:
        [CpG IDs + β-values] → Pretrained Encoder → mean pool → MLP head → age
    """

    def __init__(
        self,
        encoder,
        num_cpg_sites: int = 8000,
        hidden_size: int = 512,
        head_hidden_size: int = 256,
        head_dropout: float = 0.1,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        max_steps: int = 10000,
        age_mean: float = 0.0,
        age_std: float = 1.0,
        freeze_encoder: bool = True,
        unfreeze_encoder_epoch: int = 5,
        use_huber_loss: bool = False,
        huber_delta: float = 2.0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['encoder'])

        self.encoder = encoder
        self.age_mean = age_mean
        self.age_std = age_std

        # Optionally freeze encoder (will be unfrozen at unfreeze_encoder_epoch)
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            logger.info(f"Encoder frozen (will unfreeze at epoch {unfreeze_encoder_epoch})")

        # MLP head takes encoder output (hidden_size=512) as input
        self.age_head = nn.Sequential(
            nn.Linear(hidden_size, head_hidden_size),
            nn.LayerNorm(head_hidden_size),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden_size, head_hidden_size // 2),
            nn.LayerNorm(head_hidden_size // 2),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden_size // 2, 1),
        )

        if use_huber_loss:
            self.loss_fn = nn.HuberLoss(delta=huber_delta)
            logger.info(f"Using HuberLoss with delta={huber_delta}")
        else:
            self.loss_fn = nn.MSELoss()
            logger.info("Using MSELoss")

        # Accumulate predictions for epoch-level R² computation
        self._val_preds = []
        self._val_labels = []
        self._test_preds = []
        self._test_labels = []

        head_params = sum(p.numel() for p in self.age_head.parameters())
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Encoder params: {encoder_params:,}")
        logger.info(f"Age head params: {head_params:,}")
        logger.info(f"Trainable params: {trainable:,}")
        logger.info(f"Freeze encoder: {freeze_encoder}, unfreeze at epoch {unfreeze_encoder_epoch}")

    def forward(self, input_ids, attention_mask=None):
        # input_ids shape: [batch, 2, seq_len]
        # Field 0: CpG site token IDs (discrete)
        # Field 1: beta values (continuous)
        #
        # Pass through the pretrained encoder which computes:
        #   h_i = CpG_embed(site_i) + beta_embed(β_i) + pos_embed(i)
        # then runs through 6 transformer layers.

        batch_size = input_ids.size(0)
        seq_length = input_ids.size(2)

        # Ensure attention_mask is 2D [batch, seq_len]
        if attention_mask is not None and attention_mask.dim() == 3:
            attention_mask = attention_mask[:, 0, :]
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length), device=input_ids.device
            )

        # Pass through pretrained encoder (uses CpG IDs + beta values)
        encoder_output = self.encoder(input_ids, attention_mask=attention_mask)
        sequence_output = encoder_output.last_hidden_state  # [batch, seq_len, hidden]

        # Mean pooling (respecting attention mask)
        mask_expanded = attention_mask.unsqueeze(-1)  # [batch, seq_len, 1]
        sum_hidden = (sequence_output * mask_expanded).sum(dim=1)
        count = mask_expanded.sum(dim=1).clamp(min=1e-9)
        pooled = sum_hidden / count  # [batch, hidden_size]

        # Age prediction head
        age_pred = self.age_head(pooled)

        return age_pred

    def on_train_epoch_start(self):
        """Unfreeze encoder after N epochs and add params to optimizer."""
        epoch = self.current_epoch
        if (self.hparams.freeze_encoder and
                epoch == self.hparams.unfreeze_encoder_epoch):
            logger.info("=" * 70)
            logger.info(f"[EPOCH {epoch}] UNFREEZING encoder")
            logger.info("=" * 70)
            for param in self.encoder.parameters():
                param.requires_grad = True

            # Add encoder params to optimizer (they were excluded at init)
            optimizer = self.optimizers()
            no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
            encoder_decay = []
            encoder_no_decay = []
            for name, param in self.encoder.named_parameters():
                if not param.requires_grad:
                    continue
                if any(nd in name for nd in no_decay):
                    encoder_no_decay.append(param)
                else:
                    encoder_decay.append(param)

            encoder_lr = self.hparams.learning_rate * 0.1
            if encoder_decay:
                optimizer.add_param_group({
                    "params": encoder_decay,
                    "weight_decay": self.hparams.weight_decay,
                    "lr": encoder_lr,
                })
            if encoder_no_decay:
                optimizer.add_param_group({
                    "params": encoder_no_decay,
                    "weight_decay": 0.0,
                    "lr": encoder_lr,
                })

            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            logger.info(f"[EPOCH {epoch}] Trainable params after unfreeze: {trainable:,}")
            logger.info(f"[EPOCH {epoch}] Encoder LR: {encoder_lr} (10x lower than head)")
            logger.info("=" * 70)

    def _shared_step(self, batch, stage: str):
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
        labels = batch["labels"].float().view(-1, 1)

        # DEBUG: Print batch info on first step to verify data pipeline
        if not hasattr(self, '_debug_printed') or not self._debug_printed:
            self._debug_printed = True
            logger.info("=" * 70)
            logger.info("DEBUG: BATCH INSPECTION")
            logger.info(f"  input_ids shape: {input_ids.shape}")
            logger.info(f"  input_ids dtype: {input_ids.dtype}")
            logger.info(f"  batch keys: {list(batch.keys())}")
            if input_ids.dim() == 3:
                # Multi-field: [batch, num_fields, seq_len]
                logger.info(f"  Field 0 (cpg_sites) - first 10 values: {input_ids[0, 0, :10].tolist()}")
                logger.info(f"  Field 1 (beta_values) - first 10 values: {input_ids[0, 1, :10].tolist()}")
                # Check if field 1 varies between samples
                f1_sample0 = input_ids[0, 1, :10].tolist()
                f1_sample1 = input_ids[1, 1, :10].tolist() if input_ids.shape[0] > 1 else f1_sample0
                logger.info(f"  Field 1 sample 0: {f1_sample0}")
                logger.info(f"  Field 1 sample 1: {f1_sample1}")
                logger.info(f"  Field 1 same across samples? {f1_sample0 == f1_sample1}")
                logger.info(f"  Field 1 min={input_ids[:, 1, :].min():.4f}, max={input_ids[:, 1, :].max():.4f}, std={input_ids[:, 1, :].std():.4f}")
            elif input_ids.dim() == 2:
                logger.info(f"  WARNING: input_ids is 2D! Shape: {input_ids.shape}")
                logger.info(f"  First 10 values: {input_ids[0, :10].tolist()}")
            if attention_mask is not None:
                logger.info(f"  attention_mask shape: {attention_mask.shape}")
                logger.info(f"  attention_mask sum (non-pad tokens): {attention_mask[0].sum().item()}")
            logger.info(f"  labels (first 5): {labels[:5, 0].tolist()}")
            logger.info(f"  labels std: {labels.std():.4f}")
            logger.info(f"  age_mean={self.age_mean:.2f}, age_std={self.age_std:.2f}")
            logger.info("=" * 70)

        # Forward pass
        predictions = self(input_ids, attention_mask)

        # DEBUG: Check predictions on first few steps
        if not hasattr(self, '_debug_pred_count'):
            self._debug_pred_count = 0
        if self._debug_pred_count < 3:
            self._debug_pred_count += 1
            preds_flat = predictions.detach()[:5, 0].tolist()
            labels_flat = labels[:5, 0].tolist()
            logger.info(f"DEBUG step {self._debug_pred_count}: preds={preds_flat}, labels={labels_flat}")

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

        # Accumulate for epoch-level R² (per-batch R² is unreliable)
        self._val_preds.append(preds.detach())
        self._val_labels.append(labels.detach())

        return loss

    def on_validation_epoch_end(self):
        if self._val_preds:
            all_preds = torch.cat(self._val_preds, dim=0)
            all_labels = torch.cat(self._val_labels, dim=0)

            ss_res = torch.sum((all_labels - all_preds) ** 2)
            ss_tot = torch.sum((all_labels - all_labels.mean()) ** 2)
            r2 = 1 - ss_res / (ss_tot + 1e-8)
            self.log("val/r2", r2, prog_bar=True)

            # Also log epoch-level MAE for accuracy
            epoch_mae = torch.abs(all_preds - all_labels).mean()
            self.log("val/mae_epoch", epoch_mae)

        self._val_preds.clear()
        self._val_labels.clear()

    def test_step(self, batch, batch_idx):
        loss, mae, preds, labels = self._shared_step(batch, "test")

        self.log("test/mae", mae, on_epoch=True)

        # Accumulate for epoch-level R²
        self._test_preds.append(preds.detach())
        self._test_labels.append(labels.detach())

        return loss

    def on_test_epoch_end(self):
        if self._test_preds:
            all_preds = torch.cat(self._test_preds, dim=0)
            all_labels = torch.cat(self._test_labels, dim=0)

            ss_res = torch.sum((all_labels - all_preds) ** 2)
            ss_tot = torch.sum((all_labels - all_labels.mean()) ** 2)
            r2 = 1 - ss_res / (ss_tot + 1e-8)
            self.log("test/r2", r2)

            epoch_mae = torch.abs(all_preds - all_labels).mean()
            self.log("test/mae_epoch", epoch_mae)

            logger.info(f"Test MAE: {epoch_mae:.2f} years, R2: {r2:.4f}")

        self._test_preds.clear()
        self._test_labels.clear()

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]

        # Start with age head params (always trainable)
        # Encoder params are added later via on_train_epoch_start when unfrozen
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.age_head.named_parameters()
                           if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
                "lr": self.hparams.learning_rate,
            },
            {
                "params": [p for n, p in self.age_head.named_parameters()
                           if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": self.hparams.learning_rate,
            },
        ]

        # If encoder is not frozen, include it from the start
        if not self.hparams.freeze_encoder:
            encoder_lr = self.hparams.learning_rate * 0.1
            optimizer_grouped_parameters.extend([
                {
                    "params": [p for n, p in self.encoder.named_parameters()
                               if p.requires_grad and not any(nd in n for nd in no_decay)],
                    "weight_decay": self.hparams.weight_decay,
                    "lr": encoder_lr,
                },
                {
                    "params": [p for n, p in self.encoder.named_parameters()
                               if p.requires_grad and any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                    "lr": encoder_lr,
                },
            ])

        # Filter out empty groups
        optimizer_grouped_parameters = [
            g for g in optimizer_grouped_parameters if len(g["params"]) > 0
        ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,  # Default LR (will be overridden by group LRs)
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
    # Check if WandB is enabled (support both nested and flat config)
    wandb_enabled = False
    if hasattr(cfg, 'track_wandb') and cfg.track_wandb.get('enabled', False):
        wandb_enabled = True
    elif cfg.get('wandb_enabled', False):
        wandb_enabled = True

    if wandb_enabled:
        try:
            import wandb
            from pytorch_lightning.loggers import WandbLogger

            # Get WandB settings from nested or flat config
            if hasattr(cfg, 'track_wandb'):
                project = cfg.track_wandb.get('project', 'methylation-age')
                entity = cfg.track_wandb.get('entity', None)
                run_name = cfg.track_wandb.get('name', None)
            else:
                project = cfg.get('wandb_project', 'methylation-age')
                entity = cfg.get('wandb_entity', None)
                run_name = cfg.get('wandb_name', None)

            # Create WandB logger
            wandb_logger = WandbLogger(
                project=project,
                entity=entity,
                name=run_name,
                save_dir=cfg.output_directory,
                log_model=True,  # Log model checkpoints
            )

            # Log all hyperparameters
            wandb_logger.experiment.config.update(OmegaConf.to_container(cfg, resolve=True))

            logger.info(f"WandB logging enabled - Project: {project}")
            return wandb_logger
        except ImportError:
            logger.warning("WandB not installed, using TensorBoard")
        except Exception as e:
            logger.warning(f"WandB setup failed: {e}, using TensorBoard")

    from pytorch_lightning.loggers import TensorBoardLogger
    logger.info("Using TensorBoard logger")
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
        mlm=False,  # Disable MLM for fine-tuning
        collation_strategy="sequence_classification",
    )
    data_module.setup()

    # Setup model config
    # Hydra returns a partial when _partial_: true, so we need to call it with fields
    model_config_partial = hydra.utils.instantiate(cfg.model)
    model_config = model_config_partial(fields=fields)

    # Load pretrained encoder or create new one
    from bmfm_targets.models.predictive.scbert.modeling_scbert import SCBertModel

    if cfg.checkpoint_path and cfg.checkpoint_path != "null":
        logger.info(f"Loading pretrained checkpoint: {cfg.checkpoint_path}")
        # Load from MLMTrainingModule checkpoint (PyTorch Lightning checkpoint)
        from bmfm_targets.training.modules.masked_language_modeling import MLMTrainingModule
        from bmfm_targets.config import TrainerConfig, SCBertConfig, FieldInfo

        # Add safe globals for PyTorch 2.6+ compatibility
        import torch.serialization
        torch.serialization.add_safe_globals([SCBertConfig, TrainerConfig, FieldInfo])

        # Create a trainer config (needed for loading)
        trainer_config = TrainerConfig(
            learning_rate=cfg.trainer.learning_rate,
            weight_decay=cfg.trainer.weight_decay,
            warmup_steps=cfg.trainer.warmup_steps,
            losses=[{"name": "cross_entropy"}],  # Required for MLMTrainingModule
        )

        # Extract the encoder (SCBertModel) from the MLMTrainingModule
        # The model structure is: MLMTrainingModule.model (SCBertForMaskedLM) -> .scbert (SCBertModel)
        # Null out checkpoint to prevent SCBertForMaskedLM.__init__ from double-loading
        model_config.checkpoint = None
        pretrained_module = MLMTrainingModule.load_from_checkpoint(
            cfg.checkpoint_path,
            model_config=model_config,
            trainer_config=trainer_config,
            tokenizer=tokenizer,
            weights_only=False,
        )
        encoder = pretrained_module.model.scbert
        logger.info("Loaded pretrained encoder (CpG IDs + beta values + transformer layers)")
    else:
        logger.info("Training from scratch (no pretraining)")
        encoder = SCBertModel(model_config)

    # Create regression model
    freeze_encoder = cfg.get('freeze_encoder', True)
    unfreeze_encoder_epoch = cfg.get('unfreeze_encoder_epoch', 5)
    use_huber_loss = cfg.get('use_huber_loss', False)
    huber_delta = cfg.get('huber_delta', 2.0)

    effective_batch = cfg.data_module.batch_size * cfg.accumulate_grad_batches
    steps_per_epoch = len(data_module.train_dataset) // effective_batch
    total_steps = cfg.finetune_epochs * steps_per_epoch

    logger.info(f"Dataset size: {len(data_module.train_dataset)} train samples")
    logger.info(f"Effective batch size: {effective_batch}")
    logger.info(f"Steps per epoch: {steps_per_epoch}")
    logger.info(f"Total steps: {total_steps}")
    logger.info(f"Age stats: mean={data_module.age_mean:.2f}, std={data_module.age_std:.2f}")
    logger.info(f"Freeze encoder: {freeze_encoder}, unfreeze at epoch {unfreeze_encoder_epoch}")
    logger.info(f"Loss: {'Huber(delta=' + str(huber_delta) + ')' if use_huber_loss else 'MSE'}")

    num_cpg_sites = cfg.data_module.max_length - 2
    logger.info(f"Num CpG sites: {num_cpg_sites}")
    logger.info(f"Pipeline: [CpG IDs + beta values] -> Encoder ({model_config.hidden_size}d) -> mean pool -> MLP head -> age")

    model = MethylationAgeRegressor(
        encoder=encoder,
        num_cpg_sites=num_cpg_sites,
        hidden_size=model_config.hidden_size,
        head_hidden_size=cfg.regression_head.hidden_size,
        head_dropout=cfg.regression_head.dropout,
        learning_rate=cfg.trainer.learning_rate,
        weight_decay=cfg.trainer.weight_decay,
        warmup_steps=cfg.trainer.warmup_steps,
        max_steps=total_steps,
        age_mean=data_module.age_mean,
        age_std=data_module.age_std,
        freeze_encoder=freeze_encoder,
        unfreeze_encoder_epoch=unfreeze_encoder_epoch,
        use_huber_loss=use_huber_loss,
        huber_delta=huber_delta,
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
