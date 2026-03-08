#!/usr/bin/env python3
"""
Frozen fine-tuning for 21k WCED pretrained encoder — Method 1 style.

Mirrors the best 8k approach (R²=0.9327):
  - Encoder: completely frozen (27M params, zero gradient forever)
  - Input:   fixed 8000 CpGs from the 21k vocabulary, same every batch
  - Head:    MLP(512 → 256 → 128 → 1), only trainable part
  - Pooling: CLS (pooler_output) — WCED trained CLS to encode global info
  - No decoder, no reconstruction loss, no WCEDCollator

Why this should beat finetune_wced_21k.py (Method 2 style):
  - Uses 8000 CpGs/step vs 4000 random (2× more information)
  - Frozen encoder = no forgetting of pretrained representation
  - Fixed subset = stable consistent gradients for the head
  - On 8k data this approach won by +0.017 R² over Method 2

The 8k pipeline is NOT touched.
"""

# =============================================================================
# CRITICAL: patch torch.load BEFORE any other imports
# =============================================================================
import torch
import torch.serialization

_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load
torch.serialization.load = _patched_torch_load
# =============================================================================

import logging
import sys
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent))

from bmfm_methylation.tokenizer import (
    extract_cpg_sites_from_h5ad,
    create_methylation_multifield_tokenizer,
)
from bmfm_methylation.data_module_21k import MethylationDataModule21k

logger = logging.getLogger(__name__)


class MethylationAgeRegressorFrozen(pl.LightningModule):
    """
    Frozen WCED encoder + MLP head for age regression.

    Uses pooler_output (CLS) — WCED explicitly trained CLS to encode
    the full methylation profile of each sample.

    Encoder is permanently frozen. Only the 165k MLP head trains.
    """

    def __init__(
        self,
        encoder,
        hidden_size: int = 512,
        head_hidden_size: int = 256,
        head_dropout: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        max_steps: int = 10000,
        age_mean: float = 0.0,
        age_std: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['encoder'])

        self.encoder = encoder
        self.age_mean = age_mean
        self.age_std = age_std

        # Freeze encoder permanently
        for param in self.encoder.parameters():
            param.requires_grad = False

        # MLP age head: 512 → 256 → 128 → 1
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

        self.loss_fn = nn.MSELoss()

        self._val_preds = []
        self._val_labels = []
        self._test_preds = []
        self._test_labels = []

        head_params    = sum(p.numel() for p in self.age_head.parameters())
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        trainable      = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Frozen WCED-21k fine-tuning:")
        logger.info(f"  Encoder params:  {encoder_params:,}  (FROZEN)")
        logger.info(f"  Age head params: {head_params:,}  (trainable)")
        logger.info(f"  Total trainable: {trainable:,}")

    def _encode(self, cpg_ids, beta_values, attention_mask=None):
        """Run frozen encoder, return CLS via pooler_output."""
        input_ids = torch.stack([cpg_ids.float(), beta_values], dim=1)
        batch_size = input_ids.size(0)
        seq_length = input_ids.size(2)

        if attention_mask is not None and attention_mask.dim() == 3:
            attention_mask = attention_mask[:, 0, :]
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length), device=input_ids.device
            )

        with torch.no_grad():
            encoder_output = self.encoder(input_ids, attention_mask=attention_mask)

        return encoder_output.pooler_output  # [batch, hidden_size]

    def forward(self, cpg_ids, beta_values, attention_mask=None):
        cls = self._encode(cpg_ids, beta_values, attention_mask)
        return self.age_head(cls)

    def _shared_step(self, batch):
        cpg_ids       = batch["cpg_ids"]
        beta_values   = batch["beta_values"]
        attention_mask = batch.get("attention_mask")
        labels        = batch["labels"].float().view(-1, 1)

        age_pred = self(cpg_ids, beta_values, attention_mask)
        loss = self.loss_fn(age_pred, labels)

        preds_denorm  = age_pred * self.age_std + self.age_mean
        labels_denorm = labels   * self.age_std + self.age_mean
        mae = torch.abs(preds_denorm - labels_denorm).mean()

        return loss, preds_denorm, labels_denorm, mae

    def training_step(self, batch, batch_idx):
        loss, _, _, mae = self._shared_step(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/mae",  mae,  on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, labels, mae = self._shared_step(batch)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        self.log("val/mae",  mae,  on_epoch=True, prog_bar=True)
        self._val_preds.append(preds.detach())
        self._val_labels.append(labels.detach())
        return loss

    def on_validation_epoch_end(self):
        if self._val_preds:
            all_preds  = torch.cat(self._val_preds,  dim=0)
            all_labels = torch.cat(self._val_labels, dim=0)
            ss_res = torch.sum((all_labels - all_preds) ** 2)
            ss_tot = torch.sum((all_labels - all_labels.mean()) ** 2)
            r2 = 1 - ss_res / (ss_tot + 1e-8)
            self.log("val/r2",        r2,                                         prog_bar=True)
            self.log("val/mae_epoch", torch.abs(all_preds - all_labels).mean())
        self._val_preds.clear()
        self._val_labels.clear()

    def test_step(self, batch, batch_idx):
        loss, preds, labels, mae = self._shared_step(batch)
        self.log("test/mae", mae, on_epoch=True)
        self._test_preds.append(preds.detach())
        self._test_labels.append(labels.detach())
        return loss

    def on_test_epoch_end(self):
        if self._test_preds:
            all_preds  = torch.cat(self._test_preds,  dim=0)
            all_labels = torch.cat(self._test_labels, dim=0)
            ss_res = torch.sum((all_labels - all_preds) ** 2)
            ss_tot = torch.sum((all_labels - all_labels.mean()) ** 2)
            r2 = 1 - ss_res / (ss_tot + 1e-8)
            self.log("test/r2",        r2)
            epoch_mae = torch.abs(all_preds - all_labels).mean()
            self.log("test/mae_epoch", epoch_mae)
            logger.info(f"Test MAE: {epoch_mae:.2f} years, R²: {r2:.4f}")
        self._test_preds.clear()
        self._test_labels.clear()

    def configure_optimizers(self):
        # Only head parameters — encoder is frozen
        no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
        param_groups = [
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
        param_groups = [g for g in param_groups if len(g["params"]) > 0]

        optimizer = torch.optim.AdamW(
            param_groups,
            lr=self.hparams.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        def lr_lambda(step):
            if step < self.hparams.warmup_steps:
                return float(step) / float(max(1, self.hparams.warmup_steps))
            progress = float(step - self.hparams.warmup_steps) / \
                       float(max(1, self.hparams.max_steps - self.hparams.warmup_steps))
            return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(3.14159 * progress)).item()))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }


def setup_tokenizer(cfg: DictConfig):
    tokenizer_path = Path(cfg.tokenizer_path)
    if tokenizer_path.exists() and (tokenizer_path / "tokenizers").exists():
        from bmfm_targets.tokenization import MultiFieldTokenizer
        return MultiFieldTokenizer.from_pretrained(str(tokenizer_path))
    cpg_sites = extract_cpg_sites_from_h5ad(cfg.data_path)
    return create_methylation_multifield_tokenizer(
        cpg_sites=cpg_sites,
        output_dir=str(tokenizer_path),
    )


def setup_wandb(cfg: DictConfig):
    if hasattr(cfg, 'track_wandb') and cfg.track_wandb.get('enabled', False):
        try:
            from pytorch_lightning.loggers import WandbLogger
            return WandbLogger(
                project=cfg.track_wandb.get('project', 'finetune-frozen-21k'),
                entity=cfg.track_wandb.get('entity', None),
                name=cfg.track_wandb.get('name', None),
                save_dir=cfg.output_directory,
                log_model=True,
            )
        except Exception as e:
            logger.warning(f"WandB setup failed: {e}, using TensorBoard")
    from pytorch_lightning.loggers import TensorBoardLogger
    return TensorBoardLogger(cfg.output_directory, name="finetune_frozen_21k")


@hydra.main(
    config_path=str(Path(__file__).parent / "configs"),
    config_name="finetune_config",
    version_base="1.2",
)
def main(cfg: DictConfig):
    logger.info("=" * 70)
    logger.info("FROZEN WCED-21k FINE-TUNING — Method 1 style")
    logger.info("Encoder: frozen | Input: 8k fixed | Head only trains")
    logger.info("=" * 70)
    logger.info(f"\nConfiguration:\n{OmegaConf.to_yaml(cfg)}")

    if hasattr(cfg, 'seed') and cfg.seed:
        pl.seed_everything(cfg.seed.seed_value, workers=True)

    output_dir = Path(cfg.output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = setup_tokenizer(cfg)

    from bmfm_targets.config import FieldInfo
    fields = []
    for field_cfg in cfg.fields:
        field_dict = OmegaConf.to_container(field_cfg)
        field_dict.pop('_target_', None)
        fields.append(FieldInfo(**field_dict))

    # -------------------------------------------------------------------------
    # Data module — standard collator (no WCEDCollator needed)
    # Fixed 8000 CpGs from 21k vocab, same every batch → stable head gradients
    # MethylationDataModule21k auto-creates valid split
    # -------------------------------------------------------------------------
    subset_k = cfg.data_module.get('subset_k', 8000)

    data_module = MethylationDataModule21k(
        tokenizer=tokenizer,
        fields=fields,
        h5ad_path=cfg.data_path,
        train_split="train",
        val_split="valid",
        test_split="test",
        batch_size=cfg.data_module.batch_size,
        num_workers=cfg.data_module.num_workers,
        max_length=cfg.data_module.max_length,
        mlm=False,
        collation_strategy="sequence_classification",
        subset_k=subset_k,
        fixed_subset=cfg.data_module.get('fixed_subset', True),
        fixed_subset_seed=cfg.data_module.get('fixed_subset_seed', 42),
        use_wced_collator=False,   # Standard collator — no decoder needed
    )
    data_module.setup()

    model_config_partial = hydra.utils.instantiate(cfg.model)
    model_config = model_config_partial(fields=fields)

    # -------------------------------------------------------------------------
    # Load WCED-21k checkpoint — extract encoder only
    # -------------------------------------------------------------------------
    if not cfg.checkpoint_path or cfg.checkpoint_path == "null":
        raise ValueError("checkpoint_path is required.")

    logger.info(f"Loading WCED-21k checkpoint: {cfg.checkpoint_path}")

    from bmfm_methylation.wced_module import WCEDTrainingModule
    from bmfm_methylation.config import PretrainingConfig
    from bmfm_targets.config import SCBertConfig, TrainerConfig

    torch.serialization.add_safe_globals([SCBertConfig, TrainerConfig, FieldInfo])

    wced_config = PretrainingConfig(mode="wced")
    model_config.checkpoint = None

    pretrained_module = WCEDTrainingModule.load_from_checkpoint(
        cfg.checkpoint_path,
        model_config=model_config,
        pretrain_config=wced_config,
    )

    encoder = pretrained_module.encoder  # has learned cpg_scale + add_forward
    logger.info(f"Encoder loaded: {sum(p.numel() for p in encoder.parameters()):,} params")
    if hasattr(encoder.embeddings, 'cpg_scale'):
        logger.info(f"Learned cpg_scale: {encoder.embeddings.cpg_scale.item():.4f}")

    # -------------------------------------------------------------------------
    # Build model, trainer, run
    # -------------------------------------------------------------------------
    effective_batch  = cfg.data_module.batch_size * cfg.accumulate_grad_batches
    steps_per_epoch  = len(data_module.train_dataset) // effective_batch
    total_steps      = cfg.finetune_epochs * steps_per_epoch

    logger.info(f"Train: {len(data_module.train_dataset)} samples")
    logger.info(f"Val:   {len(data_module.val_dataset)} samples")
    logger.info(f"Effective batch: {effective_batch}, total steps: {total_steps}")
    logger.info(f"Age stats: mean={data_module.age_mean:.2f}, std={data_module.age_std:.2f}")
    logger.info(f"Input: {subset_k} fixed CpGs (max_pos_emb safe: {subset_k} < 8001)")

    model = MethylationAgeRegressorFrozen(
        encoder=encoder,
        hidden_size=model_config.hidden_size,
        head_hidden_size=cfg.regression_head.hidden_size,
        head_dropout=cfg.regression_head.dropout,
        learning_rate=cfg.trainer.learning_rate,
        weight_decay=cfg.trainer.weight_decay,
        warmup_steps=cfg.trainer.warmup_steps,
        max_steps=total_steps,
        age_mean=data_module.age_mean,
        age_std=data_module.age_std,
    )

    wandb_logger = setup_wandb(cfg)

    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=output_dir / "finetune_frozen_21k" / "checkpoints",
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
        gradient_clip_val=1.0,
        logger=wandb_logger,
        callbacks=callbacks,
        default_root_dir=str(output_dir / "finetune_frozen_21k"),
        log_every_n_steps=10,
    )

    logger.info("Starting frozen fine-tuning...")
    trainer.fit(model, data_module)

    logger.info("Running test evaluation...")
    trainer.test(model, data_module)

    best_ckpt = trainer.checkpoint_callback.best_model_path
    logger.info(f"Done. Best checkpoint: {best_ckpt}")
    logger.info(f"Best val/mae: {trainer.checkpoint_callback.best_model_score:.4f}")

    return best_ckpt


if __name__ == "__main__":
    main()
