#!/usr/bin/env python3
"""
Fine-tuning script for MethylLlama — WCED pretrained encoder → age prediction.

Mirrors finetune_wced.py but loads WCEDLlamaModule checkpoint instead of
WCEDTrainingModule (no bmfm_targets dependency).

Strategy (WCED-correct):
  Keep the pretrained decoder as a reconstruction regularizer so the encoder
  cannot forget its global methylation representation.

  Loss = age_MSE(CLS → head → age)
       + recon_weight × recon_MSE(CLS → decoder → all_betas)

Pipeline:
  Random 4k CpGs → MethylLlamaModel → CLS ─┬─► MLP head → age (primary)
                                            └─► Decoder → all 8k betas (regularizer)

Usage:
  python -m bmfm_methylation.llama_methyl.finetune_wced_llama \\
      data_path=/path/to/methylation.h5ad \\
      checkpoint_path=/path/to/wced_llama_pretrain.ckpt \\
      output_directory=./outputs/llama_finetune

Key differences from finetune_wced.py (SCBert WCED):
  1. Loads WCEDLlamaModule (not WCEDTrainingModule)
  2. Encoder is MethylLlamaModel (no SCBertModel dependency)
  3. No need to patch embeddings — architecture is built-in
  4. CLS pooling via pooler_output (same as WCED SCBert)
"""

# =============================================================================
# Patch torch.load BEFORE any other imports
# =============================================================================
import torch
import torch.serialization

_orig_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_load(*args, **kwargs)
torch.load = _patched_load
torch.serialization.load = _patched_load
# =============================================================================

import logging
import sys
from pathlib import Path
from typing import Optional

import hydra
import pytorch_lightning as pl
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf


from bmfm_methylation.shared.tokenizer import (
    extract_cpg_sites_from_h5ad,
    create_methylation_multifield_tokenizer,
)
from bmfm_methylation.shared.data_module import MethylationDataModule, WCEDCollator

from .model import MethylLlamaModel, MethylLlamaConfig
from .wced_llama import WCEDLlamaModule, WCEDDecoder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fine-tuning Lightning Module
# ---------------------------------------------------------------------------

class MethylationAgeRegressorLlama(pl.LightningModule):
    """
    Age regression fine-tuning module for WCED-pretrained MethylLlamaModel.

    WCED-correct: keeps pretrained decoder as reconstruction regularizer.
    CLS pooling (pooler_output) — matches WCED pretraining objective.

    Multi-task loss:
        total = age_MSE + recon_weight × recon_MSE(non-input CpGs only)
    """

    def __init__(
        self,
        encoder: MethylLlamaModel,
        decoder: Optional[WCEDDecoder] = None,
        hidden_size: int = 512,
        head_hidden_size: int = 256,
        head_dropout: float = 0.1,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        max_steps: int = 10000,
        age_mean: float = 0.0,
        age_std: float = 1.0,
        freeze_encoder: bool = True,
        unfreeze_encoder_epoch: int = 9999,
        recon_weight: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["encoder", "decoder"])

        self.encoder = encoder
        self.decoder = decoder  # Kept as regularizer (frozen or unfrozen)
        self.age_mean = age_mean
        self.age_std  = age_std
        self.recon_weight = recon_weight

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            logger.info(f"Encoder frozen (unfreeze at epoch {unfreeze_encoder_epoch})")

        if decoder is not None:
            # Decoder is a regularizer — freeze it to prevent overfitting on age data
            for p in self.decoder.parameters():
                p.requires_grad = False
            logger.info("Decoder frozen (reconstruction regularizer)")

        # MLP age head: hidden → head_hidden → head_hidden//2 → 1
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

        self.age_loss_fn   = nn.MSELoss()
        self.recon_loss_fn = nn.MSELoss(reduction="none")

    def on_train_epoch_start(self):
        """Unfreeze encoder after warmup epochs."""
        epoch = self.current_epoch
        if epoch == self.hparams.unfreeze_encoder_epoch:
            logger.info(f"Epoch {epoch}: unfreezing encoder")
            for p in self.encoder.parameters():
                p.requires_grad = True

    def _encode_cls(
        self,
        cpg_ids: torch.Tensor,
        beta_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Run encoder and return CLS pooler_output."""
        input_ids = torch.stack([cpg_ids.float(), beta_values], dim=1)  # [B, 2, L]
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return out.pooler_output  # [B, D]

    def _shared_step(self, batch, stage: str):
        cpg_ids      = batch["cpg_ids"]
        beta_values  = batch["beta_values"]
        attn_mask    = batch.get("attention_mask")
        all_betas    = batch.get("all_betas")
        input_mask   = batch.get("input_mask")
        age_labels   = batch["age"]

        # Encode
        cls = self._encode_cls(cpg_ids, beta_values, attn_mask)  # [B, D]

        # Age prediction (primary task)
        age_pred_norm = self.age_head(cls).squeeze(-1)  # [B]  (normalized scale)
        age_pred = age_pred_norm * self.age_std + self.age_mean   # real age scale

        # Age loss on valid (non-NaN) samples
        valid = ~torch.isnan(age_labels)
        if valid.any():
            age_loss = self.age_loss_fn(age_pred[valid], age_labels[valid].float())
        else:
            age_loss = torch.tensor(0.0, device=cls.device)

        # Reconstruction regularizer (optional, only if decoder + all_betas available)
        recon_loss = torch.tensor(0.0, device=cls.device)
        if self.decoder is not None and all_betas is not None and input_mask is not None:
            with torch.set_grad_enabled(self.recon_weight > 0 and self.training):
                predicted_betas = self.decoder(cls)  # [B, vocab_size]
                non_input = ~input_mask
                loss_per_cpg = self.recon_loss_fn(predicted_betas, all_betas)
                masked = loss_per_cpg * non_input.float()
                recon_loss = masked.sum() / non_input.float().sum().clamp(min=1)

        loss = age_loss + self.recon_weight * recon_loss

        # Metrics (denormalized)
        with torch.no_grad():
            if valid.any():
                mae = torch.abs(age_pred[valid] - age_labels[valid]).mean()
                r2_num = ((age_pred[valid] - age_labels[valid]) ** 2).sum()
                r2_den = ((age_labels[valid] - age_labels[valid].mean()) ** 2).sum()
                r2 = 1.0 - r2_num / (r2_den + 1e-8)
            else:
                mae = torch.tensor(0.0)
                r2  = torch.tensor(0.0)

        return {
            "loss":       loss,
            "age_loss":   age_loss,
            "recon_loss": recon_loss,
            "mae":        mae,
            "r2":         r2,
        }

    def training_step(self, batch, batch_idx):
        out = self._shared_step(batch, "train")
        self.log("train/loss",       out["loss"],       on_step=True,  on_epoch=True, prog_bar=True)
        self.log("train/age_loss",   out["age_loss"],   on_step=False, on_epoch=True)
        self.log("train/recon_loss", out["recon_loss"], on_step=False, on_epoch=True)
        self.log("train/mae",        out["mae"],        on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/r2",         out["r2"],         on_step=False, on_epoch=True)
        return out["loss"]

    def validation_step(self, batch, batch_idx):
        out = self._shared_step(batch, "val")
        self.log("val/loss",       out["loss"],       on_epoch=True, prog_bar=True)
        self.log("val/age_loss",   out["age_loss"],   on_epoch=True)
        self.log("val/recon_loss", out["recon_loss"], on_epoch=True)
        self.log("val/mae",        out["mae"],        on_epoch=True, prog_bar=True)
        self.log("val/r2",         out["r2"],         on_epoch=True)
        return out["loss"]

    def test_step(self, batch, batch_idx):
        out = self._shared_step(batch, "test")
        self.log("test/loss",       out["loss"],       on_epoch=True)
        self.log("test/age_loss",   out["age_loss"],   on_epoch=True)
        self.log("test/recon_loss", out["recon_loss"], on_epoch=True)
        self.log("test/mae",        out["mae"],        on_epoch=True)
        self.log("test/r2",         out["r2"],         on_epoch=True)
        return out["loss"]

    def configure_optimizers(self):
        # Only optimize trainable parameters
        trainable = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable,
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        # Cosine annealing
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_steps,
            eta_min=self.hparams.learning_rate * 0.01,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_wced_llama_checkpoint(checkpoint_path: str) -> WCEDLlamaModule:
    """Load a WCEDLlamaModule from checkpoint."""
    logger.info(f"Loading WCEDLlamaModule from {checkpoint_path}")
    module = WCEDLlamaModule.load_from_checkpoint(
        checkpoint_path,
        map_location="cpu",
    )
    logger.info("Checkpoint loaded successfully")
    return module


# ---------------------------------------------------------------------------
# Tokenizer & Data helpers
# ---------------------------------------------------------------------------

def setup_tokenizer(cfg: DictConfig):
    tokenizer_path = Path(cfg.tokenizer_path)
    if tokenizer_path.exists() and (tokenizer_path / "tokenizers").exists():
        from bmfm_targets.tokenization import MultiFieldTokenizer
        return MultiFieldTokenizer.from_pretrained(str(tokenizer_path))
    cpg_sites = extract_cpg_sites_from_h5ad(cfg.data_path)
    tokenizer = create_methylation_multifield_tokenizer(
        cpg_sites=cpg_sites,
        output_dir=str(tokenizer_path),
    )
    return tokenizer


def setup_wandb(cfg: DictConfig):
    if hasattr(cfg, "track_wandb") and cfg.track_wandb.get("enabled", False):
        try:
            from pytorch_lightning.loggers import WandbLogger
            return WandbLogger(
                project=cfg.track_wandb.get("project", "methylation-llama-finetune"),
                entity=cfg.track_wandb.get("entity"),
                name=cfg.track_wandb.get("name", "llama_wced_finetune"),
                save_dir=cfg.output_directory,
            )
        except Exception as e:
            logger.warning(f"WandB error: {e}")
    from pytorch_lightning.loggers import TensorBoardLogger
    return TensorBoardLogger(cfg.output_directory, name="finetune_llama")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(
    config_path="configs",
    config_name="finetune_llama",
    version_base="1.2",
)
def main(cfg: DictConfig):
    logger.info("=" * 70)
    logger.info("METHYLLAMA WCED FINE-TUNING")
    logger.info("=" * 70)
    logger.info(f"\n{OmegaConf.to_yaml(cfg)}")

    if hasattr(cfg, "seed") and cfg.seed:
        pl.seed_everything(cfg.seed.get("seed_value", 42), workers=True)

    output_dir = Path(cfg.output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Tokenizer
    tokenizer = setup_tokenizer(cfg)

    # Build FieldInfo
    from bmfm_targets.config import FieldInfo
    fields = []
    if hasattr(cfg, "fields") and cfg.fields:
        for fc in cfg.fields:
            fd = OmegaConf.to_container(fc)
            fd.pop("_target_", None)
            fields.append(FieldInfo(**fd))

    # Data settings
    dm_cfg = cfg.get("data_module", {})
    subset_k         = dm_cfg.get("subset_k", 8000)
    fixed_subset_seed = dm_cfg.get("fixed_subset_seed", 42)
    wced_input_ratio  = cfg.get("wced_input_ratio", 0.5)
    vocab_size        = subset_k

    # Age normalization
    age_mean = cfg.get("age_mean", 0.0)
    age_std  = cfg.get("age_std", 1.0)
    logger.info(f"Age normalization: mean={age_mean:.2f}, std={age_std:.2f}")

    # Data module (uses WCEDCollator)
    data_module = MethylationDataModule(
        tokenizer=tokenizer,
        fields=fields,
        h5ad_path=cfg.data_path,
        train_split="train",
        val_split="valid",
        test_split="test",
        batch_size=dm_cfg.get("batch_size", 32),
        num_workers=dm_cfg.get("num_workers", 4),
        max_length=dm_cfg.get("max_length", 8002),
        mlm=False,
        collation_strategy="language_modeling",
        subset_k=subset_k,
        fixed_subset=True,
        fixed_subset_seed=fixed_subset_seed,
    )

    def _wrap_collator():
        cpg_sites = None
        for ds in [data_module.train_dataset, data_module.val_dataset, data_module.test_dataset]:
            if ds is not None:
                cpg_sites = ds.cpg_sites
                break
        if cpg_sites is None:
            raise ValueError("No CpG site list found for WCEDCollator")

        wced_collator = WCEDCollator(
            tokenizer=data_module.tokenizer,
            cpg_sites=cpg_sites,
            vocab_size=vocab_size,
            input_ratio=wced_input_ratio,
            fixed_subset_seed=fixed_subset_seed,
            contrastive=False,   # No contrastive in fine-tuning
        )
        data_module.collator = wced_collator

    orig_setup = data_module.setup

    def _setup_with_wrap(stage=None):
        orig_setup(stage)
        _wrap_collator()

    data_module.setup = _setup_with_wrap
    data_module.setup()

    # Load pretrained checkpoint
    checkpoint_path = cfg.get("checkpoint_path")
    if not checkpoint_path:
        raise ValueError("checkpoint_path is required for fine-tuning")

    pretrained = load_wced_llama_checkpoint(checkpoint_path)
    encoder = pretrained.encoder
    decoder = pretrained.decoder   # Keep for reconstruction regularizer

    # Fine-tuning module
    ft_cfg = cfg.get("finetune", {})
    module = MethylationAgeRegressorLlama(
        encoder=encoder,
        decoder=decoder,
        hidden_size=encoder.config.hidden_size,
        head_hidden_size=ft_cfg.get("head_hidden_size", 256),
        head_dropout=ft_cfg.get("head_dropout", 0.1),
        learning_rate=ft_cfg.get("learning_rate", 1e-4),
        weight_decay=ft_cfg.get("weight_decay", 0.01),
        warmup_steps=ft_cfg.get("warmup_steps", 500),
        max_steps=ft_cfg.get("max_steps", 10000),
        age_mean=age_mean,
        age_std=age_std,
        freeze_encoder=ft_cfg.get("freeze_encoder", True),
        unfreeze_encoder_epoch=ft_cfg.get("unfreeze_encoder_epoch", 9999),
        recon_weight=ft_cfg.get("recon_weight", 0.1),
    )

    n_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in module.parameters())
    logger.info(f"Trainable params: {n_trainable:,} / {n_total:,}")

    # Logger and callbacks
    exp_logger = setup_wandb(cfg)
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=str(output_dir / "checkpoints"),
            filename="epoch={epoch}-val_mae={val/mae:.4f}",
            monitor="val/mae",
            mode="min",
            save_top_k=3,
            auto_insert_metric_name=False,
        ),
        pl.callbacks.EarlyStopping(
            monitor="val/mae",
            patience=cfg.get("early_stop_patience", 30),
            mode="min",
        ),
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
    ]

    trainer = pl.Trainer(
        max_epochs=cfg.get("finetune_epochs", 100),
        accelerator="auto",
        devices="auto",
        precision=cfg.get("precision", "16-mixed"),
        accumulate_grad_batches=cfg.get("accumulate_grad_batches", 4),
        gradient_clip_val=cfg.get("gradient_clip_val", 1.0),
        logger=exp_logger,
        callbacks=callbacks,
        log_every_n_steps=10,
    )

    logger.info("Starting fine-tuning...")
    trainer.fit(module, datamodule=data_module)

    if data_module.test_dataset is not None:
        logger.info("Running test evaluation...")
        trainer.test(module, datamodule=data_module)

    logger.info(f"Fine-tuning complete. Checkpoints: {output_dir / 'checkpoints'}")


if __name__ == "__main__":
    main()
