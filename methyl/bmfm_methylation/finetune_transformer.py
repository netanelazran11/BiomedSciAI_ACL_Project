#!/usr/bin/env python3
"""
Value-Only Transformer for Methylation Age Prediction

Tests whether a Transformer using only β-value embeddings + positional embeddings
(no CpG ID embeddings) can predict age. Uses the pretrained SCBertModel encoder
but skips the CpG ID word embedding field -- only the continuous value encoder
and position embeddings contribute to the input representation.

Forward pass:
    BEFORE (original SCBert):  h_i = CpG_ID_embed(s_i) + β_value_embed(β_i) + pos_embed(i)
    HERE  (value-only):        h_i = β_value_embed(β_i) + pos_embed(i)

Architecture:
    Input β-values → ContinuousValueEncoder (pretrained) → + pos_embed (pretrained)
    → LayerNorm + Dropout (pretrained) → SCBertEncoder (pretrained, 6L/8H/512d)
    → mean pooling → Age MLP head (trained from scratch) → predicted age

Usage:
    python -m bmfm_methylation.finetune_transformer \
        data_path=/path/to/methylation.h5ad \
        checkpoint_path=/path/to/pretrained.ckpt \
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


class TransformerValueOnlyAgeRegressor(pl.LightningModule):
    """
    Transformer-based age regressor that uses only β-value embeddings.

    Loads the pretrained SCBertModel but bypasses the CpG ID word embeddings.
    The forward pass manually calls the continuous value encoder (field 1),
    adds position embeddings, applies LayerNorm + dropout, then runs the
    pretrained Transformer encoder. Mean-pooled output feeds an MLP age head.
    """

    def __init__(
        self,
        encoder,  # SCBertModel (pretrained)
        hidden_size: int = 512,
        head_hidden_size: int = 256,
        head_dropout: float = 0.1,
        learning_rate: float = 5e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 200,
        max_steps: int = 10000,
        age_mean: float = 0.0,
        age_std: float = 1.0,
        freeze_encoder: bool = True,
        unfreeze_encoder_epoch: int = 5,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['encoder'])

        self.encoder = encoder
        self.age_mean = age_mean
        self.age_std = age_std

        # Freeze the CpG ID word embedding (field 0) permanently -- not used
        cpg_embed = self.encoder.embeddings.cpg_sites_embeddings
        for param in cpg_embed.parameters():
            param.requires_grad = False

        # Optionally freeze the rest of the encoder
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            logger.info("Encoder fully frozen (will unfreeze at epoch "
                        f"{unfreeze_encoder_epoch})")

        # Age regression MLP head (trained from scratch)
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

        # Accumulate predictions for epoch-level R²
        self._val_preds = []
        self._val_labels = []
        self._test_preds = []
        self._test_labels = []

        head_params = sum(p.numel() for p in self.age_head.parameters())
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("=" * 70)
        print("[MODEL] TransformerValueOnlyAgeRegressor created")
        print(f"  Encoder params:   {encoder_params:,}")
        print(f"  Age head params:  {head_params:,}")
        print(f"  Trainable params: {trainable:,}")
        print(f"  Frozen params:    {encoder_params + head_params - trainable:,}")
        print(f"  Freeze encoder:   {freeze_encoder}")
        print(f"  Unfreeze epoch:   {unfreeze_encoder_epoch}")
        print(f"  Hidden size:      {hidden_size}")
        print(f"  Head hidden:      {head_hidden_size}")
        print(f"  Head dropout:     {head_dropout}")
        print(f"  Learning rate:    {learning_rate}")
        print(f"  Loss function:    MSELoss")
        print("=" * 70)

    def forward(self, input_ids, attention_mask=None):
        """
        Value-only Transformer forward pass.

        Instead of calling encoder(input_ids) which sums CpG ID + β-value
        embeddings, we manually:
          1. Extract β-values from input_ids[:, 1, :]
          2. Run them through the pretrained continuous value encoder
          3. Add pretrained position embeddings
          4. Apply pretrained LayerNorm + dropout
          5. Run through the pretrained Transformer encoder layers
          6. Mean-pool and predict age
        """
        if not hasattr(self, '_fwd_count'):
            self._fwd_count = 0
        self._fwd_count += 1
        is_first = (self._fwd_count <= 2)

        # Validate input shape
        if input_ids.dim() != 3 or input_ids.size(1) != 2:
            raise ValueError(
                f"[FORWARD ERROR] Expected input_ids shape [batch, 2, seq_len], "
                f"got {input_ids.shape}. Field 0=cpg_ids, Field 1=beta_values"
            )

        embeddings_layer = self.encoder.embeddings

        # --- Step 1: Get β-value embeddings (field 1) ---
        beta_field = input_ids[:, 1, :].float()  # [batch, seq_len]
        beta_embeds = embeddings_layer.beta_values_embeddings(beta_field)
        # [batch, seq_len, hidden_size]

        if is_first:
            print(f"[FORWARD #{self._fwd_count}] Step 1: beta_values -> ContinuousValueEncoder")
            print(f"  input_ids shape: {input_ids.shape}")
            print(f"  beta_field min={beta_field.min():.4f}, max={beta_field.max():.4f}, "
                  f"mean={beta_field.mean():.4f}")
            print(f"  beta_embeds shape: {beta_embeds.shape}, "
                  f"norm={beta_embeds[0].norm():.4f}")

        # --- Step 2: Add position embeddings ---
        seq_length = input_ids.size(2)
        if embeddings_layer.position_embedding_type is not None:
            position_ids = embeddings_layer.position_ids[:, :seq_length]
            position_embeds = embeddings_layer.position_embeddings(position_ids)
            hidden_states = beta_embeds + position_embeds
        else:
            hidden_states = beta_embeds

        if is_first:
            print(f"[FORWARD #{self._fwd_count}] Step 2: + position embeddings")
            print(f"  hidden_states norm: {hidden_states[0].norm():.4f}")

        # --- Step 3: LayerNorm + dropout (from pretrained embeddings layer) ---
        hidden_states = embeddings_layer.LayerNorm(hidden_states)
        hidden_states = embeddings_layer.dropout(hidden_states)

        if is_first:
            print(f"[FORWARD #{self._fwd_count}] Step 3: LayerNorm + dropout")
            print(f"  hidden_states norm after LN: {hidden_states[0].norm():.4f}")

        # --- Step 4: Build attention mask ---
        batch_size = input_ids.size(0)
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length), device=input_ids.device
            )
        # Ensure attention_mask is 2D [batch, seq_len] for get_extended_attention_mask
        if attention_mask.dim() == 3:
            # If mask is [batch, num_fields, seq_len], take first field or any field (they should be same)
            attention_mask = attention_mask[:, 0, :]
        extended_attention_mask = self.encoder.get_extended_attention_mask(
            attention_mask, (batch_size, seq_length)
        )

        if is_first:
            print(f"[FORWARD #{self._fwd_count}] Step 4: Attention mask")
            print(f"  attention_mask shape: {attention_mask.shape}")
            print(f"  extended_attention_mask shape: {extended_attention_mask.shape}")

        # --- Step 5: Run through Transformer encoder ---
        head_mask = self.encoder.get_head_mask(
            None, self.encoder.config.num_hidden_layers
        )
        encoder_outputs = self.encoder.encoder(
            hidden_states,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
        )
        sequence_output = encoder_outputs.last_hidden_state
        # [batch, seq_len, hidden_size]

        if is_first:
            print(f"[FORWARD #{self._fwd_count}] Step 5: Transformer encoder "
                  f"({self.encoder.config.num_hidden_layers} layers)")
            print(f"  encoder output norm: {sequence_output[0].norm():.4f}")

        # --- Step 6: Mean pooling (ignore padding if mask provided) ---
        mask_expanded = attention_mask.unsqueeze(-1)  # [batch, seq_len, 1]
        sum_hidden = (sequence_output * mask_expanded).sum(dim=1)
        count = mask_expanded.sum(dim=1).clamp(min=1e-9)
        pooled = sum_hidden / count  # [batch, hidden_size]

        if is_first:
            print(f"[FORWARD #{self._fwd_count}] Step 6: Mean pooling -> {pooled.shape}")
            print(f"  pooled norm: {pooled[0].norm():.4f}")

        # --- Step 7: Age prediction head ---
        age_pred = self.age_head(pooled)  # [batch, 1]

        if is_first:
            print(f"[FORWARD #{self._fwd_count}] Step 7: Age head -> predictions")
            print(f"  predictions (first 5): {age_pred[:5, 0].detach().tolist()}")
            print(f"  (CpG ID embeddings were NOT used)")

        return age_pred

    def on_train_epoch_start(self):
        """Unfreeze encoder after N epochs."""
        epoch = self.current_epoch
        print(f"\n[EPOCH {epoch}] Starting training epoch {epoch}")

        # Print trainable parameter summary at start of each epoch
        encoder_trainable = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        head_trainable = sum(p.numel() for p in self.age_head.parameters() if p.requires_grad)
        print(f"[EPOCH {epoch}] Trainable params: encoder={encoder_trainable:,}, head={head_trainable:,}")

        if (self.hparams.freeze_encoder and
                epoch == self.hparams.unfreeze_encoder_epoch):
            print("=" * 70)
            print(f"[EPOCH {epoch}] UNFREEZING encoder (except CpG ID embeddings)")
            print("=" * 70)
            for param in self.encoder.parameters():
                param.requires_grad = True
            # Keep CpG ID embedding frozen -- it's not used
            cpg_embed = self.encoder.embeddings.cpg_sites_embeddings
            for param in cpg_embed.parameters():
                param.requires_grad = False
            trainable = sum(p.numel() for p in self.parameters()
                            if p.requires_grad)
            encoder_trainable_after = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
            print(f"[EPOCH {epoch}] Trainable params after unfreeze: {trainable:,}")
            print(f"[EPOCH {epoch}] Encoder trainable: {encoder_trainable_after:,}")
            print("=" * 70)

    def _shared_step(self, batch, stage: str):
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
        labels = batch["labels"].float().view(-1, 1)

        # Validate batch data
        if "labels" not in batch:
            raise ValueError(f"[{stage.upper()} ERROR] Batch missing 'labels' key. "
                             f"Available keys: {list(batch.keys())}")

        # Check for NaN/Inf in inputs
        if torch.isnan(input_ids).any() or torch.isinf(input_ids).any():
            print(f"[{stage.upper()} WARNING] NaN or Inf detected in input_ids!")

        if torch.isnan(labels).any() or torch.isinf(labels).any():
            print(f"[{stage.upper()} WARNING] NaN or Inf detected in labels!")

        # Debug logging on first step
        if not hasattr(self, '_debug_printed') or not self._debug_printed:
            self._debug_printed = True
            print("=" * 70)
            print(f"[DEBUG {stage.upper()}] TRANSFORMER VALUE-ONLY BATCH INSPECTION")
            print(f"  Batch keys: {list(batch.keys())}")
            print(f"  input_ids shape: {input_ids.shape}")
            print(f"  input_ids dtype: {input_ids.dtype}")
            if input_ids.dim() == 3:
                print(f"  Field 0 (cpg_sites) first 10: "
                      f"{input_ids[0, 0, :10].tolist()}")
                print(f"  Field 1 (beta_values) first 10: "
                      f"{input_ids[0, 1, :10].tolist()}")
                beta_field = input_ids[:, 1, :]
                print(f"  Field 1 (beta) stats: min={beta_field.min():.4f}, "
                      f"max={beta_field.max():.4f}, mean={beta_field.float().mean():.4f}")
                # Check if beta values are in expected range [0, 1]
                if beta_field.min() < -0.1 or beta_field.max() > 1.1:
                    print(f"  [WARNING] Beta values outside expected range [0,1]!")
            if attention_mask is not None:
                print(f"  attention_mask shape: {attention_mask.shape}")
            print(f"  labels shape: {labels.shape}")
            print(f"  labels (first 5, normalized): {labels[:5, 0].tolist()}")
            labels_denorm = labels * self.age_std + self.age_mean
            print(f"  labels (first 5, denormalized): {labels_denorm[:5, 0].tolist()}")
            print(f"  age_mean={self.age_mean:.2f}, age_std={self.age_std:.2f}")
            print("=" * 70)
            logger.info("=" * 70)
            logger.info("DEBUG: TRANSFORMER VALUE-ONLY BATCH INSPECTION")
            logger.info(f"  input_ids shape: {input_ids.shape}")
            logger.info(f"  input_ids dtype: {input_ids.dtype}")
            if input_ids.dim() == 3:
                logger.info(f"  Field 0 (cpg_sites) first 10: "
                            f"{input_ids[0, 0, :10].tolist()}")
                logger.info(f"  Field 1 (beta_values) first 10: "
                            f"{input_ids[0, 1, :10].tolist()}")
                logger.info(f"  Field 1 min={input_ids[:, 1, :].min():.4f}, "
                            f"max={input_ids[:, 1, :].max():.4f}")
            if attention_mask is not None:
                logger.info(f"  attention_mask shape: {attention_mask.shape}")
            logger.info(f"  labels (first 5): {labels[:5, 0].tolist()}")
            logger.info(f"  age_mean={self.age_mean:.2f}, age_std={self.age_std:.2f}")
            logger.info("=" * 70)

        predictions = self(input_ids, attention_mask)

        # Loss on normalized values
        loss = self.loss_fn(predictions, labels)

        # Check for NaN/Inf in loss or predictions
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"[{stage.upper()} ERROR] NaN/Inf loss detected!")
            print(f"  loss={loss.item()}")
            print(f"  predictions range: [{predictions.min():.4f}, {predictions.max():.4f}]")
            print(f"  labels range: [{labels.min():.4f}, {labels.max():.4f}]")
            raise ValueError("Training stopped due to NaN/Inf loss")

        # Denormalize for metrics
        preds_denorm = predictions * self.age_std + self.age_mean
        labels_denorm = labels * self.age_std + self.age_mean
        mae = torch.abs(preds_denorm - labels_denorm).mean()

        return loss, mae, preds_denorm, labels_denorm

    def training_step(self, batch, batch_idx):
        loss, mae, _, _ = self._shared_step(batch, "train")
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/mae", mae, on_step=False, on_epoch=True, prog_bar=True)

        # Log gradient stats periodically (every 100 batches)
        if batch_idx % 100 == 0:
            # Check head gradients
            head_grad_norm = 0.0
            head_grad_count = 0
            for p in self.age_head.parameters():
                if p.grad is not None:
                    head_grad_norm += p.grad.norm().item() ** 2
                    head_grad_count += 1
            head_grad_norm = head_grad_norm ** 0.5 if head_grad_count > 0 else 0.0

            # Check encoder gradients (should be 0 if frozen)
            encoder_grad_norm = 0.0
            encoder_grad_count = 0
            for p in self.encoder.parameters():
                if p.grad is not None:
                    encoder_grad_norm += p.grad.norm().item() ** 2
                    encoder_grad_count += 1
            encoder_grad_norm = encoder_grad_norm ** 0.5 if encoder_grad_count > 0 else 0.0

            self.log("debug/head_grad_norm", head_grad_norm)
            self.log("debug/encoder_grad_norm", encoder_grad_norm)

            if batch_idx == 0:
                print(f"[TRAIN STEP {batch_idx}] loss={loss:.4f}, mae={mae:.2f}, "
                      f"head_grad={head_grad_norm:.4f}, encoder_grad={encoder_grad_norm:.4f}")

        return loss

    def validation_step(self, batch, batch_idx):
        loss, mae, preds, labels = self._shared_step(batch, "val")
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        self.log("val/mae", mae, on_epoch=True, prog_bar=True)
        self._val_preds.append(preds.detach())
        self._val_labels.append(labels.detach())
        return loss

    def on_validation_epoch_end(self):
        if self._val_preds:
            all_preds = torch.cat(self._val_preds, dim=0)
            all_labels = torch.cat(self._val_labels, dim=0)
            ss_res = torch.sum((all_labels - all_preds) ** 2)
            ss_tot = torch.sum((all_labels - all_labels.mean()) ** 2)
            # Robust R² calculation
            r2 = 1 - ss_res / max(ss_tot, 1e-8)
            self.log("val/r2", r2, prog_bar=True)
            epoch_mae = torch.abs(all_preds - all_labels).mean()
            self.log("val/mae_epoch", epoch_mae)

            # Also log standard deviation of predictions (detect collapse)
            pred_std = all_preds.std()
            self.log("val/pred_std", pred_std)

            print("=" * 70)
            print(f"[VAL EPOCH {self.current_epoch}] SUMMARY")
            print(f"  MAE:        {epoch_mae:.2f} years")
            print(f"  R²:         {r2:.4f}")
            print(f"  Pred range: [{all_preds.min():.1f}, {all_preds.max():.1f}] "
                  f"(std={pred_std:.2f})")
            print(f"  Label range:[{all_labels.min():.1f}, {all_labels.max():.1f}] "
                  f"(std={all_labels.std():.2f})")
            # Warning if predictions collapse to constant
            if pred_std < 1.0:
                print(f"  [WARNING] Prediction std={pred_std:.2f} is very low - "
                      f"model may be collapsing to constant!")
            # Compare to baselines
            print(f"  Baselines:  MLP MAE=3.62 | Ridge MAE=4.49")
            if epoch_mae < 3.62:
                print(f"  [SUCCESS] Beating MLP baseline!")
            elif epoch_mae < 4.49:
                print(f"  [PROGRESS] Beating Ridge, approaching MLP...")
            print("=" * 70)
        self._val_preds.clear()
        self._val_labels.clear()

    def test_step(self, batch, batch_idx):
        loss, mae, preds, labels = self._shared_step(batch, "test")
        self.log("test/mae", mae, on_epoch=True)
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
            print("=" * 70)
            print(f"[TEST RESULTS] MAE={epoch_mae:.2f} years | R2={r2:.4f}")
            print(f"  Predictions: mean={all_preds.mean():.1f}, "
                  f"std={all_preds.std():.1f}, "
                  f"range=[{all_preds.min():.1f}, {all_preds.max():.1f}]")
            print(f"  Labels:      mean={all_labels.mean():.1f}, "
                  f"std={all_labels.std():.1f}, "
                  f"range=[{all_labels.min():.1f}, {all_labels.max():.1f}]")
            print("=" * 70)
        self._test_preds.clear()
        self._test_labels.clear()

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]

        # Separate encoder and head parameters for potential different LRs
        encoder_params_decay = []
        encoder_params_no_decay = []
        head_params_decay = []
        head_params_no_decay = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("age_head."):
                if any(nd in name for nd in no_decay):
                    head_params_no_decay.append(param)
                else:
                    head_params_decay.append(param)
            else:
                if any(nd in name for nd in no_decay):
                    encoder_params_no_decay.append(param)
                else:
                    encoder_params_decay.append(param)

        optimizer_grouped_parameters = [
            {
                "params": encoder_params_decay,
                "weight_decay": self.hparams.weight_decay,
                "lr": self.hparams.learning_rate,
            },
            {
                "params": encoder_params_no_decay,
                "weight_decay": 0.0,
                "lr": self.hparams.learning_rate,
            },
            {
                "params": head_params_decay,
                "weight_decay": self.hparams.weight_decay,
                "lr": self.hparams.learning_rate,
            },
            {
                "params": head_params_no_decay,
                "weight_decay": 0.0,
                "lr": self.hparams.learning_rate,
            },
        ]

        # Filter out empty groups
        optimizer_grouped_parameters = [
            g for g in optimizer_grouped_parameters if len(g["params"]) > 0
        ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        # Cosine decay with warmup
        def lr_lambda(current_step):
            if current_step < self.hparams.warmup_steps:
                return float(current_step) / float(
                    max(1, self.hparams.warmup_steps)
                )
            progress = float(current_step - self.hparams.warmup_steps) / float(
                max(1, self.hparams.max_steps - self.hparams.warmup_steps)
            )
            return max(
                0.0,
                0.5 * (1.0 + torch.cos(
                    torch.tensor(3.14159 * progress)
                ).item()),
            )

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
    wandb_enabled = False
    if hasattr(cfg, 'track_wandb') and cfg.track_wandb.get('enabled', False):
        wandb_enabled = True
    elif cfg.get('wandb_enabled', False):
        wandb_enabled = True

    if wandb_enabled:
        try:
            import wandb
            from pytorch_lightning.loggers import WandbLogger

            # Determine if running from pretrained checkpoint or from scratch
            has_pretrain = (
                hasattr(cfg, 'checkpoint_path')
                and cfg.checkpoint_path
                and str(cfg.checkpoint_path).lower() != "null"
            )
            mode_tag = "pretrained" if has_pretrain else "from-scratch"

            if hasattr(cfg, 'track_wandb'):
                project = cfg.track_wandb.get('project', 'methylation-age')
                entity = cfg.track_wandb.get('entity', None)
                run_name = cfg.track_wandb.get('name', None)
            else:
                project = cfg.get('wandb_project', 'methylation-age')
                entity = cfg.get('wandb_entity', None)
                run_name = cfg.get('wandb_name', None)

            # Append mode tag to run name
            if run_name:
                run_name = f"{run_name}-{mode_tag}"
            else:
                run_name = f"finetune-{mode_tag}"

            wandb_logger = WandbLogger(
                project=project,
                entity=entity,
                name=run_name,
                save_dir=cfg.output_directory,
                log_model=True,
            )
            wandb_logger.experiment.config.update(
                OmegaConf.to_container(cfg, resolve=True)
            )
            logger.info(f"WandB logging enabled - Project: {project}")
            return wandb_logger
        except ImportError:
            logger.warning("WandB not installed, using TensorBoard")
        except Exception as e:
            logger.warning(f"WandB setup failed: {e}, using TensorBoard")

    from pytorch_lightning.loggers import TensorBoardLogger
    logger.info("Using TensorBoard logger")
    return TensorBoardLogger(cfg.output_directory, name="finetune_transformer")


@hydra.main(
    config_path="configs",
    config_name="finetune_config",
    version_base="1.2"
)
def main(cfg: DictConfig):
    """Main training function for value-only Transformer."""
    print("\n" + "=" * 70)
    print("  TRANSFORMER VALUE-ONLY AGE FINE-TUNING")
    print("  Architecture: beta_embed + pos_embed -> Transformer -> mean pool -> MLP -> age")
    print("  CpG ID embeddings are NOT used")
    print("=" * 70 + "\n")

    # Set seed
    if hasattr(cfg, 'seed') and cfg.seed:
        pl.seed_everything(cfg.seed.seed_value, workers=True)
        print(f"[INIT] Random seed: {cfg.seed.seed_value}")

    # Setup output directory
    output_dir = Path(cfg.output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INIT] Output directory: {output_dir}")

    # ---- STEP 1: Tokenizer ----
    print("\n" + "-" * 70)
    print("[STEP 1/6] Loading tokenizer...")
    print("-" * 70)
    tokenizer = setup_tokenizer(cfg)
    print(f"[STEP 1/6] Tokenizer ready")

    # Instantiate fields
    from bmfm_targets.config import FieldInfo
    fields = []
    for field_cfg in cfg.fields:
        field_dict = OmegaConf.to_container(field_cfg)
        field_dict.pop('_target_', None)
        fields.append(FieldInfo(**field_dict))
    print(f"[STEP 1/6] Fields: {[f.field_name for f in fields]}")

    # ---- STEP 2: Data ----
    print("\n" + "-" * 70)
    print("[STEP 2/6] Loading data...")
    print("-" * 70)
    print(f"[STEP 2/6] Data path: {cfg.data_path}")
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
        mlm=False,
        collation_strategy="sequence_classification",
    )
    data_module.setup()
    print(f"[STEP 2/6] Data loaded:")
    print(f"  Train samples: {len(data_module.train_dataset):,}")
    print(f"  Valid samples: {len(data_module.val_dataset):,}")
    print(f"  Test samples:  {len(data_module.test_dataset):,}")
    print(f"  CpG sites:     {data_module.train_dataset.num_cpg_sites:,}")
    print(f"  Age stats:     mean={data_module.age_mean:.2f}, std={data_module.age_std:.2f}")
    print(f"  Batch size:    {cfg.data_module.batch_size}")

    # ---- STEP 3: Model config ----
    print("\n" + "-" * 70)
    print("[STEP 3/6] Building model config...")
    print("-" * 70)
    model_config_partial = hydra.utils.instantiate(cfg.model)
    model_config = model_config_partial(fields=fields)
    print(f"[STEP 3/6] SCBert config:")
    print(f"  Hidden size:      {model_config.hidden_size}")
    print(f"  Num layers:       {model_config.num_hidden_layers}")
    print(f"  Num heads:        {model_config.num_attention_heads}")
    print(f"  FFN intermediate: {model_config.intermediate_size}")
    print(f"  Max positions:    {model_config.max_position_embeddings}")

    # ---- STEP 4: Load checkpoint ----
    print("\n" + "-" * 70)
    print("[STEP 4/6] Loading pretrained encoder...")
    print("-" * 70)
    from bmfm_targets.models.predictive.scbert.modeling_scbert import SCBertModel

    if cfg.checkpoint_path and cfg.checkpoint_path != "null":
        print(f"[STEP 4/6] Checkpoint: {cfg.checkpoint_path}")
        from bmfm_targets.training.modules.masked_language_modeling import (
            MLMTrainingModule,
        )
        from bmfm_targets.config import TrainerConfig, SCBertConfig, FieldInfo

        import torch.serialization
        torch.serialization.add_safe_globals(
            [SCBertConfig, TrainerConfig, FieldInfo]
        )

        trainer_config = TrainerConfig(
            learning_rate=cfg.trainer.learning_rate,
            weight_decay=cfg.trainer.weight_decay,
            warmup_steps=cfg.trainer.warmup_steps,
            losses=[{"name": "cross_entropy"}],
        )

        print(f"[STEP 4/6] Loading MLMTrainingModule from checkpoint...")
        pretrained_module = MLMTrainingModule.load_from_checkpoint(
            cfg.checkpoint_path,
            model_config=model_config,
            trainer_config=trainer_config,
            tokenizer=tokenizer,
            weights_only=False,
        )
        encoder = pretrained_module.model.scbert
        print(f"[STEP 4/6] Encoder loaded from pretrained checkpoint")
        print(f"  Encoder params: {sum(p.numel() for p in encoder.parameters()):,}")

        # Verify checkpoint compatibility
        ckpt_path_str = str(cfg.checkpoint_path).lower()
        if "valueonly" in ckpt_path_str or "value_only" in ckpt_path_str:
            print(f"  [OK] Checkpoint appears to be from value-only pretraining")
        else:
            print(f"  [WARNING] Checkpoint path doesn't contain 'valueonly'.")
            print(f"            Make sure this is from value-only pretraining!")
            print(f"            Standard pretrained models train CpG ID embeddings,")
            print(f"            which are not used in this fine-tuning script.")
    else:
        print(f"[STEP 4/6] No checkpoint - training from scratch!")
        encoder = SCBertModel(model_config)
        print(f"  Encoder params: {sum(p.numel() for p in encoder.parameters()):,}")

    # ---- STEP 5: Create model ----
    print("\n" + "-" * 70)
    print("[STEP 5/6] Creating TransformerValueOnlyAgeRegressor...")
    print("-" * 70)
    freeze_encoder = cfg.get('freeze_encoder', True)
    unfreeze_encoder_epoch = cfg.get('unfreeze_encoder_epoch', 5)

    effective_batch = cfg.data_module.batch_size * cfg.accumulate_grad_batches
    steps_per_epoch = len(data_module.train_dataset) // effective_batch
    total_steps = cfg.finetune_epochs * steps_per_epoch

    print(f"[STEP 5/6] Training schedule:")
    print(f"  Max epochs:          {cfg.finetune_epochs}")
    print(f"  Effective batch:     {effective_batch} (batch={cfg.data_module.batch_size} x accum={cfg.accumulate_grad_batches})")
    print(f"  Steps per epoch:     {steps_per_epoch}")
    print(f"  Total steps:         {total_steps}")
    print(f"  Freeze encoder:      {freeze_encoder} (unfreeze at epoch {unfreeze_encoder_epoch})")
    print(f"  Early stop patience: {cfg.early_stopping.patience}")
    print(f"  Accelerator:         {'GPU' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"  GPU:                 {torch.cuda.get_device_name(0)}")
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU memory:          {gpu_mem:.1f} GB")

    model = TransformerValueOnlyAgeRegressor(
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
        freeze_encoder=freeze_encoder,
        unfreeze_encoder_epoch=unfreeze_encoder_epoch,
    )

    # Setup trainer
    wandb_logger = setup_wandb(cfg)

    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=output_dir / "finetune_transformer" / "checkpoints",
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
    print(f"[STEP 5/6] Checkpoints: {output_dir / 'finetune_transformer' / 'checkpoints'}")

    trainer = pl.Trainer(
        max_epochs=cfg.finetune_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed",
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        logger=wandb_logger,
        callbacks=callbacks,
        default_root_dir=str(output_dir / "finetune_transformer"),
        log_every_n_steps=10,
    )

    # ---- STEP 6: Train ----
    print("\n" + "-" * 70)
    print("[STEP 6/6] Starting training...")
    print("-" * 70)
    print("[STEP 6/6] Watch for:")
    print("  - [FORWARD #1] messages  = model architecture is correct")
    print("  - [VAL EPOCH] MAE going DOWN = model is learning")
    print("  - [EPOCH N] UNFREEZING   = encoder starts training")
    print("  - Compare with: MLP MAE=3.62, Ridge MAE=4.49")
    print("-" * 70 + "\n")

    trainer.fit(model, data_module)

    # Test
    print("\n[TEST] Running test evaluation...")
    test_results = trainer.test(model, data_module)

    # Report results
    best_ckpt = trainer.checkpoint_callback.best_model_path
    print("\n" + "=" * 70)
    print("  FINE-TUNING COMPLETE")
    print("=" * 70)
    print(f"  Best checkpoint: {best_ckpt}")
    if trainer.checkpoint_callback.best_model_score is not None:
        best_mae = trainer.checkpoint_callback.best_model_score
        print(f"  Best val/mae:    {best_mae:.4f}")
        print(f"  Baselines:       MLP MAE=3.62 | Ridge MAE=4.49")
        if best_mae < 3.62:
            print(f"  [SUCCESS] Transformer beats MLP baseline! ({3.62 - best_mae:.2f} years better)")
        elif best_mae < 4.49:
            print(f"  [PROGRESS] Transformer beats Ridge but not MLP yet")
        else:
            print(f"  [NEEDS WORK] Transformer not beating baselines yet")
            print(f"  Possible issues:")
            print(f"    - Learning rate too high/low (current: {cfg.trainer.learning_rate})")
            print(f"    - Unfreeze epoch too early/late (current: {unfreeze_encoder_epoch})")
            print(f"    - Not enough pretraining")
            print(f"    - Data normalization issues")
    else:
        print(f"  [WARNING] No best checkpoint score found")

    print("=" * 70)
    print("  TROUBLESHOOTING CHECKLIST")
    print("=" * 70)
    print("  If MAE is not improving:")
    print("    1. Check [FORWARD #1] prints - are beta values in [0,1]?")
    print("    2. Check [VAL EPOCH] pred_std - is it > 5? (otherwise collapse)")
    print("    3. Check debug/head_grad_norm - should be > 0 always")
    print("    4. Check debug/encoder_grad_norm - should be 0 before unfreeze,")
    print("       then > 0 after unfreeze epoch")
    print("    5. Run with different learning rates: 1e-4, 5e-4, 1e-3")
    print("=" * 70 + "\n")

    return best_ckpt


if __name__ == "__main__":
    main()
