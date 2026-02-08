#!/usr/bin/env python3
"""
Fine-tuning script for Methylation Age Prediction with Scaled CpG ID Embeddings.

This script uses the FULL pretrained transformer architecture but scales down
the CpG ID embeddings to prevent them from dominating the beta value signal.

Architecture:
    h_i = α * CpG_embed + β_embed + pos_embed → Transformer → mean pool → MLP → age

    where α is a learnable parameter initialized to 0.1

This approach:
- Uses all 6 pretrained transformer layers
- Keeps some positional signal from CpG IDs
- Lets the model learn how much to rely on CpG IDs vs beta values
- Is a minimal change to the existing architecture

Usage:
    python -m bmfm_methylation.finetune_transformer_scaled \
        data_path=/path/to/methylation.h5ad \
        checkpoint_path=/path/to/pretrained.ckpt \
        output_directory=./outputs
"""

# =============================================================================
# CRITICAL: This patch MUST be BEFORE any other imports!
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
import numpy as np
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

# Re-apply patch after imports
torch.load = _patched_torch_load

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from bmfm_methylation.tokenizer import (
    extract_cpg_sites_from_h5ad,
    create_methylation_multifield_tokenizer,
)
from bmfm_methylation.data_module import MethylationDataModule

logger = logging.getLogger(__name__)


class TransformerScaledCpGAgeRegressor(pl.LightningModule):
    """
    Transformer-based age regressor with configurable embedding combination.

    Supports three modes:
    1. ADD mode:      h_i = α * CpG_embed + β_embed
    2. MULTIPLY mode: h_i = CpG_embed * (β + offset) (scGPT style, with offset fix)
    3. BINNED mode:   h_i = CpG_embed + bin_embed(β) (like scGPT category mode)

    MULTIPLY mode scales the CpG embedding by the beta value + offset to prevent
    vanishing gradients for low methylation sites.

    BINNED mode discretizes beta values into N bins and uses learned embeddings
    for each bin, similar to scGPT's category value encoder.
    """

    def __init__(
        self,
        encoder,
        hidden_size: int = 512,
        head_hidden_size: int = 256,
        head_dropout: float = 0.1,
        learning_rate: float = 5e-4,
        head_learning_rate: float = 5e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 200,
        max_steps: int = 10000,
        age_mean: float = 0.0,
        age_std: float = 1.0,
        freeze_encoder: bool = True,
        unfreeze_encoder_epoch: int = 5,
        initial_cpg_scale: float = 0.1,
        use_huber_loss: bool = False,
        huber_delta: float = 2.0,
        combine_style: str = "multiply",  # "add", "multiply", or "binned"
        n_bins: int = 51,  # Number of bins for "binned" mode (like scGPT)
        beta_offset: float = 0.5,  # Offset for multiply mode: β→β+offset
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['encoder'])

        self.encoder = encoder
        self.hidden_size = hidden_size
        self.age_mean = age_mean
        self.age_std = age_std
        self.freeze_encoder = freeze_encoder
        self.unfreeze_encoder_epoch = unfreeze_encoder_epoch
        self.combine_style = combine_style
        self.n_bins = n_bins
        self.beta_offset = beta_offset

        # =====================================================================
        # EMBEDDING COMBINATION STYLE
        # =====================================================================
        logger.info(f"Combine style: {combine_style}")
        print(f"[INIT] Combine style: {combine_style}")
        if combine_style == "multiply":
            print(f"[INIT] Using MULTIPLY: h = CpG_embed * (β + {beta_offset}) (scGPT style with offset)")
        elif combine_style == "binned":
            print(f"[INIT] Using BINNED: h = CpG_embed + bin_embed(β) (scGPT category style, {n_bins} bins)")
        else:
            print(f"[INIT] Using ADD: h = α * CpG_embed + β_embed")

        # =====================================================================
        # BINNED MODE: Create bin embeddings (like scGPT category encoder)
        # =====================================================================
        if combine_style == "binned":
            # Bins: 0 = special (MASK, etc.), 1-50 = quantized beta values
            # This is exactly like scGPT's n_input_bins=51
            self.bin_embeddings = nn.Embedding(n_bins, hidden_size)
            nn.init.normal_(self.bin_embeddings.weight, mean=0, std=0.02)
            self.bin_layer_norm = nn.LayerNorm(hidden_size)
            print(f"[INIT] Created bin embeddings: {n_bins} bins × {hidden_size} dim")

        # =====================================================================
        # LEARNABLE CpG SCALE (only used in ADD mode)
        # =====================================================================
        self.cpg_scale = nn.Parameter(torch.tensor(initial_cpg_scale))
        logger.info(f"Initial CpG scale: {initial_cpg_scale}")
        print(f"[INIT] CpG scale parameter initialized to: {initial_cpg_scale}")

        # =====================================================================
        # Age prediction head (MLP)
        # =====================================================================
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

        # Loss function
        if use_huber_loss:
            self.loss_fn = nn.HuberLoss(delta=huber_delta)
            logger.info(f"Using HuberLoss with delta={huber_delta}")
        else:
            self.loss_fn = nn.MSELoss()
            logger.info("Using MSELoss")

        # Freeze encoder initially if requested
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            logger.info(f"Encoder frozen (will unfreeze at epoch {unfreeze_encoder_epoch})")
            print(f"[INIT] Encoder FROZEN - will unfreeze at epoch {unfreeze_encoder_epoch}")

        # Metrics accumulators
        self._val_preds = []
        self._val_labels = []
        self._test_preds = []
        self._test_labels = []

        # Log parameter counts
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        head_params = sum(p.numel() for p in self.age_head.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        logger.info(f"Encoder params: {encoder_params:,}")
        logger.info(f"Age head params: {head_params:,}")
        logger.info(f"CpG scale param: 1")
        logger.info(f"Trainable params: {trainable_params:,}")
        print(f"[INIT] Encoder params: {encoder_params:,}")
        print(f"[INIT] Age head params: {head_params:,}")
        print(f"[INIT] Trainable params: {trainable_params:,}")

    def _create_sinusoidal_embeddings(self, seq_length, hidden_size, device):
        """Create sinusoidal position embeddings (Vaswani et al., 2017)."""
        position = torch.arange(seq_length, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2, dtype=torch.float, device=device)
            * (-np.log(10000.0) / hidden_size)
        )
        pe = torch.zeros(seq_length, hidden_size, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass with scaled CpG ID embeddings.

        Instead of the standard:
            h_i = CpG_embed + β_embed + pos_embed

        We use:
            h_i = α * CpG_embed + β_embed + pos_embed

        where α is learnable and initialized to 0.1
        """
        batch_size = input_ids.size(0)
        seq_length = input_ids.size(2)

        # Get the embeddings layer
        embeddings_layer = self.encoder.embeddings

        # Debug: Print available attributes on first forward pass
        if not hasattr(self, '_debug_embeddings_printed'):
            self._debug_embeddings_printed = True
            print("\n" + "=" * 70)
            print("DEBUG: SCEmbeddingsLayer structure")
            print("=" * 70)
            print(f"Type: {type(embeddings_layer)}")
            print(f"Attributes: {[a for a in dir(embeddings_layer) if not a.startswith('_')]}")
            print(f"Named children:")
            for name, child in embeddings_layer.named_children():
                print(f"  {name}: {type(child).__name__}")
            print("=" * 70 + "\n")

        # =================================================================
        # Step 1: Get CpG ID embeddings (Field 0)
        # =================================================================
        cpg_ids = input_ids[:, 0, :].long()  # [batch, seq_len]
        cpg_embeds = embeddings_layer.cpg_sites_embeddings(cpg_ids)  # [batch, seq_len, 512]

        # =================================================================
        # Step 2: Get beta values (Field 1)
        # =================================================================
        beta_values = input_ids[:, 1, :].float()  # [batch, seq_len]

        # =================================================================
        # Step 3: Combine embeddings based on style
        # =================================================================
        if self.combine_style == "multiply":
            # MULTIPLY mode (scGPT style): h = CpG_embed * β_value
            #
            # IMPORTANT FIX: Match pretraining behavior exactly!
            # - MASK tokens (β = -1): CpG_embed + mask_embed (ADD style)
            # - Special tokens (CLS, PAD, etc.): Full CpG_embed (scale = 1.0)
            # - Real values (0-1): CpG_embed * β (MULTIPLY style)
            #
            # Additional improvement: Use (β + offset) to prevent near-zero embeddings
            # for low methylation sites. β=0.1 → (0.1 + 0.5) = 0.6 scale factor

            # Use instance beta_offset: β=0→offset, β=1→1+offset, then clamp to [0.3, 1.5]

            # Identify token types
            mask_token_mask = (beta_values == -1)  # MASK tokens
            other_special_mask = (beta_values < 0) & ~mask_token_mask  # CLS, PAD, etc.
            real_value_mask = (beta_values >= 0)  # Real beta values

            # Initialize hidden states
            hidden_states = torch.zeros_like(cpg_embeds)

            # For MASK tokens: use CpG + beta_embed (ADD style, matching pretraining)
            if mask_token_mask.any():
                mask_positions = mask_token_mask.nonzero(as_tuple=False)
                batch_idx = mask_positions[:, 0]
                seq_idx = mask_positions[:, 1]
                masked_beta_vals = beta_values[batch_idx, seq_idx]
                # Get mask embedding from the continuous encoder
                masked_beta_2d = masked_beta_vals.unsqueeze(0)
                mask_embeds = embeddings_layer.beta_values_embeddings(masked_beta_2d).squeeze(0)
                hidden_states[batch_idx, seq_idx] = cpg_embeds[batch_idx, seq_idx] + mask_embeds

            # For other special tokens: use full CpG embedding
            if other_special_mask.any():
                special_positions = other_special_mask.nonzero(as_tuple=False)
                batch_idx = special_positions[:, 0]
                seq_idx = special_positions[:, 1]
                hidden_states[batch_idx, seq_idx] = cpg_embeds[batch_idx, seq_idx]

            # For real beta values: use MULTIPLY with offset to prevent vanishing
            if real_value_mask.any():
                real_positions = real_value_mask.nonzero(as_tuple=False)
                batch_idx = real_positions[:, 0]
                seq_idx = real_positions[:, 1]
                beta_vals = beta_values[batch_idx, seq_idx]
                # Apply offset and clamp: β=0→offset, β=1→1+offset, clamp to [0.3, 1.5]
                beta_scale = torch.clamp(beta_vals + self.beta_offset, 0.3, 1.5).unsqueeze(-1)
                hidden_states[batch_idx, seq_idx] = cpg_embeds[batch_idx, seq_idx] * beta_scale

            if not hasattr(self, '_multiply_debug_printed'):
                self._multiply_debug_printed = True
                n_mask = mask_token_mask.sum().item()
                n_special = other_special_mask.sum().item()
                n_real = real_value_mask.sum().item()
                print(f"\n[MULTIPLY MODE - FIXED]")
                print(f"  Token breakdown: MASK={n_mask}, Special={n_special}, Real={n_real}")
                print(f"  Beta offset: {self.beta_offset} (so β=0→{self.beta_offset} scale, β=1→{1+self.beta_offset} scale)")
                print(f"  MASK handling: CpG + mask_embed (matching pretraining)")
                print(f"  Real beta scaling: clamp(β+{self.beta_offset}, 0.3, 1.5)")
                print(f"  cpg_embeds norm: {cpg_embeds[0, 0].norm():.4f}")

        elif self.combine_style == "binned":
            # BINNED mode (like scGPT category encoder)
            # Discretize beta values into bins and use learned embeddings
            #
            # Binning scheme (like scGPT):
            # - Bin 0: Special tokens (MASK, CLS, PAD, etc.)
            # - Bins 1-50: Quantized beta values (0-1 range divided into 50 bins)

            # Create bin indices
            bin_indices = torch.zeros_like(beta_values, dtype=torch.long)

            # Identify token types
            special_mask = (beta_values < 0)  # All special tokens
            real_mask = (beta_values >= 0)  # Real beta values

            # Special tokens → bin 0
            bin_indices[special_mask] = 0

            # Real values → bins 1 to n_bins-1
            if real_mask.any():
                real_betas = beta_values[real_mask]
                # Clamp to [0, 1] and scale to [1, n_bins-1]
                clamped = torch.clamp(real_betas, 0.0, 1.0)
                # Map 0-1 to 1-(n_bins-1): β=0→1, β=1→(n_bins-1)
                bin_idx = (clamped * (self.n_bins - 2) + 1).long()
                bin_idx = torch.clamp(bin_idx, 1, self.n_bins - 1)
                bin_indices[real_mask] = bin_idx

            # Get bin embeddings
            bin_embeds = self.bin_embeddings(bin_indices)  # [batch, seq, hidden]
            bin_embeds = self.bin_layer_norm(bin_embeds)

            # Combine: CpG embedding + bin embedding (ADD style like scGPT)
            hidden_states = cpg_embeds + bin_embeds

            if not hasattr(self, '_binned_debug_printed'):
                self._binned_debug_printed = True
                n_special = special_mask.sum().item()
                n_real = real_mask.sum().item()
                print(f"\n[BINNED MODE]")
                print(f"  Token breakdown: Special={n_special}, Real={n_real}")
                print(f"  Number of bins: {self.n_bins}")
                print(f"  Bin index range: [{bin_indices.min().item()}, {bin_indices.max().item()}]")
                print(f"  cpg_embeds norm: {cpg_embeds[0, 0].norm():.4f}")
                print(f"  bin_embeds norm: {bin_embeds[0, 0].norm():.4f}")
                print(f"  hidden_states norm: {hidden_states[0, 0].norm():.4f}")

        else:
            # ADD mode: h = α * CpG_embed + β_embed
            scaled_cpg_embeds = self.cpg_scale * cpg_embeds
            beta_embeds = embeddings_layer.beta_values_embeddings(beta_values)
            hidden_states = scaled_cpg_embeds + beta_embeds

        # Apply LayerNorm and dropout
        hidden_states = embeddings_layer.LayerNorm(hidden_states)
        hidden_states = embeddings_layer.dropout(hidden_states)

        # =================================================================
        # Step 5: Pass through transformer encoder
        # =================================================================
        # Create attention mask using the encoder's helper method
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=input_ids.device)

        # Handle 3D attention mask (multi-field)
        if attention_mask.dim() == 3:
            attention_mask = attention_mask[:, 0, :]

        extended_attention_mask = self.encoder.get_extended_attention_mask(
            attention_mask, (batch_size, seq_length)
        )

        # Create head mask
        head_mask = self.encoder.get_head_mask(
            None, self.encoder.config.num_hidden_layers
        )

        # Pass through encoder layers
        encoder_outputs = self.encoder.encoder(
            hidden_states,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
        )

        sequence_output = encoder_outputs[0]  # [batch, seq_len, 512]

        # =================================================================
        # Step 6: Mean pooling (exclude padding)
        # =================================================================
        if attention_mask is not None:
            # Mask out padding tokens
            mask_expanded = attention_mask.unsqueeze(-1).float()  # [batch, seq_len, 1]
            sum_embeddings = torch.sum(sequence_output * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled_output = sum_embeddings / sum_mask
        else:
            pooled_output = sequence_output.mean(dim=1)

        # =================================================================
        # Step 7: Age prediction head
        # =================================================================
        age_pred = self.age_head(pooled_output)

        return age_pred

    def on_train_epoch_start(self):
        """Unfreeze encoder at specified epoch."""
        current_epoch = self.current_epoch

        # Log CpG scale at start of each epoch
        cpg_scale_value = self.cpg_scale.item()
        logger.info(f"[Epoch {current_epoch}] CpG scale = {cpg_scale_value:.6f}")
        print(f"[EPOCH {current_epoch}] CpG scale = {cpg_scale_value:.6f}")

        if self.freeze_encoder and current_epoch == self.unfreeze_encoder_epoch:
            logger.info("=" * 70)
            logger.info(f"[EPOCH {current_epoch}] UNFREEZING encoder")
            logger.info("=" * 70)
            print(f"\n{'='*70}")
            print(f"[EPOCH {current_epoch}] UNFREEZING ENCODER")
            print(f"{'='*70}\n")

            for param in self.encoder.parameters():
                param.requires_grad = True

            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            logger.info(f"[EPOCH {current_epoch}] Trainable params after unfreeze: {trainable:,}")
            print(f"[EPOCH {current_epoch}] Trainable params: {trainable:,}")

    def _shared_step(self, batch, stage: str):
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
        labels = batch["labels"].float().view(-1, 1)

        # Debug: print batch info on first step
        if not hasattr(self, '_debug_printed') or not self._debug_printed:
            self._debug_printed = True
            logger.info("=" * 70)
            logger.info("DEBUG: BATCH INSPECTION (Scaled CpG Model)")
            logger.info(f"  input_ids shape: {input_ids.shape}")
            logger.info(f"  CpG scale: {self.cpg_scale.item():.6f}")
            if input_ids.dim() == 3:
                logger.info(f"  Field 0 (cpg_sites) first 5: {input_ids[0, 0, :5].tolist()}")
                logger.info(f"  Field 1 (beta_values) first 5: {input_ids[0, 1, :5].tolist()}")
            logger.info(f"  labels first 5: {labels[:5, 0].tolist()}")
            logger.info("=" * 70)
            print(f"\n{'='*70}")
            print(f"DEBUG: BATCH INSPECTION (Scaled CpG Model)")
            print(f"  input_ids shape: {input_ids.shape}")
            print(f"  CpG scale: {self.cpg_scale.item():.6f}")
            print(f"  labels std: {labels.std().item():.4f}")
            print(f"{'='*70}\n")

        # Forward pass
        predictions = self(input_ids, attention_mask)

        # Debug predictions on first few steps
        if not hasattr(self, '_debug_pred_count'):
            self._debug_pred_count = 0
        if self._debug_pred_count < 5:
            self._debug_pred_count += 1
            preds_denorm_sample = predictions[:5, 0] * self.age_std + self.age_mean
            labels_denorm_sample = labels[:5, 0] * self.age_std + self.age_mean
            print(f"\n[DEBUG STEP {self._debug_pred_count}]")
            print(f"  Predictions (normalized): {predictions[:5, 0].tolist()}")
            print(f"  Predictions (denorm age): {preds_denorm_sample.tolist()}")
            print(f"  Labels (denorm age):      {labels_denorm_sample.tolist()}")
            print(f"  Preds std: {predictions.std().item():.6f}")
            print(f"  Labels std: {labels.std().item():.6f}")

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
        self.log("cpg_scale", self.cpg_scale.item(), on_step=False, on_epoch=True, prog_bar=True)

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
            r2 = 1 - ss_res / (ss_tot + 1e-8)
            self.log("val/r2", r2, prog_bar=True)

            # Log CpG scale
            self.log("val/cpg_scale", self.cpg_scale.item())

            epoch_mae = torch.abs(all_preds - all_labels).mean()
            logger.info(f"[Val] MAE={epoch_mae:.2f}, R²={r2:.4f}, CpG_scale={self.cpg_scale.item():.6f}")
            print(f"[VAL] MAE={epoch_mae:.2f} years, R²={r2:.4f}, CpG_scale={self.cpg_scale.item():.6f}")

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

            logger.info(f"Test MAE: {epoch_mae:.2f} years, R²: {r2:.4f}, CpG_scale: {self.cpg_scale.item():.6f}")
            print(f"\n{'='*70}")
            print(f"TEST RESULTS")
            print(f"  MAE: {epoch_mae:.2f} years")
            print(f"  R²: {r2:.4f}")
            print(f"  Final CpG scale: {self.cpg_scale.item():.6f}")
            print(f"{'='*70}\n")

        self._test_preds.clear()
        self._test_labels.clear()

    def configure_optimizers(self):
        """Setup optimizer with different LR for encoder vs head."""
        no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]

        optimizer_grouped_parameters = [
            # Encoder parameters (lower LR)
            {
                "params": [p for n, p in self.encoder.named_parameters()
                           if p.requires_grad and not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
                "lr": self.hparams.learning_rate * 0.1,  # Lower LR for pretrained
            },
            {
                "params": [p for n, p in self.encoder.named_parameters()
                           if p.requires_grad and any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": self.hparams.learning_rate * 0.1,
            },
            # Age head parameters (higher LR)
            {
                "params": [p for n, p in self.age_head.named_parameters()
                           if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
                "lr": self.hparams.head_learning_rate,
            },
            {
                "params": [p for n, p in self.age_head.named_parameters()
                           if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": self.hparams.head_learning_rate,
            },
            # CpG scale parameter (same as head LR)
            {
                "params": [self.cpg_scale],
                "weight_decay": 0.0,
                "lr": self.hparams.head_learning_rate,
            },
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
                return float(current_step) / float(max(1, self.hparams.warmup_steps))
            progress = float(current_step - self.hparams.warmup_steps) / \
                       float(max(1, self.hparams.max_steps - self.hparams.warmup_steps))
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

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
        print(f"[SETUP] Loading tokenizer from {tokenizer_path}")
        from bmfm_targets.tokenization import MultiFieldTokenizer
        tokenizer = MultiFieldTokenizer.from_pretrained(str(tokenizer_path))
    else:
        logger.info(f"Creating new tokenizer from {cfg.data_path}")
        print(f"[SETUP] Creating new tokenizer")
        cpg_sites = extract_cpg_sites_from_h5ad(cfg.data_path)
        tokenizer = create_methylation_multifield_tokenizer(
            cpg_sites=cpg_sites,
            output_dir=str(tokenizer_path),
        )

    return tokenizer


def setup_wandb(cfg: DictConfig):
    """Setup WandB logging if enabled."""
    wandb_enabled = False
    if hasattr(cfg, 'track_wandb') and cfg.track_wandb.get('enabled', False):
        wandb_enabled = True

    if wandb_enabled:
        try:
            import wandb
            from pytorch_lightning.loggers import WandbLogger

            project = cfg.track_wandb.get('project', 'methylation-age-scaled')
            entity = cfg.track_wandb.get('entity', None)
            run_name = cfg.track_wandb.get('name', None)

            wandb_logger = WandbLogger(
                project=project,
                entity=entity,
                name=run_name,
                save_dir=cfg.output_directory,
                log_model=True,
            )

            wandb_logger.experiment.config.update(OmegaConf.to_container(cfg, resolve=True))

            logger.info(f"WandB logging enabled - Project: {project}")
            print(f"[SETUP] WandB enabled - Project: {project}")
            return wandb_logger
        except Exception as e:
            logger.warning(f"WandB setup failed: {e}")
            print(f"[SETUP] WandB failed: {e}")

    from pytorch_lightning.loggers import TensorBoardLogger
    return TensorBoardLogger(cfg.output_directory, name="finetune_scaled")


@hydra.main(
    config_path="configs",
    config_name="finetune_transformer_scaled_config",
    version_base="1.2"
)
def main(cfg: DictConfig):
    """Main fine-tuning function with scaled CpG embeddings."""
    print("\n" + "=" * 70)
    print("TRANSFORMER FINE-TUNING WITH SCALED CpG EMBEDDINGS")
    print("=" * 70)

    logger.info("=" * 70)
    logger.info("TRANSFORMER FINE-TUNING WITH SCALED CpG EMBEDDINGS")
    logger.info("=" * 70)
    logger.info(f"\nConfiguration:\n{OmegaConf.to_yaml(cfg)}")

    # Set seed
    if hasattr(cfg, 'seed') and cfg.seed:
        pl.seed_everything(cfg.seed.seed_value, workers=True)
        print(f"[SETUP] Seed: {cfg.seed.seed_value}")

    # Setup output directory
    output_dir = Path(cfg.output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[SETUP] Output: {output_dir}")

    # Setup tokenizer
    tokenizer = setup_tokenizer(cfg)

    # Setup fields
    from bmfm_targets.config import FieldInfo
    fields = []
    for field_cfg in cfg.fields:
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
        mlm=False,
        collation_strategy="sequence_classification",
    )
    data_module.setup()

    print(f"[DATA] Train: {len(data_module.train_dataset)} samples")
    print(f"[DATA] Val: {len(data_module.val_dataset)} samples")
    print(f"[DATA] Test: {len(data_module.test_dataset)} samples")

    # Setup model config
    model_config_partial = hydra.utils.instantiate(cfg.model)
    model_config = model_config_partial(fields=fields)

    # Load pretrained encoder
    from bmfm_targets.models.predictive.scbert.modeling_scbert import SCBertModel

    if cfg.checkpoint_path and cfg.checkpoint_path != "null":
        print(f"[MODEL] Loading checkpoint: {cfg.checkpoint_path}")
        logger.info(f"Loading pretrained checkpoint: {cfg.checkpoint_path}")

        from bmfm_targets.training.modules.masked_language_modeling import MLMTrainingModule
        from bmfm_targets.config import TrainerConfig, SCBertConfig, FieldInfo

        torch.serialization.add_safe_globals([SCBertConfig, TrainerConfig, FieldInfo])

        trainer_config = TrainerConfig(
            learning_rate=cfg.trainer.learning_rate,
            weight_decay=cfg.trainer.weight_decay,
            warmup_steps=cfg.trainer.warmup_steps,
            losses=[{"name": "cross_entropy"}],
        )

        pretrained_module = MLMTrainingModule.load_from_checkpoint(
            cfg.checkpoint_path,
            model_config=model_config,
            trainer_config=trainer_config,
            tokenizer=tokenizer,
            weights_only=False,
        )
        encoder = pretrained_module.model.scbert
        print("[MODEL] Loaded pretrained encoder successfully")
        logger.info("Loaded pretrained encoder (CpG IDs + beta values + transformer layers)")

        # Debug: Check encoder weights
        cpg_embed_weight = encoder.embeddings.cpg_sites_embeddings.weight
        print(f"[DEBUG] CpG embeddings shape: {cpg_embed_weight.shape}")
        print(f"[DEBUG] CpG embeddings norm: {cpg_embed_weight.norm():.4f}")
        print(f"[DEBUG] CpG embeddings mean: {cpg_embed_weight.mean():.6f}")
        print(f"[DEBUG] CpG embeddings std: {cpg_embed_weight.std():.6f}")
    else:
        print("[MODEL] Training from scratch (no checkpoint)")
        encoder = SCBertModel(model_config)

    # Get age statistics for denormalization
    age_mean = data_module.train_dataset.age_mean
    age_std = data_module.train_dataset.age_std
    print(f"[DATA] Age stats: mean={age_mean:.2f}, std={age_std:.2f}")

    # Calculate training steps
    train_size = len(data_module.train_dataset)
    batch_size = cfg.data_module.batch_size
    accum = cfg.get('accumulate_grad_batches', 1)
    steps_per_epoch = train_size // (batch_size * accum)
    max_steps = steps_per_epoch * cfg.finetune_epochs

    print(f"[TRAIN] Steps per epoch: {steps_per_epoch}")
    print(f"[TRAIN] Total steps: {max_steps}")

    # Create model
    initial_cpg_scale = cfg.get('initial_cpg_scale', 0.1)
    combine_style = cfg.get('combine_style', 'multiply')
    n_bins = cfg.get('n_bins', 51)  # Number of bins for binned mode (like scGPT)
    beta_offset = cfg.get('beta_offset', 0.5)  # Offset for multiply mode

    print(f"[MODEL] Combine style: {combine_style}")
    print(f"[MODEL] Initial CpG scale: {initial_cpg_scale}")
    if combine_style == "binned":
        print(f"[MODEL] Number of bins: {n_bins}")
    if combine_style == "multiply":
        print(f"[MODEL] Beta offset: {beta_offset}")

    model = TransformerScaledCpGAgeRegressor(
        encoder=encoder,
        hidden_size=cfg.model.hidden_size,
        head_hidden_size=cfg.regression_head.hidden_size,
        head_dropout=cfg.regression_head.dropout,
        learning_rate=cfg.trainer.learning_rate,
        head_learning_rate=cfg.trainer.learning_rate,
        weight_decay=cfg.trainer.weight_decay,
        warmup_steps=cfg.trainer.warmup_steps,
        max_steps=max_steps,
        age_mean=age_mean,
        age_std=age_std,
        freeze_encoder=cfg.get('freeze_encoder', True),
        unfreeze_encoder_epoch=cfg.get('unfreeze_encoder_epoch', 5),
        initial_cpg_scale=initial_cpg_scale,
        use_huber_loss=cfg.get('use_huber_loss', False),
        huber_delta=cfg.get('huber_delta', 2.0),
        combine_style=combine_style,
        n_bins=n_bins,
        beta_offset=beta_offset,
    )

    if combine_style == "multiply":
        print(f"[MODEL] Architecture: h = CpG_embed * (β + {beta_offset}) → Transformer → MLP → age")
        print(f"[MODEL] (scGPT style with offset - β=0→{beta_offset}, β=1→{1+beta_offset})")
    elif combine_style == "binned":
        print(f"[MODEL] Architecture: h = CpG_embed + bin_embed(β) → Transformer → MLP → age")
        print(f"[MODEL] (scGPT category style - {n_bins} bins)")
    else:
        print(f"[MODEL] Architecture: h = α*CpG_embed + β_embed → Transformer → MLP → age")
        print(f"[MODEL] α (CpG scale) is LEARNABLE, initialized to {initial_cpg_scale}")

    # Setup logger
    exp_logger = setup_wandb(cfg)

    # Callbacks
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="epoch={epoch}-val_mae={val/mae:.4f}-cpg_scale={val/cpg_scale:.4f}",
        monitor="val/mae",
        mode="min",
        save_top_k=3,
        auto_insert_metric_name=False,
    )

    early_stopping = pl.callbacks.EarlyStopping(
        monitor=cfg.early_stopping.monitor,
        patience=cfg.early_stopping.patience,
        mode=cfg.early_stopping.mode,
        verbose=True,
    )

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")

    # Trainer
    trainer = pl.Trainer(
        default_root_dir=str(output_dir),
        max_epochs=cfg.finetune_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed",
        accumulate_grad_batches=cfg.get('accumulate_grad_batches', 1),
        val_check_interval=0.25,
        callbacks=[checkpoint_callback, early_stopping, lr_monitor],
        logger=exp_logger,
        log_every_n_steps=10,
        enable_progress_bar=True,
    )

    # Train
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70 + "\n")
    logger.info("Starting fine-tuning with scaled CpG embeddings...")

    trainer.fit(model, data_module)

    # Test
    print("\n" + "=" * 70)
    print("RUNNING TEST EVALUATION")
    print("=" * 70 + "\n")
    logger.info("Running test evaluation...")

    trainer.test(model, data_module, ckpt_path="best")

    # Summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Best val/mae: {checkpoint_callback.best_model_score:.4f}")
    print(f"Final CpG scale: {model.cpg_scale.item():.6f}")
    print("=" * 70 + "\n")

    logger.info(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    logger.info(f"Best val/mae: {checkpoint_callback.best_model_score}")


if __name__ == "__main__":
    main()
