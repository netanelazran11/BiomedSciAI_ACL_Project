"""
WCED Training Module - Whole Cell Expression Decoder for Methylation

The key insight: The decoder must know WHICH CpG it's predicting for.
We achieve this by using the SAME CpG embeddings in both encoder and decoder.

Architecture:
    Encoder:
        [CpG_1+β_1, CpG_2+β_2, ..., CpG_n+β_n] → Transformer → [CLS]

    Decoder (for each CpG_i):
        query = CpG_embedding(cpg_id_i)  # Same embeddings as encoder!
        predicted_β_i = MLP(CrossAttention(query, [CLS]))

This creates:
1. Information bottleneck: ALL info compressed into [CLS]
2. CpG-aware decoding: Decoder knows which CpG it's predicting via shared embeddings
3. Proper WCED objective: Reconstruct ALL betas from global representation
"""

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
import pytorch_lightning as pl
from scipy.stats import pearsonr
import numpy as np

from bmfm_targets.config import SCBertConfig
from bmfm_targets.models.predictive.scbert.modeling_scbert import SCBertModel

from .config import PretrainingConfig

logger = logging.getLogger(__name__)


class WCEDDecoder(nn.Module):
    """
    WCED Decoder: CpG-aware decoder using shared embeddings.

    Key insight: Use the SAME CpG embeddings from encoder as queries.
    This way the decoder knows exactly which CpG it's predicting for.

    Architecture:
        For each CpG in the sequence:
            1. Get CpG embedding (shared with encoder)
            2. Cross-attend to [CLS] to get context
            3. Project to beta value
    """

    def __init__(
        self,
        cpg_embeddings: nn.Embedding,  # Shared from encoder!
        hidden_size: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Store reference to encoder's CpG embeddings (shared weights!)
        self.cpg_embeddings = cpg_embeddings
        self.hidden_size = hidden_size

        # Cross-attention: CpG queries attend to [CLS]
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Layer norm after attention
        self.norm = nn.LayerNorm(hidden_size)

        # Output projection: attended representation → beta value
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),  # Beta values are in [0, 1]
        )

        self._init_weights()

    def _init_weights(self):
        # Initialize output projection (cross-attention has its own init)
        for module in self.output_proj.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        cls_embedding: torch.Tensor,
        cpg_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict beta values using CpG embeddings as queries to [CLS].

        Args:
            cls_embedding: [batch, hidden_size] - [CLS] representation
            cpg_ids: [batch, seq_len] - CpG IDs at each position
            attention_mask: [batch, seq_len] - Mask for valid positions

        Returns:
            predicted_betas: [batch, seq_len] - Predicted beta values
        """
        batch_size, seq_len = cpg_ids.shape

        # Get CpG embeddings as queries (using SHARED encoder embeddings)
        # These embeddings know the identity of each CpG
        queries = self.cpg_embeddings(cpg_ids.long())  # [batch, seq_len, hidden]

        # [CLS] is the key and value (contains all compressed info)
        cls_expanded = cls_embedding.unsqueeze(1)  # [batch, 1, hidden]

        # Cross-attention: each CpG query asks [CLS] for its beta value
        # Q: [batch, seq_len, hidden], K/V: [batch, 1, hidden]
        attended, _ = self.cross_attention(
            query=queries,
            key=cls_expanded,
            value=cls_expanded,
        )  # [batch, seq_len, hidden]

        # Residual connection + layer norm
        attended = self.norm(queries + attended)

        # Project to beta values
        predicted_betas = self.output_proj(attended).squeeze(-1)  # [batch, seq_len]

        return predicted_betas


class WCEDTrainingModule(pl.LightningModule):
    """
    WCED Training Module - Whole Cell Expression Decoder for Methylation.

    Architecture:
        Encoder: [CpG_1+β_1, ..., CpG_n+β_n] → Transformer → [CLS]
        Decoder: CpG_embed(cpg_id) → CrossAttention([CLS]) → predicted_β

    The decoder shares CpG embeddings with encoder, so it knows exactly
    which CpG it's predicting for at each position.
    """

    def __init__(
        self,
        model_config: SCBertConfig,
        pretrain_config: Optional[PretrainingConfig] = None,
        learning_rate: float = 5e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 100,
        lr_decay_steps: int = 10000,
        num_heads: int = 8,
        betas: tuple = (0.9, 0.999),
        epsilon: float = 1e-8,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model_config', 'pretrain_config'])

        self.model_config = model_config
        if pretrain_config is None:
            pretrain_config = PretrainingConfig(mode="wced")
        self.pretrain_config = pretrain_config

        # Encoder
        self.encoder = SCBertModel(model_config, add_pooling_layer=True)

        # Apply ADD fusion stabilization
        self._patch_embeddings_add_stabilized()

        # Get the CpG embeddings from encoder (to share with decoder)
        cpg_embeddings = self.encoder.embeddings.cpg_sites_embeddings

        # WCED Decoder: uses shared CpG embeddings for CpG-aware prediction
        self.decoder = WCEDDecoder(
            cpg_embeddings=cpg_embeddings,  # Shared with encoder!
            hidden_size=model_config.hidden_size,
            num_heads=num_heads,
            dropout=pretrain_config.decoder_dropout,
        )

        # Loss function
        self.loss_fn = nn.MSELoss(reduction='none')

        # Log model info
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        # Note: CpG embeddings are shared, so counted only once in encoder
        shared_params = sum(p.numel() for p in cpg_embeddings.parameters())

        logger.info(f"WCED Training Module initialized:")
        logger.info(f"  Encoder params: {encoder_params:,}")
        logger.info(f"  Decoder params (excluding shared): {decoder_params - shared_params:,}")
        logger.info(f"  Shared CpG embeddings: {shared_params:,}")
        logger.info(f"  Total params: {encoder_params + decoder_params - shared_params:,}")
        logger.info(f"  Mode: WCED (CpG-aware decoder with shared embeddings)")

    def _patch_embeddings_add_stabilized(self, initial_cpg_scale: float = 0.1):
        """Patch embeddings to use ADD fusion with learnable CpG scaling."""
        embeddings_layer = self.encoder.embeddings
        embeddings_layer.cpg_scale = nn.Parameter(torch.tensor(float(initial_cpg_scale)))

        def add_forward(input_ids, position_ids=None, inputs_embeds=None):
            if inputs_embeds is not None:
                return inputs_embeds

            batch_size, num_fields, seq_length = input_ids.shape

            # Field 0: CpG IDs
            cpg_ids = input_ids[:, 0, :].long()
            cpg_embeds = embeddings_layer.cpg_sites_embeddings(cpg_ids)

            # Field 1: beta values
            beta_values = input_ids[:, 1, :].float()
            beta_values_clean = beta_values.clone()
            beta_values_clean[beta_values_clean < 0] = 0.0
            beta_embeds = embeddings_layer.beta_values_embeddings(beta_values_clean)

            hidden_states = embeddings_layer.cpg_scale * cpg_embeds + beta_embeds

            if embeddings_layer.position_embedding_type is not None:
                if position_ids is None:
                    position_ids = embeddings_layer.position_ids[:, :seq_length]
                position_embeddings = embeddings_layer.position_embeddings(position_ids)
                hidden_states = hidden_states + position_embeddings

            hidden_states = embeddings_layer.LayerNorm(hidden_states)
            hidden_states = embeddings_layer.dropout(hidden_states)
            return hidden_states

        embeddings_layer.forward = add_forward

    def forward(
        self,
        cpg_ids: torch.Tensor,
        beta_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for WCED.

        Args:
            cpg_ids: [batch, seq_len] - CpG site token IDs
            beta_values: [batch, seq_len] - Beta values
            attention_mask: [batch, seq_len] - Attention mask

        Returns:
            Dict with predicted_betas and cls_embedding
        """
        batch_size, seq_len = cpg_ids.shape

        # Build BMFM-style input: [batch, 2, seq_len]
        input_ids = torch.stack([cpg_ids.float(), beta_values], dim=1)

        # Encode - get [CLS] representation
        encoder_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Extract [CLS] embedding (global representation)
        cls_embedding = encoder_output.pooler_output  # [batch, hidden]

        # Decode: use CpG IDs to query [CLS] for each beta value
        predicted_betas = self.decoder(
            cls_embedding=cls_embedding,
            cpg_ids=cpg_ids,
            attention_mask=attention_mask,
        )  # [batch, seq_len]

        return {
            "predicted_betas": predicted_betas,
            "cls_embedding": cls_embedding,
        }

    def _shared_step(self, batch: Dict[str, torch.Tensor], stage: str) -> Dict[str, torch.Tensor]:
        """Shared step for train/val/test."""
        cpg_ids = batch["cpg_ids"]
        beta_values = batch["beta_values"]
        attention_mask = batch.get("attention_mask")

        # Forward pass
        outputs = self(cpg_ids, beta_values, attention_mask)
        predicted_betas = outputs["predicted_betas"]

        # Target: original beta values
        target_betas = beta_values.clone()

        # Create mask for valid positions (non-CLS, non-PAD, valid beta values)
        # Position 0 is CLS (beta = -2.0), PAD positions have beta = -3.0
        valid_mask = (target_betas >= 0) & (attention_mask == 1)

        # Compute loss only on valid positions
        loss_per_pos = self.loss_fn(predicted_betas, target_betas.clamp(0, 1))
        loss = (loss_per_pos * valid_mask.float()).sum() / valid_mask.float().sum().clamp(min=1)

        # Compute metrics on valid positions
        with torch.no_grad():
            valid_pred = predicted_betas[valid_mask]
            valid_target = target_betas[valid_mask].clamp(0, 1)
            mae = torch.abs(valid_pred - valid_target).mean()
            mse = ((valid_pred - valid_target) ** 2).mean()

        return {
            "loss": loss,
            "mae": mae,
            "mse": mse,
            "predicted_betas": predicted_betas,
            "target_betas": target_betas,
            "valid_mask": valid_mask,
        }

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        outputs = self._shared_step(batch, "train")

        self.log("train/loss", outputs["loss"], on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/mae", outputs["mae"], on_step=False, on_epoch=True)
        self.log("train/mse", outputs["mse"], on_step=False, on_epoch=True)

        return outputs["loss"]

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        outputs = self._shared_step(batch, "val")

        self.log("validation/loss", outputs["loss"], on_epoch=True, prog_bar=True)
        self.log("validation/mae", outputs["mae"], on_epoch=True)
        self.log("validation/mse", outputs["mse"], on_epoch=True)

        return outputs["loss"]

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        outputs = self._shared_step(batch, "test")

        self.log("test/loss", outputs["loss"], on_epoch=True)
        self.log("test/mae", outputs["mae"], on_epoch=True)
        self.log("test/mse", outputs["mse"], on_epoch=True)

        # Compute PCC
        valid_mask = outputs["valid_mask"]
        pred = outputs["predicted_betas"][valid_mask].detach().cpu().numpy()
        target = outputs["target_betas"][valid_mask].clamp(0, 1).detach().cpu().numpy()
        if len(pred) > 1:
            pcc, _ = pearsonr(pred, target)
            self.log("test/pcc", pcc, on_epoch=True)

        return outputs["loss"]

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            betas=self.hparams.betas,
            eps=self.hparams.epsilon,
        )

        lr_decay_steps = self.hparams.lr_decay_steps
        if lr_decay_steps <= 0:
            if self.trainer is not None and self.trainer.estimated_stepping_batches is not None:
                lr_decay_steps = int(self.trainer.estimated_stepping_batches)
            else:
                lr_decay_steps = 300 * 45

        warmup_steps = self.hparams.warmup_steps

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, lr_decay_steps - warmup_steps))
            progress = min(progress, 1.0)
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

    def get_encoder(self) -> SCBertModel:
        """Get the pretrained encoder for downstream tasks."""
        return self.encoder
