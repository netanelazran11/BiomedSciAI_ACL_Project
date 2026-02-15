"""
WCED Training Module - PyTorch Lightning module for WCED pretraining

This module implements the Whole Cell Expression Decoder (WCED) pretraining
strategy for methylation data:

1. Encode all CpG sites with their beta values (NO masking)
2. Extract the [CLS] token representation
3. Decode ALL beta values from [CLS] using a deep MLP
4. Compute MSE loss between predicted and actual beta values

This creates a global bottleneck that forces the [CLS] token to aggregate
information from the entire methylation profile into a single vector.

Comparison with MLM:
- MLM: Predicts MASKED positions only → per-token representations
- WCED: Predicts ALL positions from [CLS] → global representation

After WCED pretraining, [CLS] can be directly used for downstream tasks
without additional pooling strategies.
"""

import logging
from typing import Dict, Any, Optional, List

import torch
import torch.nn as nn
import pytorch_lightning as pl
from scipy.stats import pearsonr
import numpy as np

from bmfm_targets.config import SCBertConfig
from bmfm_targets.models.predictive.scbert.modeling_scbert import SCBertModel

from .decoders import WCEDDecoder, WCEDLoss
from .config import PretrainingConfig

logger = logging.getLogger(__name__)


class WCEDTrainingModule(pl.LightningModule):
    """
    PyTorch Lightning module for WCED pretraining.

    Architecture:
        [CpG IDs + Beta Values] → Encoder → [CLS] → Decoder → Reconstructed Betas

    The encoder processes all CpG sites with their beta values (no masking).
    The [CLS] token is extracted and passed through the WCED decoder.
    The decoder reconstructs all beta values, and MSE loss is computed.

    Args:
        model_config: SCBertConfig for the encoder
        pretrain_config: PretrainingConfig with WCED settings
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for AdamW
        warmup_steps: Number of warmup steps
        lr_decay_steps: Total steps for LR decay
        num_cpg_sites: Number of CpG sites to reconstruct
    """

    def __init__(
        self,
        model_config: SCBertConfig,
        pretrain_config: Optional[PretrainingConfig] = None,
        learning_rate: float = 5e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 100,
        lr_decay_steps: int = 10000,
        num_cpg_sites: int = 2048,
        betas: tuple = (0.9, 0.999),
        epsilon: float = 1e-8,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model_config', 'pretrain_config'])

        # Store configs
        self.model_config = model_config
        if pretrain_config is None:
            pretrain_config = PretrainingConfig(mode="wced")
        self.pretrain_config = pretrain_config

        # Encoder (BMFM SCBertModel)
        self.encoder = SCBertModel(model_config, add_pooling_layer=True)

        # Apply ADD fusion stabilization (scale down CpG embeddings)
        self._patch_embeddings_add_stabilized()

        # WCED Decoder
        if pretrain_config.use_positional_decoder:
            from .decoders import WCEDWithPositionalDecoder
            self.decoder = WCEDWithPositionalDecoder(
                hidden_size=model_config.hidden_size,
                num_cpg_sites=num_cpg_sites,
                dropout=pretrain_config.decoder_dropout,
            )
            logger.info("Using positional WCED decoder (attention-based)")
        else:
            self.decoder = WCEDDecoder(
                hidden_size=model_config.hidden_size,
                num_cpg_sites=num_cpg_sites,
                decoder_hidden_sizes=pretrain_config.decoder_hidden_sizes,
                dropout=pretrain_config.decoder_dropout,
            )
            logger.info(f"Using standard WCED decoder: {model_config.hidden_size} → {pretrain_config.decoder_hidden_sizes} → {num_cpg_sites}")

        # Loss function
        self.loss_fn = WCEDLoss(reduction='mean')

        # Log model info
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        logger.info(f"WCED Training Module initialized:")
        logger.info(f"  Encoder params: {encoder_params:,}")
        logger.info(f"  Decoder params: {decoder_params:,}")
        logger.info(f"  Total params: {encoder_params + decoder_params:,}")
        logger.info(f"  Pretraining mode: WCED (reconstruct ALL from [CLS])")

    def _patch_embeddings_add_stabilized(self, initial_cpg_scale: float = 0.1):
        """
        Patch embeddings to use ADD fusion with learnable CpG scaling.

        h = alpha * CpG_embed + beta_embed
        """
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
            # Replace special sentinels with neutral value
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
            beta_values: [batch, seq_len] - Beta values (including CLS/PAD sentinels)
            attention_mask: [batch, seq_len] - Attention mask

        Returns:
            Dict with:
                - predicted_betas: [batch, num_cpg_sites] - Reconstructed values
                - cls_embedding: [batch, hidden_size] - [CLS] representation
        """
        batch_size, seq_len = cpg_ids.shape

        # Build BMFM-style input: [batch, 2, seq_len]
        input_ids = torch.stack([cpg_ids.float(), beta_values], dim=1)

        # Encode
        encoder_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Extract [CLS] token (pooler_output)
        cls_embedding = encoder_output.pooler_output  # [batch, hidden_size]

        # Decode to all beta values
        predicted_betas = self.decoder(cls_embedding)  # [batch, num_cpg_sites]

        return {
            "predicted_betas": predicted_betas,
            "cls_embedding": cls_embedding,
        }

    def _extract_target_betas(
        self,
        beta_values: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract target beta values for loss computation.

        Args:
            beta_values: [batch, seq_len] - Input beta values with sentinels
            attention_mask: [batch, seq_len] - Attention mask

        Returns:
            target_betas: [batch, num_cpg_sites] - Clean target values
        """
        # Skip CLS token (position 0), extract content tokens
        # Beta values for CLS is sentinel (-2.0), PAD is sentinel (-3.0)
        content_betas = beta_values[:, 1:]  # Skip CLS
        content_mask = attention_mask[:, 1:]  # Skip CLS

        batch_size = beta_values.size(0)
        num_cpg_sites = self.decoder.num_cpg_sites

        # Initialize target tensor
        target_betas = torch.zeros(batch_size, num_cpg_sites, device=beta_values.device)

        # Copy valid beta values (those that are >= 0, not sentinels)
        for i in range(batch_size):
            valid_mask = (content_betas[i] >= 0) & (content_mask[i] == 1)
            valid_values = content_betas[i][valid_mask]
            # Pad or truncate to num_cpg_sites
            n_valid = min(len(valid_values), num_cpg_sites)
            target_betas[i, :n_valid] = valid_values[:n_valid]

        return target_betas

    def _shared_step(self, batch: Dict[str, torch.Tensor], stage: str) -> Dict[str, torch.Tensor]:
        """Shared step for train/val/test."""
        cpg_ids = batch["cpg_ids"]
        beta_values = batch["beta_values"]
        attention_mask = batch.get("attention_mask")

        # Forward pass
        outputs = self(cpg_ids, beta_values, attention_mask)
        predicted_betas = outputs["predicted_betas"]

        # Extract target betas (skip CLS, handle sentinels)
        target_betas = self._extract_target_betas(beta_values, attention_mask)

        # Create valid mask for loss (positions with real beta values)
        valid_mask = (target_betas > 0) | (target_betas == 0)  # All non-sentinel values

        # Compute loss
        loss = self.loss_fn(predicted_betas, target_betas, valid_mask.float())

        # Compute metrics
        with torch.no_grad():
            mae = torch.abs(predicted_betas - target_betas).mean()
            mse = ((predicted_betas - target_betas) ** 2).mean()

        return {
            "loss": loss,
            "mae": mae,
            "mse": mse,
            "predicted_betas": predicted_betas,
            "target_betas": target_betas,
        }

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        outputs = self._shared_step(batch, "train")

        self.log("train/loss", outputs["loss"], on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/mae", outputs["mae"], on_step=False, on_epoch=True)
        self.log("train/mse", outputs["mse"], on_step=False, on_epoch=True)

        return outputs["loss"]

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        outputs = self._shared_step(batch, "val")

        self.log("validation/loss", outputs["loss"], on_epoch=True, prog_bar=True)
        self.log("validation/mae", outputs["mae"], on_epoch=True)
        self.log("validation/mse", outputs["mse"], on_epoch=True)

        return outputs["loss"]

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step."""
        outputs = self._shared_step(batch, "test")

        self.log("test/loss", outputs["loss"], on_epoch=True)
        self.log("test/mae", outputs["mae"], on_epoch=True)
        self.log("test/mse", outputs["mse"], on_epoch=True)

        # Compute PCC for test
        pred = outputs["predicted_betas"].detach().cpu().numpy().flatten()
        target = outputs["target_betas"].detach().cpu().numpy().flatten()
        if len(pred) > 1:
            pcc, _ = pearsonr(pred, target)
            self.log("test/pcc", pcc, on_epoch=True)

        return outputs["loss"]

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        # AdamW optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            betas=self.hparams.betas,
            eps=self.hparams.epsilon,
        )

        # Cosine scheduler with warmup
        def lr_lambda(current_step):
            if current_step < self.hparams.warmup_steps:
                return float(current_step) / float(max(1, self.hparams.warmup_steps))
            progress = float(current_step - self.hparams.warmup_steps) / \
                       float(max(1, self.hparams.lr_decay_steps - self.hparams.warmup_steps))
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
