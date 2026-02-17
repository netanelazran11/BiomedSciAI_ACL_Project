"""
WCED Training Module - Whole Cell Expression Decoder for Methylation

Based on the original BMFM WCED implementation:
1. Input: Random subset of CpGs (e.g., 80%)
2. Encoder: Processes input → CLS hidden state
3. Decoder: Linear(CLS) → ALL CpG betas (entire vocabulary)
4. Loss: MSE only on non-input CpGs (the 20% not in input)

This forces the CLS token to learn a global representation that can
predict CpGs it hasn't seen in the input.

Architecture:
    Input:  [CLS, CpG_a, CpG_b, ...] (random 80% subset)
    Encoder: Transformer → hidden_states
    Decoder: Linear(hidden_states[0]) → [β_0, β_1, ..., β_vocab_size]
    Loss:   MSE on non-input CpGs only
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
    WCED Decoder: Simple linear layer from CLS to entire vocabulary.

    This is the correct WCED architecture from the original BMFM:
    - Input: CLS hidden state [batch, hidden_size]
    - Output: Predicted betas for ALL CpGs [batch, vocab_size]

    Each output neuron corresponds to a fixed CpG in the vocabulary.
    """

    def __init__(
        self,
        hidden_size: int = 512,
        vocab_size: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        # Simple linear decoder with optional hidden layer
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, vocab_size),
            nn.Sigmoid(),  # Beta values are in [0, 1]
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.decoder.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, cls_hidden: torch.Tensor) -> torch.Tensor:
        """
        Predict ALL beta values from CLS hidden state.

        Args:
            cls_hidden: [batch, hidden_size] - CLS token representation

        Returns:
            predicted_betas: [batch, vocab_size] - Predicted beta for each CpG
        """
        return self.decoder(cls_hidden)


class WCEDTrainingModule(pl.LightningModule):
    """
    WCED Training Module - Correct implementation based on original BMFM.

    Architecture:
        Input:  Random 80% of CpGs with their beta values
        Encoder: Transformer → CLS hidden state
        Decoder: Linear(CLS) → ALL vocab_size beta predictions
        Loss:   MSE only on non-input CpGs (forces learning)

    Key insight: Loss is computed ONLY on CpGs NOT in the input.
    This forces the model to learn patterns, not just copy.
    """

    def __init__(
        self,
        model_config: SCBertConfig,
        pretrain_config: Optional[PretrainingConfig] = None,
        learning_rate: float = 5e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 100,
        lr_decay_steps: int = 10000,
        vocab_size: int = 2048,
        betas: tuple = (0.9, 0.999),
        epsilon: float = 1e-8,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model_config', 'pretrain_config'])

        self.model_config = model_config
        if pretrain_config is None:
            pretrain_config = PretrainingConfig(mode="wced")
        self.pretrain_config = pretrain_config
        self.vocab_size = vocab_size

        # Encoder
        self.encoder = SCBertModel(model_config, add_pooling_layer=True)

        # Apply ADD fusion stabilization
        self._patch_embeddings_add_stabilized()

        # WCED Decoder: Simple linear from CLS to vocab_size
        self.decoder = WCEDDecoder(
            hidden_size=model_config.hidden_size,
            vocab_size=vocab_size,
            dropout=pretrain_config.decoder_dropout,
        )

        # Loss function
        self.loss_fn = nn.MSELoss(reduction='none')

        # Log model info
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())

        logger.info(f"WCED Training Module initialized:")
        logger.info(f"  Encoder params: {encoder_params:,}")
        logger.info(f"  Decoder params: {decoder_params:,}")
        logger.info(f"  Total params: {encoder_params + decoder_params:,}")
        logger.info(f"  Vocab size: {vocab_size}")
        logger.info(f"  Mode: WCED (Linear decoder from CLS to all CpGs)")

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
            cpg_ids: [batch, seq_len] - CpG site token IDs (input subset)
            beta_values: [batch, seq_len] - Beta values (input subset)
            attention_mask: [batch, seq_len] - Attention mask

        Returns:
            Dict with predicted_betas and cls_embedding
        """
        batch_size, seq_len = cpg_ids.shape

        # Build BMFM-style input: [batch, 2, seq_len]
        input_ids = torch.stack([cpg_ids.float(), beta_values], dim=1)

        # Encode
        encoder_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Get CLS hidden state (position 0 or pooler output)
        cls_embedding = encoder_output.pooler_output  # [batch, hidden]

        # Decode: CLS → all vocab_size betas
        predicted_betas = self.decoder(cls_embedding)  # [batch, vocab_size]

        return {
            "predicted_betas": predicted_betas,
            "cls_embedding": cls_embedding,
        }

    def _shared_step(self, batch: Dict[str, torch.Tensor], stage: str) -> Dict[str, torch.Tensor]:
        """Shared step for train/val/test."""
        cpg_ids = batch["cpg_ids"]
        beta_values = batch["beta_values"]
        attention_mask = batch.get("attention_mask")
        all_betas = batch["all_betas"]        # [batch, vocab_size] - target
        input_mask = batch["input_mask"]      # [batch, vocab_size] - True if in input

        # Forward pass
        outputs = self(cpg_ids, beta_values, attention_mask)
        predicted_betas = outputs["predicted_betas"]  # [batch, vocab_size]

        # Loss ONLY on non-input CpGs (the ones model must infer)
        non_input_mask = ~input_mask  # [batch, vocab_size]

        # Compute loss
        loss_per_cpg = self.loss_fn(predicted_betas, all_betas)  # [batch, vocab_size]

        # Mask to only non-input CpGs
        masked_loss = loss_per_cpg * non_input_mask.float()
        loss = masked_loss.sum() / non_input_mask.float().sum().clamp(min=1)

        # Compute metrics on non-input CpGs
        with torch.no_grad():
            non_input_pred = predicted_betas[non_input_mask]
            non_input_target = all_betas[non_input_mask]
            mae = torch.abs(non_input_pred - non_input_target).mean()
            mse = ((non_input_pred - non_input_target) ** 2).mean()

            # Also compute metrics on ALL CpGs for comparison
            all_mae = torch.abs(predicted_betas - all_betas).mean()
            all_mse = ((predicted_betas - all_betas) ** 2).mean()

        return {
            "loss": loss,
            "mae": mae,           # MAE on non-input only
            "mse": mse,           # MSE on non-input only
            "all_mae": all_mae,   # MAE on all CpGs
            "all_mse": all_mse,   # MSE on all CpGs
            "predicted_betas": predicted_betas,
            "target_betas": all_betas,
            "non_input_mask": non_input_mask,
        }

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        outputs = self._shared_step(batch, "train")

        self.log("train/loss", outputs["loss"], on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/mae", outputs["mae"], on_step=False, on_epoch=True)
        self.log("train/mse", outputs["mse"], on_step=False, on_epoch=True)
        self.log("train/all_mae", outputs["all_mae"], on_step=False, on_epoch=True)

        return outputs["loss"]

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        outputs = self._shared_step(batch, "val")

        self.log("validation/loss", outputs["loss"], on_epoch=True, prog_bar=True)
        self.log("validation/mae", outputs["mae"], on_epoch=True)
        self.log("validation/mse", outputs["mse"], on_epoch=True)
        self.log("validation/all_mae", outputs["all_mae"], on_epoch=True)

        return outputs["loss"]

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        outputs = self._shared_step(batch, "test")

        self.log("test/loss", outputs["loss"], on_epoch=True)
        self.log("test/mae", outputs["mae"], on_epoch=True)
        self.log("test/mse", outputs["mse"], on_epoch=True)
        self.log("test/all_mae", outputs["all_mae"], on_epoch=True)

        # Compute PCC on non-input CpGs
        non_input_mask = outputs["non_input_mask"]
        pred = outputs["predicted_betas"][non_input_mask].detach().cpu().numpy()
        target = outputs["target_betas"][non_input_mask].detach().cpu().numpy()
        if len(pred) > 1:
            pcc, _ = pearsonr(pred, target)
            self.log("test/pcc", pcc, on_epoch=True)

        # PCC on all CpGs
        all_pred = outputs["predicted_betas"].detach().cpu().numpy().flatten()
        all_target = outputs["target_betas"].detach().cpu().numpy().flatten()
        if len(all_pred) > 1:
            all_pcc, _ = pearsonr(all_pred, all_target)
            self.log("test/all_pcc", all_pcc, on_epoch=True)

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
