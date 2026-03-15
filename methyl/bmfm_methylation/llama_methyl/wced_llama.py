"""
WCED Training Module for MethylLlama.

Mirrors WCEDTrainingModule (bmfm_methylation/wced_module.py) but uses
MethylLlamaModel instead of SCBertModel.

Key simplifications vs the original:
  - No monkey-patching: cpg_scale and ScaleAdaptEncoder are built into MethylLlamaEmbeddings
  - No _patch_embeddings_add_stabilized() needed
  - No post-hoc ScaleAdapt patch: encoder always uses sinusoidal beta encoder

Architecture:
    Input:  Subset of CpGs (default 50% = 4000 of 8000)
    Encoder:  MethylLlamaModel → CLS embedding (pooler_output)
    Decoder:  Linear(512→512→8000) + Sigmoid → predicted all-CpG betas
    ProjectionHead: MLP(512→128) → L2-normalized projections
    AgeHead: MLP(512→1) → age prediction (forces CLS to encode age)
    Loss: Recon_MSE + λ_contrastive × InfoNCE + λ_age × Age_MSE

Where:
    Recon_MSE = MSE on NON-input CpGs only (forces encoder to use all input to predict rest)
    InfoNCE = contrastive between two random 50% views of same sample (optional)
    Age_MSE = auxiliary supervised age prediction (prevents CLS collapse)
"""

import logging
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from scipy.stats import pearsonr

from .model import MethylLlamaModel, MethylLlamaConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# WCED Decoder
# ---------------------------------------------------------------------------

class WCEDDecoder(nn.Module):
    """
    Decode CLS embedding → all CpG beta values.

    Architecture: Linear(hidden→hidden) → LayerNorm → GELU → Dropout
                → Linear(hidden→vocab_size) → Sigmoid

    Sigmoid ensures output is in [0, 1], matching beta value range.
    """
    def __init__(self, hidden_size: int = 512, vocab_size: int = 8000, dropout: float = 0.1):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, vocab_size),
            nn.Sigmoid(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.decoder.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, cls_hidden: torch.Tensor) -> torch.Tensor:
        return self.decoder(cls_hidden)  # [B, vocab_size]


# ---------------------------------------------------------------------------
# Projection Head (contrastive learning)
# ---------------------------------------------------------------------------

class ProjectionHead(nn.Module):
    """
    SimCLR-style projection head for InfoNCE contrastive loss.
    Maps CLS → lower-dimensional normalized space.
    """
    def __init__(self, input_dim: int = 512, hidden_dim: int = 256, output_dim: int = 128):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projection(x), dim=-1)  # L2-normalized [B, output_dim]


# ---------------------------------------------------------------------------
# Main Lightning Module
# ---------------------------------------------------------------------------

class WCEDLlamaModule(pl.LightningModule):
    """
    Multi-task WCED pretraining using MethylLlamaModel.

    Identical training loop to WCEDTrainingModule but the encoder is
    MethylLlamaModel (RMSNorm + Pre-LN + SwiGLU + RoPE + ScaleAdapt).

    No monkey-patching: all architectural improvements are built into
    MethylLlamaModel.__init__.

    Usage:
        config = MethylLlamaConfig(hidden_size=512, ...)
        model = WCEDLlamaModule(model_config=config, vocab_size=8000)
        trainer.fit(model, datamodule)
    """

    def __init__(
        self,
        model_config: MethylLlamaConfig,
        learning_rate: float = 5e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 100,
        lr_decay_steps: int = 10000,
        vocab_size: int = 8000,
        contrastive_weight: float = 0.0,
        contrastive_temp: float = 0.1,
        normalize_loss: bool = False,
        age_weight: float = 1.0,
        decoder_dropout: float = 0.1,
        betas: tuple = (0.9, 0.99),    # β₂=0.99 (upstream BMFM convention)
        epsilon: float = 1e-8,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model_config"])

        self.model_config = model_config
        self.vocab_size = vocab_size
        self.contrastive_weight = contrastive_weight
        self.contrastive_temp = contrastive_temp
        self.normalize_loss = normalize_loss
        self.age_weight = age_weight

        # Encoder: MethylLlamaModel with built-in ScaleAdapt + cpg_scale + RoPE
        self.encoder = MethylLlamaModel(model_config)

        # Decoder: CLS → all-CpG beta predictions
        self.decoder = WCEDDecoder(
            hidden_size=model_config.hidden_size,
            vocab_size=vocab_size,
            dropout=decoder_dropout,
        )

        # Projection head for contrastive learning (InfoNCE)
        self.projection_head = ProjectionHead(
            input_dim=model_config.hidden_size,
            hidden_dim=model_config.hidden_size // 2,
            output_dim=128,
        )

        # Age prediction head (auxiliary task to prevent CLS collapse)
        self.age_head = nn.Sequential(
            nn.Linear(model_config.hidden_size, model_config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(model_config.hidden_size // 2, 1),
        )

        # Loss functions
        self.recon_loss_fn = nn.MSELoss(reduction="none")
        self.age_loss_fn   = nn.MSELoss()

        # Log parameter counts
        n_enc  = sum(p.numel() for p in self.encoder.parameters())
        n_dec  = sum(p.numel() for p in self.decoder.parameters())
        n_proj = sum(p.numel() for p in self.projection_head.parameters())
        n_age  = sum(p.numel() for p in self.age_head.parameters())
        logger.info(
            f"WCEDLlamaModule: encoder={n_enc:,}, decoder={n_dec:,}, "
            f"proj={n_proj:,}, age={n_age:,}, total={n_enc+n_dec+n_proj+n_age:,}"
        )
        logger.info(
            f"  age_weight={age_weight}, contrastive_weight={contrastive_weight}, "
            f"normalize_loss={normalize_loss}"
        )

    # -------------------------------------------------------------------------
    # Encode one view
    # -------------------------------------------------------------------------

    def encode(
        self,
        cpg_ids: torch.Tensor,       # [B, L]
        beta_values: torch.Tensor,   # [B, L]
        attention_mask: Optional[torch.Tensor] = None,  # [B, L]
    ) -> Dict[str, torch.Tensor]:
        """Encode one view → CLS embedding, beta predictions, age prediction."""
        # Stack dual-field input: [B, 2, L]
        input_ids = torch.stack([cpg_ids.float(), beta_values], dim=1)

        encoder_out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        cls_embedding  = encoder_out.pooler_output              # [B, D]
        projection     = self.projection_head(cls_embedding)    # [B, 128]
        predicted_betas = self.decoder(cls_embedding)           # [B, vocab_size]
        predicted_age  = self.age_head(cls_embedding).squeeze(-1)  # [B]

        return {
            "cls_embedding":   cls_embedding,
            "projection":      projection,
            "predicted_betas": predicted_betas,
            "predicted_age":   predicted_age,
        }

    def forward(
        self,
        cpg_ids: torch.Tensor,
        beta_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Inference forward pass."""
        return self.encode(cpg_ids, beta_values, attention_mask)

    # -------------------------------------------------------------------------
    # InfoNCE contrastive loss
    # -------------------------------------------------------------------------

    def info_nce_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Symmetric InfoNCE (cross-entropy on similarity matrix).

        z1, z2: [B, dim] — L2-normalized projections from two views.
        Positive pairs: (z1[i], z2[i]); negatives: (z1[i], z2[j≠i]).
        """
        B = z1.shape[0]
        sim = torch.matmul(z1, z2.T) / self.contrastive_temp  # [B, B]
        labels = torch.arange(B, device=z1.device)
        loss_12 = F.cross_entropy(sim, labels)
        loss_21 = F.cross_entropy(sim.T, labels)
        return (loss_12 + loss_21) / 2

    # -------------------------------------------------------------------------
    # Optional per-sample normalization (removes "predict averages" shortcut)
    # -------------------------------------------------------------------------

    def _normalize_per_sample(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Normalize each sample to zero-mean unit-variance over non-input positions."""
        if mask is not None:
            mf = mask.float()
            n   = mf.sum(dim=1, keepdim=True).clamp(min=1)
            mean = (x * mf).sum(dim=1, keepdim=True) / n
            var  = ((x - mean) ** 2 * mf).sum(dim=1, keepdim=True) / n
            std  = (var + 1e-8).sqrt()
            return (x - mean) / std * mf
        else:
            return (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-8)

    # -------------------------------------------------------------------------
    # Shared train/val/test step
    # -------------------------------------------------------------------------

    def _shared_step(self, batch: Dict[str, torch.Tensor], stage: str) -> Dict:
        cpg_ids_v1     = batch["cpg_ids"]
        beta_values_v1 = batch["beta_values"]
        attention_mask_v1 = batch.get("attention_mask")
        input_mask_v1  = batch["input_mask"]    # [B, vocab_size] True=in input
        all_betas      = batch["all_betas"]     # [B, vocab_size] full targets
        age_labels     = batch.get("age")       # [B] or None

        # Encode view 1
        out_v1 = self.encode(cpg_ids_v1, beta_values_v1, attention_mask_v1)
        pred_v1 = out_v1["predicted_betas"]
        z1      = out_v1["projection"]
        age_pred_v1 = out_v1["predicted_age"]

        # Reconstruction loss on non-input CpGs only
        non_input_mask_v1 = ~input_mask_v1  # True = not in input → must reconstruct

        if self.normalize_loss:
            pred_norm   = self._normalize_per_sample(pred_v1,  non_input_mask_v1)
            target_norm = self._normalize_per_sample(all_betas, non_input_mask_v1)
            loss_per_cpg = self.recon_loss_fn(pred_norm, target_norm)
        else:
            loss_per_cpg = self.recon_loss_fn(pred_v1, all_betas)

        masked_loss   = loss_per_cpg * non_input_mask_v1.float()
        recon_loss_v1 = masked_loss.sum() / non_input_mask_v1.float().sum().clamp(min=1)

        # Age loss — skip NaN (e.g. sperm, placenta samples)
        if age_labels is not None and self.age_weight > 0:
            valid = ~torch.isnan(age_labels)
            if valid.any():
                age_loss = self.age_loss_fn(
                    age_pred_v1[valid], age_labels[valid].float()
                )
            else:
                age_loss = torch.tensor(0.0, device=recon_loss_v1.device)
        else:
            age_loss = torch.tensor(0.0, device=recon_loss_v1.device)

        # Contrastive mode: encode view 2 and compute InfoNCE on CLS
        if "cpg_ids_v2" in batch and self.contrastive_weight > 0:
            out_v2 = self.encode(
                batch["cpg_ids_v2"],
                batch["beta_values_v2"],
                batch.get("attention_mask_v2"),
            )
            pred_v2 = out_v2["predicted_betas"]

            # Reconstruction loss on view 2
            non_input_v2 = ~batch["input_mask_v2"]
            if self.normalize_loss:
                pn2 = self._normalize_per_sample(pred_v2,  non_input_v2)
                tn2 = self._normalize_per_sample(all_betas, non_input_v2)
                lpc2 = self.recon_loss_fn(pn2, tn2)
            else:
                lpc2 = self.recon_loss_fn(pred_v2, all_betas)
            recon_loss_v2 = (lpc2 * non_input_v2.float()).sum() / non_input_v2.float().sum().clamp(min=1)

            recon_loss = (recon_loss_v1 + recon_loss_v2) / 2

            # InfoNCE on normalized CLS (not on projection head — prevents head collapse)
            cls1_norm = F.normalize(out_v1["cls_embedding"], dim=-1)
            cls2_norm = F.normalize(out_v2["cls_embedding"], dim=-1)
            contrastive_loss = self.info_nce_loss(cls1_norm, cls2_norm)

            loss = recon_loss + self.contrastive_weight * contrastive_loss + self.age_weight * age_loss
            predicted_betas = pred_v1
            non_input_mask  = non_input_mask_v1
        else:
            recon_loss       = recon_loss_v1
            contrastive_loss = torch.tensor(0.0, device=recon_loss_v1.device)
            loss = recon_loss + self.age_weight * age_loss
            predicted_betas = pred_v1
            non_input_mask  = non_input_mask_v1

        # Metrics (no-grad)
        with torch.no_grad():
            ni_pred   = predicted_betas[non_input_mask]
            ni_target = all_betas[non_input_mask]
            mae = torch.abs(ni_pred - ni_target).mean()
            mse = ((ni_pred - ni_target) ** 2).mean()
            all_mae = torch.abs(predicted_betas - all_betas).mean()
            age_mae = (
                torch.abs(age_pred_v1 - age_labels).mean()
                if age_labels is not None
                else torch.tensor(0.0, device=mae.device)
            )

        return {
            "loss":             loss,
            "recon_loss":       recon_loss,
            "contrastive_loss": contrastive_loss,
            "age_loss":         age_loss,
            "age_mae":          age_mae,
            "mae":              mae,
            "mse":              mse,
            "all_mae":          all_mae,
            "predicted_betas":  predicted_betas,
            "target_betas":     all_betas,
            "non_input_mask":   non_input_mask,
            "cls_embedding":    out_v1["cls_embedding"],
            "predicted_age":    age_pred_v1,
        }

    # -------------------------------------------------------------------------
    # Lightning hooks
    # -------------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        out = self._shared_step(batch, "train")
        self.log("train/loss",             out["loss"],             on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/recon_loss",       out["recon_loss"],       on_step=False, on_epoch=True)
        self.log("train/age_loss",         out["age_loss"],         on_step=False, on_epoch=True)
        self.log("train/age_mae",          out["age_mae"],          on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/contrastive_loss", out["contrastive_loss"], on_step=False, on_epoch=True)
        self.log("train/mae",              out["mae"],              on_step=False, on_epoch=True)
        self.log("train/all_mae",          out["all_mae"],          on_step=False, on_epoch=True)
        return out["loss"]

    def validation_step(self, batch, batch_idx):
        out = self._shared_step(batch, "val")
        self.log("validation/loss",             out["loss"],             on_epoch=True, prog_bar=True)
        self.log("validation/recon_loss",       out["recon_loss"],       on_epoch=True)
        self.log("validation/age_loss",         out["age_loss"],         on_epoch=True)
        self.log("validation/age_mae",          out["age_mae"],          on_epoch=True, prog_bar=True)
        self.log("validation/contrastive_loss", out["contrastive_loss"], on_epoch=True)
        self.log("validation/mae",              out["mae"],              on_epoch=True)
        self.log("validation/all_mae",          out["all_mae"],          on_epoch=True)

        # Diagnostic: CLS diversity
        if batch_idx == 0:
            with torch.no_grad():
                cls = out["cls_embedding"]
                cls_var = cls.var(dim=0).mean()
                cls_norm = F.normalize(cls, dim=-1)
                sim_mat = torch.matmul(cls_norm, cls_norm.T)
                triu_mask = torch.triu(torch.ones_like(sim_mat), diagonal=1).bool()
                mean_sim = sim_mat[triu_mask].mean()
                self.log("validation/cls_variance",  cls_var,  on_epoch=True)
                self.log("validation/cls_similarity", mean_sim, on_epoch=True)

                pred_var   = out["predicted_betas"].var(dim=0).mean()
                target_var = out["target_betas"].var(dim=0).mean()
                self.log("validation/pred_var_ratio", pred_var / (target_var + 1e-8), on_epoch=True)

        return out["loss"]

    def test_step(self, batch, batch_idx):
        out = self._shared_step(batch, "test")
        self.log("test/loss",             out["loss"],             on_epoch=True)
        self.log("test/recon_loss",       out["recon_loss"],       on_epoch=True)
        self.log("test/age_loss",         out["age_loss"],         on_epoch=True)
        self.log("test/age_mae",          out["age_mae"],          on_epoch=True)
        self.log("test/contrastive_loss", out["contrastive_loss"], on_epoch=True)
        self.log("test/mae",              out["mae"],              on_epoch=True)
        self.log("test/all_mae",          out["all_mae"],          on_epoch=True)

        # PCC on non-input positions
        non_input_mask = out["non_input_mask"]
        pred   = out["predicted_betas"][non_input_mask].detach().cpu().numpy()
        target = out["target_betas"][non_input_mask].detach().cpu().numpy()
        if len(pred) > 1:
            pcc, _ = pearsonr(pred, target)
            self.log("test/pcc", pcc, on_epoch=True)

        all_pred   = out["predicted_betas"].detach().cpu().numpy().flatten()
        all_target = out["target_betas"].detach().cpu().numpy().flatten()
        if len(all_pred) > 1:
            all_pcc, _ = pearsonr(all_pred, all_target)
            self.log("test/all_pcc", all_pcc, on_epoch=True)

        return out["loss"]

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            betas=self.hparams.betas,
            eps=self.hparams.epsilon,
        )

        # Cosine LR schedule with warmup
        lr_decay_steps = self.hparams.lr_decay_steps
        if lr_decay_steps <= 0:
            if self.trainer is not None:
                lr_decay_steps = int(self.trainer.estimated_stepping_batches)
            else:
                lr_decay_steps = 300 * 45
        warmup_steps = self.hparams.warmup_steps

        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / max(1, warmup_steps)
            progress = min(
                float(step - warmup_steps) / max(1, lr_decay_steps - warmup_steps),
                1.0,
            )
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    def get_encoder(self) -> MethylLlamaModel:
        """Return pretrained encoder for downstream tasks."""
        return self.encoder
