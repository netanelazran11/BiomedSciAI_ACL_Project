#!/usr/bin/env python3
"""
Fine-tuning script for Methylation Age Prediction — 21k WCED pretrained encoder.

Identical to finetune_wced.py except:
  1. Uses MethylationDataModule21k (auto-creates valid split from train)
  2. wced_input_ratio is read from cfg (default 0.187 = 4000/21368)
     — CRITICAL: 21k encoder has max_position_embeddings=8002 (trained with
       ~4000 tokens/view). wced_input_ratio=0.5 would create 10,684 tokens
       and exceed the position embedding table. Use 0.187 to match pretraining.

The 8k pipeline (finetune_wced.py / finetune_wced.sh) is NOT touched.

Pipeline:
    Random 4k CpGs (of 21k) → Encoder → CLS ─┬─► MLP head → age (primary)
                                             └─► Decoder → all 21k betas (regularizer)
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
import sys
from pathlib import Path
from typing import Optional

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


class MethylationAgeRegressorWCED(pl.LightningModule):
    """
    WCED-correct fine-tuning module for methylation age regression.

    Keeps the pretrained WCED decoder as a reconstruction regularizer so the
    encoder is not free to forget its global methylation representation.

    Multi-task loss:
        total = age_MSE(CLS → head → age) + recon_weight * recon_MSE(CLS → decoder → all_betas)

    The reconstruction loss is computed only on NON-input CpGs (same as WCED
    pretraining), preventing trivial copying and forcing CLS to stay globally
    informative about the full methylation profile.
    """

    def __init__(
        self,
        encoder,
        decoder=None,
        hidden_size: int = 512,
        head_hidden_size: int = 256,
        head_dropout: float = 0.1,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        max_steps: int = 10000,
        age_mean: float = 0.0,
        age_std: float = 1.0,
        freeze_encoder: bool = False,
        unfreeze_encoder_epoch: int = 9999,
        use_huber_loss: bool = False,
        huber_delta: float = 2.0,
        recon_weight: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['encoder', 'decoder'])

        self.encoder = encoder
        self.decoder = decoder
        self.age_mean = age_mean
        self.age_std = age_std
        self.recon_weight = recon_weight

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            logger.info(f"Encoder frozen (unfreeze at epoch {unfreeze_encoder_epoch})")

        # MLP age head: hidden_size → head_hidden_size → head_hidden_size//2 → 1
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

        self.loss_fn = nn.HuberLoss(delta=huber_delta) if use_huber_loss else nn.MSELoss()
        self.recon_loss_fn = nn.MSELoss(reduction='none')

        self._val_preds = []
        self._val_labels = []
        self._test_preds = []
        self._test_labels = []

        head_params = sum(p.numel() for p in self.age_head.parameters())
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters()) if decoder else 0
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        logger.info(f"WCED-21k Fine-tuning module:")
        logger.info(f"  Encoder params:  {encoder_params:,}")
        logger.info(f"  Decoder params:  {decoder_params:,}  {'(regularizer)' if decoder else '(not used)'}")
        logger.info(f"  Age head params: {head_params:,}")
        logger.info(f"  Trainable:       {trainable:,}")
        logger.info(f"  Recon weight:    {recon_weight}")

    def _encode(self, cpg_ids, beta_values, attention_mask=None):
        """Run encoder, return CLS embedding (pooler_output)."""
        input_ids = torch.stack([cpg_ids.float(), beta_values], dim=1)
        batch_size = input_ids.size(0)
        seq_length = input_ids.size(2)

        if attention_mask is not None and attention_mask.dim() == 3:
            attention_mask = attention_mask[:, 0, :]
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length), device=input_ids.device
            )

        encoder_output = self.encoder(input_ids, attention_mask=attention_mask)
        return encoder_output.pooler_output  # [batch, hidden_size]

    def forward(self, cpg_ids, beta_values, attention_mask=None):
        cls = self._encode(cpg_ids, beta_values, attention_mask)
        return self.age_head(cls)

    def on_train_epoch_start(self):
        epoch = self.current_epoch
        if (self.hparams.freeze_encoder and
                epoch == self.hparams.unfreeze_encoder_epoch):
            logger.info(f"[EPOCH {epoch}] Unfreezing encoder")
            for param in self.encoder.parameters():
                param.requires_grad = True
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            logger.info(f"[EPOCH {epoch}] Trainable params: {trainable:,}")

    def _shared_step(self, batch, stage: str):
        cpg_ids = batch["cpg_ids"]
        beta_values = batch["beta_values"]
        attention_mask = batch.get("attention_mask")

        if "age" in batch:
            labels = batch["age"].float().view(-1, 1)
        else:
            labels = batch["labels"].float().view(-1, 1)

        cls_embedding = self._encode(cpg_ids, beta_values, attention_mask)

        age_pred = self.age_head(cls_embedding)
        age_loss = self.loss_fn(age_pred, labels)

        recon_loss = torch.tensor(0.0, device=cls_embedding.device)
        if self.decoder is not None and "all_betas" in batch:
            predicted_betas = self.decoder(cls_embedding)
            all_betas = batch["all_betas"]
            input_mask = batch.get("input_mask")

            recon_per_cpg = self.recon_loss_fn(predicted_betas, all_betas)

            if input_mask is not None:
                non_input = ~input_mask
                recon_loss = (
                    (recon_per_cpg * non_input.float()).sum()
                    / non_input.float().sum().clamp(min=1)
                )
            else:
                recon_loss = recon_per_cpg.mean()

        total_loss = age_loss + self.recon_weight * recon_loss

        preds_denorm = age_pred * self.age_std + self.age_mean
        labels_denorm = labels * self.age_std + self.age_mean
        mae = torch.abs(preds_denorm - labels_denorm).mean()

        return total_loss, age_loss, recon_loss, preds_denorm, labels_denorm, mae

    def training_step(self, batch, batch_idx):
        total_loss, age_loss, recon_loss, _, _, mae = self._shared_step(batch, "train")
        self.log("train/loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/age_loss", age_loss, on_step=False, on_epoch=True)
        self.log("train/recon_loss", recon_loss, on_step=False, on_epoch=True)
        self.log("train/mae", mae, on_step=False, on_epoch=True, prog_bar=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        total_loss, age_loss, recon_loss, preds, labels, mae = self._shared_step(batch, "val")
        self.log("val/loss", total_loss, on_epoch=True, prog_bar=True)
        self.log("val/age_loss", age_loss, on_epoch=True)
        self.log("val/recon_loss", recon_loss, on_epoch=True)
        self.log("val/mae", mae, on_epoch=True, prog_bar=True)
        self._val_preds.append(preds.detach())
        self._val_labels.append(labels.detach())
        return total_loss

    def on_validation_epoch_end(self):
        if self._val_preds:
            all_preds = torch.cat(self._val_preds, dim=0)
            all_labels = torch.cat(self._val_labels, dim=0)
            ss_res = torch.sum((all_labels - all_preds) ** 2)
            ss_tot = torch.sum((all_labels - all_labels.mean()) ** 2)
            r2 = 1 - ss_res / (ss_tot + 1e-8)
            self.log("val/r2", r2, prog_bar=True)
            self.log("val/mae_epoch", torch.abs(all_preds - all_labels).mean())
        self._val_preds.clear()
        self._val_labels.clear()

    def test_step(self, batch, batch_idx):
        total_loss, age_loss, recon_loss, preds, labels, mae = self._shared_step(batch, "test")
        self.log("test/mae", mae, on_epoch=True)
        self._test_preds.append(preds.detach())
        self._test_labels.append(labels.detach())
        return total_loss

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
            logger.info(f"Test MAE: {epoch_mae:.2f} years, R²: {r2:.4f}")
        self._test_preds.clear()
        self._test_labels.clear()

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
        encoder_lr = self.hparams.learning_rate * 0.01  # 1e-5 when head_lr=1e-3

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
            {
                "params": [p for n, p in self.encoder.named_parameters()
                           if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
                "lr": encoder_lr,
            },
            {
                "params": [p for n, p in self.encoder.named_parameters()
                           if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": encoder_lr,
            },
        ]

        if self.decoder is not None:
            optimizer_grouped_parameters.extend([
                {
                    "params": [p for n, p in self.decoder.named_parameters()
                               if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.hparams.weight_decay,
                    "lr": encoder_lr,
                },
                {
                    "params": [p for n, p in self.decoder.named_parameters()
                               if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                    "lr": encoder_lr,
                },
            ])

        optimizer_grouped_parameters = [
            g for g in optimizer_grouped_parameters if len(g["params"]) > 0
        ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        def lr_lambda(current_step):
            if current_step < self.hparams.warmup_steps:
                return float(current_step) / float(max(1, self.hparams.warmup_steps))
            progress = float(current_step - self.hparams.warmup_steps) / \
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
                project=cfg.track_wandb.get('project', 'methylation-age-wced-21k'),
                entity=cfg.track_wandb.get('entity', None),
                name=cfg.track_wandb.get('name', None),
                save_dir=cfg.output_directory,
                log_model=True,
            )
        except Exception as e:
            logger.warning(f"WandB setup failed: {e}, using TensorBoard")
    from pytorch_lightning.loggers import TensorBoardLogger
    return TensorBoardLogger(cfg.output_directory, name="finetune_wced_21k")


@hydra.main(
    config_path=str(Path(__file__).parent / "configs"),
    config_name="finetune_config",
    version_base="1.2",
)
def main(cfg: DictConfig):
    logger.info("=" * 70)
    logger.info("METHYLATION AGE FINE-TUNING — WCED-21k (encoder + decoder)")
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
    # Data module — MethylationDataModule21k auto-creates valid split if missing.
    # subset_k must match decoder vocab_size (21368 for 21k pretraining).
    # wced_input_ratio from cfg (default 0.187 = 4000/21368 tokens/view,
    # matching pretraining and staying within max_position_embeddings=8002).
    # -------------------------------------------------------------------------
    subset_k = cfg.data_module.get('subset_k', 21368)
    wced_input_ratio = cfg.data_module.get('wced_input_ratio', 0.187)

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
        fixed_subset=cfg.data_module.get('fixed_subset', False),
        fixed_subset_seed=cfg.data_module.get('fixed_subset_seed', 42),
        use_wced_collator=True,
        wced_input_ratio=wced_input_ratio,
    )
    data_module.setup()

    model_config_partial = hydra.utils.instantiate(cfg.model)
    model_config = model_config_partial(fields=fields)

    # -------------------------------------------------------------------------
    # Load 21k WCED pretrained checkpoint — extract BOTH encoder AND decoder
    # -------------------------------------------------------------------------
    if not cfg.checkpoint_path or cfg.checkpoint_path == "null":
        raise ValueError("checkpoint_path is required for WCED-21k fine-tuning.")

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

    encoder = pretrained_module.encoder
    decoder = pretrained_module.decoder

    logger.info("WCED-21k encoder + decoder loaded successfully")
    logger.info(f"  Encoder params: {sum(p.numel() for p in encoder.parameters()):,}")
    logger.info(f"  Decoder params: {sum(p.numel() for p in decoder.parameters()):,}")
    if hasattr(encoder.embeddings, 'cpg_scale'):
        logger.info(f"  Learned cpg_scale: {encoder.embeddings.cpg_scale.item():.4f}")
    logger.info(f"  Decoder vocab_size: {pretrained_module.vocab_size}")
    logger.info(f"  WCEDCollator subset_k: {subset_k}")
    logger.info(f"  WCEDCollator input_ratio: {wced_input_ratio} (~{int(subset_k * wced_input_ratio)} tokens/view)")

    if pretrained_module.vocab_size != subset_k:
        raise ValueError(
            f"Mismatch! Decoder was trained with vocab_size={pretrained_module.vocab_size} "
            f"but data module uses subset_k={subset_k}. "
            f"Set subset_k={pretrained_module.vocab_size} in finetune_wced_21k.sh."
        )

    # -------------------------------------------------------------------------
    # Fine-tuning model
    # -------------------------------------------------------------------------
    freeze_encoder = cfg.get('freeze_encoder', False)
    unfreeze_encoder_epoch = cfg.get('unfreeze_encoder_epoch', 9999)
    recon_weight = cfg.get('recon_weight', 0.1)

    effective_batch = cfg.data_module.batch_size * cfg.accumulate_grad_batches
    steps_per_epoch = len(data_module.train_dataset) // effective_batch
    total_steps = cfg.finetune_epochs * steps_per_epoch

    logger.info(f"Train samples: {len(data_module.train_dataset)}")
    logger.info(f"Val samples: {len(data_module.val_dataset)}")
    logger.info(f"Effective batch: {effective_batch}, steps/epoch: {steps_per_epoch}, total: {total_steps}")
    logger.info(f"Age stats: mean={data_module.age_mean:.2f}, std={data_module.age_std:.2f}")
    logger.info(f"Recon weight: {recon_weight}")

    model = MethylationAgeRegressorWCED(
        encoder=encoder,
        decoder=decoder,
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
        use_huber_loss=cfg.get('use_huber_loss', False),
        huber_delta=cfg.get('huber_delta', 2.0),
        recon_weight=recon_weight,
    )

    wandb_logger = setup_wandb(cfg)

    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=output_dir / "finetune_wced_21k" / "checkpoints",
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
        default_root_dir=str(output_dir / "finetune_wced_21k"),
        log_every_n_steps=10,
    )

    logger.info("Starting WCED-21k fine-tuning...")
    trainer.fit(model, data_module)

    logger.info("Running test evaluation...")
    trainer.test(model, data_module)

    best_ckpt = trainer.checkpoint_callback.best_model_path
    logger.info(f"Fine-tuning complete. Best checkpoint: {best_ckpt}")
    logger.info(f"Best val/mae: {trainer.checkpoint_callback.best_model_score:.4f}")

    return best_ckpt


if __name__ == "__main__":
    main()
