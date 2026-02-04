#!/usr/bin/env python3
"""
Value-Only Transformer Pretraining for Methylation Data

Pretrains an SCBertModel using Masked Value Modeling but with ONLY
β-value embeddings + position embeddings (no CpG ID embeddings).

The standard pretraining uses:
    h_i = CpG_ID_embed(s_i) + β_value_embed(β_i) + pos_embed(i)

This script uses:
    h_i = β_value_embed(β_i) + pos_embed(i)

The CpG ID field is still present in the config (for checkpoint compatibility)
but its contribution is zeroed out via a monkey-patched embeddings forward pass.
The masking task remains the same: predict masked β-values (regression).

After pretraining, fine-tune with finetune_transformer.py.

Usage:
    python -m bmfm_methylation.pretrain_transformer_valueonly \
        data_path=/path/to/methylation.h5ad \
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
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from bmfm_methylation.tokenizer import (
    extract_cpg_sites_from_h5ad,
    create_methylation_multifield_tokenizer,
)
from bmfm_methylation.data_module import MethylationDataModule

# Import BMFM training modules
from bmfm_targets.training.modules.masked_language_modeling import MLMTrainingModule
from bmfm_targets.config import TrainerConfig

logger = logging.getLogger(__name__)


def patch_embeddings_valueonly(scbert_model):
    """
    Monkey-patch the SCEmbeddingsLayer forward pass to skip CpG ID embeddings.

    Instead of summing field 0 (CpG IDs) + field 1 (β-values), only use field 1.
    The position embeddings, LayerNorm, and dropout are applied as normal.

    Also freezes the CpG ID embedding parameters to save memory.
    """
    print("=" * 70)
    print("[PATCH] Applying value-only embedding patch...")
    print("[PATCH] BEFORE: h = CpG_ID_embed + beta_embed + pos_embed")
    print("[PATCH] AFTER:  h = beta_embed + pos_embed  (CpG ID SKIPPED)")
    print("=" * 70)

    embeddings = scbert_model.embeddings

    # Freeze CpG ID embedding (field 0) -- not used, save memory
    cpg_embed = embeddings.cpg_sites_embeddings
    for param in cpg_embed.parameters():
        param.requires_grad = False
    print(f"[PATCH] Frozen CpG ID embedding: {sum(p.numel() for p in cpg_embed.parameters()):,} params")

    # Store reference to the original forward for potential restoration
    embeddings._original_forward = embeddings.forward
    embeddings._fwd_call_count = 0

    def valueonly_forward(
        input_ids,
        position_ids=None,
        inputs_embeds=None,
    ):
        """Value-only embedding: skip CpG ID field, only use β-values + position."""
        if inputs_embeds is not None:
            return inputs_embeds

        # Only compute β-value embeddings (field index 1)
        # Skip field 0 (CpG ID embeddings) entirely
        beta_embeds = embeddings.beta_values_embeddings(
            input_ids[:, 1, :].float()
        )

        seq_length = input_ids.size(2)
        updated_embeddings = beta_embeds

        # Add position embeddings (same as original)
        if embeddings.position_embedding_type is not None:
            if position_ids is None:
                position_ids = embeddings.position_ids[:, :seq_length]
            position_embeds = embeddings.position_embeddings(position_ids)
            updated_embeddings = updated_embeddings + position_embeds

        # LayerNorm + dropout (same as original)
        updated_embeddings = embeddings.LayerNorm(updated_embeddings)
        updated_embeddings = embeddings.dropout(updated_embeddings)

        # Log first call to confirm the patch is active
        embeddings._fwd_call_count += 1
        if embeddings._fwd_call_count == 1:
            print("=" * 70)
            print("[PATCH VERIFY] First forward pass through value-only embeddings")
            print(f"  input_ids shape: {input_ids.shape}")
            print(f"  beta field ([:, 1, :]) min={input_ids[:, 1, :].min():.4f}, "
                  f"max={input_ids[:, 1, :].max():.4f}, "
                  f"mean={input_ids[:, 1, :].float().mean():.4f}")
            print(f"  beta_embeds shape: {beta_embeds.shape}")
            print(f"  output shape: {updated_embeddings.shape}")
            print(f"  output norm (first sample): {updated_embeddings[0].norm():.4f}")
            print("[PATCH VERIFY] CpG ID field was NOT used (correct)")
            print("=" * 70)

        return updated_embeddings

    # Replace the forward method
    embeddings.forward = valueonly_forward

    # Log parameter counts
    total = sum(p.numel() for p in scbert_model.parameters())
    frozen = sum(p.numel() for p in cpg_embed.parameters())
    trainable = sum(p.numel() for p in scbert_model.parameters() if p.requires_grad)
    print(f"[PATCH] SCBert total params: {total:,}")
    print(f"[PATCH] CpG embed (frozen):  {frozen:,}")
    print(f"[PATCH] Trainable params:    {trainable:,}")
    print("[PATCH] Value-only patch applied successfully")
    print("=" * 70)


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
    if hasattr(cfg, 'track_wandb') and cfg.track_wandb.get('enabled', False):
        try:
            import wandb
            from pytorch_lightning.loggers import WandbLogger

            wandb_logger = WandbLogger(
                project=cfg.track_wandb.get('project', 'methylation-pretrain-valueonly'),
                entity=cfg.track_wandb.get('entity'),
                name=cfg.track_wandb.get('name', 'methylation_valueonly_pretrain'),
                save_dir=cfg.output_directory,
            )
            return wandb_logger
        except ImportError:
            logger.warning("WandB not installed, using TensorBoard")

    from pytorch_lightning.loggers import TensorBoardLogger
    return TensorBoardLogger(cfg.output_directory, name="pretrain_valueonly")


@hydra.main(
    config_path="configs",
    config_name="pretrain_config",
    version_base="1.2"
)
def main(cfg: DictConfig):
    """Main value-only pretraining function."""
    print("\n" + "=" * 70)
    print("  VALUE-ONLY TRANSFORMER PRETRAINING")
    print("  Task: Masked Value Modeling (predict masked beta-values)")
    print("  Architecture: h = beta_embed + pos_embed (NO CpG ID embed)")
    print("  Training from SCRATCH (no checkpoint)")
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
    print(f"[STEP 1/6] Tokenizer ready. Fields: {list(tokenizer.tokenizers.keys())}")

    # Instantiate fields
    from bmfm_targets.config import FieldInfo
    fields = []
    for field_cfg in cfg.fields:
        field_dict = OmegaConf.to_container(field_cfg)
        field_dict.pop('_target_', None)
        fields.append(FieldInfo(**field_dict))
    print(f"[STEP 1/6] Fields configured: {[f.field_name for f in fields]}")
    for f in fields:
        print(f"  - {f.field_name}: strategy={f.tokenization_strategy}, "
              f"is_masked={f.is_masked}, is_input={f.is_input}")

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
        mlm=True,
        change_ratio=cfg.data_module.change_ratio,
        mask_ratio=cfg.data_module.mask_ratio,
        switch_ratio=cfg.data_module.switch_ratio,
        collation_strategy="language_modeling",
    )
    data_module.setup()
    print(f"[STEP 2/6] Data loaded successfully:")
    print(f"  Train samples: {len(data_module.train_dataset):,}")
    print(f"  Valid samples: {len(data_module.val_dataset):,}")
    print(f"  Test samples:  {len(data_module.test_dataset):,}")
    print(f"  CpG sites:     {data_module.train_dataset.num_cpg_sites:,}")
    print(f"  Max seq length: {cfg.data_module.max_length}")
    print(f"  Batch size:     {cfg.data_module.batch_size}")
    print(f"  MLM mask ratio: {cfg.data_module.change_ratio}")

    # ---- STEP 3: Model ----
    print("\n" + "-" * 70)
    print("[STEP 3/6] Creating model...")
    print("-" * 70)
    model_config_partial = hydra.utils.instantiate(cfg.model)
    model_config = model_config_partial(fields=fields)
    print(f"[STEP 3/6] SCBert config:")
    print(f"  Hidden size:      {model_config.hidden_size}")
    print(f"  Num layers:       {model_config.num_hidden_layers}")
    print(f"  Num heads:        {model_config.num_attention_heads}")
    print(f"  FFN intermediate: {model_config.intermediate_size}")
    print(f"  Max positions:    {model_config.max_position_embeddings}")
    print(f"  Dropout:          {model_config.hidden_dropout_prob}")

    # Setup trainer config
    losses = OmegaConf.to_container(cfg.trainer.losses) if hasattr(cfg.trainer, 'losses') else [{"name": "mse", "field_name": "beta_values"}]

    metrics = None
    if hasattr(cfg.trainer, 'metrics') and cfg.trainer.metrics:
        metrics = OmegaConf.to_container(cfg.trainer.metrics)

    batch_prediction_behavior = None
    if hasattr(cfg.trainer, 'batch_prediction_behavior'):
        batch_prediction_behavior = cfg.trainer.batch_prediction_behavior

    trainer_config = TrainerConfig(
        learning_rate=cfg.trainer.learning_rate,
        weight_decay=cfg.trainer.weight_decay,
        warmup_steps=cfg.trainer.warmup_steps,
        lr_decay_steps=cfg.trainer.lr_decay_steps,
        betas=tuple(cfg.trainer.betas),
        epsilon=cfg.trainer.epsilon,
        losses=losses,
        metrics=metrics,
        batch_prediction_behavior=batch_prediction_behavior,
    )
    print(f"[STEP 3/6] Optimizer config:")
    print(f"  Learning rate:  {cfg.trainer.learning_rate}")
    print(f"  Weight decay:   {cfg.trainer.weight_decay}")
    print(f"  Warmup steps:   {cfg.trainer.warmup_steps}")
    print(f"  Betas:          {list(cfg.trainer.betas)}")
    print(f"  Losses:         {losses}")

    # Create MLMTrainingModule (same as standard pretraining)
    model = MLMTrainingModule(
        model_config=model_config,
        trainer_config=trainer_config,
        tokenizer=tokenizer,
    )
    print(f"[STEP 3/6] MLMTrainingModule created")
    print(f"  Model structure: MLMTrainingModule -> SCBertForMaskedLM -> SCBertModel")

    # ---- STEP 4: Apply value-only patch ----
    print("\n" + "-" * 70)
    print("[STEP 4/6] Applying value-only embedding patch...")
    print("-" * 70)
    # The model structure is: MLMTrainingModule.model (SCBertForMaskedLM)
    #   -> .scbert (SCBertModel) -> .embeddings (SCEmbeddingsLayer)
    patch_embeddings_valueonly(model.model.scbert)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[STEP 4/6] Final model stats:")
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Frozen parameters:    {total_params - trainable_params:,}")

    # ---- STEP 5: Training setup ----
    print("\n" + "-" * 70)
    print("[STEP 5/6] Setting up training...")
    print("-" * 70)
    wandb_logger = setup_wandb(cfg)

    effective_batch = cfg.data_module.batch_size * cfg.accumulate_grad_batches
    steps_per_epoch = len(data_module.train_dataset) // effective_batch
    total_steps = cfg.pretrain_epochs * steps_per_epoch

    early_stop_patience = cfg.get("early_stop_patience", 20)

    print(f"[STEP 5/6] Training schedule:")
    print(f"  Max epochs:           {cfg.pretrain_epochs}")
    print(f"  Batch size:           {cfg.data_module.batch_size}")
    print(f"  Grad accumulation:    {cfg.accumulate_grad_batches}")
    print(f"  Effective batch:      {effective_batch}")
    print(f"  Steps per epoch:      {steps_per_epoch}")
    print(f"  Total steps:          {total_steps}")
    print(f"  Early stop patience:  {early_stop_patience}")
    print(f"  Precision:            16-mixed")
    print(f"  Accelerator:          {'GPU' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"  GPU:                  {torch.cuda.get_device_name(0)}")
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
        print(f"  GPU memory:           {gpu_mem:.1f} GB")

    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=output_dir / "pretrain_valueonly" / "checkpoints",
            filename="epoch={epoch}-val_loss={validation/loss:.4f}",
            monitor="validation/loss",
            mode="min",
            save_top_k=3,
            save_last=True,
        ),
        pl.callbacks.EarlyStopping(
            monitor="validation/loss",
            patience=early_stop_patience,
            mode="min",
            verbose=True,
        ),
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
    ]
    print(f"[STEP 5/6] Checkpoints: {output_dir / 'pretrain_valueonly' / 'checkpoints'}")

    trainer = pl.Trainer(
        max_epochs=cfg.pretrain_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=cfg.task[0].precision if isinstance(cfg.task, list) else "16-mixed",
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        logger=wandb_logger,
        callbacks=callbacks,
        default_root_dir=str(output_dir / "pretrain_valueonly"),
        log_every_n_steps=10,
    )

    # ---- STEP 6: Train ----
    print("\n" + "-" * 70)
    print("[STEP 6/6] Starting training...")
    print("-" * 70)
    print("[STEP 6/6] Training is running. Watch for:")
    print("  - [PATCH VERIFY] message = patch is working correctly")
    print("  - validation/loss going DOWN = model is learning")
    print("  - validation/loss stuck or UP = something is wrong")
    print("-" * 70 + "\n")

    trainer.fit(model, data_module)

    # ---- Results ----
    best_ckpt = trainer.checkpoint_callback.best_model_path
    print("\n" + "=" * 70)
    print("  PRETRAINING COMPLETE")
    print("=" * 70)
    print(f"  Best checkpoint: {best_ckpt}")
    if trainer.checkpoint_callback.best_model_score is not None:
        print(f"  Best val loss:   {trainer.checkpoint_callback.best_model_score:.6f}")
    print("=" * 70)

    # Test evaluation
    print("\n[TEST] Running test evaluation...")
    if best_ckpt:
        test_results = trainer.test(model, data_module, ckpt_path=best_ckpt)
        print("\n[TEST] Results:")
        for result in test_results:
            for key, value in result.items():
                print(f"  {key}: {value:.6f}")
    else:
        print("[TEST] WARNING: No best checkpoint found, using current weights")
        test_results = trainer.test(model, data_module)

    print("\n" + "=" * 70)
    print("  NEXT STEP: Fine-tune for age prediction")
    print("=" * 70)
    print(f"  python -m bmfm_methylation.finetune_transformer \\")
    print(f"      data_path={cfg.data_path} \\")
    print(f"      checkpoint_path={best_ckpt}")
    print("=" * 70 + "\n")

    return best_ckpt


if __name__ == "__main__":
    main()
