#!/usr/bin/env python3
"""
Training script for Methylation Age Prediction using BMFM Encoder

This script trains the methylation age prediction model using the original
BMFM SCBertModel encoder with a regression head.

Usage:
    python train.py --data /path/to/methylation.h5ad --wandb

Requirements:
    - bmfm_targets (original BMFM code)
    - pytorch-lightning
    - wandb (optional, for logging)
    - scanpy (for h5ad loading)
    - torchmetrics
"""

import argparse
import os
from datetime import datetime

import torch


def main():
    parser = argparse.ArgumentParser(
        description="Train Methylation Age Prediction Model (BMFM Encoder)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data arguments
    parser.add_argument("--data", type=str, required=True,
                        help="Path to h5ad file with methylation data")
    parser.add_argument("--train-split", type=str, default="train",
                        help="Name of training split in obs['split']")
    parser.add_argument("--val-split", type=str, default="valid",
                        help="Name of validation split in obs['split']")
    parser.add_argument("--test-split", type=str, default="test",
                        help="Name of test split in obs['split']")
    parser.add_argument("--max-cpg", type=int, default=8000,
                        help="Maximum number of CpG sites to use")

    # Model arguments
    parser.add_argument("--hidden-size", type=int, default=512,
                        help="Hidden size of transformer")
    parser.add_argument("--num-layers", type=int, default=6,
                        help="Number of transformer layers")
    parser.add_argument("--num-heads", type=int, default=8,
                        help="Number of attention heads")
    parser.add_argument("--intermediate-size", type=int, default=2048,
                        help="Intermediate size in feed-forward layers")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout probability")
    parser.add_argument("--no-flash-attention", action="store_true",
                        help="Disable Flash Attention (use standard attention)")

    # Training arguments
    parser.add_argument("--epochs", type=int, default=100,
                        help="Maximum number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="Weight decay for AdamW")
    parser.add_argument("--warmup-steps", type=int, default=1000,
                        help="Number of warmup steps")
    parser.add_argument("--early-stopping", type=int, default=10,
                        help="Early stopping patience (epochs)")
    parser.add_argument("--precision", type=str, default="16-mixed",
                        choices=["32", "16-mixed", "bf16-mixed"],
                        help="Training precision")
    parser.add_argument("--accumulate-grad", type=int, default=1,
                        help="Gradient accumulation steps")

    # Logging arguments
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="methylation-age",
                        help="WandB project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="WandB entity (username or team name)")
    parser.add_argument("--wandb-run-name", type=str, default=None,
                        help="WandB run name (auto-generated if not provided)")
    parser.add_argument("--log-dir", type=str, default="./logs",
                        help="Directory for logs and checkpoints")
    parser.add_argument("--exp-name", type=str, default=None,
                        help="Experiment name (auto-generated if not provided)")

    # Hardware arguments
    parser.add_argument("--accelerator", type=str, default="auto",
                        choices=["auto", "gpu", "cpu", "mps"],
                        help="Hardware accelerator")
    parser.add_argument("--devices", type=int, default=1,
                        help="Number of devices to use")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of data loading workers")

    args = parser.parse_args()

    # Generate experiment name if not provided
    if args.exp_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.exp_name = f"methylation_L{args.num_layers}_H{args.hidden_size}_{timestamp}"

    if args.wandb_run_name is None:
        args.wandb_run_name = args.exp_name

    # Print configuration
    print("=" * 70)
    print("METHYLATION AGE PREDICTION - BMFM ENCODER")
    print("=" * 70)
    print(f"\nData:")
    print(f"  File: {args.data}")
    print(f"  Splits: train={args.train_split}, val={args.val_split}, test={args.test_split}")
    print(f"  Max CpG sites: {args.max_cpg}")
    print(f"\nModel (Original BMFM SCBertModel):")
    print(f"  Hidden size: {args.hidden_size}")
    print(f"  Layers: {args.num_layers}")
    print(f"  Attention heads: {args.num_heads}")
    print(f"  Intermediate size: {args.intermediate_size}")
    print(f"  Dropout: {args.dropout}")
    print(f"  Flash Attention: {'Disabled' if args.no_flash_attention else 'Enabled'}")
    print(f"\nTraining:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Precision: {args.precision}")
    print(f"  Early stopping patience: {args.early_stopping}")
    print(f"\nLogging:")
    print(f"  WandB: {'Enabled' if args.wandb else 'Disabled'}")
    if args.wandb:
        print(f"  WandB entity: {args.wandb_entity or '(default)'}")
        print(f"  WandB project: {args.wandb_project}")
        print(f"  WandB run name: {args.wandb_run_name}")
    print(f"  Log directory: {args.log_dir}")
    print(f"  Experiment name: {args.exp_name}")
    print(f"\nHardware:")
    print(f"  Accelerator: {args.accelerator}")
    print(f"  Devices: {args.devices}")
    print("=" * 70)

    # Import after parsing to fail fast on bad arguments
    from bmfm_methylation import create_methylation_config, train_methylation_model

    # Create model config
    config = create_methylation_config(
        num_cpg_sites=args.max_cpg,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        hidden_dropout_prob=args.dropout,
        attention_probs_dropout_prob=args.dropout,
        max_position_embeddings=args.max_cpg + 10,
        use_flash_attention=not args.no_flash_attention,
    )

    # Train model
    trainer, module = train_methylation_model(
        data_path=args.data,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
        config=config,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        max_cpg=args.max_cpg,
        num_workers=args.num_workers,
        accelerator=args.accelerator,
        precision=args.precision,
        log_dir=args.log_dir,
        experiment_name=args.exp_name,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        early_stopping_patience=args.early_stopping,
        devices=args.devices,
        accumulate_grad_batches=args.accumulate_grad,
    )

    # Print results
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    # Get best checkpoint path
    best_checkpoint = trainer.checkpoint_callback.best_model_path
    last_checkpoint = trainer.checkpoint_callback.last_model_path

    print(f"\nCheckpoints saved:")
    print(f"  Best model: {best_checkpoint}")
    print(f"  Last model: {last_checkpoint}")
    print(f"  Best val/mae: {trainer.checkpoint_callback.best_model_score:.4f}")

    # Print final metrics if available
    if hasattr(trainer, 'logged_metrics') and trainer.logged_metrics:
        print(f"\nFinal metrics:")
        for key, value in trainer.logged_metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f}")

    print("\n" + "=" * 70)
    print("To load the best model:")
    print(f"  from bmfm_methylation import MethylationAgeLightningModule")
    print(f"  module = MethylationAgeLightningModule.load_from_checkpoint('{best_checkpoint}')")
    print("=" * 70)

    return trainer, module


if __name__ == "__main__":
    main()
