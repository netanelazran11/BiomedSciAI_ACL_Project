"""
Command-line interface for BMFM Methylation Encoder

This CLI wraps the original BMFM SCBertModel for methylation age prediction.
"""

import argparse
import torch

from .config import create_methylation_config
from .model import MethylationAgeModel


def main():
    parser = argparse.ArgumentParser(description="BMFM Encoder for Methylation Data")
    parser.add_argument("--test", action="store_true", help="Run model test")

    # Data options (two modes)
    parser.add_argument("--data", type=str, help="Path to h5ad file with splits (train/valid/test)")
    parser.add_argument("--train", type=str, help="Path to training h5ad (separate file mode)")
    parser.add_argument("--val", type=str, help="Path to validation h5ad (separate file mode)")
    parser.add_argument("--test-data", type=str, help="Path to test h5ad (separate file mode)")

    # Split names (for single-file mode)
    parser.add_argument("--train-split", type=str, default="train", help="Train split name")
    parser.add_argument("--val-split", type=str, default="valid", help="Validation split name")
    parser.add_argument("--test-split", type=str, default="test", help="Test split name")

    # Training options
    parser.add_argument("--epochs", type=int, default=100, help="Max epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max-cpg", type=int, default=8000, help="Max CpG sites")
    parser.add_argument("--hidden-size", type=int, default=512, help="Hidden size")
    parser.add_argument("--layers", type=int, default=6, help="Number of layers")
    parser.add_argument("--heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--log-dir", type=str, default="./logs", help="Log directory")
    parser.add_argument("--exp-name", type=str, default="methylation_age", help="Experiment name")
    parser.add_argument("--wandb", action="store_true", help="Use Weights & Biases")
    parser.add_argument("--precision", type=str, default="16-mixed", help="Training precision")

    args = parser.parse_args()

    if args.data or (args.train and args.val):
        # Training mode
        try:
            from .trainer import train_methylation_model
        except ImportError:
            print("ERROR: pytorch_lightning required for training.")
            print("Install with: pip install pytorch-lightning torchmetrics")
            return

        # Create config using original BMFM SCBertConfig
        config = create_methylation_config(
            num_cpg_sites=args.max_cpg,
            num_hidden_layers=args.layers,
            num_attention_heads=args.heads,
            hidden_size=args.hidden_size,
            intermediate_size=args.hidden_size * 4,
            max_position_embeddings=args.max_cpg + 10
        )

        # Train - determine mode
        if args.data:
            # Single file with splits
            trainer, module = train_methylation_model(
                data_path=args.data,
                train_split=args.train_split,
                val_split=args.val_split,
                test_split=args.test_split,
                config=config,
                max_epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                max_cpg=args.max_cpg,
                log_dir=args.log_dir,
                experiment_name=args.exp_name,
                use_wandb=args.wandb,
                precision=args.precision
            )
        else:
            # Separate files
            trainer, module = train_methylation_model(
                train_adata=args.train,
                val_adata=args.val,
                test_adata=args.test_data,
                config=config,
                max_epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                max_cpg=args.max_cpg,
                log_dir=args.log_dir,
                experiment_name=args.exp_name,
                use_wandb=args.wandb,
                precision=args.precision
            )

        print("\nTraining complete!")
        print(f"Best model saved at: {trainer.checkpoint_callback.best_model_path}")

    else:
        # Test mode (default)
        print("=" * 60)
        print("BMFM Encoder for Methylation - Test")
        print("=" * 60)

        # Create config using original BMFM SCBertConfig
        config = create_methylation_config(
            num_cpg_sites=8000,
            num_hidden_layers=6,
            num_attention_heads=8,
            hidden_size=512,
            intermediate_size=2048,
            max_position_embeddings=8010
        )

        print(f"\nConfig:")
        print(f"  Layers: {config.num_hidden_layers}")
        print(f"  Heads: {config.num_attention_heads}")
        print(f"  Hidden size: {config.hidden_size}")
        print(f"  Intermediate: {config.intermediate_size}")

        # Create model
        model = MethylationAgeModel(config)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nModel parameters:")
        print(f"  Total: {total_params:,}")
        print(f"  Trainable: {trainable_params:,}")

        # Test forward pass
        batch_size = 4
        seq_len = 8000

        cpg_ids = torch.randint(0, 8000, (batch_size, seq_len))
        beta_values = torch.rand(batch_size, seq_len)

        print(f"\nTest input:")
        print(f"  CpG IDs shape: {cpg_ids.shape}")
        print(f"  Beta values shape: {beta_values.shape}")

        # Forward pass
        model.eval()
        with torch.no_grad():
            age_pred = model(cpg_ids, beta_values)

        print(f"\nOutput:")
        print(f"  Predicted ages shape: {age_pred.shape}")
        print(f"  Sample predictions: {age_pred.squeeze().tolist()}")

        # Test Lightning module if available
        try:
            from .lightning_module import MethylationAgeLightningModule
            if MethylationAgeLightningModule is not None:
                print(f"\n{'='*60}")
                print("Lightning Module Test")
                print("=" * 60)

                module = MethylationAgeLightningModule(config)
                print(f"Lightning module created successfully!")
                print(f"Optimizer: AdamW with LR={module.learning_rate}")
                print(f"Scheduler: {module.scheduler_type} with {module.warmup_steps} warmup steps")
        except ImportError:
            pass

        print("\n" + "=" * 60)
        print("SUCCESS! Model is ready for training on methylation data.")
        print("=" * 60)
        print("\nUsage examples:")
        print("  # Test model architecture:")
        print("  python -m bmfm_methylation --test")
        print("")
        print("  # Train on single h5ad with splits (your data format):")
        print("  python -m bmfm_methylation \\")
        print("      --data methylation.h5ad \\")
        print("      --train-split train \\")
        print("      --val-split valid \\")
        print("      --epochs 100 \\")
        print("      --batch-size 32")
        print("")
        print("  # Train on separate files:")
        print("  python -m bmfm_methylation \\")
        print("      --train train.h5ad \\")
        print("      --val val.h5ad \\")
        print("      --epochs 100")


if __name__ == "__main__":
    main()
