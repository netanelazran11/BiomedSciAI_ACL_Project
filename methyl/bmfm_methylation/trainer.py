"""
Training utilities for methylation age prediction

This module provides high-level training functions that use the original
BMFM SCBertModel encoder wrapped for methylation data.
"""

from typing import Optional, Tuple, Dict, Any, Union
from torch.utils.data import DataLoader

from .config import create_methylation_config, SCBertConfig
from .dataset import MethylationDataset
from .lightning_module import MethylationAgeLightningModule

# Try to import PyTorch Lightning
try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import (
        ModelCheckpoint,
        EarlyStopping,
        LearningRateMonitor,
        RichProgressBar
    )
    from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
    HAS_LIGHTNING = True
except ImportError:
    HAS_LIGHTNING = False
    pl = None


def create_trainer(
    max_epochs: int = 100,
    accelerator: str = "auto",
    devices: int = 1,
    precision: str = "16-mixed",
    log_dir: str = "./logs",
    experiment_name: str = "methylation_age",
    use_wandb: bool = False,
    wandb_project: str = "methylation-age",
    wandb_entity: str = None,
    early_stopping_patience: int = 10,
    checkpoint_metric: str = "val/mae",
    checkpoint_mode: str = "min",
    gradient_clip_val: float = 1.0,
    accumulate_grad_batches: int = 1,
    val_check_interval: float = 1.0,
    log_every_n_steps: int = 50
) -> "pl.Trainer":
    """
    Create a configured PyTorch Lightning Trainer.

    Args:
        max_epochs: Maximum training epochs
        accelerator: "auto", "gpu", "cpu", "mps"
        devices: Number of devices
        precision: "32", "16-mixed", "bf16-mixed"
        log_dir: Directory for logs and checkpoints
        experiment_name: Name for this experiment
        use_wandb: Whether to use Weights & Biases logging
        wandb_project: WandB project name
        early_stopping_patience: Epochs without improvement before stopping
        checkpoint_metric: Metric to monitor for checkpointing
        checkpoint_mode: "min" or "max"
        gradient_clip_val: Gradient clipping value
        accumulate_grad_batches: Gradient accumulation steps
        val_check_interval: Validation frequency
        log_every_n_steps: Logging frequency

    Returns:
        Configured pl.Trainer instance
    """
    if not HAS_LIGHTNING:
        raise ImportError("pytorch_lightning required. pip install pytorch-lightning")

    # Callbacks
    callbacks = []

    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{log_dir}/{experiment_name}/checkpoints",
        filename="{epoch}-{val/mae:.4f}",
        monitor=checkpoint_metric,
        mode=checkpoint_mode,
        save_top_k=3,
        save_last=True,
        verbose=True
    )
    callbacks.append(checkpoint_callback)

    # Early stopping
    early_stop_callback = EarlyStopping(
        monitor=checkpoint_metric,
        patience=early_stopping_patience,
        mode=checkpoint_mode,
        verbose=True
    )
    callbacks.append(early_stop_callback)

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)

    # Progress bar
    try:
        progress_bar = RichProgressBar()
        callbacks.append(progress_bar)
    except ImportError:
        pass  # Rich not installed

    # Logger
    if use_wandb:
        try:
            import wandb
            logger = WandbLogger(
                project=wandb_project,
                entity=wandb_entity,  # Your WandB username or team
                name=experiment_name,
                save_dir=log_dir,
                log_model=True,  # Log model checkpoints to WandB
            )
        except ImportError:
            print("wandb not installed, falling back to TensorBoard")
            logger = TensorBoardLogger(log_dir, name=experiment_name)
    else:
        logger = TensorBoardLogger(log_dir, name=experiment_name)

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=gradient_clip_val,
        accumulate_grad_batches=accumulate_grad_batches,
        val_check_interval=val_check_interval,
        log_every_n_steps=log_every_n_steps,
        deterministic=False,
        enable_progress_bar=True
    )

    return trainer


def train_methylation_model(
    train_adata=None,
    val_adata=None,
    test_adata=None,
    data_path: Optional[str] = None,
    train_split: str = "train",
    val_split: str = "valid",
    test_split: str = "test",
    config: Optional[SCBertConfig] = None,
    max_epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    max_cpg: int = 8000,
    num_workers: int = 4,
    accelerator: str = "auto",
    precision: str = "16-mixed",
    log_dir: str = "./logs",
    experiment_name: str = "methylation_age",
    use_wandb: bool = False,
    wandb_project: str = "methylation-age",
    wandb_entity: str = None,
    early_stopping_patience: int = 10,
    normalize_age: bool = True,
    **trainer_kwargs
) -> Tuple["pl.Trainer", "MethylationAgeLightningModule"]:
    """
    High-level function to train a methylation age prediction model.

    Supports two modes:
    1. Single h5ad with splits: Pass data_path and split names
    2. Separate files: Pass train_adata, val_adata, test_adata

    Args:
        train_adata: Training AnnData or path (mode 2)
        val_adata: Validation AnnData or path (mode 2)
        test_adata: Optional test AnnData (mode 2)
        data_path: Path to h5ad with all splits (mode 1)
        train_split: Name of train split (default: "train")
        val_split: Name of validation split (default: "valid")
        test_split: Name of test split (default: "test")
        config: SCBertConfig for methylation (uses defaults if None)
        max_epochs: Maximum training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        max_cpg: Maximum CpG sites
        num_workers: DataLoader workers
        accelerator: Device accelerator
        precision: Training precision
        log_dir: Logging directory
        experiment_name: Experiment name
        use_wandb: Use Weights & Biases
        wandb_project: WandB project name
        early_stopping_patience: Early stopping patience
        normalize_age: Whether to normalize age targets
        **trainer_kwargs: Additional trainer arguments

    Returns:
        Tuple of (trainer, module)

    Example (single file with splits):
        >>> trainer, module = train_methylation_model(
        ...     data_path="methylation.h5ad",
        ...     train_split="train",
        ...     val_split="valid",
        ...     max_epochs=50
        ... )

    Example (separate files):
        >>> trainer, module = train_methylation_model(
        ...     train_adata="train.h5ad",
        ...     val_adata="val.h5ad",
        ...     max_epochs=50
        ... )
    """
    if not HAS_LIGHTNING:
        raise ImportError("pytorch_lightning required. pip install pytorch-lightning")

    # Default config using original BMFM SCBertConfig
    if config is None:
        config = create_methylation_config(
            num_cpg_sites=max_cpg,
            max_position_embeddings=max_cpg + 10
        )

    # Determine data source mode
    if data_path is not None:
        # Mode 1: Single file with splits
        print(f"Loading data from: {data_path}")
        train_dataset = MethylationDataset(
            data_path,
            max_cpg=max_cpg,
            normalize_age=normalize_age,
            split=train_split
        )
        val_dataset = MethylationDataset(
            data_path,
            max_cpg=max_cpg,
            normalize_age=normalize_age,
            age_mean=train_dataset.age_mean,
            age_std=train_dataset.age_std,
            split=val_split
        )
        # Check if test split exists
        has_test = True
        try:
            test_dataset = MethylationDataset(
                data_path,
                max_cpg=max_cpg,
                normalize_age=normalize_age,
                age_mean=train_dataset.age_mean,
                age_std=train_dataset.age_std,
                split=test_split
            )
        except ValueError:
            has_test = False
            test_dataset = None
    elif train_adata is not None and val_adata is not None:
        # Mode 2: Separate files
        train_dataset = MethylationDataset(
            train_adata,
            max_cpg=max_cpg,
            normalize_age=normalize_age
        )
        val_dataset = MethylationDataset(
            val_adata,
            max_cpg=max_cpg,
            normalize_age=normalize_age,
            age_mean=train_dataset.age_mean,
            age_std=train_dataset.age_std
        )
        has_test = test_adata is not None
        test_dataset = None
    else:
        raise ValueError("Must provide either data_path OR (train_adata and val_adata)")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # Calculate max_steps for scheduler
    steps_per_epoch = len(train_loader)
    max_steps = max_epochs * steps_per_epoch

    # Create Lightning module
    module = MethylationAgeLightningModule(
        config=config,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=min(1000, steps_per_epoch),
        max_steps=max_steps,
        normalize_targets=normalize_age,
        age_mean=train_dataset.age_mean,
        age_std=train_dataset.age_std
    )

    # Create trainer
    trainer = create_trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        precision=precision,
        log_dir=log_dir,
        experiment_name=experiment_name,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        early_stopping_patience=early_stopping_patience,
        **trainer_kwargs
    )

    # Log hyperparameters to WandB
    if use_wandb and trainer.logger is not None:
        trainer.logger.log_hyperparams({
            "model/num_hidden_layers": config.num_hidden_layers,
            "model/num_attention_heads": config.num_attention_heads,
            "model/hidden_size": config.hidden_size,
            "model/intermediate_size": config.intermediate_size,
            "data/num_cpg_sites": max_cpg,
            "data/train_samples": len(train_dataset),
            "data/val_samples": len(val_dataset),
            "data/age_mean": train_dataset.age_mean,
            "data/age_std": train_dataset.age_std,
            "training/batch_size": batch_size,
            "training/learning_rate": learning_rate,
            "training/weight_decay": weight_decay,
            "training/max_epochs": max_epochs,
            "training/precision": precision,
        })

    # Train
    print(f"\n{'='*60}")
    print("Starting Training")
    print(f"{'='*60}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"CpG sites: {max_cpg}")
    print(f"Batch size: {batch_size}")
    print(f"Max epochs: {max_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"{'='*60}\n")

    trainer.fit(module, train_loader, val_loader)

    # Test if available
    if has_test:
        if test_dataset is None and test_adata is not None:
            # Mode 2: create test dataset from separate file
            test_dataset = MethylationDataset(
                test_adata,
                max_cpg=max_cpg,
                normalize_age=normalize_age,
                age_mean=train_dataset.age_mean,
                age_std=train_dataset.age_std
            )

        if test_dataset is not None:
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers
            )
            trainer.test(module, test_loader)

    return trainer, module
