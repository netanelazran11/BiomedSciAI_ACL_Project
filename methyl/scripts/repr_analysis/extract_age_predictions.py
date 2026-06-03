#!/usr/bin/env python3
"""
extract_age_predictions.py
==========================
Run the full fine-tuned MethylationAgeRegressorLlama model on the 19k finetune
dataset and save per-sample predicted age.

Output: <outdir>/age_predictions.csv  with columns:
  sample_id, actual_age, predicted_age, split, tissue

Usage (on cluster):
  python scripts/repr_analysis/extract_age_predictions.py \
      --checkpoint  outputs/finetune-llama-small/.../epoch=127-val_medae=3.5625.ckpt \
      --data        /path/to/finetuning_19608.h5ad \
      --tokenizer   tokenizer_llama_pretrain49k \
      --outdir      outputs/repr_analysis/age_predictions_JOBID \
      --batch_size  64 \
      --device      cuda
"""

import argparse, logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  required=True)
    p.add_argument("--data",        required=True)
    p.add_argument("--tokenizer",   required=True)
    p.add_argument("--outdir",      default="outputs/repr_analysis/age_predictions")
    p.add_argument("--batch_size",  type=int, default=64)
    p.add_argument("--device",      default="cuda")
    p.add_argument("--metadata",    default=None,
                   help="External metadata CSV.gz for tissue labels")
    p.add_argument("--metadata_id_col", default="GSM_ID")
    return p.parse_args()


def load_model(ckpt_path: str, device: str):
    from bmfm_methylation.llama.finetune_llama import MethylationAgeRegressorLlama
    log.info(f"Loading fine-tuned model from {ckpt_path}")
    model = MethylationAgeRegressorLlama.load_from_checkpoint(
        ckpt_path, map_location=device, strict=False
    )
    model.eval()
    model.to(device)
    log.info(f"  age_mean={model.age_mean:.2f}  age_std={model.age_std:.2f}")
    return model


def main():
    args   = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")

    # Load model
    model = load_model(args.checkpoint, device)

    # Load data module
    from bmfm_methylation.llama.finetune_llama import MethylLlamaDataModule
    dm = MethylLlamaDataModule(
        data_path=args.data,
        tokenizer_path=args.tokenizer,
        batch_size=args.batch_size,
        num_workers=4,
        subset_k=8000,
        max_length=8002,
    )
    dm.setup("predict")

    # Full dataset (train + val + test)
    from bmfm_methylation.llama.finetune_llama import MethylLlamaDataset
    import anndata
    adata = anndata.read_h5ad(args.data)
    log.info(f"h5ad: {adata.shape[0]} samples × {adata.shape[1]} CpGs")

    # Use the predict dataloader if available, else iterate train+val+test
    all_sample_ids = []
    all_actual     = []
    all_predicted  = []
    all_splits     = []

    for split_name in ["train", "val", "test"]:
        try:
            if split_name == "train":
                loader = dm.train_dataloader()
            elif split_name == "val":
                loader = dm.val_dataloader()
            else:
                loader = dm.test_dataloader()
        except Exception:
            continue

        log.info(f"  Running {split_name} split ({len(loader)} batches)...")
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                attn_mask = batch.get("attention_mask")
                if attn_mask is not None:
                    attn_mask = attn_mask.to(device)
                ages_batch = batch.get("age", batch.get("labels"))

                out = model(input_ids=input_ids,
                            attention_mask=attn_mask)

                # out is the dict from validation_step's shared_step
                # Actually call the shared forward
                cls = model._encode(input_ids, attn_mask)
                age_pred_norm = model.age_head(cls).squeeze(-1)
                age_pred_yr   = age_pred_norm.detach().cpu() * model.age_std + model.age_mean

                sample_ids = batch.get("sample_id", [f"{split_name}_{i}"
                                                       for i in range(len(age_pred_yr))])
                actual_yr  = (ages_batch.float() * model.age_std + model.age_mean
                              if ages_batch is not None
                              else torch.full_like(age_pred_yr, float("nan")))

                all_sample_ids.extend(sample_ids if isinstance(sample_ids, list)
                                      else sample_ids.tolist())
                all_predicted.extend(age_pred_yr.numpy().tolist())
                all_actual.extend(actual_yr.numpy().tolist())
                all_splits.extend([split_name] * len(age_pred_yr))

    df = pd.DataFrame({
        "sample_id":    all_sample_ids,
        "actual_age":   all_actual,
        "predicted_age": all_predicted,
        "split":        all_splits,
    })

    # Join tissue if external metadata provided
    if args.metadata and Path(args.metadata).exists():
        ext = pd.read_csv(args.metadata)
        ext = ext.drop_duplicates(subset=args.metadata_id_col).set_index(args.metadata_id_col)
        if "tissue" in ext.columns:
            df = df.set_index("sample_id").join(ext[["tissue"]], how="left").reset_index()

    out_path = outdir / "age_predictions.csv"
    df.to_csv(out_path, index=False)
    log.info(f"Saved {len(df)} predictions → {out_path}")

    # Quick metrics
    for split_name in ["train", "val", "test"]:
        mask = (df["split"] == split_name) & df["actual_age"].notna()
        if mask.sum() > 0:
            from sklearn.metrics import r2_score, median_absolute_error
            r2    = r2_score(df.loc[mask, "actual_age"], df.loc[mask, "predicted_age"])
            medae = median_absolute_error(df.loc[mask, "actual_age"], df.loc[mask, "predicted_age"])
            log.info(f"  [{split_name:5s}] R²={r2:.3f}  MedAE={medae:.2f} yr  n={mask.sum()}")


if __name__ == "__main__":
    main()
