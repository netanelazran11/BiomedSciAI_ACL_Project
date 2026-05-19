"""
Convert finetuning_49k.h5ad to train/val/test parquet files for MethylGPT fine-tuning.
Each parquet file has two columns:
  - "age": float
  - "data": numpy array of 49,156 beta values (NaN preserved — handled by MethylGPT collater)
"""
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from pathlib import Path

DATA = "/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_finetune_49k_h5ad/finetuning_49k.h5ad"
OUT_DIR = Path("/sci/labs/benjamin.yakir/netanel.azran/repos/MethylGPT-Thesis/data/finetune_49k_parquet")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Loading {DATA}...")
adata = sc.read_h5ad(DATA)
print(f"  {adata.n_obs} samples x {adata.n_vars} CpGs")
print(f"  obs columns: {list(adata.obs.columns)}")

X = adata.X.toarray() if sp.issparse(adata.X) else np.array(adata.X, dtype=np.float32)
ages = adata.obs["age"].values.astype(np.float64)

if "split" not in adata.obs.columns:
    raise ValueError("No 'split' column in adata.obs — expected train/val/test labels")

splits = adata.obs["split"].values
print(f"\nSplit distribution:\n{pd.Series(splits).value_counts().to_string()}")

SPLIT_MAP = {
    "train": ["train"],
    "val":   ["val", "valid", "validation"],
    "test":  ["test"],
}

for out_name, split_labels in SPLIT_MAP.items():
    mask = np.isin(splits, split_labels) & ~np.isnan(ages)
    X_split = X[mask].astype(np.float32)
    ages_split = ages[mask]

    df = pd.DataFrame({
        "age":  ages_split,
        "data": [row for row in X_split],
    })

    out_file = OUT_DIR / f"finetune49k_{out_name}.parquet"
    df.to_parquet(out_file, index=False)
    print(f"  Saved {out_name}: {len(df)} samples -> {out_file}")

print("\nDone.")
