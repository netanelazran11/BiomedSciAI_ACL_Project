"""
Compare train/val/test sample IDs between MethylGPT (parquet) and MethylLlama (h5ad).
Verifies both models used identical splits for fair comparison.
"""

import pandas as pd
import scanpy as sc

PARQUET_DIR = "/sci/labs/benjamin.yakir/netanel.azran/MethylGPT/data/19k_data/finetuning_data"
H5AD_PATH   = "/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_finetune_19k_h5ad/finetuning_19608_clean_stratified_no_outliers.h5ad"

# ── MethylGPT (parquet) ───────────────────────────────────────────────────────
print("Loading MethylGPT parquet files...")
gpt = {}
for split in ("train", "valid", "test"):
    df = pd.read_parquet(f"{PARQUET_DIR}/{split}.parquet")
    # sample ID is usually index or a column — try both
    if "sample_id" in df.columns:
        ids = set(df["sample_id"].astype(str))
    elif "GSM" in str(df.index[0]) or "GSE" in str(df.index[0]):
        ids = set(df.index.astype(str))
    else:
        ids = set(df.index.astype(str))
    gpt[split] = ids
    print(f"  MethylGPT  {split:5s}: {len(ids):5d} samples  (example: {list(ids)[:2]})")

# ── MethylLlama (h5ad) ───────────────────────────────────────────────────────
print("\nLoading MethylLlama h5ad...")
adata = sc.read_h5ad(H5AD_PATH)
print(f"  Total samples: {len(adata)}")
print(f"  Split column values: {adata.obs['split'].value_counts().to_dict()}")

llama = {}
for split in ("train", "valid", "test"):
    ids = set(adata.obs[adata.obs["split"] == split].index.astype(str))
    llama[split] = ids
    print(f"  MethylLlama {split:5s}: {len(ids):5d} samples  (example: {list(ids)[:2]})")

# ── Compare ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SPLIT COMPARISON")
print("=" * 60)
all_match = True
for split in ("train", "valid", "test"):
    g = gpt[split]
    l = llama[split]
    overlap  = len(g & l)
    only_gpt = len(g - l)
    only_llm = len(l - g)
    identical = (g == l)
    all_match = all_match and identical
    print(f"\n{split.upper()}:")
    print(f"  MethylGPT : {len(g):5d} samples")
    print(f"  MethylLlama: {len(l):5d} samples")
    print(f"  Overlap   : {overlap:5d}")
    print(f"  Only GPT  : {only_gpt:5d}")
    print(f"  Only Llama: {only_llm:5d}")
    print(f"  IDENTICAL : {'YES ✓' if identical else 'NO ✗'}")
    if only_gpt > 0 and only_gpt <= 5:
        print(f"  GPT-only IDs: {list(g - l)}")
    if only_llm > 0 and only_llm <= 5:
        print(f"  Llama-only IDs: {list(l - g)}")

print("\n" + "=" * 60)
print(f"ALL SPLITS IDENTICAL: {'YES ✓' if all_match else 'NO ✗ — comparison may be unfair!'}")
print("=" * 60)
