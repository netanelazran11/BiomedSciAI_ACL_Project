"""
Compare train/val/test sample IDs between MethylGPT (parquet) and MethylLlama (h5ad).
Verifies both models used identical splits for fair comparison.
"""

import h5py
import pandas as pd
import pyarrow.parquet as pq

PARQUET_DIR = "/sci/labs/benjamin.yakir/netanel.azran/MethylGPT/data/19k_data/finetuning_data"
H5AD_PATH   = "/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_finetune_19k_h5ad/finetuning_19608_clean_stratified_no_outliers.h5ad"

# ── MethylGPT (parquet) — read schema + first column only ────────────────────
print("Loading MethylGPT parquet files (schema inspect)...")
gpt = {}
for split in ("train", "valid", "test"):
    path = f"{PARQUET_DIR}/{split}.parquet"
    pf = pq.ParquetFile(path)
    col_names = pf.schema_arrow.names
    print(f"  {split} columns: {col_names[:8]}")
    # Read only the first column to get row count + index
    tbl = pf.read(columns=[col_names[0]])
    df  = tbl.to_pandas()
    ids = set(df.index.astype(str))
    if len(ids) == 1 and list(ids)[0] in ("0", ""):
        # index is RangeIndex, IDs must be in a column
        ids = set(df[col_names[0]].astype(str))
    gpt[split] = ids
    print(f"  MethylGPT  {split:5s}: {len(ids):5d} samples  (example: {list(ids)[:3]})")

# ── MethylLlama (h5ad) — read only obs via h5py, no matrix load ──────────────
print("\nLoading MethylLlama h5ad (obs only via h5py)...")
with h5py.File(H5AD_PATH, "r") as f:
    # obs index (sample IDs)
    obs_grp = f["obs"]
    print(f"  obs keys: {list(obs_grp.keys())[:10]}")
    # index is stored as _index or the first string dataset
    if "_index" in obs_grp:
        all_ids = [x.decode() if isinstance(x, bytes) else x for x in obs_grp["_index"][:]]
    else:
        idx_key = list(obs_grp.keys())[0]
        all_ids = [x.decode() if isinstance(x, bytes) else x for x in obs_grp[idx_key][:]]
    # split column
    split_vals = [x.decode() if isinstance(x, bytes) else x for x in obs_grp["split"]["codes"][:]]
    split_cats = [x.decode() if isinstance(x, bytes) else x for x in obs_grp["split"]["categories"][:]]
    splits_decoded = [split_cats[c] for c in split_vals]

print(f"  Total samples: {len(all_ids)}")
from collections import Counter
print(f"  Split counts: {dict(Counter(splits_decoded))}")

llama = {}
for split in ("train", "valid", "test"):
    ids = set(sid for sid, sp in zip(all_ids, splits_decoded) if sp == split)
    llama[split] = ids
    print(f"  MethylLlama {split:5s}: {len(ids):5d} samples  (example: {list(ids)[:3]})")

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
