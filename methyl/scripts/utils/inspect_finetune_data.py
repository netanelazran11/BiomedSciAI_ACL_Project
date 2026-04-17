"""
Inspect the 49k fine-tuning parquet data (nested format: id, data, age).
Run on cluster: python3 scripts/utils/inspect_finetune_data.py
"""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path

DATA_DIR = Path("/sci/labs/benjamin.yakir/netanel.azran/repos/MethylGPT-Thesis/data/finetuning_data_49k")
PRETRAIN_PROBES = "/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_pretrain_type3_h5ad/probe_ids_type3_pretrain.csv"

# ── 1. Schema ──────────────────────────────────────────────────────────────
for split in ["train", "valid", "test"]:
    pf = pq.ParquetFile(DATA_DIR / f"{split}.parquet")
    print(f"{split}: {pf.metadata.num_rows} rows × {pf.metadata.num_columns} cols")

# ── 2. Read a few rows of the data column only ─────────────────────────────
print("\n--- Reading 5 rows of 'data' column ---")
pf = pq.ParquetFile(DATA_DIR / "train.parquet")
batch = next(pf.iter_batches(batch_size=5, columns=["data", "age", "id"]))
df = batch.to_pandas()

print(f"id sample:   {df['id'].tolist()}")
print(f"age sample:  {df['age'].tolist()}")

# Inspect the 'data' field structure
first = df["data"].iloc[0]
print(f"\ndata dtype:  {type(first)}")
arr = np.array(first, dtype=np.float32)
print(f"data length: {len(arr)}")
print(f"data range:  [{np.nanmin(arr):.4f}, {np.nanmax(arr):.4f}]")
print(f"data mean:   {np.nanmean(arr):.4f}")
print(f"NaN count:   {np.isnan(arr).sum()} / {len(arr)} ({100*np.isnan(arr).mean():.1f}%)")
print(f"Zero count:  {(arr==0).sum()} / {len(arr)} ({100*(arr==0).mean():.1f}%)")

# Check across all 5 samples
print("\n--- NaN/zero per sample (first 5) ---")
for i, row in df.iterrows():
    a = np.array(row["data"], dtype=np.float32)
    print(f"  sample {i}: len={len(a)}  NaN={np.isnan(a).sum()}  zero={(a==0).sum()}  age={row['age']:.1f}")

# ── 3. Check cpg_mapping ───────────────────────────────────────────────────
print("\n--- cpg_mapping/ directory ---")
mapping_dir = DATA_DIR / "cpg_mapping"
if mapping_dir.exists():
    for f in sorted(mapping_dir.iterdir()):
        print(f"  {f.name}  ({f.stat().st_size/1e3:.1f} KB)")
    # Try reading the mapping
    for f in sorted(mapping_dir.iterdir()):
        try:
            if f.suffix == ".csv":
                m = pd.read_csv(f, nrows=5)
            elif f.suffix == ".parquet":
                m = pd.read_parquet(f).head(5)
            elif f.suffix in (".json", ".txt"):
                print(f"  {f.name} content preview:")
                print(open(f).read()[:300])
                continue
            else:
                continue
            print(f"\n  {f.name} — shape: {m.shape}")
            print(m.to_string())
        except Exception as e:
            print(f"  {f.name}: error {e}")
else:
    print("  Not found")

# ── 4. Overlap with pretrain vocab ─────────────────────────────────────────
print("\n--- Pretrain vocab overlap ---")
try:
    pretrain_probes = pd.read_csv(PRETRAIN_PROBES)["illumina_probe_id"].tolist()
    print(f"Pretrain vocab: {len(pretrain_probes)} CpGs")
    print(f"Finetune data:  {len(arr)} values per sample")
    if len(arr) == len(pretrain_probes):
        print("Sizes match exactly — likely same 49k CpG set")
    else:
        print(f"Size mismatch: {len(arr)} vs {len(pretrain_probes)}")
except Exception as e:
    print(f"Could not read pretrain probes: {e}")

print("\nDONE")
