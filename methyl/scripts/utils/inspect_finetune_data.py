"""
Inspect the 49k fine-tuning parquet data.
Run on cluster: python scripts/utils/inspect_finetune_data.py
"""
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path("/sci/labs/benjamin.yakir/netanel.azran/repos/MethylGPT-Thesis/data/finetuning_data_49k")
PRETRAIN_PROBES = "/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_pretrain_type3_h5ad/probe_ids_type3_pretrain.csv"

for split in ["train", "valid", "test"]:
    path = DATA_DIR / f"{split}.parquet"
    if not path.exists():
        path = Path(str(path) + "")  # symlink resolve
    print(f"\n{'='*60}")
    print(f"  {split.upper()}: {path}")
    print(f"{'='*60}")

    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    print(f"  Total rows: {pf.metadata.num_rows}")
    print(f"  Row groups: {pf.metadata.num_row_groups}")
    df = next(pf.iter_batches(batch_size=100)).to_pandas()  # read only first 100 rows
    print(f"  Shape:   {df.shape}")
    print(f"  Columns: {list(df.columns[:10])}{'...' if len(df.columns)>10 else ''}")
    print(f"  Dtypes:\n{df.dtypes.value_counts().to_string()}")

    # Check for age/label column
    label_cols = [c for c in df.columns if any(k in c.lower() for k in ["age","label","target","y"])]
    print(f"  Label columns: {label_cols}")
    for col in label_cols:
        print(f"    {col}: min={df[col].min():.1f}  max={df[col].max():.1f}  mean={df[col].mean():.1f}  nulls={df[col].isna().sum()}")

    # Check CpG columns (non-label numeric columns)
    cpg_cols = [c for c in df.columns if c not in label_cols and df[c].dtype in [np.float32, np.float64, "float32", "float64"]]
    print(f"  CpG columns: {len(cpg_cols)}")

    if cpg_cols:
        cpg_data = df[cpg_cols].values.astype(np.float32)
        nan_pct   = np.isnan(cpg_data).mean() * 100
        zero_pct  = (cpg_data == 0).mean() * 100
        print(f"  NaN  %:  {nan_pct:.2f}%")
        print(f"  Zero %:  {zero_pct:.2f}%")
        print(f"  Value range: [{np.nanmin(cpg_data):.4f}, {np.nanmax(cpg_data):.4f}]")
        print(f"  Mean:  {np.nanmean(cpg_data):.4f}  Std: {np.nanstd(cpg_data):.4f}")

        # Per-column NaN analysis
        col_nan = np.isnan(cpg_data).mean(axis=0)
        all_nan_cols  = (col_nan == 1.0).sum()
        high_nan_cols = (col_nan > 0.5).sum()
        zero_nan_cols = (col_nan == 0.0).sum()
        print(f"  Cols with ALL NaN:   {all_nan_cols}")
        print(f"  Cols with >50% NaN:  {high_nan_cols}")
        print(f"  Cols with 0% NaN:    {zero_nan_cols}")

    # Check overlap with pretrain vocab
    pretrain_probes = set(pd.read_csv(PRETRAIN_PROBES)["illumina_probe_id"].tolist())
    overlap = set(cpg_cols) & pretrain_probes
    only_in_ft = set(cpg_cols) - pretrain_probes
    print(f"\n  Pretrain vocab size:  {len(pretrain_probes)}")
    print(f"  Finetune CpG cols:    {len(cpg_cols)}")
    print(f"  Overlap:              {len(overlap)} ({100*len(overlap)/max(len(cpg_cols),1):.1f}% of finetune in pretrain)")
    print(f"  Only in finetune:     {len(only_in_ft)}")

    # Sample rows
    print(f"\n  First 3 rows (first 5 cols):")
    print(df.iloc[:3, :5].to_string())
    break  # only inspect train for speed

print("\n\nDONE")
