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
    n_rows = pf.metadata.num_rows
    n_cols = pf.metadata.num_columns
    all_cols = pf.schema_arrow.names
    print(f"  Total rows:    {n_rows}")
    print(f"  Total columns: {n_cols}")
    print(f"  First 10 cols: {all_cols[:10]}")
    print(f"  Last  10 cols: {all_cols[-10:]}")

    # Read only label-like columns (cheap — small subset)
    label_candidates = [c for c in all_cols if any(k in c.lower() for k in ["age","label","target","split","id","sample"])]
    print(f"\n  Label-like columns: {label_candidates}")

    if label_candidates:
        df_labels = pq.read_table(path, columns=label_candidates).to_pandas()
        for col in label_candidates:
            try:
                print(f"    {col}: dtype={df_labels[col].dtype}  sample={df_labels[col].iloc[:5].tolist()}")
                if pd.api.types.is_numeric_dtype(df_labels[col]):
                    print(f"      min={df_labels[col].min():.2f}  max={df_labels[col].max():.2f}  mean={df_labels[col].mean():.2f}  nulls={df_labels[col].isna().sum()}")
            except Exception as e:
                print(f"    {col}: error {e}")

    # Read only 5 CpG columns to check NaN/zero pattern (cheap)
    cpg_candidates = [c for c in all_cols if c not in label_candidates]
    sample_cpgs = cpg_candidates[:5] + cpg_candidates[len(cpg_candidates)//2:len(cpg_candidates)//2+5]
    print(f"\n  Sampling 10 CpG columns to check NaN/zero pattern...")
    df_cpg = pq.read_table(path, columns=sample_cpgs).to_pandas()
    for col in sample_cpgs:
        nan_pct  = df_cpg[col].isna().mean() * 100
        zero_pct = (df_cpg[col] == 0).mean() * 100
        print(f"    {col}: NaN={nan_pct:.1f}%  zero={zero_pct:.1f}%  sample={df_cpg[col].dropna().iloc[:3].round(4).tolist()}")

    # Check overlap with pretrain vocab
    pretrain_probes = set(pd.read_csv(PRETRAIN_PROBES)["illumina_probe_id"].tolist())
    cpg_cols_set = set(cpg_candidates)
    overlap = pretrain_probes & cpg_cols_set
    print(f"\n  Pretrain vocab:    {len(pretrain_probes)}")
    print(f"  Finetune CpG cols: {len(cpg_candidates)}")
    print(f"  Overlap:           {len(overlap)} ({100*len(overlap)/max(len(cpg_candidates),1):.1f}%)")
    break  # only inspect train for speed

print("\n\nDONE")
