import h5py
import os

path = "/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_pretrain_type3_h5ad/methylgpt_pretrain_type3.h5ad"
print(f"File size: {os.path.getsize(path)/1e9:.2f} GB\n")

def print_tree(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"  DATASET  {name:60s}  shape={obj.shape}  dtype={obj.dtype}")
    else:
        print(f"  GROUP    {name}")

with h5py.File(path, "r") as f:
    print("Top-level keys:", list(f.keys()))
    print()
    f.visititems(print_tree)
