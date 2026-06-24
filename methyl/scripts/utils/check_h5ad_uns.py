import h5py
import numpy as np

h5ad = "/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_pretrain_type3_h5ad/methylgpt_pretrain_type3.h5ad"
with h5py.File(h5ad, "r") as f:
    print("Top-level keys:", list(f.keys()))
    print("\n=== uns values ===")
    for k in f["uns"].keys():
        v = f["uns"][k]
        try:
            print(f"  uns['{k}']: {v[()]}")
        except Exception:
            print(f"  uns['{k}']: (group) keys={list(v.keys())}")
    print("\n=== layers ===")
    print("layer keys:", list(f["layers"].keys()))
