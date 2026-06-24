import h5py

h5ad = "/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_pretrain_type3_h5ad/methylgpt_pretrain_type3.h5ad"
with h5py.File(h5ad, "r") as f:
    print("obs keys:", list(f["obs"].keys()))
    if "_index" in f["obs"]:
        idx = f["obs"]["_index"]
        print(f"obs_names type: {idx.dtype}, first 5: {list(idx[:5])}")
    for k in f["obs"].keys():
        v = f["obs"][k]
        print(f"  obs['{k}']: dtype={v.dtype}, shape={v.shape}, first 3={list(v[:3])}")
