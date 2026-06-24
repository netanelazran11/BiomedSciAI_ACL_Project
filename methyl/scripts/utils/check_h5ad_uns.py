import h5py

h5ad = "/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_pretrain_type3_h5ad/methylgpt_pretrain_type3.h5ad"
with h5py.File(h5ad, "r") as f:
    print("Top-level keys:", list(f.keys()))
    if "uns" in f:
        print("uns keys:", list(f["uns"].keys()))
    if "obsm" in f:
        print("obsm keys:", list(f["obsm"].keys()))
    if "obsp" in f:
        print("obsp keys:", list(f["obsp"].keys()))
