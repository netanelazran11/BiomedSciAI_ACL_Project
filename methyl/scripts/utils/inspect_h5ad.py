import anndata
import os

path = "/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_pretrain_type3_h5ad/methylgpt_pretrain_type3.h5ad"
size_gb = os.path.getsize(path) / 1e9
print(f"File size: {size_gb:.2f} GB")

a = anndata.read_h5ad(path, backed="r")
print("Shape:", a.shape)
print("obs columns:", a.obs.columns.tolist())
print("var columns:", a.var.columns.tolist())
print("obs head:")
print(a.obs.head())
print("var head:")
print(a.var.head())
