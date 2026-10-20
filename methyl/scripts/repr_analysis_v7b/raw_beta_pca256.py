#!/usr/bin/env python3
"""
Raw-methylation baseline representation for Fig. 3: PCA-256 of the raw beta values of the 10,988
AltuMAge profiles, the same dimension as the CLS representation, rows aligned to metadata.csv.

Unsupervised, like the CLS representation: missing values are replaced by the CpG's mean over all
profiles, beta values are centred per CpG (not scaled), and PCA (256 components, randomized solver,
random_state 0) is fitted on all profiles; no label is used. Probes, CIs and neighbourhood statistics
are computed afterwards with the same code as for CLS (fig3_baseline_probes.py, pinned environment).

Usage (cluster): sbatch scripts/repr_analysis_v7b/run_raw_beta_pca256.sh
Writes figures/v7b_pretrain_cls/raw_pca256.npy (float32) and raw_pca256_info.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--h5ad", required=True)
ap.add_argument("--dir", default="figures/v7b_pretrain_cls")
ap.add_argument("--n_components", type=int, default=256)
a = ap.parse_args()

from bmfm_methylation.shared.data_module import _read_h5ad_robust  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
import sklearn  # noqa: E402

d = Path(a.dir)
meta = pd.read_csv(d / "metadata.csv")
adata = _read_h5ad_robust(a.h5ad)
names = np.asarray(adata.obs_names).astype(str)
ids = meta["sample_id"].astype(str).values
assert len(set(names)) == len(names), "duplicate obs names in h5ad"
missing = set(ids) - set(names)
assert not missing, f"{len(missing)} metadata profiles not in h5ad"
pos = pd.Index(names).get_indexer(ids)
X = adata.X
X = np.asarray(X.todense() if hasattr(X, "todense") else X, dtype=np.float32)[pos]
assert X.shape[0] == len(meta) == 10988, X.shape
n_nan = int(np.isnan(X).sum())
col_mean = np.nanmean(X, axis=0)
all_nan = np.isnan(col_mean)
col_mean[all_nan] = 0.5
r, c = np.where(np.isnan(X))
X[r, c] = col_mean[c]
X -= X.mean(axis=0, keepdims=True)
pca = PCA(n_components=a.n_components, svd_solver="randomized", random_state=0)
Z = pca.fit_transform(X).astype(np.float32)
np.save(d / "raw_pca256.npy", Z)
info = {"script": "raw_beta_pca256.py", "h5ad": a.h5ad, "n_profiles": int(Z.shape[0]), "n_cpgs": int(X.shape[1]),
        "n_components": a.n_components, "missing_values_imputed": n_nan, "cpgs_all_missing": int(all_nan.sum()),
        "explained_variance_total": float(pca.explained_variance_ratio_.sum()),
        "explained_variance_first10": [float(v) for v in pca.explained_variance_ratio_[:10]],
        "rows": "aligned to metadata.csv sample_id (asserted)", "sklearn": sklearn.__version__}
(d / "raw_pca256_info.json").write_text(json.dumps(info, indent=1))
print(json.dumps(info, indent=1))
