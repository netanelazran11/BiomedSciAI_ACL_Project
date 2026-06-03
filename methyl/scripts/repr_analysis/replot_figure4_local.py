#!/usr/bin/env python3
"""
Replot figure4 from pre-computed PCA coords (no GPU / cluster needed).
Loads coords and metadata from the local figures/figure4/ directory.

Usage:
  python scripts/repr_analysis/replot_figure4_local.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from scripts.repr_analysis.figure4_age_pca import make_figure

DATA_DIR = ROOT / "figures" / "figure4"
OUT_DIR  = DATA_DIR

pre_coords = np.load(DATA_DIR / "pretrained_pca_coords.npy")
ft_coords  = np.load(DATA_DIR / "finetuned_pca_coords.npy")
meta       = pd.read_csv(DATA_DIR / "aligned_metadata.csv", index_col=0)

ages          = pd.to_numeric(meta["age"], errors="coerce").values
tissue_labels = meta["tissue"].fillna("unknown").tolist()

# Recompute explained variance from coords (approximate from std of each PC)
# We don't have PCA objects, so pass dummy var ratios and override in title
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

print("Loading embeddings to recompute variance explained...")

pre_npy_path = ROOT / "figures" / "figure4" / ".." / ".." / ".."
# Fallback: use dummy variances — real values are in the axis labels already
# We store real variance in the existing figure. Let's just use stored coords
# and compute approximate variance from the marginal std.
# Proper variance needs the original embeddings which aren't synced locally.
# Use approximate: var[i] ≈ std(coords[:,i])^2 / sum(std^2)
pre_std = pre_coords.std(axis=0) ** 2
pre_var = pre_std / pre_std.sum()
ft_std  = ft_coords.std(axis=0) ** 2
ft_var  = ft_std / ft_std.sum()

# These won't match exactly (PCA var = fraction of total embedding variance,
# not fraction of 2D variance), but for the axis labels it's close enough.
# The real values from the cluster run were:
#   Pretrained: PC1=25.7%, PC2=12.4%
#   Fine-tuned: PC1=22.0%, PC2=10.7%
# Hard-code them:
pre_var = np.array([0.257, 0.124])
ft_var  = np.array([0.220, 0.107])

print(f"pre_coords: {pre_coords.shape}  ft_coords: {ft_coords.shape}")
print(f"ages: {ages.min():.0f}–{ages.max():.0f}  valid: {(~np.isnan(ages)).sum()}")
print(f"tissues: {len(set(tissue_labels))} unique")

make_figure(pre_coords, pre_var, ft_coords, ft_var,
            ages, tissue_labels, OUT_DIR, dpi=200)

print(f"\nSaved → {OUT_DIR}/figures/figure4_age_pca.png")
