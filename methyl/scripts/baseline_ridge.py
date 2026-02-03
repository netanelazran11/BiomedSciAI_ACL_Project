#!/usr/bin/env python3
"""
Baseline: Ridge regression on raw beta values for age prediction.

This script establishes the performance floor. If Ridge can predict age
well (MAE < 5 years), the data contains an age signal and the deep model
should be able to match or beat it.

Usage:
    python scripts/baseline_ridge.py /path/to/methylation.h5ad
"""

import sys
import numpy as np
import scanpy as sc
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy import stats


def main():
    if len(sys.argv) < 2:
        print("Usage: python baseline_ridge.py <h5ad_path>")
        sys.exit(1)

    h5ad_path = sys.argv[1]
    print(f"Loading data from: {h5ad_path}")

    adata = sc.read_h5ad(h5ad_path)
    print(f"Total samples: {len(adata)}")
    print(f"Total CpG sites: {adata.shape[1]}")
    print(f"Columns: {list(adata.obs.columns)}")

    # Split data
    if "split" in adata.obs.columns:
        train_mask = adata.obs["split"] == "train"
        val_mask = adata.obs["split"] == "valid"
        test_mask = adata.obs["split"] == "test"
    else:
        print("ERROR: No 'split' column found in obs")
        sys.exit(1)

    print(f"\nSplit sizes:")
    print(f"  Train: {train_mask.sum()}")
    print(f"  Valid: {val_mask.sum()}")
    print(f"  Test:  {test_mask.sum()}")

    # Get X and y
    X_train = adata[train_mask].X
    X_val = adata[val_mask].X
    X_test = adata[test_mask].X

    if hasattr(X_train, 'toarray'):
        X_train = X_train.toarray()
        X_val = X_val.toarray()
        X_test = X_test.toarray()

    y_train = adata[train_mask].obs["age"].values.astype(np.float64)
    y_val = adata[val_mask].obs["age"].values.astype(np.float64)
    y_test = adata[test_mask].obs["age"].values.astype(np.float64)

    print(f"\nAge distribution (train):")
    print(f"  Mean:   {y_train.mean():.1f}")
    print(f"  Std:    {y_train.std():.1f}")
    print(f"  Min:    {y_train.min():.1f}")
    print(f"  Max:    {y_train.max():.1f}")
    print(f"  Median: {np.median(y_train):.1f}")

    # Baseline: predict mean
    mean_pred = np.full_like(y_test, y_train.mean())
    print(f"\n{'='*60}")
    print("BASELINE: Predict Mean Age")
    print(f"{'='*60}")
    print(f"  MAE:  {mean_absolute_error(y_test, mean_pred):.2f} years")
    print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, mean_pred)):.2f} years")
    print(f"  R2:   {r2_score(y_test, mean_pred):.4f}")

    # Ridge regression with different alphas
    print(f"\n{'='*60}")
    print("RIDGE REGRESSION")
    print(f"{'='*60}")
    best_mae = float('inf')
    best_alpha = None

    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_train, y_train)

        val_preds = ridge.predict(X_val)
        val_mae = mean_absolute_error(y_val, val_preds)
        val_r2 = r2_score(y_val, val_preds)

        test_preds = ridge.predict(X_test)
        test_mae = mean_absolute_error(y_test, test_preds)
        test_r2 = r2_score(y_test, test_preds)

        print(f"\n  alpha={alpha:>8.2f} | Val MAE={val_mae:6.2f}, R2={val_r2:.4f} | Test MAE={test_mae:6.2f}, R2={test_r2:.4f}")

        if val_mae < best_mae:
            best_mae = val_mae
            best_alpha = alpha

    # Best Ridge
    print(f"\n{'='*60}")
    print(f"BEST RIDGE (alpha={best_alpha})")
    print(f"{'='*60}")
    ridge = Ridge(alpha=best_alpha)
    ridge.fit(X_train, y_train)

    for split_name, X, y in [("Train", X_train, y_train), ("Valid", X_val, y_val), ("Test", X_test, y_test)]:
        preds = ridge.predict(X)
        mae = mean_absolute_error(y, preds)
        rmse = np.sqrt(mean_squared_error(y, preds))
        r2 = r2_score(y, preds)
        pcc = stats.pearsonr(y, preds)[0]
        print(f"  {split_name:5s}: MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.4f}, PCC={pcc:.4f}")

    # ElasticNet for comparison
    print(f"\n{'='*60}")
    print("ELASTICNET (alpha=1.0, l1_ratio=0.5)")
    print(f"{'='*60}")
    enet = ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=10000)
    enet.fit(X_train, y_train)

    for split_name, X, y in [("Train", X_train, y_train), ("Valid", X_val, y_val), ("Test", X_test, y_test)]:
        preds = enet.predict(X)
        mae = mean_absolute_error(y, preds)
        r2 = r2_score(y, preds)
        pcc = stats.pearsonr(y, preds)[0]
        print(f"  {split_name:5s}: MAE={mae:.2f}, R2={r2:.4f}, PCC={pcc:.4f}")

    n_nonzero = np.sum(np.abs(enet.coef_) > 1e-8)
    print(f"  Non-zero coefficients: {n_nonzero} / {len(enet.coef_)}")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"If Ridge MAE < 5 years  -> Data is GOOD, deep model should work")
    print(f"If Ridge MAE 5-10 years -> Data is OK, deep model might help")
    print(f"If Ridge MAE > 15 years -> Data lacks age signal, fix data first")


if __name__ == "__main__":
    main()
