"""
ElasticNet baseline for age prediction on AltumAge 21k methylation data.

Purpose: establish a linear baseline on raw CpG beta values using the
identical dataset, splits, and filters as MethylLlama fine-tuning.
Reports MedAE / MAE / R² so results are directly comparable to V5/V6.

ElasticNet chosen because:
  - Standard in methylation aging literature (AltumAge itself uses it)
  - Handles high-dimensional input (21,368 features) via L1 sparsity
  - alpha=0.01, l1_ratio=0.5: equal L1+L2, mild regularization

NaN handling: simple column-mean imputation on train set, applied to val/test.
"""

import argparse
import json
import logging
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import pearsonr
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — same logic as DataModule to ensure identical preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def compute_dedup_exclusions(pairs_csv: str) -> set:
    """Connected-component dedup: keep alphabetically-first ID per group."""
    df = pd.read_csv(pairs_csv)
    if df.empty or "id_a" not in df.columns:
        return set()

    adj = defaultdict(set)
    all_nodes = set()
    for _, row in df.iterrows():
        a, b = str(row["id_a"]), str(row["id_b"])
        adj[a].add(b)
        adj[b].add(a)
        all_nodes.add(a)
        all_nodes.add(b)

    visited, to_exclude = set(), set()
    for start in sorted(all_nodes):
        if start in visited:
            continue
        component, queue = [], deque([start])
        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            component.append(node)
            for nb in adj[node]:
                if nb not in visited:
                    queue.append(nb)
        component.sort()
        to_exclude.update(component[1:])
    return to_exclude


def median_absolute_error(y_true, y_pred):
    return float(np.median(np.abs(np.array(y_true) - np.array(y_pred))))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ElasticNet age baseline on 21k methylation data")
    parser.add_argument("--h5ad",    required=True, help="Path to altumage_21k_3way.h5ad")
    parser.add_argument("--dedup",   required=True, help="Path to duplicate_pairs.csv")
    parser.add_argument("--outdir",  required=True, help="Output directory for results")
    parser.add_argument("--alpha",   type=float, default=0.01,
                        help="ElasticNet alpha (overall regularization strength). Default: 0.01")
    parser.add_argument("--l1_ratio", type=float, default=0.5,
                        help="ElasticNet l1_ratio (0=Ridge, 1=Lasso). Default: 0.5")
    parser.add_argument("--age_col",   default="age")
    parser.add_argument("--split_col", default="split")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    logger.info(f"Loading {args.h5ad}")
    adata = sc.read_h5ad(args.h5ad)
    logger.info(f"  shape: {adata.shape[0]:,} × {adata.shape[1]:,}")

    # ── 2. Age outlier filter (same as DataModule) ────────────────────────────
    age_vals = pd.to_numeric(adata.obs[args.age_col], errors="coerce")
    keep = (age_vals >= 0) & (age_vals <= 120)
    n_removed = int((~keep).sum())
    logger.info(f"Age filter: removing {n_removed} samples (age<0 or age>120)")
    adata = adata[keep.values].copy()

    # ── 3. Duplicate exclusion (same as DataModule) ───────────────────────────
    exclude_ids = compute_dedup_exclusions(args.dedup)
    keep_mask = np.array([sid not in exclude_ids for sid in adata.obs_names])
    n_dup = int((~keep_mask).sum())
    logger.info(f"Duplicate exclusion: removing {n_dup} samples")
    adata = adata[keep_mask].copy()
    logger.info(f"Clean dataset: {adata.shape[0]:,} samples")

    # ── 4. Split into train / valid / test ────────────────────────────────────
    splits = adata.obs[args.split_col]
    for name in ("train", "valid", "test"):
        n = int((splits == name).sum())
        logger.info(f"  {name}: {n:,} samples")

    idx_train = splits == "train"
    idx_valid = splits == "valid"
    idx_test  = splits == "test"

    X_train = adata[idx_train].X
    X_valid = adata[idx_valid].X
    X_test  = adata[idx_test].X

    # Convert sparse → dense if needed
    if hasattr(X_train, "toarray"):
        X_train = X_train.toarray()
        X_valid = X_valid.toarray()
        X_test  = X_test.toarray()

    X_train = X_train.astype(np.float32)
    X_valid = X_valid.astype(np.float32)
    X_test  = X_test.astype(np.float32)

    y_train = adata[idx_train].obs[args.age_col].values.astype(np.float32)
    y_valid = adata[idx_valid].obs[args.age_col].values.astype(np.float32)
    y_test  = adata[idx_test].obs[args.age_col].values.astype(np.float32)

    logger.info(f"X_train: {X_train.shape}, y_train: mean={y_train.mean():.1f}yr std={y_train.std():.1f}yr")

    # ── 5. NaN imputation (column mean from train set) ────────────────────────
    nan_cols = np.where(np.isnan(X_train).any(axis=0))[0]
    if len(nan_cols) > 0:
        logger.info(f"Imputing {len(nan_cols)} CpGs with column mean from train set")
        col_means = np.nanmean(X_train, axis=0)
        for col in nan_cols:
            X_train[np.isnan(X_train[:, col]), col] = col_means[col]
            X_valid[np.isnan(X_valid[:, col]), col] = col_means[col]
            X_test[np.isnan(X_test[:, col]),  col] = col_means[col]
    else:
        logger.info("No NaN values found — no imputation needed")

    # ── 6. Feature scaling (StandardScaler fit on train) ──────────────────────
    logger.info("Fitting StandardScaler on train set")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)
    X_test  = scaler.transform(X_test)

    # ── 7. Fit ElasticNet ─────────────────────────────────────────────────────
    logger.info(
        f"Fitting ElasticNet(alpha={args.alpha}, l1_ratio={args.l1_ratio}, max_iter=10000)"
    )
    model = ElasticNet(
        alpha=args.alpha,
        l1_ratio=args.l1_ratio,
        max_iter=10000,
        tol=1e-4,
        random_state=42,
        selection="random",   # faster convergence for high-dimensional data
    )
    model.fit(X_train, y_train)
    n_nonzero = int(np.sum(model.coef_ != 0))
    logger.info(f"ElasticNet fitted: {n_nonzero:,} / {X_train.shape[1]:,} non-zero coefficients")

    # ── 8. Evaluate ───────────────────────────────────────────────────────────
    def evaluate(X, y, name):
        pred = model.predict(X)
        mae    = float(mean_absolute_error(y, pred))
        medae  = median_absolute_error(y, pred)
        r2     = float(r2_score(y, pred))
        pcc, _ = pearsonr(y, pred)
        logger.info(
            f"  {name:5s}  MAE={mae:.2f}yr  MedAE={medae:.2f}yr  R²={r2:.4f}  PCC={pcc:.4f}"
        )
        return {"mae": mae, "medae": medae, "r2": r2, "pcc": float(pcc), "n": int(len(y))}

    logger.info("=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    results = {
        "model": "ElasticNet",
        "alpha": args.alpha,
        "l1_ratio": args.l1_ratio,
        "n_features": int(X_train.shape[1]),
        "n_nonzero_coef": n_nonzero,
        "train": evaluate(X_train, y_train, "train"),
        "valid": evaluate(X_valid, y_valid, "valid"),
        "test":  evaluate(X_test,  y_test,  "test"),
    }

    # ── 9. Save results ───────────────────────────────────────────────────────
    results_path = outdir / "elasticnet_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    # Summary for quick comparison with MethylLlama
    summary = (
        f"\n{'='*60}\n"
        f"ElasticNet Baseline Summary\n"
        f"{'='*60}\n"
        f"alpha={args.alpha}, l1_ratio={args.l1_ratio}\n"
        f"Non-zero CpGs: {n_nonzero:,} / {X_train.shape[1]:,}\n"
        f"\n"
        f"{'Split':<8} {'MAE':>8} {'MedAE':>8} {'R²':>8}\n"
        f"{'-'*36}\n"
        f"{'train':<8} {results['train']['mae']:>8.2f} {results['train']['medae']:>8.2f} {results['train']['r2']:>8.4f}\n"
        f"{'valid':<8} {results['valid']['mae']:>8.2f} {results['valid']['medae']:>8.2f} {results['valid']['r2']:>8.4f}\n"
        f"{'test':<8} {results['test']['mae']:>8.2f} {results['test']['medae']:>8.2f} {results['test']['r2']:>8.4f}\n"
        f"{'='*60}\n"
        f"MethylLlama V5 reference: MedAE=3.65yr, R²=0.905\n"
    )
    print(summary)
    with open(outdir / "elasticnet_summary.txt", "w") as f:
        f.write(summary)


if __name__ == "__main__":
    main()
