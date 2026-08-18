"""
ElasticNet baseline for age prediction on AltumAge 21k methylation data.

Matches the exact evaluation protocol used for MethylLlama/MethylGPT:
same split column, same duplicate-pair exclusion (dataset_fingerprint_outputs/
duplicate_pairs.csv), same age-outlier filter (age<0 or age>120 removed) --
so the resulting test set is the identical 2,149 samples used in the
paired-bootstrap comparison, not the raw 2,264.

Hyperparameters (alpha, l1_ratio) are chosen by a small grid search selected
on VALIDATION MedAE (never test), matching how every MethylLlama/MethylGPT
checkpoint in this project is selected. alpha=0.01 is included as one of the
grid points.
"""

import argparse
import json
import logging
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


def median_absolute_error(y_true, y_pred):
    return float(np.median(np.abs(np.array(y_true) - np.array(y_pred))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5ad",     required=True)
    parser.add_argument("--outdir",   required=True)
    parser.add_argument("--alphas",    type=float, nargs="+",
                         default=[0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0],
                         help="Grid of alpha values to search; best chosen by validation MedAE.")
    parser.add_argument("--l1_ratios", type=float, nargs="+",
                         default=[0.1, 0.5, 0.9, 1.0],
                         help="Grid of l1_ratio values to search; best chosen by validation MedAE.")
    parser.add_argument("--age_col",   default="age")
    parser.add_argument("--split_col", default="split")
    parser.add_argument("--duplicate_pairs_csv",
                         default=None,
                         help="Path to duplicate_pairs.csv -- if set, excludes one ID per "
                              "duplicate pair, matching the MethylLlama/MethylGPT eval protocol.")
    parser.add_argument("--filter_age_outliers", action="store_true", default=True)
    parser.add_argument("--no_filter_age_outliers", dest="filter_age_outliers", action="store_false")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ── 0. Fail loudly if inputs don't exist -- don't burn a job on a typo'd path ──
    h5ad_path = Path(args.h5ad)
    if not h5ad_path.exists():
        raise FileNotFoundError(f"--h5ad not found: {h5ad_path}")
    if args.duplicate_pairs_csv is not None and not Path(args.duplicate_pairs_csv).exists():
        raise FileNotFoundError(f"--duplicate_pairs_csv not found: {args.duplicate_pairs_csv}")

    # ── 1. Load raw data ──────────────────────────────────────────────────────
    logger.info(f"Loading {args.h5ad}")
    adata = sc.read_h5ad(args.h5ad)
    logger.info(f"  Total samples : {adata.shape[0]:,}")
    logger.info(f"  Total CpGs    : {adata.shape[1]:,}")

    # ── 1b. Apply the same exclusions as the MethylLlama/MethylGPT eval set ───
    n_before = adata.shape[0]
    if args.filter_age_outliers and args.age_col in adata.obs.columns:
        age_vals = adata.obs[args.age_col].astype(float)
        keep = (age_vals >= 0) & (age_vals <= 120)
        n_removed = int((~keep).sum())
        adata = adata[keep].copy()
        logger.info(f"Age outlier filter: removed {n_removed} samples (age<0 or age>120)")

    if args.duplicate_pairs_csv is not None:
        from bmfm_methylation.shared.data_module import _compute_dedup_exclusions
        exclude_ids = _compute_dedup_exclusions(args.duplicate_pairs_csv)
        keep_mask = ~adata.obs_names.isin(exclude_ids)
        n_removed = int((~keep_mask).sum())
        adata = adata[keep_mask].copy()
        logger.info(f"Duplicate filter: removed {n_removed} samples ({len(exclude_ids)} IDs in dedup set)")

    logger.info(f"Total after filtering: {adata.shape[0]:,} / {n_before:,} samples")

    # ── 2. Split by existing split column ─────────────────────────────────────
    splits = adata.obs[args.split_col]
    for name in ("train", "valid", "test"):
        logger.info(f"  {name:5s} : {int((splits == name).sum()):,} samples")

    idx_train = splits == "train"
    idx_valid = splits == "valid"
    idx_test  = splits == "test"

    def to_dense(X):
        if hasattr(X, "toarray"):
            X = X.toarray()
        return X.astype(np.float32)

    X_train = to_dense(adata[idx_train].X)
    X_valid = to_dense(adata[idx_valid].X)
    X_test  = to_dense(adata[idx_test].X)

    y_train = adata[idx_train].obs[args.age_col].values.astype(np.float32)
    y_valid = adata[idx_valid].obs[args.age_col].values.astype(np.float32)
    y_test  = adata[idx_test].obs[args.age_col].values.astype(np.float32)

    logger.info(f"Train age: mean={y_train.mean():.1f}yr  std={y_train.std():.1f}yr")

    # ── 3. Feature scaling (StandardScaler — required for ElasticNet) ─────────
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)
    X_test  = scaler.transform(X_test)

    # ── 4. Grid search over (alpha, l1_ratio), select by VALIDATION MedAE ─────
    # Selecting on validation (never test) matches how every MethylLlama/MethylGPT
    # checkpoint in this project is chosen (best-val_medae), so the reported
    # ElasticNet number is comparable on equal footing, not cherry-picked on test.
    logger.info(f"Grid search: {len(args.alphas)} alphas x {len(args.l1_ratios)} l1_ratios "
                f"= {len(args.alphas) * len(args.l1_ratios)} fits")
    grid_results = []
    best = None  # (val_medae, alpha, l1_ratio, model, n_nonzero)
    for alpha in args.alphas:
        for l1_ratio in args.l1_ratios:
            model = ElasticNet(
                alpha=alpha,
                l1_ratio=l1_ratio,
                max_iter=10000,
                tol=1e-4,
                random_state=42,
                selection="random",
            )
            model.fit(X_train, y_train)
            val_pred = model.predict(X_valid)
            val_medae = median_absolute_error(y_valid, val_pred)
            val_mae = float(mean_absolute_error(y_valid, val_pred))
            n_nonzero = int(np.sum(model.coef_ != 0))
            grid_results.append({
                "alpha": alpha, "l1_ratio": l1_ratio,
                "val_medae": val_medae, "val_mae": val_mae, "n_nonzero_coef": n_nonzero,
            })
            logger.info(f"  alpha={alpha:<8} l1_ratio={l1_ratio:<5} "
                        f"val_medae={val_medae:.3f}yr  val_mae={val_mae:.3f}yr  "
                        f"nonzero={n_nonzero}/{X_train.shape[1]}")
            if best is None or val_medae < best[0]:
                best = (val_medae, alpha, l1_ratio, model, n_nonzero)

    best_val_medae, best_alpha, best_l1_ratio, model, n_nonzero = best
    logger.info(f"Best: alpha={best_alpha}, l1_ratio={best_l1_ratio} (val_medae={best_val_medae:.3f}yr)")

    # ── 5. Evaluate the selected model ────────────────────────────────────────
    def evaluate(X, y, name):
        pred  = model.predict(X)
        mae   = float(mean_absolute_error(y, pred))
        medae = median_absolute_error(y, pred)
        r2    = float(r2_score(y, pred))
        pcc,_ = pearsonr(y, pred)
        logger.info(f"  {name:5s}  MAE={mae:.2f}yr  MedAE={medae:.2f}yr  R²={r2:.4f}  PCC={pcc:.4f}")
        return {"mae": mae, "medae": medae, "r2": r2, "pcc": float(pcc), "n": int(len(y))}

    logger.info("=" * 60)
    logger.info("RESULTS (best grid point, selected on validation MedAE)")
    logger.info("=" * 60)
    results = {
        "model": "ElasticNet",
        "alpha": best_alpha,
        "l1_ratio": best_l1_ratio,
        "n_features": int(X_train.shape[1]),
        "n_nonzero_coef": n_nonzero,
        "grid_search": grid_results,
        "train": evaluate(X_train, y_train, "train"),
        "valid": evaluate(X_valid, y_valid, "valid"),
        "test":  evaluate(X_test,  y_test,  "test"),
    }

    # ── 6. Save ───────────────────────────────────────────────────────────────
    with open(outdir / "elasticnet_results.json", "w") as f:
        json.dump(results, f, indent=2)

    summary = (
        f"\n{'='*60}\n"
        f"ElasticNet Baseline — selected alpha={best_alpha}, l1_ratio={best_l1_ratio} "
        f"(best of {len(grid_results)} grid points by validation MedAE)\n"
        f"Non-zero CpGs : {n_nonzero:,} / {X_train.shape[1]:,}\n"
        f"{'='*60}\n"
        f"{'Split':<8} {'N':>6} {'MAE':>8} {'MedAE':>8} {'R²':>8}\n"
        f"{'-'*42}\n"
        f"{'train':<8} {results['train']['n']:>6} {results['train']['mae']:>8.2f} {results['train']['medae']:>8.2f} {results['train']['r2']:>8.4f}\n"
        f"{'valid':<8} {results['valid']['n']:>6} {results['valid']['mae']:>8.2f} {results['valid']['medae']:>8.2f} {results['valid']['r2']:>8.4f}\n"
        f"{'test':<8} {results['test']['n']:>6} {results['test']['mae']:>8.2f} {results['test']['medae']:>8.2f} {results['test']['r2']:>8.4f}\n"
        f"{'='*60}\n"
    )
    print(summary)
    with open(outdir / "elasticnet_summary.txt", "w") as f:
        f.write(summary)


if __name__ == "__main__":
    main()
