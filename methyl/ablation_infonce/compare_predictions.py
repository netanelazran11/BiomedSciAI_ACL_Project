#!/usr/bin/env python3
"""
Compare two per-subject age-prediction files written by
scripts/repr_analysis/extract_kfold_test_predictions.py (sample_id, true_age, predicted_age, ...).

  --mode repro   Same model, two environments: are the predictions identical?
                 Reports max |difference| and the three metrics for both files. PASS when the
                 sample sets are identical and max |difference| < --tol (years).
  --mode paired  Two models on the same test subjects (e.g. pilot λ=0.05 vs λ=0): metrics per
                 model and a paired bootstrap over subjects of (B minus A) for MAE, MedAE and R².

Run from methyl/ (CPU, seconds):
  python -m ablation_infonce.compare_predictions --mode repro  --a <stored.csv> --b <new.csv>
  python -m ablation_infonce.compare_predictions --mode paired --a <lam0.csv> --b <lam005.csv> \
      --label_a "lambda=0" --label_b "lambda=0.05" --out <summary.json>
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["repro", "paired"], required=True)
ap.add_argument("--a", required=True)
ap.add_argument("--b", required=True)
ap.add_argument("--label_a", default="A")
ap.add_argument("--label_b", default="B")
ap.add_argument("--tol", type=float, default=0.01, help="repro: max allowed |difference| in years")
ap.add_argument("--n_boot", type=int, default=10000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default=None)
a = ap.parse_args()


def r2(t, p):
    return 1.0 - np.sum((t - p) ** 2) / np.sum((t - t.mean()) ** 2)


def metrics(t, p):
    e = np.abs(p - t)
    return {"mae": float(e.mean()), "medae": float(np.median(e)), "r2": float(r2(t, p)), "n": int(len(t))}


A = pd.read_csv(a.a, dtype={"sample_id": str}).set_index("sample_id")
B = pd.read_csv(a.b, dtype={"sample_id": str}).set_index("sample_id")
for name, df in (("a", A), ("b", B)):
    if df.index.duplicated().any() or df["predicted_age"].isna().any():
        raise SystemExit(f"file {name}: duplicate ids or missing predictions")
same_ids = set(A.index) == set(B.index)
common = A.index.intersection(B.index)
A, B = A.loc[common], B.loc[common]
if not np.allclose(A["true_age"].values, B["true_age"].values):
    raise SystemExit("true_age differs between the files for the same subjects")
t = A["true_age"].to_numpy(float)
pa, pb = A["predicted_age"].to_numpy(float), B["predicted_age"].to_numpy(float)

res = {"mode": a.mode, "file_a": a.a, "file_b": a.b, "same_sample_set": bool(same_ids), "n_common": int(len(common)),
       a.label_a: metrics(t, pa), a.label_b: metrics(t, pb)}

if a.mode == "repro":
    d = np.abs(pa - pb)
    res.update({"max_abs_diff_years": float(d.max()), "mean_abs_diff_years": float(d.mean()), "tol": a.tol,
                "PASS": bool(same_ids and d.max() < a.tol)})
else:
    rng = np.random.default_rng(a.seed)
    n = len(t)
    boot = {"mae": [], "medae": [], "r2": []}
    for _ in range(a.n_boot):
        i = rng.integers(0, n, n)
        ma, mb = metrics(t[i], pa[i]), metrics(t[i], pb[i])
        for k in boot:
            boot[k].append(mb[k] - ma[k])
    point = {k: res[a.label_b][k] - res[a.label_a][k] for k in boot}
    res["difference_b_minus_a"] = {
        k: {"point": float(point[k]), "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
            "ci_excludes_0": bool(np.percentile(v, 2.5) > 0 or np.percentile(v, 97.5) < 0)}
        for k, v in boot.items()}
    res["note"] = "Negative MAE/MedAE difference or positive R² difference means B is better."

print(json.dumps(res, indent=1))
if a.out:
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
if a.mode == "repro" and not res["PASS"]:
    raise SystemExit("REPRODUCIBILITY CHECK FAILED")
