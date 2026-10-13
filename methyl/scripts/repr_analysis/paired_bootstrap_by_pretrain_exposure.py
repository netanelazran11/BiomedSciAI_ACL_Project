#!/usr/bin/env python3
"""
paired_bootstrap_by_pretrain_exposure.py
=========================================
Does the MethylLlama-vs-MethylGPT advantage on the 2,149-subject test set depend
on whether a subject's profile was part of the pretraining corpus?

Inputs (both already produced; nothing is re-run):
  outputs/bootstrap_predictions/paired_per_subject_predictions.csv   fold-averaged predictions, both models
  outputs/pretrain_overlap/overlap_by_id.csv                         exact id overlap + pretraining split
                                                                     (scripts/utils/pretrain_overlap_by_id.py)

Same statistics as Table 1 (scripts/repr_analysis/paired_bootstrap_comparison.py): percentile
bootstrap over subjects, 10,000 resamples, seed 0, identical resample indices for both models.
Signs: error differences are MethylGPT minus MethylLlama, R2 is MethylLlama minus MethylGPT, so
positive always favours MethylLlama.

Subsets
  all                     the 2,149 test subjects (must reproduce Table 1)
  in_pretrain_train       profile was in the pretraining TRAIN split (seen during optimisation)
  not_in_pretrain_train   profile never contributed a gradient (absent, or in valid / held-out)
  untouched               absent from the corpus or in its held-out split (not used even for
                          checkpoint selection)
  not_in_corpus           absent from the pretraining corpus altogether

Output: outputs/pretrain_overlap/paired_bootstrap_by_exposure.json (+ a printed table)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


def metrics(y, p):
    e = np.abs(p - y)
    return float(np.median(e)), float(e.mean()), float(r2_score(y, p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", default="outputs/bootstrap_predictions/paired_per_subject_predictions.csv")
    ap.add_argument("--overlap", default="outputs/pretrain_overlap/overlap_by_id.csv")
    ap.add_argument("--out", default="outputs/pretrain_overlap/paired_bootstrap_by_exposure.json")
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    pred = pd.read_csv(a.predictions, dtype={"sample_id": str})
    ov = pd.read_csv(a.overlap, dtype={"sample_id": str}).fillna({"pretrain_split": ""})
    assert pred["sample_id"].is_unique and ov["sample_id"].is_unique
    df = pred.merge(ov[["sample_id", "in_pretraining_corpus", "pretrain_split", "dataset", "tissue_type"]],
                    on="sample_id", how="left", validate="one_to_one")
    assert df["in_pretraining_corpus"].notna().all(), "test subjects missing from the overlap file"
    assert len(df) == 2149, len(df)
    in_train = df["pretrain_split"].str.contains("train")
    subsets = {
        "all": np.ones(len(df), bool),
        "in_pretrain_train": in_train.values,
        "not_in_pretrain_train": (~in_train).values,
        "untouched": (~df["in_pretraining_corpus"].astype(bool) | (df["pretrain_split"] == "heldout")).values,
        "not_in_corpus": (~df["in_pretraining_corpus"].astype(bool)).values,
    }

    out = {"n_boot": a.n_boot, "seed": a.seed,
           "sign_convention": "error: MethylGPT - MethylLlama; R2: MethylLlama - MethylGPT; positive favours MethylLlama",
           "exposure_counts": df["pretrain_split"].replace("", "not in corpus").value_counts().to_dict(),
           "subsets": {}}
    print(f"{'subset':<24}{'n':>6} {'age mean':>9} | {'Llama MedAE/MAE/R2':<24} {'GPT MedAE/MAE/R2':<24} | paired diff [95% CI]")
    for name, mask in subsets.items():
        s = df[mask]
        y, pl, pg = (s[c].to_numpy(float) for c in ("true_age", "predicted_age_llama", "predicted_age_gpt"))
        n = len(s)
        ml, mg = metrics(y, pl), metrics(y, pg)
        rng = np.random.default_rng(a.seed)      # same stream start for every subset
        d = np.empty((a.n_boot, 3))
        for b in range(a.n_boot):
            i = rng.integers(0, n, n)
            x, z = metrics(y[i], pl[i]), metrics(y[i], pg[i])
            d[b] = (z[0] - x[0], z[1] - x[1], x[2] - z[2])
        lo, hi = np.percentile(d, [2.5, 97.5], axis=0)
        point = (mg[0] - ml[0], mg[1] - ml[1], ml[2] - mg[2])
        out["subsets"][name] = {
            "n": int(n), "age_mean": float(y.mean()), "age_sd": float(y.std(ddof=1)),
            "n_studies": int(s["dataset"].nunique()),
            "methyllama": dict(zip(("medae", "mae", "r2"), ml)), "methylgpt": dict(zip(("medae", "mae", "r2"), mg)),
            "paired_difference": {k: {"estimate": float(point[j]), "ci_95": [float(lo[j]), float(hi[j])],
                                      "excludes_zero": bool(lo[j] > 0 or hi[j] < 0)}
                                  for j, k in enumerate(("medae", "mae", "r2"))},
        }
        print(f"{name:<24}{n:>6} {y.mean():>9.1f} | {ml[0]:.3f} / {ml[1]:.3f} / {ml[2]:.4f}   "
              f"{mg[0]:.3f} / {mg[1]:.3f} / {mg[2]:.4f}   | MedAE {point[0]:.3f} [{lo[0]:.3f}, {hi[0]:.3f}]  "
              f"MAE {point[1]:.3f} [{lo[1]:.3f}, {hi[1]:.3f}]  R2 {point[2]:.4f} [{lo[2]:.4f}, {hi[2]:.4f}]")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\nexposure of the 2,149 test subjects: {out['exposure_counts']}")
    print(f"Saved -> {a.out}")


if __name__ == "__main__":
    main()
