#!/usr/bin/env python3
"""
Collect the pilot evaluation into one table (CSV + JSON) and apply the pre-registered decision rule.

Reads, per arm directory under --eval-root (from evaluate_pilot.sh, rsynced to the Mac):
  view_consistency/per_run/{overlap,disjoint}_seed0.json   retrieval on the full held-out partition
  hard_negative.json                                        hard_negative_eval.py --out (run locally first)
  reconstruction/reconstruction_withheld_summary.json       withheld-CpG reconstruction
  frozen_probes/analysis_summary.json                       geometry + frozen probes (AltumAge)

Run from methyl/:
  for d in outputs/ablation_infonce/pilot_eval/*/; do
    python -m ablation_infonce.hard_negative_eval --emb-dir "$d/view_consistency/per_run/" --out "$d/hard_negative.json"
  done
  python -m ablation_infonce.summarize_pilot

Decision rule (fixed before the results; seed noise = |seed42 - seed43| of the same λ):
  InfoNCE adds something  <=>  λ=0.05 beats λ=0 on disjoint hard-negative top-1 (same study+tissue)
                               by more than BOTH seed gaps, and recon PCC / frozen probes are not worse
                               by more than the seed gap.
  -> a full-corpus recon-only run is worth its cost if the pilot shows such a gap OR is inconclusive
     (gap within noise); it is NOT needed if λ=0 already matches λ=0.05 everywhere within noise AND the
     paper drops the claim that InfoNCE is what produces identity/stability.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--eval-root", default="outputs/ablation_infonce/pilot_eval")
ap.add_argument("--out", default="outputs/ablation_infonce/pilot_summary")
a = ap.parse_args()


def load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


rows = []
for d in sorted(Path(a.eval_root).iterdir()):
    if not d.is_dir():
        continue
    r = {"arm": d.name}
    for cond in ("overlap", "disjoint"):
        v = load(d / "view_consistency" / "per_run" / f"{cond}_seed0.json")
        if v:
            p = v["pools"]["all"]
            r[f"{cond}_top1"] = p["retrieval_at1"]
            r[f"{cond}_matched_cos"] = p["matched_cos"]
            r[f"{cond}_unmatched_cos"] = p["unmatched_cos"]
            if "fully_measured" in v["pools"]:
                r[f"{cond}_top1_fully_measured"] = v["pools"]["fully_measured"]["retrieval_at1"]
    h = load(d / "hard_negative.json")
    if h and "disjoint_seed0" in h:
        pools = h["disjoint_seed0"]["pools"]
        for key, col in [("random", "hn_random"), ("same study", "hn_study"),
                         ("same study+tissue", "hn_study_tissue"),
                         ("same study+tissue+sex", "hn_study_tissue_sex"),
                         ("K nearest neighbours of target", "hn_knn")]:
            if key in pools:
                r[f"{col}_top1"] = pools[key]["top1"]
                if pools[key].get("median_margin") is not None:
                    r[f"{col}_margin"] = pools[key]["median_margin"]
    rc = load(d / "reconstruction" / "reconstruction_withheld_summary.json")
    if rc:
        r["recon_pcc"] = rc["pearson_withheld_obs_vs_pred"]
        r["recon_raw_mse"] = rc["raw_mse_model"]["mean"]
        r["recon_ratio_vs_mean_baseline"] = rc["ratio_model_vs_b_mean"]
    fp = load(d / "frozen_probes" / "analysis_summary.json")
    if fp:
        g = fp["geometry_cls"]
        r["eff_rank"] = g["effective_rank"]
        r["anisotropy"] = g["anisotropy_mean_cos"]
        r["age_ridge_r2"] = fp["age_probe_cls"]["linear_ridge_r2"]
        r["age_ridge_medae"] = fp["age_probe_cls"]["linear_ridge_medae"]
        for c, v in fp.get("class_probes_cls", {}).items():
            r[f"{c}_bal_acc"] = v["balanced_acc"]
    rows.append(r)

df = pd.DataFrame(rows).set_index("arm")
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
df.to_csv(out / "pilot_table.csv")
pd.set_option("display.width", 200)
print(df.T.to_string(float_format=lambda x: f"{x:.4f}"))

# Decision rule
key = "hn_study_tissue_top1"
ep = [i for i in df.index if i.startswith("lam") and "_ep" in i]
get = lambda lam, seed, col: next((df.loc[i, col] for i in ep
                                   if i.startswith(f"lam{lam}_seed{seed}_") and col in df.columns), None)
verdict = {"metric": key}
try:
    a0, a0b = get("0.0", 42, key), get("0.0", 43, key)
    c0, c0b = get("0.05", 42, key), get("0.05", 43, key)
    gap = ((c0 + c0b) - (a0 + a0b)) / 2
    noise = max(abs(a0 - a0b), abs(c0 - c0b))
    verdict.update({"lambda0_mean": (a0 + a0b) / 2, "lambda005_mean": (c0 + c0b) / 2,
                    "gap_005_minus_0": gap, "seed_noise": noise,
                    "infonce_adds_identity": bool(gap > noise),
                    "inconclusive": bool(abs(gap) <= noise)})
    for col in ("recon_pcc", "age_ridge_r2", "tissue_type_bal_acc"):
        x0, x0b, y0, y0b = (get("0.0", 42, col), get("0.0", 43, col), get("0.05", 42, col), get("0.05", 43, col))
        if None not in (x0, x0b, y0, y0b):
            verdict[f"{col}_gap_005_minus_0"] = ((y0 + y0b) - (x0 + x0b)) / 2
            verdict[f"{col}_seed_noise"] = max(abs(x0 - x0b), abs(y0 - y0b))
except TypeError:
    verdict["note"] = "seed replicates (tasks 0, 2, 4, 5) or hard_negative.json missing — rule not applied"
(out / "pilot_verdict.json").write_text(json.dumps(verdict, indent=1, default=float))
print("\nDecision rule:", json.dumps(verdict, indent=1, default=float))
