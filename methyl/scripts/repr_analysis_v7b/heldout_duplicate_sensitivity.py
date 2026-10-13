"""
heldout_duplicate_sensitivity.py
================================
Do the held-out results (Fig. 2) change when held-out profiles that duplicate another profile
are removed?

Inputs (all already produced; nothing is re-encoded)
  outputs/data_structure_audit/heldout_duplicates_pairs.csv          scripts/utils/pretrain_heldout_duplicates.py
  outputs/pretrain_overlap/pretrain_ids.csv.gz                      corpus row -> sample id
  figures/v7b_pretrain_cls/view_consistency_heldout_full/per_run/   embeddings of both views, every run
  figures/v7b_pretrain_cls/reconstruction_withheld/per_sample_metrics.csv
      rows = held-out positions rng(0).choice(16,912, 5,000), sorted (reconstruction_withheld_eval.py,
      DataLoader without shuffling)

Exclusion sets
  dup_train       held-out profiles with a duplicate in the pretraining TRAINING partition
  dup_any         ... or a duplicate inside the held-out partition (both members removed)
  near_any        dup_any plus near-duplicates (training or held-out)

Writes outputs/data_structure_audit/heldout_duplicate_sensitivity.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from view_consistency_heldout_full import pool_stats      # noqa: E402

VIEW = REPO / "figures/v7b_pretrain_cls/view_consistency_heldout_full/per_run"


def main():
    pairs = pd.read_csv(REPO / "outputs/data_structure_audit/heldout_duplicates_pairs.csv")
    ids = pd.read_csv(REPO / "outputs/pretrain_overlap/pretrain_ids.csv.gz").set_index("row").sample_id
    pairs["sid_a"], pairs["sid_b"] = ids.reindex(pairs.row_a).to_numpy(), ids.reindex(pairs.row_b).to_numpy()
    assert (pairs.sid_a == pairs.id_a).all() and (pairs.sid_b == pairs.id_b).all(), "row numbering differs"
    tr = pairs.comparison == "heldout vs train"
    ho = pairs.comparison == "heldout vs heldout"
    sets = {"dup_train": set(pairs[tr & pairs.duplicate].id_a)}
    sets["dup_any"] = sets["dup_train"] | set(pairs[ho & pairs.duplicate].id_a) | set(pairs[ho & pairs.duplicate].id_b)
    nd = pairs.duplicate | pairs.near_duplicate
    sets["near_any"] = sets["dup_any"] | set(pairs[tr & nd].id_a) | set(pairs[ho & nd].id_a) | set(pairs[ho & nd].id_b)

    out = {"excluded_profiles": {k: len(v) for k, v in sets.items()}, "cross_view": {}, "reconstruction": {}}
    runs = sorted(VIEW.glob("*_embeddings.npz"))
    assert len(runs) == 7
    for f in runs:
        z = np.load(f)
        e1, e2, sid = z["e1"].astype(np.float32), z["e2"].astype(np.float32), z["sample_ids"].astype(str)
        tag = f.name.replace("_embeddings.npz", "")
        res = {"all": pool_stats(e1, e2, np.arange(len(sid)))[0]}
        for k, v in sets.items():
            res[f"without_{k}"] = pool_stats(e1, e2, np.where(~np.isin(sid, list(v)))[0])[0]
        out["cross_view"][tag] = {p: {m: round(r[m], 6) for m in ("n_candidates", "matched_cos", "unmatched_cos",
                                                                  "retrieval_at1", "retrieval_at10", "chance_top1")}
                                  for p, r in res.items()}
        print(tag, {p: f"N={r['n_candidates']} top1={100 * r['retrieval_at1']:.2f}%" for p, r in res.items()})

    sid = np.load(runs[0])["sample_ids"].astype(str)                     # held-out partition order
    pos = np.sort(np.random.default_rng(0).choice(len(sid), size=5000, replace=False))
    pm = pd.read_csv(REPO / "figures/v7b_pretrain_cls/reconstruction_withheld/per_sample_metrics.csv")
    assert len(pm) == len(pos)
    pm["sample_id"] = sid[pos]
    for name, keep in [("all", np.ones(len(pm), bool))] + [(f"without_{k}", ~pm.sample_id.isin(v).to_numpy())
                                                           for k, v in sets.items()]:
        g = pm[keep]
        out["reconstruction"][name] = {"n": int(len(g)), **{c: round(float(g[c].mean()), 5)
                                                            for c in ("model_mse", "b_mean_mse", "b_shuffle_mse", "b_random_mse")}}
        print("reconstruction", name, out["reconstruction"][name])
    (REPO / "outputs/data_structure_audit/heldout_duplicate_sensitivity.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
