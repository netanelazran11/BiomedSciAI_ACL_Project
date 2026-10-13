"""
check_study_value_type.py
=========================
What kind of numbers does a downstream study hold?

GEO describes the GSE56105 processed file (average_beta.txt) as "Normalised Average Beta ... and
the Detection P-value": two columns per sample. If the beta and p-value columns were confused
when the collection was assembled, the downstream values of that study are detection p-values.
Detection p-values of a good array are ~0 for nearly every probe; beta values are bimodal
(many near 0, many near 1, few in between).

For the study under test and for comparison studies of the same tissue, prints per study:
fraction of values exactly 0, < 0.01, > 0.99, between 0.2 and 0.8, mean, and quantiles.

  python scripts/utils/check_study_value_type.py
(reads ~300 rows; seconds, login node is fine; no torch)
"""
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from tcga_same_participant_check import DOWN, row_names     # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--downstream", default=DOWN)
    p.add_argument("--overlap", default="outputs/pretrain_overlap/overlap_by_id.csv")
    p.add_argument("--studies", nargs="+", default=["E-GEOD-56105", "E-GEOD-40279", "GSE41037", "E-GEOD-36194"])
    p.add_argument("--max_rows", type=int, default=300)
    a = p.parse_args()
    ov = pd.read_csv(a.overlap, dtype={"sample_id": str}).set_index("sample_id")
    with h5py.File(a.downstream, "r") as f:
        n = f["X"].shape[0]
        ids, _ = row_names(f["obs"], n)
        study = ov.dataset.reindex(ids).to_numpy()
        rows = []
        for s in a.studies:
            r = np.where(study == s)[0][:a.max_rows]
            if not len(r):
                print(f"{s}: not found"); continue
            v = np.concatenate([f["X"][int(i)] for i in r]).astype(np.float64)
            v = v[np.isfinite(v)]
            q = np.quantile(v, [0.01, 0.25, 0.5, 0.75, 0.99])
            rows.append({"study": s, "profiles": len(r), "exactly_0": (v == 0).mean(), "below_0.01": (v < 0.01).mean(),
                         "above_0.99": (v > 0.99).mean(), "0.2_to_0.8": ((v > 0.2) & (v < 0.8)).mean(), "mean": v.mean(),
                         "q01": q[0], "q25": q[1], "median": q[2], "q75": q[3], "q99": q[4],
                         "min": v.min(), "max": v.max()})
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
