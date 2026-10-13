"""
pretrain_benchmark_pair_similarity.py
=====================================
Follow-up to tcga_same_participant_check.py. That run showed that no downstream TCGA profile is
value-identical to a corpus profile, but its calibration pairs mixed tissues, so it could not
tell "the same sample, processed by a different pipeline" from "another person, same tissue".

This script measures both references directly and places the TCGA pairs between them.

  K1  same accession            downstream GSM profile vs the corpus profile with the SAME accession
                                -> how similar the same sample is across the two corpora
  K2  same study, same tissue   downstream GSM profile vs the corpus profile of ANOTHER accession
                                from the same study and tissue label
                                -> how similar two different samples of one study and tissue are
  T1  TCGA, same participant    downstream profile vs corpus normal-tissue (11) profile
  T2  TCGA, same participant    downstream profile vs corpus tumour (01) profile
  T3  TCGA, other participant   downstream profile vs corpus normal-tissue profiles of OTHER
                                participants of the same project

Identification test for T1: among the corpus normal-tissue profiles of one project, is the
profile of the downstream profile's own participant the most similar one? Candidates are
defined by identifiers (project, sample type); nothing is searched outside them.

Reads single rows through h5py; no torch; the 33 GB matrix is never loaded.

  srun --mem=16G --time=2:00:00 python scripts/utils/pretrain_benchmark_pair_similarity.py

Writes outputs/data_structure_audit/pair_similarity.csv, pair_similarity_summary.json,
tcga_identification.csv. Overwrites nothing from the first run.
"""
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from tcga_same_participant_check import DOWN, PRE, cpg_names, row_names    # noqa: E402

N_REF = 1000


def stats(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    n = int(ok.sum())
    if n < 100:
        return n, np.nan, np.nan
    x, y = a[ok].astype(np.float64), b[ok].astype(np.float64)
    return n, float(np.corrcoef(x, y)[0, 1]), float(np.abs(x - y).mean())


def quantiles(s):
    return {k: float(s.quantile(q)) for k, q in [("min", 0), ("p05", .05), ("median", .5), ("p95", .95), ("max", 1)]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrain", default=PRE)
    p.add_argument("--downstream", default=DOWN)
    p.add_argument("--overlap", default="outputs/pretrain_overlap/overlap_by_id.csv")
    p.add_argument("--outdir", default="outputs/data_structure_audit")
    a = p.parse_args()
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    for f in ("pair_similarity.csv", "pair_similarity_summary.json", "tcga_identification.csv"):
        if (out / f).exists():
            raise SystemExit(f"{out / f} exists -- nothing is overwritten")
    ann = pd.read_csv(a.overlap, dtype={"sample_id": str}).set_index("sample_id")

    with h5py.File(a.pretrain, "r") as fp, h5py.File(a.downstream, "r") as fd:
        n_pre, c_pre = fp["X"].shape
        n_dn, c_dn = fd["X"].shape
        pre_ids, _ = row_names(fp["obs"], n_pre)
        dn_ids, _ = row_names(fd["obs"], n_dn)
        pre_cpg, dn_cpg = cpg_names(fp["var"], c_pre), cpg_names(fd["var"], c_dn)
        pos = {c: i for i, c in enumerate(pre_cpg)}
        keep = np.array([c in pos for c in dn_cpg])
        col_dn = np.where(keep)[0]                                   # increasing
        col_pre = np.array([pos[c] for c in dn_cpg[keep]])           # same CpG order as col_dn
        print(f"pretraining {n_pre:,} x {c_pre:,} | downstream {n_dn:,} x {c_dn:,} | shared CpGs {keep.sum():,}", flush=True)
        assert set(dn_ids) == set(ann.index), "overlap table and downstream file list different profiles"

        pre_row = pd.Series(np.arange(n_pre), index=pre_ids)
        dn_row = pd.Series(np.arange(n_dn), index=dn_ids)
        pairs = []                                                   # (kind, downstream id, corpus id)

        # K1, K2: GEO references
        rng = np.random.default_rng(0)
        both = ann[ann.index.str.startswith("GSM") & ann.index.isin(pre_row.index)]
        ref = both.sample(n=min(N_REF, len(both)), random_state=0)
        grp = both.groupby(["dataset", "tissue_type"]).groups
        for sid, r in ref.iterrows():
            pairs.append(("K1 same accession", sid, sid))
            others = [o for o in grp[(r.dataset, r.tissue_type)] if o != sid]
            if others:
                pairs.append(("K2 other accession, same study and tissue", sid, others[int(rng.integers(len(others)))]))

        # TCGA
        tc = pd.DataFrame({"sample_id": pre_ids})
        tc = tc[tc.sample_id.str.startswith("TCGA-")].assign(participant=lambda d: d.sample_id.str[:12],
                                                             sample_type=lambda d: d.sample_id.str[13:15])
        dt = ann[ann.index.str.startswith("TCGA-")]
        project = dt.dataset.to_dict()                               # participant -> project
        normals = tc[(tc.sample_type == "11") & tc.participant.isin(project)].assign(
            project=lambda d: d.participant.map(project))
        for sid in dt.index:
            own = tc[tc.participant == sid]
            for r in own.itertuples():
                if r.sample_type == "11":
                    pairs.append(("T1 TCGA same participant, corpus normal tissue", sid, r.sample_id))
                elif r.sample_type == "01":
                    pairs.append(("T2 TCGA same participant, corpus tumour", sid, r.sample_id))
            for r in normals[(normals.project == project[sid]) & (normals.participant != sid)].itertuples():
                pairs.append(("T3 TCGA other participant, same project, corpus normal tissue", sid, r.sample_id))
        pairs = pd.DataFrame(pairs, columns=["kind", "downstream_id", "corpus_id"]).drop_duplicates()
        print(pairs.kind.value_counts().to_string(), flush=True)

        def load(X, rows, cols, order=None):
            m = {}
            for i, r in enumerate(sorted(set(rows))):
                v = X[int(r)]
                m[r] = v[cols].astype(np.float32)
                if i % 500 == 0:
                    print(f"   read {i:,} rows", flush=True)
            return m
        print("reading downstream rows", flush=True)
        A = load(fd["X"], dn_row[pairs.downstream_id.unique()].to_numpy(), col_dn)
        print("reading corpus rows", flush=True)
        B = load(fp["X"], pre_row[pairs.corpus_id.unique()].to_numpy(), col_pre)

    s = [stats(A[dn_row[d]], B[pre_row[c]]) for d, c in zip(pairs.downstream_id, pairs.corpus_id)]
    pairs[["n_cpgs", "pearson_r", "mean_abs_diff"]] = s
    perm = np.random.default_rng(42).permutation(n_pre)
    split = np.empty(n_pre, dtype=object)
    split[perm[:int(0.8 * n_pre)]] = "train"
    split[perm[int(0.8 * n_pre):int(0.9 * n_pre)]] = "valid"
    split[perm[int(0.9 * n_pre):]] = "heldout"
    pairs["corpus_split"] = split[pre_row[pairs.corpus_id].to_numpy()]
    pairs["project_or_study"] = ann.dataset.reindex(pairs.downstream_id).to_numpy()
    pairs.to_csv(out / "pair_similarity.csv", index=False)

    summ = {"n_shared_cpgs": int(keep.sum()), "kinds": {}}
    for k, g in pairs.groupby("kind"):
        summ["kinds"][k] = {"pairs": int(len(g)), "pearson_r": quantiles(g.pearson_r), "mean_abs_diff": quantiles(g.mean_abs_diff)}
    k1 = pairs[pairs.kind.str.startswith("K1")]
    summ["K1_value_identical_pairs (mean_abs_diff <= 0.005 and r >= 0.9999)"] = int(
        ((k1.mean_abs_diff <= 0.005) & (k1.pearson_r >= 0.9999)).sum())

    # identification: own participant against the other normals of the project
    t = pairs[pairs.kind.str.startswith(("T1", "T3"))].copy()
    t["own"] = t.kind.str.startswith("T1")
    rows = []
    for sid, g in t.groupby("downstream_id"):
        if not g.own.any() or (~g.own).sum() == 0:
            continue
        own = g[g.own].sort_values("pearson_r", ascending=False).iloc[0]
        oth = g[~g.own]
        rows.append({"downstream_id": sid, "project": own.project_or_study, "corpus_id": own.corpus_id,
                     "corpus_split": own.corpus_split, "own_r": own.pearson_r, "own_mean_abs_diff": own.mean_abs_diff,
                     "best_other_r": oth.pearson_r.max(), "lowest_other_mean_abs_diff": oth.mean_abs_diff.min(),
                     "n_other_candidates": int(len(oth)),
                     "own_is_most_similar_by_r": bool(own.pearson_r > oth.pearson_r.max()),
                     "own_is_most_similar_by_mad": bool(own.mean_abs_diff < oth.mean_abs_diff.min())})
    ident = pd.DataFrame(rows)
    ident.to_csv(out / "tcga_identification.csv", index=False)
    if len(ident):
        both_ = ident.own_is_most_similar_by_r & ident.own_is_most_similar_by_mad
        summ["tcga_identification"] = {
            "downstream_profiles_tested": int(len(ident)),
            "median_candidates_per_profile": float(ident.n_other_candidates.median() + 1),
            "own_participant_most_similar_by_r": int(ident.own_is_most_similar_by_r.sum()),
            "own_participant_most_similar_by_mean_abs_diff": int(ident.own_is_most_similar_by_mad.sum()),
            "own_participant_most_similar_by_both": int(both_.sum()),
            "of_those_corpus_profile_in_pretraining_train": int((both_ & (ident.corpus_split == "train")).sum()),
            "by_project": {k: {"tested": int(len(g)), "own_most_similar_by_both": int(
                (g.own_is_most_similar_by_r & g.own_is_most_similar_by_mad).sum())} for k, g in ident.groupby("project")}}
    (out / "pair_similarity_summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
