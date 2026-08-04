"""
Genomic-locality decay analysis — CONTEXTUALIZED CpG embeddings vs genomic-rank distance.

Pure post-processing (no GPU, no checkpoint): reads contextual_cpg_emb.npy
(from extract_contextual_cpg.py) and the raw CpG embedding table
(cpg_embedding_matrix.npy, from extract_pretrain_cls.py), and reproduces the
"Genomic RoPE injects genomic locality" figure: mean cosine similarity between
CpG pairs as a function of genomic-rank distance, for (a) contextualized
(post-transformer) embeddings, (b) a random-pair baseline, (c) the raw
(non-contextualized) token embedding table.

Usage:
  python scripts/repr_analysis_v7b/analyze_cpg_genomic_locality.py \
    --context_dir figures/v7b_cpg_context \
    --pretrain_dir figures/v7b_pretrain_cls \
    --outdir figures/v7b_cpg_context
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--context_dir", default="figures/v7b_cpg_context")
    p.add_argument("--pretrain_dir", default="figures/v7b_pretrain_cls")
    p.add_argument("--outdir", default="figures/v7b_cpg_context")
    p.add_argument("--n_pairs_per_bin", type=int, default=4000)
    p.add_argument("--n_random_pairs", type=int, default=200_000)
    p.add_argument("--tolerance_frac", type=float, default=0.15,
                    help="relative tolerance around each target distance when sampling pairs")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def cosine_for_pairs(Xn, i, j):
    return np.sum(Xn[i] * Xn[j], axis=1)


def mean_sim_at_distance(rank, Xn, target_dist, n_pairs, tol_frac, rng):
    """Sample CpG pairs whose genomic-rank GAP (not position-index gap) is close
    to target_dist. Only ~21k/49k CpGs are present here, so consecutive sorted
    positions are ~2.3 ranks apart on average -- we search in true rank-space
    (via searchsorted on the sorted rank array), not position-index space, so
    this stays correct regardless of how sparse the subset is."""
    n = len(rank)
    order = np.argsort(rank)
    ranked = rank[order]
    tol = max(1, int(round(target_dist * tol_frac)))

    pos = rng.integers(0, n, size=n_pairs * 4)
    sign = rng.choice([-1, 1], size=n_pairs * 4)
    target_rank = ranked[pos] + sign * target_dist
    pos2 = np.searchsorted(ranked, target_rank)
    pos2 = np.clip(pos2, 0, n - 1)
    actual_gap = np.abs(ranked[pos2] - ranked[pos])
    valid = (np.abs(actual_gap - target_dist) <= tol) & (pos2 != pos)
    pos, pos2 = pos[valid], pos2[valid]
    if len(pos) > n_pairs:
        pos, pos2 = pos[:n_pairs], pos2[:n_pairs]
    if len(pos) == 0:
        return np.nan, 0
    i_cols, j_cols = order[pos], order[pos2]
    sims = cosine_for_pairs(Xn, i_cols, j_cols)
    return float(np.mean(sims)), len(pos)


def main():
    a = parse_args()
    ctx_dir = Path(a.context_dir)
    pre_dir = Path(a.pretrain_dir)
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    # ── contextualized embeddings + genomic rank (already sorted ascending) ──
    ctx = np.load(ctx_dir / "contextual_cpg_emb.npy").astype(np.float64)
    order_df = pd.read_csv(ctx_dir / "cpg_order.csv")
    rank_ctx = order_df["genomic_rank"].values.astype(np.int64)
    assert len(rank_ctx) == len(ctx), f"{len(rank_ctx)} vs {len(ctx)}"

    meta_path = ctx_dir / "contextual_cpg_meta.json"
    n_samples = json.loads(meta_path.read_text())["n_samples"] if meta_path.exists() else None

    ctx_n = ctx / (np.linalg.norm(ctx, axis=1, keepdims=True) + 1e-9)

    # ── raw (non-contextualized) embedding table, aligned to the same CpGs ───
    raw_table = np.load(pre_dir / "cpg_embedding_matrix.npy").astype(np.float64)
    align = pd.read_csv(pre_dir / "cpg_alignment.csv")  # vocab_id, cpg_name, genomic_rank
    name_to_vocab = dict(zip(align["cpg_name"], align["vocab_id"]))
    vocab_ids = order_df["cpg_name"].map(name_to_vocab).values
    missing = pd.isna(vocab_ids).sum()
    if missing:
        print(f"WARNING: {missing} CpGs missing from cpg_alignment.csv — dropping")
    keep = ~pd.isna(vocab_ids)
    raw = raw_table[vocab_ids[keep].astype(int)]
    raw_n = raw / (np.linalg.norm(raw, axis=1, keepdims=True) + 1e-9)
    rank_raw = rank_ctx[keep]

    # ── distance bins (log-spaced, matches original figure) ──────────────────
    bins = [1, 2, 3, 5, 8, 11, 20, 35, 50, 75, 100, 200, 500, 1000, 2000, 5000]

    rows = []
    for d in bins:
        sim_ctx, n1 = mean_sim_at_distance(rank_ctx, ctx_n, d, a.n_pairs_per_bin, a.tolerance_frac, rng)
        sim_raw, n2 = mean_sim_at_distance(rank_raw, raw_n, d, a.n_pairs_per_bin, a.tolerance_frac, rng)
        rows.append({"distance": d, "contextual_cos": sim_ctx, "raw_cos": sim_raw,
                     "n_pairs_contextual": n1, "n_pairs_raw": n2})
    df = pd.DataFrame(rows)

    # ── random-pair baseline (contextualized) ─────────────────────────────────
    n = len(ctx_n)
    i_rand = rng.integers(0, n, size=a.n_random_pairs)
    j_rand = rng.integers(0, n, size=a.n_random_pairs)
    valid = i_rand != j_rand
    random_baseline = float(np.mean(cosine_for_pairs(ctx_n, i_rand[valid], j_rand[valid])))

    df.to_csv(outdir / "cpg_genomic_locality_decay.csv", index=False)

    summary = {
        "n_samples_averaged": n_samples,
        "adjacent_cos_dist1": float(df.loc[df["distance"] == 1, "contextual_cos"].iloc[0]),
        "random_pair_baseline": random_baseline,
        "raw_table_mean_cos": float(np.nanmean(df["raw_cos"])),
    }
    with open(outdir / "cpg_genomic_locality_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    # ── plot (matches original figure style) ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(df["distance"], df["contextual_cos"], "o-", color="#1a4ab0", ms=5, lw=2,
            label="contextualized CpG (post-transformer)")
    ax.axhline(random_baseline, color="#c0392b", linestyle="--", lw=1.5,
               label=f"random pair baseline ({random_baseline:.3f})")
    ax.plot(df["distance"], df["raw_cos"], ":", color="#888888", lw=1.5,
            label="raw token table (no locality)")
    ax.set_xscale("log")
    ax.set_xlabel("genomic-rank distance between CpGs")
    ax.set_ylabel("mean cosine similarity of embeddings")
    n_label = f", n={n_samples} samples" if n_samples else ""
    ax.set_title(f"Genomic RoPE injects genomic locality into CpG representations{n_label}\n"
                 f"(adjacent CpGs similar; decays toward random by ~100 steps)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "cpg_genomic_locality_decay.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {outdir}/cpg_genomic_locality_decay.png")
    print(f"Saved -> {outdir}/cpg_genomic_locality_decay.csv")


if __name__ == "__main__":
    main()
