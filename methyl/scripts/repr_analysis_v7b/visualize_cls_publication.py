"""
Publication-grade CLS visualization for the V7b representation analysis.

Pure post-processing on saved arrays from extract_pretrain_cls.py (no GPU,
no checkpoint). Produces clean per-label standalone figures + the key
confound diagnostic + a CpG-embedding-matrix figure.

Figures written to <dir>/pub/:
  umap_age.png, umap_tissue.png, umap_sex.png, umap_dataset.png   (+ pca_*)
  umap_age_vs_dataset.png     age gradient beside dataset blobs (confound check)
  cpg_embedding_spectrum.png  singular-value spectrum + pairwise-cosine histogram
  pretrained_vs_finetuned.png if --finetuned_dir given (side-by-side by age+tissue)

Usage:
  python scripts/repr_analysis_v7b/visualize_cls_publication.py \
     --dir figures/v7b_pretrain_cls --top_tissues 15 --dpi 200
  # optional later, when a fine-tuned checkpoint is extracted:
  #   --finetuned_dir figures/v7b_finetuned_cls
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="figures/v7b_pretrain_cls")
    p.add_argument("--finetuned_dir", default=None, help="optional 2nd model for side-by-side")
    p.add_argument("--top_tissues", type=int, default=15)
    p.add_argument("--top_datasets", type=int, default=15)
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--point_size", type=float, default=5.0)
    p.add_argument("--age_col", default="age")
    p.add_argument("--tissue_col", default="tissue_type")
    p.add_argument("--dataset_col", default="dataset")
    p.add_argument("--sex_col", default="sex")
    return p.parse_args()


def load(d):
    d = Path(d)
    meta = pd.read_csv(d / "metadata.csv")
    coords = {}
    for name in ("pca", "umap"):
        f = d / f"cls_{name}_coords.npy"
        if f.exists():
            coords[name] = np.load(f)
    return meta, coords


def scatter_cont(ax, xy, values, title, cmap="viridis", s=5):
    v = pd.to_numeric(values, errors="coerce").values
    ok = ~np.isnan(v)
    sc = ax.scatter(xy[ok, 0], xy[ok, 1], c=v[ok], s=s, cmap=cmap, alpha=0.7, linewidths=0)
    cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cb.ax.tick_params(labelsize=8)
    ax.set_title(title, fontsize=12, weight="bold")
    ax.set_xticks([]); ax.set_yticks([])


def scatter_cat(ax, xy, labels, title, top_n, s=5, legend=True):
    labels = pd.Series(labels).astype(str).values
    top = pd.Series(labels).value_counts().head(top_n).index.tolist()
    palette = cm.get_cmap("tab20", max(len(top), 1))
    other = ~np.isin(labels, top)
    if other.any():
        ax.scatter(xy[other, 0], xy[other, 1], s=s * 0.7, c="#d9d9d9",
                   alpha=0.5, linewidths=0, label="other")
    for k, cat in enumerate(top):
        m = labels == cat
        ax.scatter(xy[m, 0], xy[m, 1], s=s, color=palette(k), alpha=0.8,
                   linewidths=0, label=str(cat)[:22])
    ax.set_title(title, fontsize=12, weight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    if legend:
        ax.legend(markerscale=2.2, fontsize=7, loc="center left",
                  bbox_to_anchor=(1.01, 0.5), framealpha=0.9, borderpad=0.4)


def main():
    a = parse_args()
    d = Path(a.dir)
    outdir = d / "pub"
    outdir.mkdir(parents=True, exist_ok=True)
    meta, coords = load(d)
    if not coords:
        raise SystemExit(f"No cls_*_coords.npy in {d} — run visualize_cls.py first.")
    print(f"Loaded {len(meta)} rows; projections: {list(coords)}")

    # ── per-label standalone figures ──────────────────────────────────────────
    for proj, xy in coords.items():
        if a.age_col in meta:
            fig, ax = plt.subplots(figsize=(6.5, 5.5))
            scatter_cont(ax, xy, meta[a.age_col], f"{proj.upper()} — age (years)", s=a.point_size)
            fig.tight_layout(); fig.savefig(outdir / f"{proj}_age.png", dpi=a.dpi, bbox_inches="tight"); plt.close(fig)
        for col, top, tag in [(a.tissue_col, a.top_tissues, "tissue"),
                              (a.dataset_col, a.top_datasets, "dataset"),
                              (a.sex_col, 4, "sex")]:
            if col in meta:
                fig, ax = plt.subplots(figsize=(8.5, 5.5))
                scatter_cat(ax, xy, meta[col], f"{proj.upper()} — {tag}",
                            top, s=a.point_size, legend=True)
                fig.tight_layout(); fig.savefig(outdir / f"{proj}_{tag}.png", dpi=a.dpi, bbox_inches="tight"); plt.close(fig)

    # ── confound diagnostic: age gradient beside dataset blobs (UMAP) ─────────
    if "umap" in coords and a.age_col in meta and a.dataset_col in meta:
        xy = coords["umap"]
        fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))
        scatter_cont(axes[0], xy, meta[a.age_col], "UMAP — age (years)", s=a.point_size)
        scatter_cat(axes[1], xy, meta[a.dataset_col], "UMAP — dataset (top)",
                    a.top_datasets, s=a.point_size, legend=True)
        fig.suptitle("Confound check: does the age gradient cross dataset boundaries?",
                     fontsize=13, weight="bold", y=1.02)
        fig.tight_layout(); fig.savefig(outdir / "umap_age_vs_dataset.png", dpi=a.dpi, bbox_inches="tight"); plt.close(fig)

    # ── CpG embedding matrix figure: SV spectrum + pairwise-cosine histogram ──
    if (d / "cpg_embedding_matrix.npy").exists() and (d / "cpg_alignment.csv").exists():
        W = np.load(d / "cpg_embedding_matrix.npy").astype(np.float64)
        align = pd.read_csv(d / "cpg_alignment.csv")
        col = "encoder_vocab_id" if "encoder_vocab_id" in align else "vocab_id"
        E = W[align[col].values]
        Ec = E - E.mean(0)
        s = np.linalg.svd(Ec, compute_uv=False)
        En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
        rng = np.random.default_rng(0)
        i, j = rng.integers(0, len(En), 40000), rng.integers(0, len(En), 40000)
        cos = np.sum(En[i] * En[j], 1)
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
        ax[0].plot(np.arange(1, len(s) + 1), s**2 / (s**2).sum(), "o-", ms=3)
        ax[0].set_yscale("log"); ax[0].set_xlabel("component"); ax[0].set_ylabel("variance frac (log)")
        ax[0].set_title(f"CpG embedding SV spectrum ({E.shape[0]} CpGs)", fontsize=11, weight="bold")
        ax[1].hist(cos, bins=80, color="#4a70cc", alpha=0.85)
        ax[1].axvline(cos.mean(), color="k", ls="--", lw=1, label=f"mean={cos.mean():.3f}")
        ax[1].set_xlabel("pairwise cosine"); ax[1].set_ylabel("count")
        ax[1].set_title("CpG embedding pairwise cosine (near-0 = orthogonal/healthy)", fontsize=11, weight="bold")
        ax[1].legend(fontsize=9)
        fig.tight_layout(); fig.savefig(outdir / "cpg_embedding_spectrum.png", dpi=a.dpi, bbox_inches="tight"); plt.close(fig)

    # ── pretrained vs finetuned side-by-side (if given) ───────────────────────
    if a.finetuned_dir:
        m2, c2 = load(a.finetuned_dir)
        if "umap" in coords and "umap" in c2:
            fig, axes = plt.subplots(2, 2, figsize=(13, 11))
            scatter_cont(axes[0, 0], coords["umap"], meta[a.age_col], "Pretrained — age", s=a.point_size)
            scatter_cont(axes[0, 1], c2["umap"], m2[a.age_col], "Fine-tuned — age", s=a.point_size)
            scatter_cat(axes[1, 0], coords["umap"], meta[a.tissue_col], "Pretrained — tissue", a.top_tissues, s=a.point_size, legend=False)
            scatter_cat(axes[1, 1], c2["umap"], m2[a.tissue_col], "Fine-tuned — tissue", a.top_tissues, s=a.point_size, legend=False)
            fig.suptitle("CLS space: before vs after fine-tuning", fontsize=14, weight="bold", y=1.01)
            fig.tight_layout(); fig.savefig(outdir / "pretrained_vs_finetuned.png", dpi=a.dpi, bbox_inches="tight"); plt.close(fig)

        # quantitative before/after: geometry + age-probe from each dir's analysis_summary.json
        import json
        s1_path, s2_path = Path(a.dir) / "analysis_summary.json", Path(a.finetuned_dir) / "analysis_summary.json"
        if s1_path.exists() and s2_path.exists():
            s1, s2 = json.loads(s1_path.read_text()), json.loads(s2_path.read_text())
            g1, g2 = s1.get("geometry_cls", {}), s2.get("geometry_cls", {})
            a1, a2 = s1.get("age_probe_cls", {}), s2.get("age_probe_cls", {})
            lines = ["CLS representation: pretrained vs fine-tuned", "=" * 55, ""]
            lines += ["Geometry:",
                      f"  {'metric':<20}{'pretrained':>14}{'fine-tuned':>14}"]
            for k in ("effective_rank", "top1_sv_frac", "anisotropy_mean_cos", "dead_dims_lt1pct"):
                lines.append(f"  {k:<20}{g1.get(k, 'NA'):>14}{g2.get(k, 'NA'):>14}")
            lines += ["", "Age probe (linear ridge / replica head — proxy, not the trained head):",
                      f"  {'metric':<20}{'pretrained':>14}{'fine-tuned':>14}"]
            for k in ("linear_ridge_r2", "replica_head_r2", "replica_head_medae"):
                lines.append(f"  {k:<20}{a1.get(k, 'NA'):>14}{a2.get(k, 'NA'):>14}")
            ah2 = s2.get("age_head_actual")
            if ah2:
                lines += ["", "Fine-tuned model's ACTUAL trained age_head (ground truth, not a proxy):",
                          f"  test MedAE={ah2['medae']}yr  MAE={ah2['mae']}yr  R2={ah2['r2']}  n={ah2['n_test']}",
                          "  (should match this fold's WandB test/medae — correctness gate)"]
            report = "\n".join(lines)
            (outdir / "comparison_pretrain_vs_finetune.txt").write_text(report)
            print("\n" + report + "\n")

    print(f"Saved publication figures → {outdir}/")
    for f in sorted(outdir.glob("*.png")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
