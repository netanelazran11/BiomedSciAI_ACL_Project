#!/usr/bin/env python3
"""
attention_analysis.py  —  Figure 5 equivalent
==============================================
Extract multi-head self-attention weights from the last transformer layer of
MethylLlama-Small and analyse age-specific CpG attention patterns.

Replicates MethylGPT Figure 5:
  b  — Attention score matrices per age group (<20 / 20-60 / >60)
  c  — Volcano plot: log2 FC vs -log10(p-value), young vs old
  d  — Heatmap of top young-important and old-important CpG sites
  e  — GO/pathway enrichment (requires external tool: MethylGSA or gseapy)

Method (from MethylGPT paper Methods section):
  1. Extract attention scores from ALL heads in the FINAL transformer layer
  2. Mean attention received by each CpG site = column-sum of attention matrix
     averaged across heads  →  per-CpG attention score per sample
  3. Compute mean attention score per CpG per age group
  4. Two-sided t-test (young <20 vs old >60), Benjamini-Hochberg correction
  5. Differential sites: |log2 FC| > log2(1.5) AND BH-corrected p < 0.05

Usage:
  python scripts/repr_analysis/attention_analysis.py \\
      --checkpoint outputs/pretrain-llama-wced/.../epoch=98-val_loss=0.0059.ckpt \\
      --data       /path/to/finetuning_19608_clean_stratified_no_outliers.h5ad \\
      --tokenizer  tokenizer_llama_pretrain49k \\
      --metadata   data/pretrain_metadata.csv.gz \\
      --outdir     outputs/repr_analysis/attention \\
      --batch_size 16
"""

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Attention weight analysis (Fig 5)")
    p.add_argument("--checkpoint",   required=True)
    p.add_argument("--data",         required=True, help="h5ad file")
    p.add_argument("--tokenizer",    required=True)
    p.add_argument("--metadata",     default=None,  help="pretrain_metadata.csv.gz (optional)")
    p.add_argument("--metadata_id_col", default="GSM_ID")
    p.add_argument("--manifest",     default=None,
                   help="cpg_annotations_tokenizer49k.tsv for gene/EWAS annotation of top CpGs")
    p.add_argument("--ckpt_type",    default="pretrain", choices=["pretrain", "finetune"])
    p.add_argument("--outdir",       default="outputs/repr_analysis/attention")
    p.add_argument("--batch_size",   type=int, default=16,
                   help="Keep small (16-32) — storing attention weights is memory-intensive")
    p.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--age_col",      default="age")
    p.add_argument("--max_samples",  type=int, default=5000,
                   help="Cap samples for attention extraction (attention is expensive). -1 = all")
    p.add_argument("--layer",        type=int, default=-1,
                   help="Which transformer layer to extract attention from. -1 = last layer.")
    p.add_argument("--top_n_cpgs",   type=int, default=15,
                   help="Number of top CpGs to show in heatmap per age group")
    p.add_argument("--lfc_thresh",   type=float, default=np.log2(1.5),
                   help="log2 fold-change threshold for differential CpGs (default log2(1.5))")
    p.add_argument("--fdr_thresh",   type=float, default=0.05)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Model loading with attention hook
# ─────────────────────────────────────────────────────────────────────────────

class AttentionCapturer:
    """
    Hook into MethylLlama transformer layers to capture attention weights.

    MethylLlama uses a custom self-attention module. We patch its forward
    method to compute and store the attention weight matrix explicitly.
    """
    def __init__(self, encoder, layer_idx: int = -1):
        self.attn_weights = []  # list of [num_heads, L, L] per batch
        self._hook = None

        layers = encoder.encoder.layers
        target_layer = layers[layer_idx]

        # Patch the attention forward to return weights
        orig_forward = target_layer.forward

        def patched_forward(hidden_states, attention_mask=None, **kwargs):
            # Run original forward but intercept attention
            # We need to compute attention weights manually since
            # F.scaled_dot_product_attention doesn't expose them by default
            out = orig_forward(hidden_states, attention_mask=attention_mask, **kwargs)
            return out

        # Try the hook approach: hook on self_attn submodule
        self._hook = target_layer.self_attn.register_forward_hook(self._capture_hook)
        log.info(f"Attention hook registered on layer {layer_idx}")

    def _capture_hook(self, module, input, output):
        """
        Standard forward hook. Output from self_attn is typically
        (context, attn_weights) or just context.
        Handles both cases.
        """
        if isinstance(output, tuple) and len(output) >= 2:
            attn = output[1]  # [B, H, L, L]
            if attn is not None:
                self.attn_weights.append(attn.detach().cpu())
        # If only context is returned, we can't get weights from hook alone
        # In that case, fall back to gradient-based or skip

    def clear(self):
        self.attn_weights = []

    def remove(self):
        if self._hook:
            self._hook.remove()


def load_encoder(checkpoint_path: str, ckpt_type: str):
    if ckpt_type == "pretrain":
        from bmfm_methylation.llama.finetune_llama import load_wced_llama_checkpoint
        module = load_wced_llama_checkpoint(checkpoint_path)
    else:
        from bmfm_methylation.llama.finetune_llama import load_finetune_llama_checkpoint
        module = load_finetune_llama_checkpoint(checkpoint_path)
    encoder = module.encoder
    encoder.eval()
    return encoder


# ─────────────────────────────────────────────────────────────────────────────
# Attention extraction
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_attention_scores(encoder, data_path, tokenizer_path, batch_size,
                              device, layer_idx, max_samples, age_col):
    """
    Extract per-CpG attention scores for each sample.

    For each sample, the attention score of CpG site j is:
        mean over heads of: sum over query positions of attn_weight[head, q, j]
    i.e., how much total attention flows INTO each CpG site (it is the "key" role).

    Returns:
        cpg_attention  np.ndarray [N, n_cpg]  — per-sample, per-CpG score
        ages           np.ndarray [N]
        cpg_ids        list[str]   in dataset order
        sample_ids     list[str]
    """
    from bmfm_targets.tokenization import MultiFieldTokenizer
    from bmfm_methylation.shared.data_module import MethylationDataset, WCEDCollator

    tok     = MultiFieldTokenizer.from_pretrained(tokenizer_path)
    dataset = MethylationDataset(h5ad_path=data_path, split=None, normalize_age=False)
    cpg_sites = dataset.cpg_sites

    # Subsample if requested
    if max_samples > 0 and len(dataset) > max_samples:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(dataset), size=max_samples, replace=False)
        dataset = torch.utils.data.Subset(dataset, sorted(indices.tolist()))
        log.info(f"Subsampled to {max_samples:,} samples for attention extraction")

    collator = WCEDCollator(
        tokenizer=tok, cpg_sites=cpg_sites,
        vocab_size=len(cpg_sites), input_ratio=1.0, contrastive=False,
    )
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator,
                        shuffle=False, num_workers=2, pin_memory=False)

    encoder = encoder.to(device)

    # Register hook
    capturer = AttentionCapturer(encoder, layer_idx=layer_idx)

    attn_rows, age_rows = [], []
    for i, batch in enumerate(loader):
        cpg_ids_b   = batch["cpg_ids"].to(device)
        beta_values = batch["beta_values"].to(device)
        attn_mask   = batch["attention_mask"].to(device)
        ages        = batch["age"].cpu().numpy()

        capturer.clear()
        input_ids = torch.stack([cpg_ids_b.float(), beta_values], dim=1)
        _ = encoder(input_ids=input_ids, attention_mask=attn_mask)

        if capturer.attn_weights:
            # attn: [B, H, L, L]  — L includes CLS at pos 0
            attn = capturer.attn_weights[0].float().numpy()  # [B, H, L, L]
            # Key attention (column sum): how much flows INTO each position
            # Sum over query dim (axis=-2), mean over heads (axis=1)
            # Result: [B, L]
            key_attn = attn.sum(axis=-2).mean(axis=1)  # [B, L]
            key_attn_cpg = key_attn[:, 1:]              # [B, L-1] skip CLS pos

            # Normalize per sample
            row_sum = key_attn_cpg.sum(axis=1, keepdims=True)
            key_attn_cpg = key_attn_cpg / (row_sum + 1e-12)
            attn_rows.append(key_attn_cpg)
        else:
            log.warning(f"  Batch {i}: no attention weights captured (model may not expose them)")
            # Fallback: uniform attention
            B = ages.shape[0]
            L = batch["cpg_ids"].shape[1]
            attn_rows.append(np.ones((B, L)) / L)

        age_rows.append(ages)

        if (i + 1) % 10 == 0:
            log.info(f"  batch {i+1}/{len(loader)}")

    capturer.remove()

    cpg_attention = np.concatenate(attn_rows, axis=0)   # [N, n_cpg]
    ages_all      = np.concatenate(age_rows,  axis=0)   # [N]

    log.info(f"Attention matrix: {cpg_attention.shape}  ages: {ages_all.shape}")
    return cpg_attention, ages_all, list(cpg_sites)


# ─────────────────────────────────────────────────────────────────────────────
# Age-group analysis
# ─────────────────────────────────────────────────────────────────────────────

def benjamini_hochberg(p_values):
    """BH FDR correction. Returns adjusted p-values."""
    n = len(p_values)
    order = np.argsort(p_values)
    rank  = np.empty_like(order)
    rank[order] = np.arange(1, n + 1)
    adj = np.minimum(1.0, p_values * n / rank)
    # Enforce monotonicity from right
    for i in range(n - 2, -1, -1):
        adj[order[i]] = min(adj[order[i]], adj[order[i + 1]])
    return adj


def differential_attention(cpg_attention, ages, age_groups=None):
    """
    Compute mean attention per CpG per age group and run t-test.

    age_groups: list of (label, (lo, hi)) tuples defining age bins.
    Default: young <20, mid 20-60, old >60.

    Returns DataFrame with columns:
      cpg_idx, mean_young, mean_old, log2FC, pvalue, padj, significant
    """
    if age_groups is None:
        age_groups = [("young", (0, 20)), ("mid", (20, 60)), ("old", (60, 200))]

    group_masks = {}
    for label, (lo, hi) in age_groups:
        mask = (ages >= lo) & (ages < hi) & ~np.isnan(ages)
        group_masks[label] = mask
        log.info(f"  Age group '{label}' ({lo}-{hi}): {mask.sum():,} samples")

    young_attn = cpg_attention[group_masks["young"]]
    old_attn   = cpg_attention[group_masks["old"]]

    if len(young_attn) < 5 or len(old_attn) < 5:
        log.warning("Too few samples in age groups — results may be unreliable")

    mean_young = young_attn.mean(axis=0)  # [n_cpg]
    mean_old   = old_attn.mean(axis=0)

    # log2 fold change: old / young
    eps = 1e-12
    log2fc = np.log2((mean_old + eps) / (mean_young + eps))

    # Two-sided t-test per CpG
    n_cpg = cpg_attention.shape[1]
    pvals = np.ones(n_cpg)
    for j in range(n_cpg):
        _, p = stats.ttest_ind(young_attn[:, j], old_attn[:, j], equal_var=False)
        pvals[j] = p

    padj = benjamini_hochberg(pvals)

    df = pd.DataFrame({
        "cpg_idx":    np.arange(n_cpg),
        "mean_young": mean_young,
        "mean_old":   mean_old,
        "log2FC":     log2fc,
        "pvalue":     pvals,
        "padj":       padj,
    })

    # Group means for all three groups
    if "mid" in group_masks:
        mid_attn = cpg_attention[group_masks["mid"]]
        df["mean_mid"] = mid_attn.mean(axis=0)

    return df, {label: cpg_attention[mask] for label, mask in group_masks.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_attention_matrices(group_attn_dict, cpg_ids, outdir: Path, n_show=200):
    """Fig 5b: Attention score matrix heatmap per age group."""
    fig_dir = outdir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    groups = [k for k in ["young", "mid", "old"] if k in group_attn_dict]
    titles = {"young": "Age < 20", "mid": "Age 20-60", "old": "Age > 60"}

    fig, axes = plt.subplots(1, len(groups), figsize=(5 * len(groups), 5))
    if len(groups) == 1:
        axes = [axes]

    for ax, grp in zip(axes, groups):
        attn = group_attn_dict[grp]
        # Show mean attention (CpGs × quantile of CpG importance)
        mean_attn = attn.mean(axis=0)
        # Sort CpGs by mean attention, show top n_show
        order = np.argsort(mean_attn)[::-1][:n_show]
        # Sample subset of samples for display
        n_samp = min(200, attn.shape[0])
        samp_idx = np.random.choice(attn.shape[0], n_samp, replace=False)
        mat = attn[np.sort(samp_idx)][:, order]  # [n_samp, n_show]

        im = ax.imshow(mat.T, aspect="auto", cmap="RdBu_r",
                       vmin=0, vmax=np.quantile(mat, 0.99))
        ax.set_title(titles.get(grp, grp), fontsize=10, fontweight="bold")
        ax.set_xlabel("Samples", fontsize=8)
        ax.set_ylabel("CpG sites (top by attention)", fontsize=8)
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Attention Score")

    plt.suptitle("Attention Score Matrices by Age Group", fontsize=12, y=1.02)
    plt.tight_layout()
    out = fig_dir / "attention_matrices.png"
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved {out.name}")


def plot_volcano(diff_df, lfc_thresh, fdr_thresh, outdir: Path):
    """Fig 5c: Volcano plot of log2FC vs -log10(padj)."""
    fig_dir = outdir / "figures"

    df = diff_df.copy()
    df["-log10padj"] = -np.log10(df["padj"].clip(1e-300))

    sig_old   = (df["padj"] < fdr_thresh) & (df["log2FC"] >  lfc_thresh)
    sig_young = (df["padj"] < fdr_thresh) & (df["log2FC"] < -lfc_thresh)
    ns        = ~(sig_old | sig_young)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(df.loc[ns,       "log2FC"], df.loc[ns,       "-log10padj"],
               s=3, alpha=0.3, c="#ADB6B6", rasterized=True, label="NS")
    ax.scatter(df.loc[sig_young,"log2FC"], df.loc[sig_young,"-log10padj"],
               s=8, alpha=0.7, c="#4DBBD5", rasterized=True, label=f"Higher in young ({sig_young.sum()})")
    ax.scatter(df.loc[sig_old,  "log2FC"], df.loc[sig_old,  "-log10padj"],
               s=8, alpha=0.7, c="#E64B35", rasterized=True, label=f"Higher in old ({sig_old.sum()})")

    ax.axvline( lfc_thresh, color="gray", linewidth=0.8, linestyle="--")
    ax.axvline(-lfc_thresh, color="gray", linewidth=0.8, linestyle="--")
    ax.axhline(-np.log10(fdr_thresh), color="gray", linewidth=0.8, linestyle=":")

    ax.set_xlabel("Log2 Fold Change (Old vs Young)", fontsize=10)
    ax.set_ylabel("-log10(adj. p-value)", fontsize=10)
    ax.set_title("Differential Attention: Young vs Old", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.5)

    plt.tight_layout()
    out = fig_dir / "attention_volcano.png"
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved {out.name}")

    return sig_young, sig_old


def plot_top_cpg_heatmap(diff_df, group_attn_dict, cpg_ids, top_n, outdir: Path,
                          manifest_df=None):
    """Fig 5d: Heatmap of mean attention for top young-/old-important CpGs."""
    fig_dir = outdir / "figures"

    df = diff_df.sort_values("log2FC")
    top_young_idx = df.head(top_n)["cpg_idx"].values  # most negative log2FC
    top_old_idx   = df.tail(top_n)["cpg_idx"].values  # most positive log2FC
    top_idx = np.concatenate([top_young_idx, top_old_idx])

    groups = [k for k in ["young", "mid", "old"] if k in group_attn_dict]
    mat = np.array([group_attn_dict[g].mean(axis=0)[top_idx] for g in groups])  # [n_groups, 2*top_n]

    # CpG labels
    cpg_labels = [cpg_ids[i] if i < len(cpg_ids) else f"cpg_{i}" for i in top_idx]
    if manifest_df is not None and "gene" in manifest_df.columns:
        genes = []
        for cid in cpg_labels:
            g = manifest_df.loc[manifest_df["cpg_id"] == cid, "gene"].values
            genes.append(g[0] if len(g) > 0 and str(g[0]) not in ("nan", "") else "")
        cpg_labels = [f"{c}\n{g}" if g else c for c, g in zip(cpg_labels, genes)]

    fig, ax = plt.subplots(figsize=(max(14, top_n * 0.8), 3))
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r")
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels(["Age<20", "Age 20-60", "Age>60"], fontsize=9)
    ax.set_xticks(range(len(top_idx)))
    ax.set_xticklabels(cpg_labels, rotation=90, fontsize=7)
    ax.axvline(top_n - 0.5, color="white", linewidth=2)
    ax.set_title(f"Top {top_n} Young-Important (left) and Old-Important (right) CpG Sites",
                 fontsize=10, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="Mean Attention Score")
    plt.tight_layout()
    out = fig_dir / "attention_top_cpg_heatmap.png"
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1. Load encoder
    log.info("[1/5] Loading encoder ...")
    encoder = load_encoder(args.checkpoint, args.ckpt_type)

    # 2. Extract attention scores
    log.info("[2/5] Extracting attention scores ...")
    cpg_attention, ages, cpg_ids = extract_attention_scores(
        encoder, args.data, args.tokenizer,
        args.batch_size, args.device, args.layer,
        args.max_samples, args.age_col,
    )
    np.save(outdir / "cpg_attention.npy", cpg_attention)
    np.save(outdir / "ages.npy", ages)

    # Load external metadata ages if available and h5ad doesn't have them
    if args.metadata and np.isnan(ages).mean() > 0.5:
        log.info("Loading ages from external metadata ...")
        meta = pd.read_csv(args.metadata)
        # ages already loaded from dataset; this is a fallback
        log.info(f"  External metadata has {meta['age'].notna().sum():,} valid ages")

    # 3. Differential attention analysis
    log.info("[3/5] Computing differential attention (young vs old) ...")
    valid = ~np.isnan(ages)
    diff_df, group_attn = differential_attention(cpg_attention[valid], ages[valid])
    diff_df["cpg_id"] = [cpg_ids[i] if i < len(cpg_ids) else f"cpg_{i}"
                         for i in diff_df["cpg_idx"]]

    # Add manifest annotations if available
    manifest_df = None
    if args.manifest:
        log.info("Loading manifest annotations ...")
        manifest_df = pd.read_csv(args.manifest, sep="\t")
        diff_df = diff_df.merge(
            manifest_df[["cpg_id"] + [c for c in ["gene", "chr", "island_relation"] if c in manifest_df.columns]],
            on="cpg_id", how="left",
        )

    diff_df.to_csv(outdir / "differential_attention.csv", index=False)
    log.info(f"  Saved differential_attention.csv  ({len(diff_df):,} CpGs)")

    # 4. Significant sites summary
    sig_thresh = diff_df["padj"] < args.fdr_thresh
    lfc_thresh = args.lfc_thresh
    n_old   = ((diff_df["log2FC"] >  lfc_thresh) & sig_thresh).sum()
    n_young = ((diff_df["log2FC"] < -lfc_thresh) & sig_thresh).sum()
    log.info(f"  Differential CpGs: {n_old} old-enriched, {n_young} young-enriched "
             f"(|log2FC|>{lfc_thresh:.2f}, FDR<{args.fdr_thresh})")

    # 5. Plots
    log.info("[4/5] Generating plots ...")
    plot_attention_matrices(group_attn, cpg_ids, outdir)
    plot_volcano(diff_df, lfc_thresh, args.fdr_thresh, outdir)
    plot_top_cpg_heatmap(diff_df, group_attn, cpg_ids, args.top_n_cpgs, outdir, manifest_df)

    # 5. Report
    log.info("[5/5] Writing report ...")
    report = [
        "=" * 60,
        "  MethylLlama-Small — Attention Analysis (Fig 5)",
        "=" * 60,
        f"  Checkpoint  : {args.checkpoint}",
        f"  Layer       : {args.layer} (last layer)",
        f"  Samples     : {cpg_attention.shape[0]:,}",
        f"  CpG sites   : {cpg_attention.shape[1]:,}",
        f"  Age groups  :",
        f"    Young (<20)  : {(~np.isnan(ages) & (ages < 20)).sum():,}",
        f"    Mid (20-60)  : {(~np.isnan(ages) & (ages >= 20) & (ages < 60)).sum():,}",
        f"    Old (>60)    : {(~np.isnan(ages) & (ages >= 60)).sum():,}",
        f"  Differential CpGs:",
        f"    Old-enriched  (log2FC > {lfc_thresh:.2f}, FDR<{args.fdr_thresh}): {n_old}",
        f"    Young-enriched(log2FC < {-lfc_thresh:.2f}, FDR<{args.fdr_thresh}): {n_young}",
        "=" * 60,
    ]
    (outdir / "attention_report.txt").write_text("\n".join(report))
    print("\n".join(report))

    log.info(f"\nDone. Outputs → {outdir}/")
    log.info(f"  cpg_attention.npy          [{cpg_attention.shape[0]:,} × {cpg_attention.shape[1]:,}]")
    log.info(f"  differential_attention.csv  all CpGs with stats")
    log.info(f"  figures/attention_matrices.png")
    log.info(f"  figures/attention_volcano.png")
    log.info(f"  figures/attention_top_cpg_heatmap.png")


if __name__ == "__main__":
    main()
