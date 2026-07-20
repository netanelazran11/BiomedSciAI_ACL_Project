"""
V7b attention analysis (genomic-RoPE correct) — the attention_v5 analysis redone.

The old scripts/repr_analysis/cls_attention_analysis.py computed attention with
SEQUENTIAL RoPE (m.rotary_emb(L), no position_ids) → invalid for the Genomic-RoPE
model, and its near-uniform result is an artifact. This version:
  - passes genomic position_ids so RoPE matches the trained model
  - uses the model's native output_attentions (per-layer CLS→CpG row, [B,H,1,L])

Two analyses:
  A. CLS attention selectivity (v5 parity): per (layer,head) Shannon entropy,
     normalized entropy, Gini, top-k mass, and the top-attended CpGs.
  B. Genomic distance-decay (the real RoPE validation): for a random subset of
     CpG *query* positions, attention weight as a function of genomic-rank distance
     to the key CpG. If Genomic RoPE learned locality, attention decays with
     genomic distance. (Needs a light hook — native path only gives the CLS row.)

Outputs (--outdir, default figures/v7b_attention/):
  cls_attention_summary.json      per-(layer,head) selectivity metrics
  cls_mean_attn.npy               (n_layers, n_heads, n_cpg) mean CLS attention
  attention_distance_decay.json   per-layer attention vs genomic-distance bins
  fig_cls_entropy_gini.png, fig_topk_mass.png, fig_distance_decay.png

Usage (cluster, GPU): see run_attention_v7b.sh
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--genomic_rank", required=True)
    p.add_argument("--outdir", default="figures/v7b_attention")
    p.add_argument("--max_samples", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--n_query", type=int, default=32, help="query CpGs sampled for decay")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ── distance-decay hook: capture subset-query attention with genomic RoPE ──────
class DecayHook:
    def __init__(self, attn_module, n_query, n_bins=40):
        from bmfm_methylation.llama.model import apply_rotary_pos_emb
        self._rope = apply_rotary_pos_emb
        self.m = attn_module
        self.n_query = n_query
        # log-spaced genomic-distance bin edges (rank units)
        self.edges = np.unique(np.round(np.logspace(0, np.log10(25000), n_bins)).astype(int))
        self.wsum = np.zeros(len(self.edges) - 1)
        self.wcnt = np.zeros(len(self.edges) - 1)
        self._orig = attn_module.forward
        attn_module.forward = self._fwd

    def _fwd(self, hidden_states, attention_mask=None, position_ids=None,
             output_attentions=False, **kw):
        out = self._orig(hidden_states, attention_mask, position_ids=position_ids,
                         output_attentions=output_attentions, **kw)
        if position_ids is None:
            return out
        with torch.no_grad():
            m = self.m
            B, L, D = hidden_states.shape
            H, Dh = m.num_heads, m.head_dim
            q = m.q_proj(hidden_states).view(B, L, H, Dh).transpose(1, 2)
            k = m.k_proj(hidden_states).view(B, L, H, Dh).transpose(1, 2)
            cos, sin = m.rotary_emb(position_ids=position_ids)
            q, k = self._rope(q, k, cos, sin)
            scale = 1.0 / math.sqrt(Dh)
            pos = position_ids  # [B, L]; 0 = CLS/pad, rank+1 for CpGs
            for b in range(B):
                real = (pos[b] > 0).nonzero(as_tuple=True)[0]  # CpG positions
                if len(real) < 2:
                    continue
                qsel = real[torch.randperm(len(real))[: self.n_query]]
                scores = (q[b, :, qsel, :] @ k[b].transpose(-2, -1)) * scale  # [H, nq, L]
                w = torch.softmax(scores.float(), dim=-1).mean(0).cpu().numpy()  # [nq, L]
                rq = pos[b, qsel].cpu().numpy()          # genomic rank+1 of queries
                rk = pos[b].cpu().numpy()                # genomic rank+1 of keys
                for i in range(len(qsel)):
                    dist = np.abs(rk - rq[i])
                    keep = (rk > 0) & (dist > 0)
                    idx = np.digitize(dist[keep], self.edges) - 1
                    ok = (idx >= 0) & (idx < len(self.wsum))
                    np.add.at(self.wsum, idx[ok], w[i][keep][ok])
                    np.add.at(self.wcnt, idx[ok], 1.0)
        return out

    def curve(self):
        mean_w = np.divide(self.wsum, self.wcnt, out=np.full_like(self.wsum, np.nan),
                           where=self.wcnt > 0)
        centers = np.sqrt(self.edges[:-1] * self.edges[1:])
        return centers, mean_w

    def remove(self):
        self.m.forward = self._orig


def gini(w):
    x = np.sort(w[w > 0])
    if len(x) == 0:
        return 0.0
    n = len(x); cum = np.cumsum(x)
    return float(1 - 2 * (cum.sum() / (n * x.sum())) + 1.0 / n)


def main():
    a = parse_args()
    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
    from torch.utils.data import DataLoader
    from bmfm_targets.tokenization import MultiFieldTokenizer
    from bmfm_methylation.shared.data_module import MethylationDataset, WCEDCollator
    from bmfm_methylation.llama.finetune_llama import load_wced_llama_checkpoint

    module = load_wced_llama_checkpoint(a.checkpoint)
    encoder = module.encoder.to(a.device).eval()
    tok = MultiFieldTokenizer.from_pretrained(a.tokenizer)
    ds = MethylationDataset(h5ad_path=a.data, split=None, normalize_age=False)
    collator = WCEDCollator(tokenizer=tok, cpg_sites=ds.cpg_sites,
                            vocab_size=len(ds.cpg_sites), input_ratio=1.0,
                            contrastive=False, genomic_rank_path=a.genomic_rank)
    loader = DataLoader(ds, batch_size=a.batch_size, collate_fn=collator,
                        shuffle=False, num_workers=0)

    layers = encoder.encoder.layers
    n_layers = len(layers)
    decay_hooks = [DecayHook(l.attn, a.n_query) for l in layers]

    cls_sum = None
    n_seen = 0
    ps_top1, ps_ent = [], []       # per-sample, per-layer concentration summaries
    with torch.no_grad():
        for batch in loader:
            if n_seen >= a.max_samples:
                break
            cpg_ids = batch["cpg_ids"].to(a.device)
            beta = batch["beta_values"].to(a.device)
            attn = batch["attention_mask"].to(a.device)
            pos = batch.get("position_ids")
            pos = pos.to(a.device) if pos is not None else None
            input_ids = torch.stack([cpg_ids.float(), beta], dim=1)
            out = encoder(input_ids=input_ids, attention_mask=attn,
                          position_ids=pos, output_attentions=True)
            # out.attentions: tuple(n_layers) of [B, H, 1, L]; CpG cols are 1:
            att = torch.stack([w[:, :, 0, 1:] for w in out.attentions], dim=1)  # [B, nL, H, ncpg]
            s = att.sum(0).cpu().numpy()
            cls_sum = s if cls_sum is None else cls_sum + s
            # per-sample per-layer concentration (mean over heads)
            aw = att.mean(2)                                     # [B, nL, ncpg]
            awn = aw / aw.sum(-1, keepdim=True).clamp(min=1e-12)
            k = max(1, int(0.01 * awn.shape[-1]))
            ps_top1.append(torch.topk(awn, k, dim=-1).values.sum(-1).cpu().numpy())   # [B, nL]
            ps_ent.append((-(awn.clamp(min=1e-12).log() * awn).sum(-1)).cpu().numpy())  # [B, nL]
            n_seen += att.shape[0]
    mean_attn = cls_sum / max(n_seen, 1)                # [nL, H, ncpg]
    np.save(outdir / "cls_mean_attn.npy", mean_attn.astype(np.float32))
    print(f"CLS attention averaged over {n_seen} samples: {mean_attn.shape}")

    # ── per-sample × layer concentration + heatmap ────────────────────────────
    ps_top1 = np.concatenate(ps_top1)[:n_seen]          # [n_seen, nL]
    ps_ent = np.concatenate(ps_ent)[:n_seen]
    np.save(outdir / "cls_persample_top1pct.npy", ps_top1.astype(np.float32))
    np.save(outdir / "cls_persample_entropy.npy", ps_ent.astype(np.float32))
    ps_meta = ds.adata.obs.iloc[:n_seen].reset_index()
    keep_cols = [c for c in ["index", "age", "tissue_type", "dataset"] if c in ps_meta.columns]
    ps_meta[keep_cols].to_csv(outdir / "cls_persample_meta.csv", index=False)
    # order samples by tissue then age so structure is visible
    tis = ps_meta["tissue_type"].astype(str).values if "tissue_type" in ps_meta else np.zeros(n_seen, str)
    agev = ps_meta["age"].values if "age" in ps_meta else np.zeros(n_seen)
    order = np.lexsort((agev, tis))
    fig, ax = plt.subplots(figsize=(6, 9))
    im = ax.imshow(ps_top1[order], aspect="auto", cmap="inferno", vmin=0, vmax=1)
    ax.set_xticks(range(mean_attn.shape[0])); ax.set_xticklabels([f"L{l}" for l in range(mean_attn.shape[0])])
    ax.set_xlabel("layer"); ax.set_ylabel("sample (grouped by tissue, then age)")
    ax.set_title("Per-sample CLS attention concentration\n(top-1% attention mass; 1=one CpG, low=spread)",
                 fontsize=11, weight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, label="top-1% attention mass")
    fig.tight_layout(); fig.savefig(outdir / "fig_persample_concentration.png", dpi=160, bbox_inches="tight"); plt.close(fig)
    print(f"per-sample concentration: mean top-1% mass by layer = "
          f"{np.round(ps_top1.mean(0), 3).tolist()}")

    # ── A. selectivity metrics ────────────────────────────────────────────────
    n_cpg = mean_attn.shape[-1]
    summary = []
    for l in range(mean_attn.shape[0]):
        for h in range(mean_attn.shape[1]):
            w = mean_attn[l, h]; w = w / (w.sum() + 1e-12)
            wpos = w[w > 0]
            ent = float(-(wpos * np.log(wpos)).sum())
            summary.append({
                "layer": l, "head": h,
                "normalized_entropy": round(ent / math.log(n_cpg), 4),
                "gini": round(gini(w), 4),
                "top10_mass": round(float(np.sort(w)[-10:].sum()), 5),
                "top100_mass": round(float(np.sort(w)[-100:].sum()), 5),
            })
    json.dump(summary, open(outdir / "cls_attention_summary.json", "w"), indent=2)

    # ── B. distance-decay ─────────────────────────────────────────────────────
    decay = {}
    for l, hk in enumerate(decay_hooks):
        c, w = hk.curve(); hk.remove()
        decay[f"layer{l}"] = {"dist": c.tolist(), "attn": np.nan_to_num(w).tolist()}
    json.dump(decay, open(outdir / "attention_distance_decay.json", "w"), indent=2)

    # ── figures ───────────────────────────────────────────────────────────────
    S = np.array([[d["normalized_entropy"] for d in summary if d["layer"] == l]
                  for l in range(n_layers)])
    G = np.array([[d["gini"] for d in summary if d["layer"] == l] for l in range(n_layers)])
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    im0 = ax[0].imshow(S, aspect="auto", cmap="viridis", vmin=0.9, vmax=1.0)
    ax[0].set_title("CLS attention normalized entropy\n(1=uniform)", fontsize=11, weight="bold")
    ax[0].set_xlabel("head"); ax[0].set_ylabel("layer"); plt.colorbar(im0, ax=ax[0], fraction=0.046)
    im1 = ax[1].imshow(G, aspect="auto", cmap="magma")
    ax[1].set_title("CLS attention Gini\n(0=uniform, 1=spike)", fontsize=11, weight="bold")
    ax[1].set_xlabel("head"); ax[1].set_ylabel("layer"); plt.colorbar(im1, ax=ax[1], fraction=0.046)
    fig.tight_layout(); fig.savefig(outdir / "fig_cls_entropy_gini.png", dpi=160, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for l in range(n_layers):
        c = np.array(decay[f"layer{l}"]["dist"]); w = np.array(decay[f"layer{l}"]["attn"])
        ax.plot(c, w, "o-", ms=3, label=f"layer {l}", alpha=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("genomic-rank distance between CpGs"); ax.set_ylabel("mean attention weight")
    ax.set_title("Genomic RoPE validation: attention vs genomic distance\n(decay = learned locality)",
                 fontsize=11, weight="bold")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(outdir / "fig_distance_decay.png", dpi=160, bbox_inches="tight"); plt.close(fig)

    # verdict on RoPE locality: near vs far attention ratio (layer-averaged)
    near, far = [], []
    for l in range(n_layers):
        c = np.array(decay[f"layer{l}"]["dist"]); w = np.array(decay[f"layer{l}"]["attn"])
        near.append(np.nanmean(w[c < 50])); far.append(np.nanmean(w[c > 5000]))
    ratio = float(np.nanmean(near) / (np.nanmean(far) + 1e-12))
    print(f"\nGenomic RoPE locality: near(<50)/far(>5000) attention ratio = {ratio:.2f}")
    print("  >1.5 = clear locality; ~1 = position-agnostic")
    print(f"CLS selectivity: mean normalized entropy = {S.mean():.4f} (1=uniform), mean Gini = {G.mean():.4f}")
    print(f"Outputs → {outdir}/")


if __name__ == "__main__":
    main()
