"""
check_contrastive.py
====================
Verify the contrastive (InfoNCE) loss in WCED pretraining is actually working.

Runs on the PRETRAINED model checkpoint (not fine-tuned).
Creates two random masked views of each val sample, extracts CLS,
then measures whether the contrastive objective is achieving its goal.

Metrics:
  1. Positive similarity    — cosine(CLS_view1, CLS_view2) same sample
  2. Negative similarity    — cosine(CLS_view1, CLS_other) different sample
  3. Alignment              — mean positive similarity (want: > 0.7)
  4. Uniformity             — log mean exp(-2 * dist²) (want: < -1.0)
  5. Retrieval accuracy     — does view1 retrieve view2 as nearest neighbour?
  6. Temperature check      — what temperature makes InfoNCE non-trivial?

Usage (on cluster):
  python check_contrastive.py \
      --checkpoint /path/to/pretrain.ckpt \
      --h5ad /path/to/data.h5ad \
      --tokenizer /path/to/tokenizer_llama_pretrain49k \
      --outdir outputs/contrastive_check \
      --n_samples 512
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ── metrics ───────────────────────────────────────────────────────────────────

def alignment(z1: torch.Tensor, z2: torch.Tensor) -> float:
    """Mean cosine similarity between paired views (want > 0.7)."""
    return F.cosine_similarity(z1, z2).mean().item()


def uniformity(z: torch.Tensor) -> float:
    """Wang & Isola (2020) uniformity = log mean exp(-2||z_i - z_j||²).
    Want < -1.0 (well spread). > -0.5 means collapse."""
    sq_dists = torch.cdist(z, z).pow(2)
    n = z.size(0)
    mask = ~torch.eye(n, dtype=torch.bool, device=z.device)
    return torch.log(torch.exp(-2 * sq_dists[mask]).mean()).item()


def retrieval_accuracy(z1: torch.Tensor, z2: torch.Tensor) -> float:
    """Fraction of samples where view1's nearest neighbour in z2 is its paired view2."""
    sim = z1 @ z2.T  # [N, N] — rows=z1, cols=z2
    top1 = sim.argmax(dim=1)  # [N]
    labels = torch.arange(z1.size(0), device=z1.device)
    return (top1 == labels).float().mean().item()


def infonce_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1) -> float:
    """Compute symmetric InfoNCE loss."""
    N = z1.size(0)
    z = torch.cat([z1, z2], dim=0)          # [2N, D]
    sim = (z @ z.T) / temperature           # [2N, 2N]
    # Mask out diagonal (self-similarity)
    mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, -1e9)
    # Labels: view1[i] matches view2[i] = position i+N
    labels = torch.cat([torch.arange(N, 2 * N), torch.arange(0, N)]).to(z.device)
    loss = F.cross_entropy(sim, labels)
    return loss.item()


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="WCED pretrain checkpoint (.ckpt)")
    parser.add_argument("--h5ad",       required=True)
    parser.add_argument("--tokenizer",  required=True)
    parser.add_argument("--outdir",     default="outputs/contrastive_check")
    parser.add_argument("--n_samples",  type=int, default=512,
                        help="Number of val samples to use")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--subset_k",   type=int, default=49156)
    parser.add_argument("--input_ratio", type=float, default=0.5,
                        help="Fraction of CpGs visible to encoder (matches pretrain)")
    parser.add_argument("--temperature", type=float, default=0.1)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # ── Load pretrained model ─────────────────────────────────────────────────
    from bmfm_methylation.llama.finetune_llama import load_wced_llama_checkpoint
    model = load_wced_llama_checkpoint(args.checkpoint)
    encoder = model.encoder.eval().to(device)
    log.info("Pretrained encoder loaded")

    # ── Load data ─────────────────────────────────────────────────────────────
    from bmfm_methylation.shared.data_module import MethylationDataset, WCEDCollator
    from bmfm_targets.tokenization import MultiFieldTokenizer

    tokenizer = MultiFieldTokenizer.from_pretrained(args.tokenizer)

    ds = MethylationDataset(h5ad_path=args.h5ad, split="train")
    n = min(args.n_samples, len(ds))
    idx = np.random.choice(len(ds), n, replace=False)
    ds_sub = Subset(ds, idx.tolist())

    cpg_sites = ds.cpg_sites

    # Two collators with different seeds → two independent random masks
    collator_v1 = WCEDCollator(
        tokenizer=tokenizer, cpg_sites=cpg_sites,
        vocab_size=args.subset_k, input_ratio=args.input_ratio,
        fixed_subset_seed=None,   # random mask each call
        contrastive=False,
    )
    collator_v2 = WCEDCollator(
        tokenizer=tokenizer, cpg_sites=cpg_sites,
        vocab_size=args.subset_k, input_ratio=args.input_ratio,
        fixed_subset_seed=None,
        contrastive=False,
    )

    loader_v1 = DataLoader(ds_sub, batch_size=args.batch_size,
                           collate_fn=collator_v1, shuffle=False, num_workers=2)
    loader_v2 = DataLoader(ds_sub, batch_size=args.batch_size,
                           collate_fn=collator_v2, shuffle=False, num_workers=2)

    # ── Extract CLS embeddings for both views ─────────────────────────────────
    all_z1, all_z2 = [], []

    with torch.no_grad():
        for batch1, batch2 in zip(loader_v1, loader_v2):
            cpg_ids1 = batch1["cpg_ids"].to(device)
            beta1    = batch1["beta_values"].to(device)
            mask1    = batch1["attention_mask"].to(device)

            cpg_ids2 = batch2["cpg_ids"].to(device)
            beta2    = batch2["beta_values"].to(device)
            mask2    = batch2["attention_mask"].to(device)

            out1 = encoder(cpg_ids=cpg_ids1, beta_values=beta1, attention_mask=mask1)
            out2 = encoder(cpg_ids=cpg_ids2, beta_values=beta2, attention_mask=mask2)

            # Extract CLS (pooler_output)
            cls1 = out1.pooler_output  # [B, 256]
            cls2 = out2.pooler_output

            all_z1.append(F.normalize(cls1, dim=-1).cpu())
            all_z2.append(F.normalize(cls2, dim=-1).cpu())

    z1 = torch.cat(all_z1, dim=0)  # [N, 256]
    z2 = torch.cat(all_z2, dim=0)

    log.info(f"Extracted CLS for {z1.size(0)} samples × 2 views")

    # ── Compute all metrics ───────────────────────────────────────────────────
    pos_sim = F.cosine_similarity(z1, z2).numpy()
    # Negative: mean cosine between view1[i] and all view2[j≠i]
    sim_matrix = (z1 @ z2.T).numpy()  # [N, N]
    np.fill_diagonal(sim_matrix, np.nan)
    neg_sim_per_sample = np.nanmean(sim_matrix, axis=1)

    align = alignment(z1.to(device), z2.to(device))
    unif  = uniformity(z1.to(device))
    retr  = retrieval_accuracy(z1.to(device), z2.to(device))
    loss  = infonce_loss(z1.to(device), z2.to(device), temperature=args.temperature)

    # ── Print results ─────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("CONTRASTIVE LOSS DIAGNOSTIC")
    print(f"  N samples     : {z1.size(0)}")
    print(f"  Input ratio   : {args.input_ratio} (random {int(args.input_ratio*100)}% CpGs visible)")
    print(f"  Temperature   : {args.temperature}")
    print("=" * 60)

    print(f"\n1. ALIGNMENT (positive cosine similarity)")
    print(f"   Mean  : {pos_sim.mean():.4f}  (want > 0.70)")
    print(f"   Std   : {pos_sim.std():.4f}")
    print(f"   Min   : {pos_sim.min():.4f}")
    print(f"   Max   : {pos_sim.max():.4f}")
    _interp_align(pos_sim.mean())

    print(f"\n2. NEGATIVE SIMILARITY (cross-sample cosine)")
    print(f"   Mean  : {neg_sim_per_sample.mean():.4f}  (want < 0.20)")
    print(f"   Std   : {neg_sim_per_sample.std():.4f}")
    _interp_negative(neg_sim_per_sample.mean())

    print(f"\n3. CONTRAST GAP (alignment - negative)")
    gap = pos_sim.mean() - neg_sim_per_sample.mean()
    print(f"   Gap   : {gap:.4f}  (want > 0.50)")
    if gap < 0.1:
        print("   ⚠ NO CONTRAST: positives and negatives are indistinguishable")
        print("     Contrastive loss is NOT working — embeddings are not aligned")
    elif gap < 0.3:
        print("   ~ WEAK CONTRAST: some alignment but negatives too similar")
        print("     Try: higher contrastive weight, projection head, larger batch")
    else:
        print("   ✓ STRONG CONTRAST: contrastive is working correctly")

    print(f"\n4. UNIFORMITY (Wang & Isola, 2020)")
    print(f"   Score : {unif:.4f}  (want < -1.0; > -0.5 = collapse)")
    if unif > -0.5:
        print("   ⚠ COLLAPSE: embeddings cluster too tightly — no diversity")
    elif unif > -1.0:
        print("   ~ MODERATE uniformity")
    else:
        print("   ✓ WELL SPREAD: embeddings uniformly distributed on sphere")

    print(f"\n5. RETRIEVAL ACCURACY")
    print(f"   Top-1 : {retr*100:.1f}%  (want > 80%; random = {100/z1.size(0):.2f}%)")
    if retr < 0.5:
        print("   ⚠ POOR: view1 cannot find its paired view2 in top-1")
        print("     Contrastive is failing — views are not close enough")
    elif retr < 0.8:
        print("   ~ MODERATE: some retrieval ability but not strong")
    else:
        print("   ✓ GOOD: views reliably find their pair")

    print(f"\n6. InfoNCE LOSS (at temperature={args.temperature})")
    random_loss = np.log(z1.size(0))
    print(f"   Current : {loss:.4f}")
    print(f"   Random  : {random_loss:.4f}  (upper bound = log(N))")
    print(f"   % of random: {loss/random_loss*100:.1f}%  (want < 30%)")
    if loss > 0.9 * random_loss:
        print("   ⚠ NEAR-RANDOM: InfoNCE is barely better than random")
    elif loss > 0.5 * random_loss:
        print("   ~ PARTIAL: some signal but much room to improve")
    else:
        print("   ✓ LOW LOSS: contrastive is learning strong alignment")

    print()
    print("=" * 60)
    print("WHAT TO DO BASED ON RESULTS")
    print("=" * 60)
    print("""
  If retrieval accuracy > 80% and alignment > 0.7:
    → Contrastive IS working. Current weight=0.1 is fine.
    → To improve: try weight=0.3 for next pretrain.

  If retrieval accuracy < 50% or alignment < 0.4:
    → Contrastive is NOT working.
    Possible causes:
      a) Temperature too high (> 0.2) → try 0.05–0.1
      b) Contrastive weight too low (0.1) → reconstruction dominates
      c) No L2 normalization before InfoNCE → add F.normalize()
      d) Positives not truly from same sample → check collator logic
    Fix: verify normalization, lower temperature, raise weight to 0.3.

  If uniformity > -0.5 (collapse):
    → Embeddings all point same direction.
    → Model ignores contrastive signal.
    Fix: add projection head, raise contrastive weight significantly.

  PROJECTION HEAD ADVICE:
    Apply InfoNCE on a 2-layer MLP projection (256→128→128),
    keep the raw CLS for downstream tasks.
    This is SimCLR design: projection absorbs contrastive geometry,
    CLS stays general.
    Worth testing if current alignment < 0.6.
""")

    # ── Save raw numbers ──────────────────────────────────────────────────────
    np.save(outdir / "pos_sim.npy", pos_sim)
    np.save(outdir / "neg_sim.npy", neg_sim_per_sample)
    with open(outdir / "contrastive_report.txt", "w") as f:
        f.write(f"alignment={align:.4f}\n")
        f.write(f"uniformity={unif:.4f}\n")
        f.write(f"retrieval_acc={retr:.4f}\n")
        f.write(f"infonce_loss={loss:.4f}\n")
        f.write(f"random_loss={random_loss:.4f}\n")
        f.write(f"contrast_gap={gap:.4f}\n")
    log.info(f"Saved to {outdir}/")


def _interp_align(v):
    if v > 0.85:
        print("   ✓ EXCELLENT: same-sample views are very similar regardless of which CpGs are visible")
    elif v > 0.70:
        print("   ✓ GOOD: views are well-aligned — contrastive is working")
    elif v > 0.50:
        print("   ~ MODERATE: partial alignment — contrastive has some effect")
    else:
        print("   ⚠ POOR: views of the same sample are NOT similar")
        print("     Contrastive loss is not working — check normalization and temperature")


def _interp_negative(v):
    if v < 0.10:
        print("   ✓ EXCELLENT: different samples are well-separated")
    elif v < 0.20:
        print("   ✓ GOOD: negatives are pushed apart")
    elif v < 0.40:
        print("   ~ MODERATE: negatives still somewhat similar — more training or higher weight needed")
    else:
        print("   ⚠ HIGH: negatives too similar — embeddings not discriminative")
        print("     May indicate tissue dominates the space (all blood looks the same)")


if __name__ == "__main__":
    main()
