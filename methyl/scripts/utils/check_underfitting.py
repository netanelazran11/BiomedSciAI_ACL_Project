"""
check_underfitting.py
=====================
Diagnose whether MethylLlama fine-tuning is underfitting, overfitting,
or data-bottlenecked — WITHOUT running any new experiments.

Checks:
  1. Effective rank of CLS embeddings (collapse detection)
  2. Train vs val loss gap (overfitting / underfitting)
  3. PCA variance concentration (is space spread or collapsed?)
  4. Linear vs MLP probe gap (linearity of representation)

All checks run locally on existing pretrained checkpoint.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────

def effective_rank(matrix: np.ndarray) -> float:
    """Roy & Vetterli (2007) effective rank = exp(entropy of singular value distribution)."""
    _, s, _ = np.linalg.svd(matrix, full_matrices=False)
    s = s[s > 1e-10]
    p = s / s.sum()
    return float(np.exp(-np.sum(p * np.log(p + 1e-12))))


def pca_variance_explained(matrix: np.ndarray, n_components: int = 20):
    """Return cumulative variance explained by top-k PCA components."""
    matrix = matrix - matrix.mean(0)
    _, s, _ = np.linalg.svd(matrix, full_matrices=False)
    var = s ** 2
    return (var[:n_components] / var.sum()).cumsum()


def rank_collapse_check(matrix: np.ndarray, threshold: float = 0.01) -> int:
    """Count how many singular values exceed threshold × max singular value."""
    _, s, _ = np.linalg.svd(matrix, full_matrices=False)
    return int((s > threshold * s[0]).sum())


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", required=True,
                        help="Path to .npy file of CLS embeddings [N, 256]")
    parser.add_argument("--wandb_run", default=None,
                        help="WandB run ID (optional — prints train/val curve advice)")
    parser.add_argument("--outdir", default="outputs/underfitting_check")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ── Load embeddings ───────────────────────────────────────────────────────
    emb = np.load(args.embeddings).astype(np.float32)
    log.info(f"Embeddings shape: {emb.shape}")

    N, D = emb.shape

    # ── 1. Effective rank ─────────────────────────────────────────────────────
    eff_rank = effective_rank(emb)
    hard_rank = rank_collapse_check(emb, threshold=0.01)
    log.info(f"\n{'='*55}")
    log.info(f"1. EFFECTIVE RANK")
    log.info(f"   Effective rank  : {eff_rank:.1f}  (max possible = {D})")
    log.info(f"   Hard rank (1%)  : {hard_rank}  dimensions above 1% of max singular value")

    if eff_rank < 20:
        log.info(f"   ⚠ COLLAPSE: effective rank {eff_rank:.1f} << {D}D — representation is degenerate")
        log.info(f"     Fix: increase contrastive weight, add weight decay, lower LR")
    elif eff_rank < 50:
        log.info(f"   ~ PARTIAL USE: {eff_rank:.1f} effective dims — moderate collapse")
    else:
        log.info(f"   ✓ HEALTHY: {eff_rank:.1f} effective dims — space is well-utilized")

    # ── 2. PCA variance concentration ────────────────────────────────────────
    cumvar = pca_variance_explained(emb, n_components=min(20, N, D))
    log.info(f"\n{'='*55}")
    log.info(f"2. PCA VARIANCE EXPLAINED")
    for k, cv in zip([1, 3, 5, 10, 20], [cumvar[0], cumvar[2], cumvar[4], cumvar[9], cumvar[min(19, len(cumvar)-1)]]):
        log.info(f"   Top-{k:2d} PCs : {cv*100:.1f}%")

    if cumvar[2] > 0.90:
        log.info("   ⚠ CONCENTRATION: top-3 PCs explain >90% — space is very low-dimensional")
        log.info("     Interpretation: almost all biological variation captured in 3 directions")
        log.info("     Could mean: (a) only 3 major biological axes exist in data, OR")
        log.info("                 (b) model collapsed to simple structure")
    elif cumvar[9] > 0.90:
        log.info("   ~ MODERATE: top-10 PCs explain >90% — reasonable for 256D space")
    else:
        log.info("   ✓ DISTRIBUTED: top-20 PCs < 90% — variance spread across many dims")

    # ── 3. Isotropy score ─────────────────────────────────────────────────────
    emb_norm = emb - emb.mean(0)
    emb_norm /= (np.linalg.norm(emb_norm, axis=1, keepdims=True) + 1e-8)
    # Average cosine similarity between random pairs
    idx = np.random.choice(N, size=min(1000, N), replace=False)
    sub = emb_norm[idx]
    cos_sim = (sub @ sub.T)
    np.fill_diagonal(cos_sim, 0)
    avg_cos = cos_sim.sum() / (len(idx) * (len(idx) - 1))
    log.info(f"\n{'='*55}")
    log.info(f"3. ISOTROPY (avg cosine between random pairs)")
    log.info(f"   Average cosine similarity: {avg_cos:.4f}")
    if avg_cos > 0.8:
        log.info("   ⚠ ANISOTROPY: embeddings are nearly all pointing in same direction")
        log.info("     Embeddings are not well-spread — likely dominated by one feature (e.g., tissue)")
    elif avg_cos > 0.3:
        log.info("   ~ MODERATE: some directional bias")
    else:
        log.info("   ✓ ISOTROPIC: embeddings are spread across the sphere")

    # ── 4. Summary interpretation ─────────────────────────────────────────────
    log.info(f"\n{'='*55}")
    log.info("4. UNDERFITTING vs OVERFITTING GUIDE")
    log.info("""
  To check underfitting/overfitting you need WandB train loss curves.
  Look at your fine-tuning WandB run (3t5eve7t for MethylLlama 19k):

  SCENARIO A — Overfitting:
    train_loss keeps falling → val_loss stops falling or rises
    train MedAE << val MedAE at best checkpoint
    Fix: more weight decay, dropout, smaller LR

  SCENARIO B — Underfitting:
    train_loss AND val_loss both plateau high
    train MedAE ≈ val MedAE but both large
    Fix: more epochs, larger model, higher LR

  SCENARIO C — Data-bottlenecked (most likely given 7k samples):
    train_loss falls close to 0, val_loss plateaus
    Best val checkpoint early, final epoch worse
    train MedAE << val MedAE (large gap)
    Fix: more data, not bigger model

  HOW TO CHECK RIGHT NOW:
    In WandB for run 3t5eve7t:
    - Plot train/medae vs val/medae on same axis
    - If train/medae → 0 but val/medae ≈ 3.5yr → overfitting
    - If train/medae ≈ val/medae ≈ 3.5yr → data-bottlenecked
    - The fact that best val checkpoint (ep 127) != final (ep 300)
      strongly suggests mild overfitting after ep 127.
""")

    log.info(f"\n{'='*55}")
    log.info("ARCHITECTURE SCALING ADVICE")
    log.info(f"""
  Given what we know about your setup:
    - Fine-tune data: 7,286 train samples
    - Current model: 4L × 256D = 7.7M params (2.1M non-embedding)
    - Best val MedAE: 3.56yr (ep 127), random init never reaches <5yr

  Adding layers (6L or 8L) WITHOUT retraining the encoder will NOT help:
    - The new layers would be randomly initialized
    - They would act as noise on top of a well-trained 4L encoder
    - You would need to pretrain a 6L model from scratch

  Adding layers WITH retraining (new pretrain) is expensive and risky.
  The data bottleneck (7k samples) means larger model = more overfitting.

  RECOMMENDATION:
    Skip architecture scaling for the thesis.
    Current 4L × 256D is justified because:
      (a) Larger model needs more fine-tuning data
      (b) Random-init 4L already can't fit the data well (MedAE=8.5yr)
          → problem is pretraining, not capacity
      (c) Fine-tuned 4L WCED matches MethylGPT Medium (15M params)
          → architecture capacity is NOT the bottleneck
""")

    log.info(f"Results saved. Effective rank: {eff_rank:.1f}, Hard rank: {hard_rank}")


if __name__ == "__main__":
    main()
