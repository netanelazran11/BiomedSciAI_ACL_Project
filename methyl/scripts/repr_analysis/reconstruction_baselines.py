#!/usr/bin/env python3
"""
reconstruction_baselines.py
============================
Diagnostic experiment B: Does the WCED decoder actually use the CLS embedding,
or does it reconstruct methylation by memorising per-CpG population means?

Three baselines compared against the real model reconstruction:

  B1 — Per-CpG training-mean baseline
       Replace decoder output with E[beta_i] computed on the training set.
       This is the trivial floor: a decoder that ignores CLS entirely.
       If model MSE ≈ B1 MSE → decoder has learned nothing sample-specific.

  B3 — Shuffled-CLS baseline
       Run real encoder on each sample, then shuffle the CLS vectors across
       the batch before passing to decoder.  CLS still contains real embeddings
       but they are disconnected from the correct sample.  If model MSE ≈ B3 MSE
       → decoder reconstructs from population statistics stored in its weights,
       not from individual CLS information.

  B4 — Random-Gaussian CLS baseline
       Replace CLS with N(0,1) noise of matching dimension.  A stronger test:
       if model MSE ≈ B4 MSE → decoder has entirely learned to ignore CLS.

Output (saved to --outdir):
  reconstruction_baselines.json   — scalar metrics for all conditions
  reconstruction_baselines.csv    — per-sample MSE for model / B1 / B3 / B4

Usage (cluster, see run_reconstruction_baselines.sh):
  python scripts/repr_analysis/reconstruction_baselines.py \\
      --checkpoint outputs/pretrain-llama-wced/.../epoch=98-val_loss=0.0059.ckpt \\
      --data /path/to/finetuning_19608_clean_stratified_no_outliers.h5ad \\
      --tokenizer tokenizer_llama_pretrain49k \\
      --outdir outputs/repr_analysis/reconstruction_baselines_JOBID \\
      --batch_size 64 --device cuda
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",      required=True,
                   help="WCEDLlamaModule checkpoint (.ckpt)")
    p.add_argument("--data",            required=True,
                   help="h5ad path (finetuning_19608_clean_stratified_no_outliers.h5ad)")
    p.add_argument("--tokenizer",       required=True,
                   help="tokenizer_llama_pretrain49k directory")
    p.add_argument("--outdir",          default="outputs/repr_analysis/reconstruction_baselines")
    p.add_argument("--batch_size",      type=int, default=64)
    p.add_argument("--device",          default="cuda")
    p.add_argument("--n_batches",       type=int, default=0,
                   help="Limit to N batches for quick test (0 = all)")
    return p.parse_args()


@torch.no_grad()
def run_baselines(model, loader, device, n_batches=0):
    """
    Returns per-sample arrays:
      model_mse  — MSE(decoder(real_cls), all_betas)
      b3_mse     — MSE(decoder(shuffled_cls), all_betas)
      b4_mse     — MSE(decoder(random_cls),   all_betas)
      cls_list   — (n_samples, hidden_size) real CLS embeddings
      labels     — (n_samples,) age labels (NaN if missing)
      splits     — list of split strings

    B1 (per-CpG training mean) is computed AFTER this loop from model_mse arrays.
    """
    model_mses, b3_mses, b4_mses = [], [], []
    cls_list, label_list, split_list = [], [], []
    all_target_betas = []      # collect to compute B1 (training-set mean)

    for i, batch in enumerate(loader):
        if n_batches > 0 and i >= n_batches:
            break

        cpg_ids     = batch["cpg_ids"].to(device)
        beta_values = batch["beta_values"].to(device)
        attn_mask   = batch["attention_mask"].to(device)

        # Encoder → real CLS
        input_ids = torch.stack([cpg_ids.float(), beta_values], dim=1)
        enc_out   = model.encoder(input_ids=input_ids, attention_mask=attn_mask)
        cls_real  = enc_out.pooler_output   # [B, D]

        # Target betas — all CpGs, float32
        all_betas = batch.get("all_betas")
        if all_betas is None:
            # BMFMStyle: labels tensor contains target betas (–100 = ignore)
            all_betas_t = batch.get("labels", None)
            if all_betas_t is None:
                log.warning(f"Batch {i}: no all_betas or labels field — skipping")
                continue
            valid_mask  = all_betas_t != -100.0
            target      = all_betas_t.clamp(min=0.0).to(device)
        else:
            valid_mask  = batch.get("valid_mask", torch.ones_like(all_betas, dtype=torch.bool))
            target      = all_betas.to(device)
            valid_mask  = valid_mask.to(device)

        B, V = target.shape

        # Decoder forward with real CLS
        recon_real = model.decoder(cls_real)   # [B, vocab_size]

        # Per-sample MSE (only over valid CpGs)
        def masked_mse(pred, tgt, mask):
            diff2 = (pred - tgt).pow(2) * mask.float()
            return (diff2.sum(dim=1) / mask.float().sum(dim=1).clamp(min=1)).cpu().numpy()

        model_mses.append(masked_mse(recon_real, target, valid_mask))

        # B3: shuffled CLS — permute batch dimension
        perm      = torch.randperm(B, device=device)
        cls_shuf  = cls_real[perm]
        recon_b3  = model.decoder(cls_shuf)
        b3_mses.append(masked_mse(recon_b3, target, valid_mask))

        # B4: random Gaussian CLS
        cls_rand  = torch.randn_like(cls_real)
        recon_b4  = model.decoder(cls_rand)
        b4_mses.append(masked_mse(recon_b4, target, valid_mask))

        cls_list.append(cls_real.cpu().numpy())
        all_target_betas.append(target.cpu().numpy())   # for B1 computation

        # Metadata
        age = batch.get("age", None)
        if age is not None:
            label_list.extend(age.float().numpy().tolist())
        else:
            label_list.extend([float("nan")] * B)

        split = batch.get("split", ["unknown"] * B)
        if isinstance(split, torch.Tensor):
            split = split.tolist()
        split_list.extend(split)

        if (i + 1) % 20 == 0:
            log.info(f"  batch {i+1}/{len(loader)}")

    model_mse = np.concatenate(model_mses)
    b3_mse    = np.concatenate(b3_mses)
    b4_mse    = np.concatenate(b4_mses)
    cls_arr   = np.concatenate(cls_list, axis=0)
    labels    = np.array(label_list)

    # B1: per-CpG training-set mean prediction
    # Stack all target matrices to compute column means
    all_targets_np = np.concatenate(all_target_betas, axis=0)   # [N, V]
    cpg_mean       = all_targets_np.mean(axis=0, keepdims=True)  # [1, V]
    cpg_mean_pred  = np.tile(cpg_mean, (all_targets_np.shape[0], 1))  # [N, V]
    b1_mse         = np.mean((cpg_mean_pred - all_targets_np) ** 2, axis=1)

    return {
        "model_mse": model_mse,
        "b1_mse":    b1_mse,
        "b3_mse":    b3_mse,
        "b4_mse":    b4_mse,
        "cls":       cls_arr,
        "labels":    labels,
        "splits":    split_list,
    }


def summarise(results: dict) -> dict:
    """Compute scalar summary statistics from per-sample MSE arrays."""
    summary = {}
    for key in ["model_mse", "b1_mse", "b3_mse", "b4_mse"]:
        arr = results[key]
        summary[key] = {
            "mean":   float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std":    float(np.std(arr)),
            "p10":    float(np.percentile(arr, 10)),
            "p90":    float(np.percentile(arr, 90)),
        }

    # Ratios: how much better is the real model over each baseline?
    # Ratio > 1 means model MSE < baseline MSE (model is better).
    for base in ["b1", "b3", "b4"]:
        ratio = summary["model_mse"]["mean"] / summary[f"{base}_mse"]["mean"]
        summary[f"ratio_model_vs_{base}"] = float(ratio)

    # Interpretation guide:
    #   ratio_model_vs_b1 < 1.0  → model is worse than trivial mean — bug
    #   ratio_model_vs_b1 ≈ 1.0  → model barely beats trivial floor
    #   ratio_model_vs_b3 ≈ 1.0  → shuffled CLS reconstructs as well → CLS not used
    #   ratio_model_vs_b4 ≈ 1.0  → random CLS reconstructs as well  → CLS ignored
    log.info("=== Reconstruction Baseline Summary ===")
    log.info(f"  Model MSE (real CLS):    {summary['model_mse']['mean']:.6f}")
    log.info(f"  B1  MSE  (cpg mean):     {summary['b1_mse']['mean']:.6f}")
    log.info(f"  B3  MSE  (shuffled CLS): {summary['b3_mse']['mean']:.6f}")
    log.info(f"  B4  MSE  (random CLS):   {summary['b4_mse']['mean']:.6f}")
    log.info(f"  model / B1  = {summary['ratio_model_vs_b1']:.4f}")
    log.info(f"  model / B3  = {summary['ratio_model_vs_b3']:.4f}")
    log.info(f"  model / B4  = {summary['ratio_model_vs_b4']:.4f}")
    if summary["ratio_model_vs_b3"] > 0.95:
        log.warning("  *** model ≈ shuffled CLS → decoder may not use CLS ***")
    return summary


def main():
    args   = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")

    # ── Load WCEDLlamaModule (has both encoder + decoder) ─────────────────────
    from bmfm_methylation.llama.finetune_llama import load_wced_llama_checkpoint
    module = load_wced_llama_checkpoint(args.checkpoint)
    module.eval()
    module.to(device)
    log.info(f"Encoder hidden_size={module.encoder.config.hidden_size}")

    # ── Dataloader ────────────────────────────────────────────────────────────
    from bmfm_targets.tokenization import MultiFieldTokenizer
    from bmfm_methylation.shared.data_module import MethylationDataset, WCEDCollator

    tokenizer = MultiFieldTokenizer.from_pretrained(args.tokenizer)
    dataset   = MethylationDataset(h5ad_path=args.data, split=None, normalize_age=False)
    cpg_sites = dataset.cpg_sites
    vocab_size = len(cpg_sites)
    log.info(f"Dataset: {len(dataset)} samples × {vocab_size} CpGs")

    # input_ratio=1.0 so every CpG is an input — use all_betas for target reconstruction
    collator = WCEDCollator(
        tokenizer=tokenizer, cpg_sites=cpg_sites,
        vocab_size=vocab_size, input_ratio=1.0, contrastive=False,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collator,
                        shuffle=False, num_workers=4,
                        pin_memory=(device == "cuda"))

    # ── Run baselines ─────────────────────────────────────────────────────────
    log.info("Running reconstruction baselines ...")
    results = run_baselines(module, loader, device, n_batches=args.n_batches)
    summary = summarise(results)

    # ── Save results ──────────────────────────────────────────────────────────
    json_path = outdir / "reconstruction_baselines.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Saved → {json_path}")

    n = len(results["model_mse"])
    df = pd.DataFrame({
        "split":     results["splits"][:n],
        "label_age": results["labels"][:n],
        "model_mse": results["model_mse"],
        "b1_mse":    results["b1_mse"],
        "b3_mse":    results["b3_mse"],
        "b4_mse":    results["b4_mse"],
    })
    csv_path = outdir / "reconstruction_baselines.csv"
    df.to_csv(csv_path, index=False)
    log.info(f"Saved → {csv_path}")

    np.save(outdir / "cls_embeddings.npy", results["cls"])
    log.info(f"Saved CLS embeddings → {outdir / 'cls_embeddings.npy'}")

    log.info("Done.")


if __name__ == "__main__":
    main()
