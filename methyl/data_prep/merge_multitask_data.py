#!/usr/bin/env python3
"""
Merge age h5ad and smoking h5ad into a single multi-task h5ad.

The age h5ad defines the pretraining CpG vocabulary (8k CpGs, specific token IDs).
The smoking h5ad (482k CpGs with sequential IDs) is filtered to the vocab intersection
and its cpg_ids are remapped to match the pretraining token IDs.

Outputs:
  --output_merged:  combined h5ad for multi-task training (age + smoking samples)
  --output_aligned: smoking h5ad filtered + remapped to pretraining vocab (for Task A/C)

Usage:
  python data_prep/merge_multitask_data.py \
      --age_h5ad /path/to/age_data.h5ad \
      --smoking_h5ad /path/to/smoking_geo/smoking_data.h5ad \
      --output_merged /path/to/merged/multitask_data.h5ad \
      --output_aligned /path/to/smoking_geo/smoking_data_aligned.h5ad
"""

import argparse
import logging
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def load_and_align_smoking(smk_adata: ad.AnnData, age_adata: ad.AnnData) -> ad.AnnData:
    """Filter smoking h5ad to pretraining vocab CpGs and remap token IDs."""
    # Build vocab: cpg_name -> pretraining token_id
    vocab = age_adata.var["cpg_id"].to_dict()  # {cpg_name: int}

    common_cpgs = [c for c in smk_adata.var_names if c in vocab]
    logger.info(
        f"CpG intersection: {len(common_cpgs)} "
        f"(age vocab={len(age_adata.var)}, smoking={len(smk_adata.var)})"
    )
    if len(common_cpgs) == 0:
        raise ValueError(
            "No CpG overlap between age and smoking datasets. "
            "Check that both use the same probe naming (e.g. cg00000029)."
        )

    smk_aligned = smk_adata[:, common_cpgs].copy()

    # Remap cpg_ids to pretraining token IDs
    smk_aligned.var["cpg_id"] = [vocab[c] for c in common_cpgs]
    logger.info(f"Aligned smoking dataset: {smk_aligned.shape}")
    return smk_aligned


def main():
    parser = argparse.ArgumentParser(description="Merge age + smoking h5ad for multi-task training")
    parser.add_argument("--age_h5ad", required=True, help="Age h5ad with pretraining vocab CpGs")
    parser.add_argument("--smoking_h5ad", required=True, help="Raw smoking h5ad (482k CpGs)")
    parser.add_argument("--output_merged", required=True, help="Output path for merged multi-task h5ad")
    parser.add_argument("--output_aligned", default=None,
                        help="Optional: save aligned smoking h5ad for Task A/C")
    parser.add_argument("--age_col", default="age")
    parser.add_argument("--sex_col", default="sex")
    args = parser.parse_args()

    logger.info("Loading age h5ad...")
    age_adata = ad.read_h5ad(args.age_h5ad)
    logger.info(f"Age dataset: {age_adata.shape}, obs cols: {list(age_adata.obs.columns)}")

    logger.info("Loading smoking h5ad...")
    smk_adata = ad.read_h5ad(args.smoking_h5ad)
    logger.info(f"Smoking dataset: {smk_adata.shape}, obs cols: {list(smk_adata.obs.columns)}")

    smk_aligned = load_and_align_smoking(smk_adata, age_adata)

    if args.output_aligned:
        out_aligned = Path(args.output_aligned)
        out_aligned.parent.mkdir(parents=True, exist_ok=True)
        smk_aligned.write_h5ad(out_aligned)
        logger.info(f"Saved aligned smoking h5ad: {out_aligned}")

    # ── Build obs for age samples ──────────────────────────────────────────────
    age_obs = pd.DataFrame(index=age_adata.obs_names)
    age_obs["age"] = (
        age_adata.obs[args.age_col].values
        if args.age_col in age_adata.obs.columns
        else np.nan
    )
    age_obs["smoking_status"] = np.nan
    age_obs["sex"] = (
        age_adata.obs[args.sex_col].values
        if args.sex_col in age_adata.obs.columns
        else np.nan
    )
    age_obs["split"] = age_adata.obs["split"].values
    age_obs["source"] = "age"

    # ── Build obs for smoking samples ─────────────────────────────────────────
    smk_obs = pd.DataFrame(index=smk_aligned.obs_names)
    smk_obs["age"] = np.nan
    smk_obs["smoking_status"] = smk_aligned.obs["smoking_status"].values
    smk_obs["sex"] = np.nan
    smk_obs["split"] = smk_aligned.obs["split"].values
    smk_obs["source"] = "smoking"

    # ── Align CpGs of age_adata to the intersection ───────────────────────────
    common_cpgs = list(smk_aligned.var_names)
    age_aligned = age_adata[:, common_cpgs].copy()

    X_age = age_aligned.X if not hasattr(age_aligned.X, "toarray") else age_aligned.X.toarray()
    X_smk = smk_aligned.X if not hasattr(smk_aligned.X, "toarray") else smk_aligned.X.toarray()
    combined_X = np.vstack([X_age, X_smk]).astype(np.float32)

    combined_obs = pd.concat([age_obs, smk_obs], axis=0)
    var = age_aligned.var[["cpg_id"]].copy()

    combined = ad.AnnData(X=combined_X, obs=combined_obs, var=var)
    logger.info(f"Merged dataset: {combined.shape}")
    logger.info(f"Split distribution:\n{combined.obs['split'].value_counts().to_dict()}")
    logger.info(f"Source distribution:\n{combined.obs['source'].value_counts().to_dict()}")

    out_merged = Path(args.output_merged)
    out_merged.parent.mkdir(parents=True, exist_ok=True)
    combined.write_h5ad(out_merged)
    logger.info(f"Saved merged h5ad: {out_merged}")


if __name__ == "__main__":
    main()
