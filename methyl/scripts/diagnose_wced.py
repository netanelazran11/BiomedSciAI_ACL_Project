#!/usr/bin/env python3
"""
WCED Diagnostic Script

Analyzes WHY CLS embeddings fail to encode sample-specific information.

Diagnostics:
1. CLS Embedding Variance: Do CLS embeddings differ across samples?
2. CLS-Age Correlation: Does CLS encode age-related information?
3. CLS Similarity: Are all CLS embeddings too similar?
4. Prediction Variance: Do predictions vary per sample or just predict averages?
5. Per-CpG Analysis: Is the model predicting per-CpG means?

Usage:
    python scripts/diagnose_wced.py --checkpoint /path/to/checkpoint.ckpt --data /path/to/data.h5ad
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import scanpy as sc
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bmfm_methylation.wced_module import WCEDTrainingModule
from bmfm_methylation.data_module import MethylationDataModule, WCEDCollator
from bmfm_methylation.tokenizer import create_methylation_multifield_tokenizer, extract_cpg_sites_from_h5ad
from bmfm_targets.config import FieldInfo

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model_and_data(checkpoint_path: str, data_path: str, vocab_size: int = 2048):
    """Load trained WCED model and prepare data."""

    # Load checkpoint
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Extract model config from checkpoint
    hparams = checkpoint.get('hyper_parameters', {})

    # Load data
    logger.info(f"Loading data: {data_path}")
    adata = sc.read_h5ad(data_path)

    # Create tokenizer
    cpg_sites = list(adata.var_names)
    tokenizer = create_methylation_multifield_tokenizer(cpg_sites=cpg_sites)

    # Create fields
    fields = [
        FieldInfo(name="cpg_sites", is_input=True, vocab_size=len(cpg_sites) + 10),
        FieldInfo(name="beta_values", is_input=True, vocab_size=1),
    ]

    # Create data module
    data_module = MethylationDataModule(
        tokenizer=tokenizer,
        fields=fields,
        h5ad_path=data_path,
        batch_size=32,
        num_workers=0,
        subset_k=vocab_size,
        mlm=False,
    )
    data_module.setup()

    # Create collator (non-contrastive for inference)
    collator = WCEDCollator(
        tokenizer=tokenizer,
        cpg_sites=cpg_sites,
        vocab_size=vocab_size,
        input_ratio=0.8,  # Use 80% as input for diagnosis
        contrastive=False,
    )

    return checkpoint, data_module, collator, adata


def extract_cls_embeddings(model, data_module, collator, device='cpu', max_samples=500):
    """Extract CLS embeddings for all samples."""

    model.eval()
    model.to(device)

    cls_embeddings = []
    predictions_all = []
    targets_all = []
    ages = []

    dataset = data_module.val_dataset or data_module.train_dataset

    logger.info(f"Extracting CLS embeddings for {min(len(dataset), max_samples)} samples...")

    with torch.no_grad():
        for i in range(min(len(dataset), max_samples)):
            example = dataset[i]
            batch = collator([example])

            # Move to device
            cpg_ids = batch['cpg_ids'].to(device)
            beta_values = batch['beta_values'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            all_betas = batch['all_betas'].to(device)

            # Forward pass
            outputs = model(cpg_ids, beta_values, attention_mask)

            cls_emb = outputs['cls_embedding'].cpu().numpy()
            pred = outputs['predicted_betas'].cpu().numpy()
            target = all_betas.cpu().numpy()

            cls_embeddings.append(cls_emb[0])
            predictions_all.append(pred[0])
            targets_all.append(target[0])

            # Get age
            if example.metadata and 'labels' in example.metadata:
                ages.append(example.metadata['labels'])
            else:
                ages.append(0.0)

    cls_embeddings = np.array(cls_embeddings)
    predictions_all = np.array(predictions_all)
    targets_all = np.array(targets_all)
    ages = np.array(ages)

    return cls_embeddings, predictions_all, targets_all, ages


def analyze_cls_variance(cls_embeddings):
    """Analyze variance in CLS embeddings."""

    print("\n" + "="*70)
    print("DIAGNOSTIC 1: CLS Embedding Variance")
    print("="*70)

    # Per-dimension variance
    var_per_dim = np.var(cls_embeddings, axis=0)
    mean_var = np.mean(var_per_dim)
    max_var = np.max(var_per_dim)
    min_var = np.min(var_per_dim)

    print(f"CLS embedding shape: {cls_embeddings.shape}")
    print(f"Variance per dimension: mean={mean_var:.6f}, max={max_var:.6f}, min={min_var:.6f}")

    # Total variance (Frobenius norm style)
    total_var = np.sum(var_per_dim)
    print(f"Total variance (sum across dims): {total_var:.6f}")

    # Compare to mean embedding
    mean_cls = np.mean(cls_embeddings, axis=0)
    distances_from_mean = np.linalg.norm(cls_embeddings - mean_cls, axis=1)

    print(f"Distance from mean CLS: mean={np.mean(distances_from_mean):.6f}, "
          f"std={np.std(distances_from_mean):.6f}")

    # Pairwise similarities
    from sklearn.metrics.pairwise import cosine_similarity
    sim_matrix = cosine_similarity(cls_embeddings)
    upper_tri = sim_matrix[np.triu_indices(len(cls_embeddings), k=1)]

    print(f"Pairwise CLS cosine similarity: mean={np.mean(upper_tri):.6f}, "
          f"std={np.std(upper_tri):.6f}, min={np.min(upper_tri):.6f}, max={np.max(upper_tri):.6f}")

    if np.mean(upper_tri) > 0.99:
        print("\n⚠️  WARNING: CLS embeddings are NEARLY IDENTICAL across samples!")
        print("   This explains why WCED fails - CLS doesn't encode sample-specific info.")
    elif np.mean(upper_tri) > 0.95:
        print("\n⚠️  WARNING: CLS embeddings are VERY SIMILAR across samples.")
    else:
        print("\n✓ CLS embeddings show reasonable variation across samples.")

    return var_per_dim, distances_from_mean, upper_tri


def analyze_cls_age_correlation(cls_embeddings, ages):
    """Check if CLS embeddings correlate with age."""

    print("\n" + "="*70)
    print("DIAGNOSTIC 2: CLS-Age Correlation")
    print("="*70)

    # PCA on CLS embeddings
    pca = PCA(n_components=min(10, cls_embeddings.shape[1]))
    cls_pca = pca.fit_transform(cls_embeddings)

    print(f"PCA explained variance (top 10): {pca.explained_variance_ratio_[:10]}")

    # Correlation of each PC with age
    print("\nCorrelation of principal components with age:")
    pc_age_corrs = []
    for i in range(min(10, cls_pca.shape[1])):
        corr, p_val = pearsonr(cls_pca[:, i], ages)
        pc_age_corrs.append(corr)
        print(f"  PC{i+1}: r={corr:.4f}, p={p_val:.4e}")

    # Best linear combination for age prediction
    from sklearn.linear_model import Ridge
    ridge = Ridge(alpha=1.0)
    ridge.fit(cls_embeddings, ages)
    age_pred = ridge.predict(cls_embeddings)

    overall_corr, _ = pearsonr(age_pred, ages)
    print(f"\nBest linear prediction of age from CLS: r={overall_corr:.4f}")

    if overall_corr < 0.3:
        print("\n⚠️  WARNING: CLS embeddings have LOW correlation with age!")
        print("   CLS is not encoding age-relevant information.")
    elif overall_corr < 0.6:
        print("\n⚠️  CLS shows MODERATE correlation with age.")
    else:
        print("\n✓ CLS shows GOOD correlation with age.")

    return pc_age_corrs, overall_corr


def analyze_predictions(predictions, targets):
    """Analyze if predictions vary per sample or just predict averages."""

    print("\n" + "="*70)
    print("DIAGNOSTIC 3: Prediction Analysis")
    print("="*70)

    # Per-CpG mean across samples
    cpg_means = np.mean(targets, axis=0)  # Ground truth means
    pred_means = np.mean(predictions, axis=0)  # Predicted means

    # Per-sample PCC
    sample_pccs = []
    for i in range(len(predictions)):
        pcc, _ = pearsonr(predictions[i], targets[i])
        sample_pccs.append(pcc)

    print(f"Per-sample PCC: mean={np.mean(sample_pccs):.4f}, std={np.std(sample_pccs):.4f}")

    # Check if predictions are just the per-CpG means
    mean_pred_pccs = []
    for i in range(len(predictions)):
        # How well do per-CpG means predict this sample?
        pcc_mean, _ = pearsonr(cpg_means, targets[i])
        mean_pred_pccs.append(pcc_mean)

    print(f"Per-CpG mean baseline PCC: mean={np.mean(mean_pred_pccs):.4f}")

    # Variance in predictions
    pred_var_per_cpg = np.var(predictions, axis=0)  # Variance across samples for each CpG
    target_var_per_cpg = np.var(targets, axis=0)

    var_ratio = np.mean(pred_var_per_cpg) / np.mean(target_var_per_cpg)
    print(f"Prediction variance / Target variance: {var_ratio:.4f}")

    if var_ratio < 0.1:
        print("\n⚠️  WARNING: Predictions have MUCH LESS variance than targets!")
        print("   Model is predicting near-constant values per CpG (averages).")
    elif var_ratio < 0.5:
        print("\n⚠️  Predictions have lower variance than targets.")
    else:
        print("\n✓ Predictions show reasonable variance.")

    # Check if model output ≈ per-CpG means
    pred_vs_cpgmean_corr, _ = pearsonr(pred_means, cpg_means)
    print(f"\nPredicted means vs Ground truth means correlation: r={pred_vs_cpgmean_corr:.4f}")

    # Per-sample deviation from mean
    print("\nSample-specific signal analysis:")
    deviations_pred = predictions - pred_means  # Deviation from predicted mean
    deviations_true = targets - cpg_means  # Deviation from true mean

    # How well do predicted deviations match true deviations?
    dev_corrs = []
    for i in range(len(predictions)):
        corr, _ = pearsonr(deviations_pred[i], deviations_true[i])
        dev_corrs.append(corr)

    print(f"Deviation correlation: mean={np.mean(dev_corrs):.4f}, std={np.std(dev_corrs):.4f}")

    if np.mean(dev_corrs) < 0.1:
        print("\n⚠️  WARNING: Model fails to capture sample-specific deviations!")
        print("   This confirms the model just predicts per-CpG averages.")

    return sample_pccs, mean_pred_pccs, var_ratio


def visualize_cls_embeddings(cls_embeddings, ages, output_dir):
    """Create visualizations of CLS embeddings."""

    print("\n" + "="*70)
    print("Creating Visualizations...")
    print("="*70)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # PCA visualization
    pca = PCA(n_components=2)
    cls_pca = pca.fit_transform(cls_embeddings)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # PCA colored by age
    scatter = axes[0].scatter(cls_pca[:, 0], cls_pca[:, 1], c=ages, cmap='viridis', alpha=0.7)
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')
    axes[0].set_title('CLS Embeddings (PCA) colored by Age')
    plt.colorbar(scatter, ax=axes[0], label='Age')

    # t-SNE
    if len(cls_embeddings) > 30:
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(cls_embeddings)-1))
        cls_tsne = tsne.fit_transform(cls_embeddings)

        scatter = axes[1].scatter(cls_tsne[:, 0], cls_tsne[:, 1], c=ages, cmap='viridis', alpha=0.7)
        axes[1].set_xlabel('t-SNE 1')
        axes[1].set_ylabel('t-SNE 2')
        axes[1].set_title('CLS Embeddings (t-SNE) colored by Age')
        plt.colorbar(scatter, ax=axes[1], label='Age')

    plt.tight_layout()
    plt.savefig(output_dir / 'cls_embeddings_visualization.png', dpi=150)
    print(f"Saved: {output_dir / 'cls_embeddings_visualization.png'}")

    # Histogram of pairwise similarities
    from sklearn.metrics.pairwise import cosine_similarity
    sim_matrix = cosine_similarity(cls_embeddings)
    upper_tri = sim_matrix[np.triu_indices(len(cls_embeddings), k=1)]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(upper_tri, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(upper_tri), color='red', linestyle='--', label=f'Mean: {np.mean(upper_tri):.4f}')
    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Count')
    ax.set_title('Pairwise CLS Embedding Similarities')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'cls_similarity_histogram.png', dpi=150)
    print(f"Saved: {output_dir / 'cls_similarity_histogram.png'}")


def main():
    parser = argparse.ArgumentParser(description='Diagnose WCED CLS embeddings')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to WCED checkpoint')
    parser.add_argument('--data', type=str, required=True, help='Path to h5ad data file')
    parser.add_argument('--output', type=str, default='./wced_diagnosis', help='Output directory')
    parser.add_argument('--vocab_size', type=int, default=2048, help='Vocabulary size')
    parser.add_argument('--max_samples', type=int, default=500, help='Max samples to analyze')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda)')

    args = parser.parse_args()

    print("="*70)
    print("WCED DIAGNOSTIC ANALYSIS")
    print("="*70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data: {args.data}")
    print(f"Vocab size: {args.vocab_size}")
    print("="*70)

    # Load model and data
    checkpoint, data_module, collator, adata = load_model_and_data(
        args.checkpoint, args.data, args.vocab_size
    )

    # Reconstruct model from checkpoint
    # This requires knowing the model config - we'll extract from checkpoint
    from bmfm_targets.config import SCBertConfig
    from bmfm_methylation.config import PretrainingConfig

    hparams = checkpoint.get('hyper_parameters', {})
    model_config = checkpoint.get('model_config', None)

    if model_config is None:
        # Create default config
        model_config = SCBertConfig(
            num_hidden_layers=6,
            num_attention_heads=8,
            hidden_size=512,
            intermediate_size=2048,
        )

    pretrain_config = PretrainingConfig(mode="wced", decoder_dropout=0.1)

    model = WCEDTrainingModule(
        model_config=model_config,
        pretrain_config=pretrain_config,
        vocab_size=args.vocab_size,
    )

    # Load state dict
    model.load_state_dict(checkpoint['state_dict'], strict=False)

    # Extract CLS embeddings
    cls_embeddings, predictions, targets, ages = extract_cls_embeddings(
        model, data_module, collator, device=args.device, max_samples=args.max_samples
    )

    # Run diagnostics
    analyze_cls_variance(cls_embeddings)
    analyze_cls_age_correlation(cls_embeddings, ages)
    analyze_predictions(predictions, targets)

    # Create visualizations
    visualize_cls_embeddings(cls_embeddings, ages, args.output)

    print("\n" + "="*70)
    print("DIAGNOSIS COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {args.output}")
    print("\nKey questions answered:")
    print("1. Do CLS embeddings vary across samples?")
    print("2. Do CLS embeddings correlate with age?")
    print("3. Does the model predict sample-specific values or just averages?")


if __name__ == "__main__":
    main()
