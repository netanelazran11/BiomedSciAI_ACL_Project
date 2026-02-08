"""
Methylation Tokenizer - Creates tokenizer for CpG sites

This module creates a tokenizer compatible with BMFM's MultiFieldTokenizer
for methylation data (CpG site IDs).
"""

import os
import json
import tempfile
from pathlib import Path
from typing import List, Optional, Dict, Any

from transformers.models.bert import BertTokenizerFast


# Special tokens matching BMFM conventions
SPECIAL_TOKENS = {
    "pad_token": "[PAD]",
    "unk_token": "[UNK]",
    "cls_token": "[CLS]",
    "sep_token": "[SEP]",
    "mask_token": "[MASK]",
}


def create_cpg_vocabulary(
    cpg_sites: List[str],
    output_dir: str,
    add_special_tokens: bool = True,
) -> str:
    """
    Create a vocabulary file for CpG sites.

    Args:
        cpg_sites: List of CpG site names (e.g., ["cg00000029", "cg00000108", ...])
        output_dir: Directory to save the vocabulary
        add_special_tokens: Whether to add special tokens

    Returns:
        Path to the created vocabulary file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vocab = []

    # Add special tokens first in BMFM order:
    # 0: [UNK], 1: [SEP], 2: [PAD], 3: [CLS], 4: [MASK]
    if add_special_tokens:
        vocab.extend(
            [
                SPECIAL_TOKENS["unk_token"],  # ID 0
                SPECIAL_TOKENS["sep_token"],  # ID 1
                SPECIAL_TOKENS["pad_token"],  # ID 2
                SPECIAL_TOKENS["cls_token"],  # ID 3
                SPECIAL_TOKENS["mask_token"], # ID 4
            ]
        )

    # Add CpG sites
    vocab.extend(cpg_sites)

    # Write vocabulary file
    vocab_file = output_dir / "vocab.txt"
    with open(vocab_file, "w") as f:
        f.write("\n".join(vocab) + "\n")

    return str(vocab_file)


def create_methylation_tokenizer(
    cpg_sites: List[str],
    output_dir: str,
    tokenizer_name: str = "cpg_sites",
) -> BertTokenizerFast:
    """
    Create a BertTokenizerFast for CpG sites.

    Args:
        cpg_sites: List of CpG site names
        output_dir: Directory to save the tokenizer
        tokenizer_name: Name for the tokenizer subdirectory

    Returns:
        BertTokenizerFast tokenizer
    """
    output_dir = Path(output_dir)
    tokenizer_dir = output_dir / "tokenizers" / tokenizer_name
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    # Create vocabulary
    vocab_file = create_cpg_vocabulary(cpg_sites, tokenizer_dir)

    # Create tokenizer
    tokenizer = BertTokenizerFast(
        vocab_file=vocab_file,
        do_lower_case=False,
        tokenize_chinese_chars=False,
        clean_text=False,
        strip_accents=False,
        **SPECIAL_TOKENS,
    )

    # Save tokenizer
    tokenizer.save_pretrained(tokenizer_dir)

    return tokenizer


def create_methylation_multifield_tokenizer(
    cpg_sites: List[str],
    output_dir: str,
) -> "MultiFieldTokenizer":
    """
    Create a MultiFieldTokenizer for methylation data.

    This creates a tokenizer with a single discrete field:
    - cpg_sites: discrete tokens for CpG site IDs

    Continuous beta values are handled by the collator/encoder (no HF tokenizer).

    Args:
        cpg_sites: List of CpG site names
        output_dir: Directory to save the tokenizer

    Returns:
        MultiFieldTokenizer instance
    """
    from bmfm_targets.tokenization import MultiFieldTokenizer

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create CpG sites tokenizer only
    create_methylation_tokenizer(
        cpg_sites=cpg_sites,
        output_dir=output_dir,
        tokenizer_name="cpg_sites",
    )

    # Create MultiFieldTokenizer
    multifield_tokenizer = MultiFieldTokenizer.from_pretrained(
        name_or_path=str(output_dir),
    )

    tok = multifield_tokenizer.tokenizers["cpg_sites"]
    assert tok.unk_token_id == 0, f"UNK id mismatch: {tok.unk_token_id}"
    assert tok.sep_token_id == 1, f"SEP id mismatch: {tok.sep_token_id}"
    assert tok.pad_token_id == 2, f"PAD id mismatch: {tok.pad_token_id}"
    assert tok.cls_token_id == 3, f"CLS id mismatch: {tok.cls_token_id}"
    assert tok.mask_token_id == 4, f"MASK id mismatch: {tok.mask_token_id}"

    return multifield_tokenizer


def extract_cpg_sites_from_h5ad(h5ad_path: str) -> List[str]:
    """
    Extract CpG site names from an h5ad file.

    Args:
        h5ad_path: Path to h5ad file

    Returns:
        List of CpG site names from var_names
    """
    import scanpy as sc

    adata = sc.read_h5ad(h5ad_path)
    cpg_sites = list(adata.var_names)

    return cpg_sites


def create_tokenizer_from_h5ad(
    h5ad_path: str,
    output_dir: str,
) -> "MultiFieldTokenizer":
    """
    Create a MultiFieldTokenizer from an h5ad file.

    Args:
        h5ad_path: Path to h5ad file
        output_dir: Directory to save the tokenizer

    Returns:
        MultiFieldTokenizer instance
    """
    # Extract CpG sites from h5ad
    cpg_sites = extract_cpg_sites_from_h5ad(h5ad_path)

    print(f"Found {len(cpg_sites)} CpG sites in {h5ad_path}")

    # Create tokenizer
    tokenizer = create_methylation_multifield_tokenizer(
        cpg_sites=cpg_sites,
        output_dir=output_dir,
    )

    print(f"Tokenizer saved to {output_dir}")

    return tokenizer


# For simple indexed CpG sites (0, 1, 2, ..., N-1)
def create_indexed_tokenizer(
    num_cpg_sites: int,
    output_dir: str,
) -> "MultiFieldTokenizer":
    """
    Create a tokenizer for indexed CpG sites (0, 1, 2, ..., N-1).

    Use this when your data uses integer indices instead of CpG names.

    Args:
        num_cpg_sites: Number of CpG sites
        output_dir: Directory to save the tokenizer

    Returns:
        MultiFieldTokenizer instance
    """
    # Create indexed CpG site names
    cpg_sites = [str(i) for i in range(num_cpg_sites)]

    return create_methylation_multifield_tokenizer(
        cpg_sites=cpg_sites,
        output_dir=output_dir,
    )
