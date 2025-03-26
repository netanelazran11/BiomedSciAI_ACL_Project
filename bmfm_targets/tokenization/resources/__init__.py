"""External resources related to tokenization."""

from .reference_data import get_hgnc_df, get_protein_coding_genes, get_ortholog_genes

__all__ = ["get_hgnc_df", "get_protein_coding_genes", "get_ortholog_genes"]
