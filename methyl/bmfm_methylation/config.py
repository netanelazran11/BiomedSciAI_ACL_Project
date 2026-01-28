"""
Methylation Configuration - Wraps original BMFM SCBertConfig

This module provides configuration for the methylation encoder by wrapping
the original BMFM SCBertConfig with appropriate field definitions for
methylation data (CpG site IDs + beta values).
"""

from dataclasses import dataclass, field
from typing import Optional, List, Any

# Import original BMFM config
from bmfm_targets.config import SCBertConfig, FieldInfo


def create_methylation_config(
    num_cpg_sites: int = 8000,
    vocab_size: Optional[int] = None,
    num_hidden_layers: int = 6,
    num_attention_heads: int = 8,
    hidden_size: int = 512,
    intermediate_size: int = 2048,
    hidden_dropout_prob: float = 0.1,
    attention_probs_dropout_prob: float = 0.1,
    max_position_embeddings: int = 8010,
    position_embedding_type: str = "absolute",
    hidden_act: str = "gelu",
    layer_norm_eps: float = 1e-12,
    initializer_range: float = 0.02,
    classifier_pooling: str = "cls",
    **kwargs
) -> SCBertConfig:
    """
    Create an SCBertConfig configured for methylation data.

    This uses the ORIGINAL BMFM SCBertConfig with fields configured for:
    - CpG site IDs (discrete tokens)
    - Beta values (continuous values 0-1)

    Args:
        num_cpg_sites: Number of CpG sites in vocabulary
        vocab_size: Override vocab size (if None, uses num_cpg_sites + 5)
        num_hidden_layers: Number of transformer layers
        num_attention_heads: Number of attention heads
        hidden_size: Hidden dimension
        intermediate_size: FFN intermediate dimension
        hidden_dropout_prob: Dropout probability
        attention_probs_dropout_prob: Attention dropout
        max_position_embeddings: Maximum sequence length
        position_embedding_type: "absolute" or "sinusoidal"
        hidden_act: Activation function
        layer_norm_eps: Layer norm epsilon
        initializer_range: Weight initialization range
        classifier_pooling: "cls" or "mean"

    Returns:
        SCBertConfig configured for methylation
    """
    # Calculate vocab size
    actual_vocab_size = vocab_size if vocab_size is not None else num_cpg_sites + 5

    # Define fields for methylation data
    fields = [
        # CpG site IDs - discrete tokens
        FieldInfo(
            field_name="cpg_sites",
            vocab_size=actual_vocab_size,
            is_input=True,
            is_decode=False,
            tokenization_strategy="tokenize",
        ),
        # Beta values - continuous values
        FieldInfo(
            field_name="beta_values",
            is_input=True,
            is_decode=False,
            tokenization_strategy="continuous_value_encoder",
            encoder_kwargs={
                "kind": "mlp",  # Simple MLP encoder for methylation
            },
        ),
    ]

    config = SCBertConfig(
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        hidden_dropout_prob=hidden_dropout_prob,
        attention_probs_dropout_prob=attention_probs_dropout_prob,
        max_position_embeddings=max_position_embeddings,
        position_embedding_type=position_embedding_type,
        hidden_act=hidden_act,
        layer_norm_eps=layer_norm_eps,
        initializer_range=initializer_range,
        classifier_pooling=classifier_pooling,
        fields=fields,
        is_decoder=False,
        add_cross_attention=False,
        **kwargs
    )

    return config


# Backwards compatibility alias - can be called as BMFMConfig(...) or used as a type hint
BMFMConfig = create_methylation_config

# Re-export SCBertConfig for type hints
__all__ = ["create_methylation_config", "BMFMConfig", "SCBertConfig", "FieldInfo"]
