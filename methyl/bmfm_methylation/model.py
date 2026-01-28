"""
Methylation Age Prediction Model - Wraps original BMFM SCBertModel
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any

# Import original BMFM model
from bmfm_targets.models.predictive.scbert.modeling_scbert import SCBertModel
from bmfm_targets.config import SCBertConfig

from .config import create_methylation_config


class MethylationEncoder(nn.Module):
    """
    Wrapper around the original BMFM SCBertModel for methylation data.

    This uses the EXACT same encoder architecture as BMFM-RNA,
    just configured for methylation inputs.
    """

    def __init__(self, config: SCBertConfig, add_pooling_layer: bool = True):
        """
        Args:
            config: SCBertConfig configured for methylation (use create_methylation_config)
            add_pooling_layer: Whether to include the pooling layer
        """
        super().__init__()
        self.config = config

        # Use the ORIGINAL BMFM SCBertModel
        self.encoder = SCBertModel(config, add_pooling_layer=add_pooling_layer)

    def forward(
        self,
        cpg_ids: torch.Tensor,
        beta_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> Dict[str, Any]:
        """
        Forward pass through the BMFM encoder.

        Args:
            cpg_ids: [batch_size, seq_len] - CpG site token indices
            beta_values: [batch_size, seq_len] - methylation beta values (0-1)
            attention_mask: [batch_size, seq_len] - 1 for valid, 0 for padding
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return all hidden states

        Returns:
            Dict with last_hidden_state, pooler_output, etc.
        """
        batch_size, seq_len = cpg_ids.shape

        # Create combined input for BMFM format: [batch, num_fields, seq_len]
        # Field 0: cpg_ids (discrete tokens - will use embedding lookup)
        # Field 1: beta_values (continuous - will use ContinuousValueEncoder)
        input_ids = torch.zeros(batch_size, 2, seq_len, device=cpg_ids.device)
        input_ids[:, 0, :] = cpg_ids.float()  # CpG site IDs
        input_ids[:, 1, :] = beta_values.float()  # Beta values (continuous 0-1)

        # Forward through original BMFM SCBertModel encoder
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        return {
            "last_hidden_state": outputs.last_hidden_state,
            "pooler_output": outputs.pooler_output,
            "hidden_states": outputs.hidden_states,
            "attentions": outputs.attentions,
        }


class MethylationAgeModel(nn.Module):
    """
    Complete model for methylation-based age prediction.

    Uses the ORIGINAL BMFM SCBertModel encoder with a regression head.

    Example:
        >>> config = create_methylation_config(num_cpg_sites=8000)
        >>> model = MethylationAgeModel(config)
        >>> age_pred = model(cpg_ids, beta_values)
    """

    def __init__(
        self,
        config: Optional[SCBertConfig] = None,
        num_cpg_sites: int = 8000,
        head_hidden_size: int = 256,
        head_dropout: float = 0.1,
        **config_kwargs
    ):
        """
        Args:
            config: SCBertConfig for methylation (if None, creates default)
            num_cpg_sites: Number of CpG sites (used if config is None)
            head_hidden_size: Hidden size for regression head
            head_dropout: Dropout in regression head
            **config_kwargs: Additional args for create_methylation_config
        """
        super().__init__()

        # Create config if not provided
        if config is None:
            config = create_methylation_config(
                num_cpg_sites=num_cpg_sites,
                **config_kwargs
            )
        self.config = config

        # BMFM Encoder (original code)
        self.encoder = MethylationEncoder(config, add_pooling_layer=True)

        # Age prediction head
        self.age_head = nn.Sequential(
            nn.Linear(config.hidden_size, head_hidden_size),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden_size, head_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden_size // 2, 1)
        )

    def forward(
        self,
        cpg_ids: torch.Tensor,
        beta_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for age prediction.

        Args:
            cpg_ids: [batch_size, seq_len] - CpG site indices
            beta_values: [batch_size, seq_len] - methylation beta values (0-1)
            attention_mask: [batch_size, seq_len] - optional mask

        Returns:
            age_pred: [batch_size, 1] - predicted ages
        """
        # Encode with BMFM
        encoder_outputs = self.encoder(
            cpg_ids, beta_values, attention_mask
        )

        # Predict age from pooled output
        pooled = encoder_outputs["pooler_output"]
        age_pred = self.age_head(pooled)

        return age_pred

    def get_embeddings(
        self,
        cpg_ids: torch.Tensor,
        beta_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get embeddings without prediction head.

        Returns:
            embeddings: [batch_size, hidden_size]
        """
        encoder_outputs = self.encoder(
            cpg_ids, beta_values, attention_mask
        )
        return encoder_outputs["pooler_output"]
