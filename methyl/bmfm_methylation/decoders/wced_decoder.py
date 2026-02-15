"""
WCED (Whole Cell Expression Decoder) for Methylation Data

This decoder implements the WCED pretraining strategy:
- Takes [CLS] token embedding as input (global representation)
- Reconstructs ALL beta values from this single embedding
- Creates a global bottleneck that forces learning compressed representations

Unlike MLM which predicts only masked positions, WCED reconstructs the entire
methylation profile from a single 512-dim vector. This forces the [CLS] token
to aggregate information from all CpG sites.

Architecture:
    [CLS] embedding (512) → MLP decoder → All beta values (num_cpg_sites)

Reference: scGPT and BMFM-RNA use similar whole-cell decoder approaches.
"""

import torch
import torch.nn as nn
from typing import Optional


class WCEDDecoder(nn.Module):
    """
    WCED Decoder: Reconstructs all beta values from [CLS] embedding.

    This creates a global bottleneck by compressing the entire methylation
    profile into a single [CLS] vector, then reconstructing from it.

    Args:
        hidden_size: Input dimension from encoder (default: 512)
        num_cpg_sites: Number of CpG sites to reconstruct (default: 8000)
        decoder_hidden_sizes: Hidden layer sizes (default: [2048, 4096])
        dropout: Dropout probability (default: 0.1)
        use_layer_norm: Whether to use LayerNorm (default: True)

    Example:
        >>> decoder = WCEDDecoder(hidden_size=512, num_cpg_sites=2048)
        >>> cls_embedding = torch.randn(32, 512)  # [batch, hidden]
        >>> predicted_betas = decoder(cls_embedding)  # [batch, 2048]
    """

    def __init__(
        self,
        hidden_size: int = 512,
        num_cpg_sites: int = 8000,
        decoder_hidden_sizes: Optional[list] = None,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_cpg_sites = num_cpg_sites

        # Default architecture: 512 → 2048 → 4096 → num_cpg_sites
        if decoder_hidden_sizes is None:
            decoder_hidden_sizes = [2048, 4096]

        # Build decoder layers
        layers = []
        in_dim = hidden_size

        for out_dim in decoder_hidden_sizes:
            layers.append(nn.Linear(in_dim, out_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(out_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            in_dim = out_dim

        # Final projection to beta values
        layers.append(nn.Linear(in_dim, num_cpg_sites))
        # Sigmoid to ensure output is in [0, 1] range (valid beta values)
        layers.append(nn.Sigmoid())

        self.decoder = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize decoder weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, cls_embedding: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct beta values from [CLS] embedding.

        Args:
            cls_embedding: [batch_size, hidden_size] - [CLS] token representation

        Returns:
            predicted_betas: [batch_size, num_cpg_sites] - Reconstructed beta values (0-1)
        """
        return self.decoder(cls_embedding)


class WCEDLoss(nn.Module):
    """
    Loss function for WCED pretraining.

    Computes MSE loss between predicted and actual beta values,
    optionally weighted by a mask for valid positions.

    Args:
        reduction: Loss reduction mode ('mean', 'sum', 'none')
    """

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
        self.mse = nn.MSELoss(reduction='none')

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute WCED loss.

        Args:
            predictions: [batch, num_cpg_sites] - Predicted beta values
            targets: [batch, num_cpg_sites] - Actual beta values
            valid_mask: [batch, num_cpg_sites] - Optional mask (1=valid, 0=invalid)

        Returns:
            loss: Scalar loss value
        """
        # Compute per-element MSE
        mse_loss = self.mse(predictions, targets)

        # Apply valid mask if provided
        if valid_mask is not None:
            mse_loss = mse_loss * valid_mask

            if self.reduction == 'mean':
                # Mean over valid positions only
                return mse_loss.sum() / valid_mask.sum().clamp(min=1)
            elif self.reduction == 'sum':
                return mse_loss.sum()
            else:
                return mse_loss
        else:
            if self.reduction == 'mean':
                return mse_loss.mean()
            elif self.reduction == 'sum':
                return mse_loss.sum()
            else:
                return mse_loss


class WCEDWithPositionalDecoder(nn.Module):
    """
    WCED Decoder with CpG positional awareness.

    This variant incorporates CpG position information into the decoder,
    allowing it to learn position-specific reconstruction patterns.

    Instead of predicting all positions from a single vector, this adds
    positional queries that attend to the [CLS] representation.

    Args:
        hidden_size: Input dimension from encoder
        num_cpg_sites: Number of CpG sites to reconstruct
        num_heads: Number of attention heads
        dropout: Dropout probability
    """

    def __init__(
        self,
        hidden_size: int = 512,
        num_cpg_sites: int = 8000,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_cpg_sites = num_cpg_sites

        # Learnable position queries for each CpG site
        self.cpg_queries = nn.Parameter(torch.randn(1, num_cpg_sites, hidden_size))

        # Cross-attention: CpG queries attend to [CLS] embedding
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Projection to beta values
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize decoder weights."""
        nn.init.normal_(self.cpg_queries, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, cls_embedding: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct beta values using positional queries.

        Args:
            cls_embedding: [batch_size, hidden_size] - [CLS] representation

        Returns:
            predicted_betas: [batch_size, num_cpg_sites] - Reconstructed values
        """
        batch_size = cls_embedding.size(0)

        # Expand CpG queries for batch
        queries = self.cpg_queries.expand(batch_size, -1, -1)  # [B, num_cpg, H]

        # [CLS] as key/value (expand to sequence of 1)
        kv = cls_embedding.unsqueeze(1)  # [B, 1, H]

        # Cross-attention
        attended, _ = self.cross_attention(queries, kv, kv)  # [B, num_cpg, H]

        # Project to beta values
        predicted_betas = self.output_proj(attended).squeeze(-1)  # [B, num_cpg]

        return predicted_betas
