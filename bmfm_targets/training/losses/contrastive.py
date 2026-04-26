"""Contrastive learning building blocks for self-supervised pretraining."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """SimCLR-style MLP projection head.

    Maps encoder embeddings to a lower-dimensional space in which the
    contrastive loss is computed. The output is L2-normalised so that
    dot products equal cosine similarities.

    Args:
    ----
        input_dim: Dimensionality of the encoder output (e.g. hidden_size).
        hidden_dim: Width of the single hidden layer.
        output_dim: Dimensionality of the projected space.

    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project and L2-normalise a batch of embeddings.

        Args:
        ----
            x: Input tensor of shape ``(batch, input_dim)``.

        Returns:
        -------
            Normalised projections of shape ``(batch, output_dim)``.

        """
        return F.normalize(self.net(x), dim=-1)


class InfoNCELoss(nn.Module):
    """Symmetric InfoNCE contrastive loss for dual-view self-supervised learning.

    Given two sets of L2-normalised embeddings ``z1`` and ``z2`` derived
    from two augmented views of the same samples, the loss maximises
    agreement between same-sample pairs (diagonal) and minimises it for
    all cross-sample pairs (off-diagonal).

    Args:
    ----
        temperature: Softmax temperature τ.  Smaller values produce
            sharper distributions and harder negatives.

    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Compute symmetric InfoNCE loss.

        Args:
        ----
            z1: Normalised projections from view 1, shape ``(batch, dim)``.
            z2: Normalised projections from view 2, shape ``(batch, dim)``.

        Returns:
        -------
            Scalar loss averaged over both directions.

        """
        batch_size = z1.shape[0]
        sim = torch.matmul(z1, z2.T) / self.temperature  # (batch, batch)
        labels = torch.arange(batch_size, device=z1.device)
        return (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2
