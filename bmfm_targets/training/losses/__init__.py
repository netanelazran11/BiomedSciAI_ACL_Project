"""Loss functions for self-supervised and contrastive pretraining."""

from .contrastive import InfoNCELoss, ProjectionHead

__all__ = ["InfoNCELoss", "ProjectionHead"]
