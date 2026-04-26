"""WCED pretraining with InfoNCE contrastive regularisation between dual views."""

import logging

import torch

from bmfm_targets.config import TrainerConfig
from bmfm_targets.config.model_config import SCModelConfigBase
from bmfm_targets.tokenization import MultiFieldTokenizer
from bmfm_targets.training import metrics
from bmfm_targets.training.losses.contrastive import InfoNCELoss, ProjectionHead

from .sequence_labeling import SequenceLabelingTrainingModule

logger = logging.getLogger(__name__)


class WCEDContrastiveTrainingModule(SequenceLabelingTrainingModule):
    """WCED pretraining with optional InfoNCE contrastive regularisation.

    Extends :class:`SequenceLabelingTrainingModule` with a projection head
    and InfoNCE loss computed between two augmented views of each sample.

    When a batch contains ``input_ids_view2`` and ``attention_mask_view2``
    the module encodes both views and adds the contrastive term to the WCED
    reconstruction loss::

        loss = wced_loss + contrastive_weight * InfoNCE(z1, z2)

    When those keys are absent the module degrades to standard WCED
    pretraining, so existing pipelines remain unaffected.

    The two views are produced upstream by :class:`MultiFieldCollator` with
    ``contrastive=True``, which applies the sequence dropout transform twice
    independently, yielding different random gene subsets as input for each.
    """

    MODELING_STRATEGY = "sequence_labeling"

    def __init__(
        self,
        model_config: SCModelConfigBase,
        trainer_config: TrainerConfig,
        tokenizer: MultiFieldTokenizer | None = None,
        label_dict: dict[str, dict[str, int]] | None = None,
        contrastive_weight: float = 0.1,
        contrastive_temp: float = 0.1,
        projection_dim: int = 128,
        **kwargs,
    ):
        """
        Initialise WCEDContrastiveTrainingModule.

        Args:
        ----
            model_config: Model architecture configuration.
            trainer_config: Training hyperparameters and settings.
            tokenizer: Optional multi-field tokenizer.
            label_dict: Optional mapping of label column names to value dicts.
            contrastive_weight: Weight λ applied to the InfoNCE term.
                Set to ``0`` to disable contrastive learning and fall back to
                standard WCED pretraining.
            contrastive_temp: Temperature τ for the InfoNCE softmax.
            projection_dim: Output dimensionality of the projection head.
            **kwargs: Forwarded to :class:`BaseTrainingModule`.

        """
        super().__init__(model_config, trainer_config, tokenizer, label_dict, **kwargs)
        hidden_size = model_config.hidden_size
        self.projection_head = ProjectionHead(hidden_size, hidden_size // 2, projection_dim)
        self.contrastive_loss_fn = InfoNCELoss(temperature=contrastive_temp)
        self.contrastive_weight = contrastive_weight

    def _is_contrastive_batch(self, batch: dict) -> bool:
        """Return True when the batch contains a second view for contrastive loss."""
        return (
            self.contrastive_weight > 0
            and "input_ids_view2" in batch
            and "attention_mask_view2" in batch
        )

    def _compute_contrastive_loss(
        self,
        embeddings_v1: torch.Tensor,
        batch: dict,
    ) -> torch.Tensor:
        """Encode view 2 and return InfoNCE loss between the two projections.

        Args:
        ----
            embeddings_v1: CLS embeddings from view 1, shape ``(batch, hidden)``.
            batch: Batch dict containing ``input_ids_view2`` and
                ``attention_mask_view2``.

        Returns:
        -------
            Scalar contrastive loss.

        """
        outputs_v2 = self.model(
            input_ids=batch["input_ids_view2"],
            attention_mask=batch["attention_mask_view2"],
        )
        z1 = self.projection_head(embeddings_v1)
        z2 = self.projection_head(outputs_v2.embeddings)
        return self.contrastive_loss_fn(z1, z2)

    def training_step(self, batch, batch_idx):
        """Training step with optional contrastive loss when view 2 is present."""
        labels = batch["labels"]
        batch_size = batch["input_ids"].shape[0]

        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=labels,
        )
        all_losses = metrics.calculate_losses(self.loss_tasks, outputs.logits, labels)

        if self._is_contrastive_batch(batch):
            c_loss = self._compute_contrastive_loss(outputs.embeddings, batch)
            all_losses["contrastive_loss"] = c_loss
            all_losses["loss"] = all_losses["loss"] + self.contrastive_weight * c_loss

        loss = all_losses["loss"]
        step_metrics = self.update_metrics(labels, outputs, split="train")

        with torch.no_grad():
            self.log_losses(all_losses, batch_size, on_step=True, on_epoch=True, sync_dist=True)
            self.log_metrics(
                step_metrics, split="train", batch_size=batch_size, suffix="_step"
            )

        if loss != 0.0:
            return loss
        return None

    def validation_step(self, batch, batch_idx):
        """Validation step with optional contrastive loss when view 2 is present."""
        labels = batch["labels"]
        batch_size = batch["input_ids"].shape[0]

        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=labels,
        )
        all_losses = metrics.calculate_losses(self.loss_tasks, outputs.logits, labels)

        if self._is_contrastive_batch(batch):
            c_loss = self._compute_contrastive_loss(outputs.embeddings, batch)
            all_losses["contrastive_loss"] = c_loss
            all_losses["loss"] = all_losses["loss"] + self.contrastive_weight * c_loss

        loss = all_losses["loss"]
        self.update_metrics(labels, outputs, split="validation")

        with torch.no_grad():
            if self.trainer_config.batch_prediction_behavior:
                self.record_batch_predictions(batch, outputs, "validation")
            self.log_losses(
                all_losses,
                batch_size,
                split="validation",
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

        return loss

    def test_step(self, batch, batch_idx):
        """Test step with optional contrastive loss when view 2 is present."""
        labels = batch["labels"]
        batch_size = batch["input_ids"].shape[0]

        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=labels,
        )
        all_losses = metrics.calculate_losses(self.loss_tasks, outputs.logits, labels)

        if self._is_contrastive_batch(batch):
            c_loss = self._compute_contrastive_loss(outputs.embeddings, batch)
            all_losses["contrastive_loss"] = c_loss
            all_losses["loss"] = all_losses["loss"] + self.contrastive_weight * c_loss

        loss = all_losses["loss"]
        step_metrics = self.update_metrics(labels, outputs, split="test")

        with torch.no_grad():
            if self.trainer_config.batch_prediction_behavior:
                self.record_batch_predictions(batch, outputs, "test")
            self.log_metrics(step_metrics, split="test", batch_size=batch_size)
            self.log_losses(
                all_losses,
                batch_size,
                split="test",
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

        return loss
