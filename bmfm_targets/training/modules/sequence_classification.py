from bmfm_targets.training.modules.base import BaseTrainingModule


class SequenceClassificationTrainingModule(BaseTrainingModule):
    MODELING_STRATEGY = "sequence_classification"
