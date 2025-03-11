import logging
import re
from collections.abc import Mapping

import pytorch_lightning as pl
import torch

from bmfm_targets.config import FieldInfo
from bmfm_targets.tokenization import MultiFieldTokenizer

logger = logging.getLogger(__name__)


class MaskingStrategy:
    def __init__(
        self,
        tokenizer: MultiFieldTokenizer,
        lookup_field_name="genes",
        pattern_weights: list[tuple[str, float]] | None = None,
        temperature: float | None = None,
        should_update_from_errors: bool = False,
        callback: pl.Callback | None = None,
        use_for_validation: bool = True,
    ):
        self.tokenizer = tokenizer
        self.lookup_field_name = lookup_field_name
        fv = tokenizer.get_field_vocab(self.lookup_field_name)
        self.token_masking_probs = torch.ones(size=(len(fv),))
        self.pattern_weights = pattern_weights
        self.temperature = temperature
        self.should_update_from_errors = should_update_from_errors
        self.use_for_validation = use_for_validation
        if callback is None and self.should_update_from_errors:
            # to avoid circular import
            from bmfm_targets.training.callbacks import TokenErrorUpdateCallback

            callback = TokenErrorUpdateCallback()
        self.callback = callback
        if self.pattern_weights:
            self.update_token_probs_from_pattern_weights()

    def get_mask_probs(self, batch):
        return self.token_masking_probs[batch[self.lookup_field_name]["input_ids"]]

    def update_token_probs_from_pattern_weights(self):
        assert self.pattern_weights is not None
        vocab = self.tokenizer.get_field_vocab(self.lookup_field_name)
        for regex, weight in self.pattern_weights:
            matching_ids = self.find_ids_matching_pattern(vocab, regex)
            self.token_masking_probs[matching_ids] *= weight

    @staticmethod
    def find_ids_matching_pattern(vocab, regex):
        pattern = re.compile(regex)
        matching_ids = [tok_id for tok, tok_id in vocab.items() if pattern.match(tok)]
        return matching_ids

    def get_trainer_callback(self):
        return self.callback

    def update_token_masking_probs(self, token_probs: dict[str, float]):
        if not self.should_update_from_errors:
            logger.warning(
                "Updating token masking probs on object that does not expect it."
                " Probably indicates misconfiguration."
            )
        ft = self.tokenizer.get_field_tokenizer(self.lookup_field_name)
        ft.backend_tokenizer.pre_tokenizer = None
        assert ft.backend_tokenizer.pre_tokenizer is None
        token_ids = ft(
            [*token_probs.keys()],
            is_split_into_words=True,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            return_tensors="pt",
        )["input_ids"].squeeze()
        token_prob_vec = torch.tensor([*token_probs.values()])
        if self.temperature is not None:
            token_prob_vec = torch.functional.F.softmax(
                token_prob_vec / self.temperature, dim=0
            )
        self.token_masking_probs[token_ids] = token_prob_vec


class Masker:
    def __init__(
        self,
        change_ratio: float,
        mask_ratio: float,
        switch_ratio: float,
        tokenizer: MultiFieldTokenizer,
        masking_strategy: MaskingStrategy | None = None,
    ):
        self.change_ratio = change_ratio
        self.mask_ratio = mask_ratio
        self.switch_ratio = switch_ratio
        self.tokenizer = tokenizer
        self.masking_strategy = masking_strategy
        self.__post_init__()

    def __post_init__(self):
        if self.change_ratio < 0 or self.change_ratio > 1:
            raise ValueError(
                "Change ratio should be between 0 and 1. 0 means no change and 1 means all tokens are changed"
            )
        if self.mask_ratio < 0 or self.mask_ratio > 1:
            raise ValueError("Mask ratio should be between 0 and 1")

        if self.switch_ratio < 0 or self.switch_ratio > 1:
            raise ValueError("Switch ratio should be between 0 and 1")

        if self.mask_ratio + self.switch_ratio > 1:
            raise ValueError(
                "Mask ratio + Switch ratio should be less than or equal to 1"
            )

    def mask_inputs(
        self,
        fields: list[FieldInfo],
        batch: dict[str, Mapping[str, torch.Tensor]],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """

        Mask inputs from all `mask_field_names`s for language modeling.


        Args:
        ----
            fields (list[FieldInfo]): all fields for the model
            batch (dict[str, Mapping[str, torch.Tensor]]): batch of fields and input_ids

        Returns:
        -------
            tuple[torch.Tensor, dict[str, torch.Tensor]]: input_ids and labels for all fields

        """
        input_fields = [f for f in fields if f.is_input]
        masked_fields = [f for f in input_fields if f.is_masked]

        labels = {}
        for field in masked_fields:
            mask_probs = self.get_mask_probs(batch, masked_fields)
            _, field_labels = self.mask_single_field(
                field,
                batch[field.field_name],
                mask_probs,
                batch.get(f"label_{field.field_name}", None),
            )
            labels[field.field_name] = field_labels
        field_input_ids = [
            batch[field.field_name]["input_ids"] for field in input_fields
        ]
        input_ids = torch.stack(field_input_ids, dim=1)
        return input_ids, labels

    def mask_single_field(
        self,
        field: FieldInfo,
        field_encoding: Mapping[str, torch.Tensor],
        mask_probs: torch.Tensor | None = None,
        label_encoding: Mapping[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Mask tensor for a single field with non-uniform masking probabilities.

        Args:
        ----
            field (FieldInfo): field config object
            field_encoding (Mapping[str, torch.Tensor]): encoding for field
            mask_probs (torch.Tensor | None): probabilities for masking specific tokens
            label_encoding (Mapping[str, torch.Tensor] | None, optional):
            labels other than input to use. Defaults to None, in which case the labels are
            the inputs (normal MLM behavior).

        Returns:
        -------
            tuple[torch.Tensor, torch.Tensor]: input_ids and label_ids after masking
        """
        input_ids = field_encoding["input_ids"]
        if label_encoding is not None:
            label_ids = label_encoding["input_ids"].clone().detach()
        else:
            label_ids = input_ids.clone().detach()
        special_tokens_mask = field_encoding["special_tokens_mask"]

        random_tensor = torch.rand_like(
            label_ids, layout=torch.strided, dtype=torch.float, device=input_ids.device
        )

        if mask_probs is None:
            mask_probs = torch.full_like(random_tensor, fill_value=self.change_ratio)
        else:
            # Normalize probabilities so that on average the right number are changed
            mask_probs *= self.change_ratio / mask_probs.mean()
            mask_probs = torch.clamp(mask_probs, 0, 1)

        mask_probs[special_tokens_mask.bool()] = 0

        # Keep indices based on probabilities
        keep_indices = random_tensor.gt(mask_probs)
        label_ids[keep_indices] = -100

        # Masking tokens which are not preserved
        mask_indices = random_tensor.le(mask_probs * self.mask_ratio)

        if field.tokenization_strategy == "continuous_value_encoder":
            mask_value = -(self.tokenizer.mask_token_id + 1)
        else:
            mask_value = self.tokenizer.mask_token_id

        input_ids[mask_indices] = mask_value

        # Switching tokens which are not preserved and not masked
        if self.switch_ratio > 0:
            random_tokens = _draw_random_tokens(
                field.field_name, self.tokenizer, label_ids
            )
            mask_or_switch_probs = mask_probs * (self.mask_ratio + self.switch_ratio)
            switch_indices = random_tensor.le(mask_or_switch_probs) & (~mask_indices)
            input_ids[switch_indices] = random_tokens[switch_indices]

        return input_ids, label_ids

    def get_mask_probs(
        self, batch: dict[str, Mapping[str, torch.Tensor]], fields: list[FieldInfo]
    ) -> torch.Tensor | None:
        if self.masking_strategy is not None:
            mask_probs = self.masking_strategy.get_mask_probs(batch)
        else:
            mask_probs = torch.ones_like(
                batch[fields[0].field_name]["input_ids"]
            ).float()
        already_masked = self.find_masked_or_unk(batch, fields)
        mask_probs[already_masked] = 0

        return mask_probs

    def find_masked_or_unk(
        self,
        batch: dict[str, Mapping[str, torch.Tensor]],
        other_mask_fields: list[FieldInfo],
    ):
        ineligible_list = []
        for field in other_mask_fields:
            mask_id = self.tokenizer.mask_token_id
            unk_id = self.tokenizer.unk_token_id
            if field.tokenization_strategy == "continuous_value_encoder":
                mask_id = -(mask_id + 1)
            already_masked = batch[field.field_name]["input_ids"] == mask_id
            if field.tokenization_strategy != "continuous_value_encoder":
                # with continuous values there are no unk and preventing unk_id=0
                # means never masking 0
                is_unk = batch[field.field_name]["input_ids"] == unk_id
                ineligible_list.append(already_masked | is_unk)
            else:
                ineligible_list.append(already_masked)
        already_masked = torch.stack(ineligible_list, dim=-1)
        return already_masked.any(dim=-1)


def _draw_random_tokens(field_name, tokenizer, label_tensor):
    min_val = len(tokenizer.all_special_ids)
    max_val = tokenizer.field_vocab_size(field_name)
    return torch.randint_like(label_tensor, low=min_val, high=max_val)
