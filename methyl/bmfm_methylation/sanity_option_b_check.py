#!/usr/bin/env python3
"""
Sanity check for Option-B (K=2048) methylation pipeline.
Run with: python -m bmfm_methylation.sanity_option_b_check
"""

import tempfile

import numpy as np
import torch

from bmfm_methylation.tokenizer import create_methylation_multifield_tokenizer
from bmfm_methylation.data_module import MethylationCollator
from bmfm_methylation.config import create_methylation_config
from bmfm_methylation.model import MethylationAgeModel

from bmfm_targets.tokenization.multifield_instance import MultiFieldInstance


def main():
    # -----------------------------
    # 1) Tokenizer checks
    # -----------------------------
    P = 8000
    cpg_sites = [f"cg{idx:08d}" for idx in range(P)]

    with tempfile.TemporaryDirectory() as tmpdir:
        tokenizer = create_methylation_multifield_tokenizer(
            cpg_sites=cpg_sites,
            output_dir=tmpdir,
        )

        tok = tokenizer.tokenizers["cpg_sites"]
        assert tok.unk_token_id == 0, f"UNK id mismatch: {tok.unk_token_id}"
        assert tok.sep_token_id == 1, f"SEP id mismatch: {tok.sep_token_id}"
        assert tok.pad_token_id == 2, f"PAD id mismatch: {tok.pad_token_id}"
        assert tok.cls_token_id == 3, f"CLS id mismatch: {tok.cls_token_id}"
        assert tok.mask_token_id == 4, f"MASK id mismatch: {tok.mask_token_id}"

        # -----------------------------
        # 2) Collator checks (Option-B)
        # -----------------------------
        B = 4
        rng = np.random.default_rng(123)
        examples = []
        for _ in range(B):
            beta_values = rng.random(P, dtype=np.float32)
            valid_mask = np.ones(P, dtype=bool)
            examples.append(
                MultiFieldInstance(
                    data={
                        "beta_values": beta_values.tolist(),
                        "valid_mask": valid_mask.tolist(),
                    },
                    metadata={},
                )
            )

        collator = MethylationCollator(
            tokenizer=tokenizer,
            cpg_sites=cpg_sites,
            k=2048,
            mask_ratio=0.3,
        )

        batch = collator(examples)

        cpg_ids = batch["cpg_ids"]
        beta_values = batch["beta_values"]
        attention_mask = batch["attention_mask"]
        labels_beta = batch["labels_beta"]
        loss_mask_beta = batch["loss_mask_beta"]

        assert cpg_ids.shape == (B, 2049)
        assert beta_values.shape == (B, 2049)
        assert attention_mask.shape == (B, 2049)
        assert labels_beta.shape == (B, 2049)
        assert loss_mask_beta.shape == (B, 2049)

        assert cpg_ids.dtype == torch.long
        assert beta_values.dtype == torch.float32
        assert loss_mask_beta.sum().item() > 0

        # -----------------------------
        # 3) Distinct subsets across samples
        # -----------------------------
        pad_id = tok.pad_token_id
        s0 = cpg_ids[0, 1:]
        s1 = cpg_ids[1, 1:]

        mask0 = s0 != pad_id
        mask1 = s1 != pad_id
        both = mask0 & mask1
        diff_count = (s0[both] != s1[both]).sum().item()
        assert diff_count >= 10, f"Subset overlap too high: diff_count={diff_count}"

        # -----------------------------
        # 4) Model forward smoke test
        # -----------------------------
        config = create_methylation_config(num_cpg_sites=P)
        model = MethylationAgeModel(config=config)

        with torch.no_grad():
            out = model(
                cpg_ids=cpg_ids,
                beta_values=beta_values,
                attention_mask=attention_mask,
            )

        # -----------------------------
        # 5) PASS summary
        # -----------------------------
        print("PASS: Token IDs ->", {
            "UNK": tok.unk_token_id,
            "SEP": tok.sep_token_id,
            "PAD": tok.pad_token_id,
            "CLS": tok.cls_token_id,
            "MASK": tok.mask_token_id,
        })
        print("PASS: Shapes ->", {
            "cpg_ids": tuple(cpg_ids.shape),
            "beta_values": tuple(beta_values.shape),
            "attention_mask": tuple(attention_mask.shape),
            "labels_beta": tuple(labels_beta.shape),
            "loss_mask_beta": tuple(loss_mask_beta.shape),
            "model_out": tuple(out.shape),
        })
        print("PASS: Masked count ->", int(loss_mask_beta.sum().item()))
        print("PASS: Subset diff count (sample0 vs sample1) ->", int(diff_count))


if __name__ == "__main__":
    main()
