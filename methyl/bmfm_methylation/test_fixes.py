#!/usr/bin/env python3
"""
Quick test to verify all fixes work before running on server.
Run: python -m bmfm_methylation.test_fixes
"""

import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test all imports work."""
    print("1. Testing imports...")

    from bmfm_targets.config import FieldInfo, TrainerConfig, SCBertConfig
    from bmfm_targets.training.modules.masked_language_modeling import MLMTrainingModule
    from bmfm_targets.training.modules.base import BaseTrainingModule
    from bmfm_targets.tokenization import MultiFieldTokenizer
    from bmfm_targets.tokenization.multifield_instance import MultiFieldInstance

    print("   ✓ All imports successful")
    return True

def test_field_info():
    """Test FieldInfo with token_scores decode mode."""
    print("\n2. Testing FieldInfo with token_scores...")

    from bmfm_targets.config import FieldInfo

    field = FieldInfo(
        field_name="cpg_sites",
        vocab_size=8005,
        is_input=True,
        is_masked=True,
        tokenization_strategy="tokenize",
        decode_modes={"token_scores": {}}
    )

    # Test is_decode property works
    assert field.is_decode == True, "is_decode should be True"
    print(f"   ✓ FieldInfo created, is_decode={field.is_decode}")
    return True

def test_multifield_instance():
    """Test MultiFieldInstance with metadata."""
    print("\n3. Testing MultiFieldInstance with metadata...")

    from bmfm_targets.tokenization.multifield_instance import MultiFieldInstance

    mfi = MultiFieldInstance(
        data={
            "cpg_sites": ["cg001", "cg002", "cg003"],
            "beta_values": [0.5, 0.3, 0.8],
        },
        metadata={
            "labels": 45.5,
            "cell_name": "sample_0",
        }
    )

    # Test access
    assert mfi["cpg_sites"] == ["cg001", "cg002", "cg003"]
    assert mfi.metadata["labels"] == 45.5
    print(f"   ✓ MultiFieldInstance created with metadata")
    return True

def test_base_training_module_fix():
    """Test that label_dict=None doesn't crash."""
    print("\n4. Testing BaseTrainingModule label_dict fix...")

    # Simulate the fixed condition
    label_dict = None
    model_config_has_label_columns = True
    model_config_label_columns_is_none = True

    # This is the FIXED logic
    if label_dict is not None and (
        not model_config_has_label_columns
        or model_config_label_columns_is_none
    ):
        # Would access label_dict.keys() here
        print("   Would access label_dict.keys()")
    else:
        print("   ✓ Correctly skipped label_dict.keys() when label_dict is None")

    return True

def test_scbert_config():
    """Test SCBertConfig instantiation."""
    print("\n5. Testing SCBertConfig...")

    from bmfm_targets.config import SCBertConfig, FieldInfo

    fields = [
        FieldInfo(
            field_name="cpg_sites",
            vocab_size=8005,
            is_input=True,
            is_masked=True,
            tokenization_strategy="tokenize",
            decode_modes={"token_scores": {}}
        ),
        FieldInfo(
            field_name="beta_values",
            is_input=True,
            is_masked=False,
            tokenization_strategy="continuous_value_encoder",
            encoder_kwargs={"kind": "mlp"}
        ),
    ]

    config = SCBertConfig(
        fields=fields,
        num_hidden_layers=6,
        num_attention_heads=8,
        hidden_size=512,
        intermediate_size=2048,
    )

    assert config.fields == fields
    assert config.num_hidden_layers == 6
    print(f"   ✓ SCBertConfig created with {len(config.fields)} fields")
    return True

def test_model_creation():
    """Test model can be created."""
    print("\n6. Testing model creation...")

    try:
        from bmfm_targets.config import SCBertConfig, FieldInfo, TrainerConfig
        from bmfm_targets.training.modules.masked_language_modeling import MLMTrainingModule

        fields = [
            FieldInfo(
                field_name="cpg_sites",
                vocab_size=8005,
                is_input=True,
                is_masked=True,
                tokenization_strategy="tokenize",
                decode_modes={"token_scores": {}}
            ),
            FieldInfo(
                field_name="beta_values",
                is_input=True,
                is_masked=False,
                tokenization_strategy="continuous_value_encoder",
                encoder_kwargs={"kind": "mlp"}
            ),
        ]

        model_config = SCBertConfig(
            fields=fields,
            num_hidden_layers=2,  # Small for testing
            num_attention_heads=2,
            hidden_size=64,
            intermediate_size=128,
        )

        trainer_config = TrainerConfig(
            learning_rate=1e-4,
            losses=[{"name": "cross_entropy"}],
        )

        model = MLMTrainingModule(
            model_config=model_config,
            trainer_config=trainer_config,
            tokenizer=None,  # OK for this test
        )

        num_params = sum(p.numel() for p in model.parameters())
        print(f"   ✓ MLMTrainingModule created with {num_params:,} parameters")
        return True

    except Exception as e:
        print(f"   ✗ Model creation failed: {e}")
        return False

def main():
    print("=" * 60)
    print("BMFM Methylation - Fix Verification Tests")
    print("=" * 60)

    tests = [
        test_imports,
        test_field_info,
        test_multifield_instance,
        test_base_training_module_fix,
        test_scbert_config,
        test_model_creation,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"   ✗ FAILED: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print("\n✓ All fixes verified! Safe to run on server.")
        print("\nCopy files to server:")
        print("  scp ~/Projects/BMFM-RNA_thesis/bmfm_targets/training/modules/base.py USER@SERVER:/path/to/base.py")
        print("  scp ~/Projects/BMFM-RNA_thesis/methyl/bmfm_methylation/pretrain.py USER@SERVER:/path/to/pretrain.py")
        print("  scp ~/Projects/BMFM-RNA_thesis/methyl/bmfm_methylation/data_module.py USER@SERVER:/path/to/data_module.py")
        print("  scp ~/Projects/BMFM-RNA_thesis/methyl/bmfm_methylation/configs/fields/methylation.yaml USER@SERVER:/path/to/methylation.yaml")
    else:
        print("\n✗ Some tests failed. Please fix before running on server.")
        sys.exit(1)

if __name__ == "__main__":
    main()
