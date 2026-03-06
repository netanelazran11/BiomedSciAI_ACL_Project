"""
Pretraining entry point for 21k methylation data.

Identical to pretrain.py except it uses MethylationDataModule21k,
which automatically creates a validation split when the h5ad file
has no 'valid' label (e.g. altumage_21k_combined.h5ad).

Usage:
    python -m bmfm_methylation.pretrain_21k \\
        data_path=/path/to/altumage_21k_pretrain.h5ad \\
        output_directory=./outputs/pretrain-wced-21k \\
        pretraining_mode=wced

The 8k pipeline (pretrain.py / pretrain_wced.sh) is NOT touched.
"""

# =============================================================================
# CRITICAL: patch torch.load BEFORE any other imports
# =============================================================================
import torch
import torch.serialization

_original_torch_load = torch.load

def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load
torch.serialization.load = _patched_torch_load
# =============================================================================

# Swap in the 21k-aware data module before pretrain.py is executed
import bmfm_methylation.pretrain as _pretrain_module
from bmfm_methylation.data_module_21k import MethylationDataModule21k

_pretrain_module.MethylationDataModule = MethylationDataModule21k

# Run the original main (Hydra config path stays relative to pretrain.py)
from bmfm_methylation.pretrain import main  # noqa: E402

main()
