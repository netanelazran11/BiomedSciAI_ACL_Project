"""
Pretraining entry point for 21k methylation data.

Identical to pretrain.py except it uses MethylationDataModule21k,
which automatically creates a validation split when the h5ad file
has no 'valid' label (e.g. altumage_21k_combined.h5ad).

Usage:
    python -m bmfm_methylation.pretrain_21k \\
        data_path=/path/to/altumage_21k_combined.h5ad \\
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

from pathlib import Path
import hydra
from omegaconf import DictConfig

# Swap in the 21k-aware data module BEFORE pretrain.py runs any logic
import bmfm_methylation.pretrain as _pretrain_module
from bmfm_methylation.data_module_21k import MethylationDataModule21k

_pretrain_module.MethylationDataModule = MethylationDataModule21k


# Own Hydra entry point with explicit absolute config path.
# This avoids Hydra's confusion about which module owns "configs/" when
# the decorated main() lives in pretrain.py but __main__ is pretrain_21k.py.
@hydra.main(
    config_path=str(Path(__file__).parent / "configs"),
    config_name="pretrain_config",
    version_base="1.2",
)
def main(cfg: DictConfig):
    # Call pretrain.py's actual logic (the unwrapped function, bypassing its
    # own Hydra decorator which would re-initialise Hydra and crash).
    _pretrain_module.main.__wrapped__(cfg)


if __name__ == "__main__":
    main()
