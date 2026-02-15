"""
Decoders for Methylation Pretraining

This package provides decoder modules for different pretraining strategies:

1. MLM (Masked Language Modeling) - Uses BMFM's built-in MLM head
   - Masks some positions, predicts only those
   - Default in BMFM

2. WCED (Whole Cell Expression Decoder) - Custom decoder
   - Uses [CLS] to reconstruct ALL positions
   - Creates global bottleneck for better [CLS] representations

Usage:
    from bmfm_methylation.decoders import WCEDDecoder, WCEDLoss

    decoder = WCEDDecoder(hidden_size=512, num_cpg_sites=2048)
    loss_fn = WCEDLoss()
"""

from .wced_decoder import (
    WCEDDecoder,
    WCEDLoss,
    WCEDWithPositionalDecoder,
)

__all__ = [
    "WCEDDecoder",
    "WCEDLoss",
    "WCEDWithPositionalDecoder",
]
