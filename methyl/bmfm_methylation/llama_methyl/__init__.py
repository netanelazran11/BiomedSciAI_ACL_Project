"""
MethylLlama — LLaMA-style encoder for methylation data.

LLaMA improvements over SCBertModel:
  - RMSNorm (Pre-LN): stable training, no careful warmup needed
  - SwiGLU MLP: gated activation, more expressive
  - RoPE: relative position in attention, no absolute pos table (saves ~4M params)
  - ScaleAdaptEncoder: built-in sinusoidal beta encoder (no monkey-patching)

Quick start:
    from bmfm_methylation.llama_methyl.model import MethylLlamaModel, MethylLlamaConfig

    config = MethylLlamaConfig(hidden_size=512, num_hidden_layers=6)
    model = MethylLlamaModel(config)

    # Input: dual-field [batch, 2, seq_len]
    import torch
    input_ids = torch.zeros(2, 2, 100)  # [batch=2, fields=2, seq_len=100]
    out = model(input_ids)
    print(out.pooler_output.shape)      # [2, 512]
"""

from .model import (
    MethylLlamaConfig,
    MethylLlamaModel,
    MethylLlamaEmbeddings,
    MethylLlamaEncoder,
    MethylLlamaPooler,
    RMSNorm,
    RotaryEmbedding,
    build_methyl_llama,
)

from .wced_llama import (
    WCEDLlamaModule,
    WCEDDecoder,
    ProjectionHead,
)

__all__ = [
    "MethylLlamaConfig",
    "MethylLlamaModel",
    "MethylLlamaEmbeddings",
    "MethylLlamaEncoder",
    "MethylLlamaPooler",
    "RMSNorm",
    "RotaryEmbedding",
    "build_methyl_llama",
    "WCEDLlamaModule",
    "WCEDDecoder",
    "ProjectionHead",
]
