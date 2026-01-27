"""
Tropical Attention: Multi-Head Attention using Tropical Geometry.

This package provides tropical attention mechanisms that use the Hilbert
projective metric instead of softmax-based attention scores.

Key Components:
- TropicalMultiheadAttention: Drop-in replacement for nn.MultiheadAttention
- TropicalTransformerEncoderLayer: Transformer encoder with tropical attention
- Tropicalize/Detropicalize: Convert between Euclidean and tropical space
- hilbert_distance: Hilbert projective metric for attention scores
- TropicalLinear: Linear projection using max-plus matmul
"""

from .layers import (
    Tropicalize,
    Detropicalize,
    hilbert_distance,
    TropicalLinear,
    TropicalMultiheadAttention,
)
from .models import TropicalTransformerEncoderLayer

__version__ = "0.1.0"

__all__ = [
    # Layers
    "Tropicalize",
    "Detropicalize",
    "hilbert_distance",
    "TropicalLinear",
    "TropicalMultiheadAttention",
    # Models
    "TropicalTransformerEncoderLayer",
]
