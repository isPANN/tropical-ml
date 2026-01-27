"""
Tropicalization and De-tropicalization Layers.

Convert between Euclidean and tropical projective space.
"""

import torch
import torch.nn as nn


class Tropicalize(nn.Module):
    """
    Convert Euclidean embeddings to tropical projective space.

    Maps positive values to log-space and normalizes so max coordinate = 0.
    This places vectors on the tropical simplex (projective equivalence class).

    Args:
        eps: Small constant for numerical stability in log

    Shape:
        - Input: (batch, seq_len, d_model)
        - Output: (batch, seq_len, d_model) in tropical simplex (max coord = 0)
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            z: (batch, seq_len, d_model) in tropical simplex (max coord = 0)
        """
        # Clamp to positive, take log
        u = torch.log(torch.clamp(x, min=self.eps))
        # Normalize: subtract max so max coord = 0
        z = u - u.max(dim=-1, keepdim=True).values
        return z

    def extra_repr(self) -> str:
        return f"eps={self.eps}"


class Detropicalize(nn.Module):
    """
    Convert tropical projective space back to Euclidean.

    Applies exp() to map from log-space back to positive reals.

    Shape:
        - Input: (batch, seq_len, d_model) in tropical space
        - Output: (batch, seq_len, d_model) in Euclidean space
    """

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (batch, seq_len, d_model) in tropical space
        Returns:
            x: (batch, seq_len, d_model) in Euclidean space
        """
        return torch.exp(z)


__all__ = ["Tropicalize", "Detropicalize"]
