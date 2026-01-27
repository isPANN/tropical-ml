"""
Tropical Linear Layer.

Linear projection using tropical matrix multiplication (max-plus semiring).
"""

import math

import torch
import torch.nn as nn

from tropical_gemm.pytorch import (
    tropical_maxplus_matmul,
    tropical_maxplus_matmul_gpu,
    GPU_AVAILABLE,
)


class TropicalLinear(nn.Module):
    """
    Linear projection using tropical matrix multiplication.

    Computes y = x ⊙ W^T where ⊙ is max-plus matmul:
        y[i,j] = max_k(x[i,k] + W[j,k])

    This is the tropical analog of a standard linear layer.

    Args:
        in_features: Size of input features
        out_features: Size of output features
        bias: If True, adds a learnable bias (combined via tropical addition)

    Shape:
        - Input: (..., in_features)
        - Output: (..., out_features)
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self._reset_parameters()

    def _reset_parameters(self):
        # Initialize similar to standard linear
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., in_features)
        Returns:
            y: (..., out_features)
        """
        # Handle batched input
        orig_shape = x.shape[:-1]
        x_2d = x.reshape(-1, x.shape[-1])  # (B*N, in_features)

        # Tropical matmul: (B*N, in) ⊙ (in, out) -> (B*N, out)
        if x.is_cuda and GPU_AVAILABLE:
            y_2d = tropical_maxplus_matmul_gpu(x_2d, self.weight.t())
        else:
            y_2d = tropical_maxplus_matmul(x_2d, self.weight.t())

        # Reshape back
        y = y_2d.reshape(*orig_shape, -1)

        if self.bias is not None:
            # Tropical addition is max
            y = torch.maximum(y, self.bias)

        return y

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


__all__ = ["TropicalLinear", "GPU_AVAILABLE"]
