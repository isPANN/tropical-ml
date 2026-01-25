"""
Tropical Layers: MaxPlus and MinPlus Affine Maps.

These layers replace traditional activation functions (ReLU, etc.) with
tropical affine transformations that use only additions and max/min operations.

Mathematical foundation:
- MaxPlusAffine: y[i] = max(max_k(x[k] + W[k,i]), b[i])
- MinPlusAffine: y[i] = min(min_k(x[k] + W[k,i]), b[i])

Key design choices:
- Square matrices (features → features): acts as activation replacement
- LayerNorm before tropical operation: stabilizes sparse gradients
- Bias as threshold via max/min: true tropical affine transformation

Reference: Luo & Fan 2021 - "Min-Max-Plus Neural Networks"
"""

import torch
import torch.nn as nn

# Import tropical-gemm backend
try:
    from tropical_gemm.pytorch import (
        tropical_maxplus_matmul,
        tropical_minplus_matmul,
        tropical_maxplus_matmul_gpu,
        tropical_minplus_matmul_gpu,
        GPU_AVAILABLE,
    )
    TROPICAL_GEMM_AVAILABLE = True
except ImportError:
    TROPICAL_GEMM_AVAILABLE = False
    GPU_AVAILABLE = False


class MaxPlusAffine(nn.Module):
    """
    MaxPlus affine layer: y[i] = max(max_k(LayerNorm(x)[k] + W[k,i]), b[i])

    This is the tropical analog of an affine transformation in the max-plus semiring.
    - LayerNorm stabilizes training (sparse gradients from max)
    - Bias acts as learned threshold via max (tropical addition)

    Args:
        features: Number of features (square matrix: features → features)
        use_gpu: Use GPU acceleration if available
        use_norm: Apply LayerNorm before tropical operation (recommended)

    Shape:
        - Input: (N, features)
        - Output: (N, features)

    Example:
        >>> layer = MaxPlusAffine(256)
        >>> x = torch.randn(32, 256)
        >>> output = layer(x)  # shape: (32, 256)
    """

    def __init__(self, features: int, use_gpu: bool = False, use_norm: bool = True):
        super().__init__()
        self.features = features
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.use_norm = use_norm

        # LayerNorm before tropical operation
        if use_norm:
            self.norm = nn.LayerNorm(features)

        # Weight matrix (square) - initialized with spread for diverse winners
        self.weight = nn.Parameter(torch.randn(features, features) * 0.5)
        # Bias as threshold (combined via max)
        self.bias = nn.Parameter(torch.zeros(features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_norm:
            x = self.norm(x)

        if self.use_gpu:
            out = tropical_maxplus_matmul_gpu(x, self.weight)
        else:
            out = tropical_maxplus_matmul(x, self.weight)

        # Tropical affine: y = max(out, bias)
        return torch.maximum(out, self.bias)

    def extra_repr(self) -> str:
        return f"features={self.features}, gpu={self.use_gpu}, norm={self.use_norm}"


class MinPlusAffine(nn.Module):
    """
    MinPlus affine layer: y[i] = min(min_k(LayerNorm(x)[k] + W[k,i]), b[i])

    Complement to MaxPlusAffine. Together they provide universal approximation.
    - MaxPlus finds "longest paths" (max over sums)
    - MinPlus finds "shortest paths" (min over sums)

    Args:
        features: Number of features (square matrix: features → features)
        use_gpu: Use GPU acceleration if available
        use_norm: Apply LayerNorm before tropical operation (recommended)

    Shape:
        - Input: (N, features)
        - Output: (N, features)

    Example:
        >>> layer = MinPlusAffine(256)
        >>> x = torch.randn(32, 256)
        >>> output = layer(x)  # shape: (32, 256)
    """

    def __init__(self, features: int, use_gpu: bool = False, use_norm: bool = True):
        super().__init__()
        self.features = features
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.use_norm = use_norm

        if use_norm:
            self.norm = nn.LayerNorm(features)

        self.weight = nn.Parameter(torch.randn(features, features) * 0.5)
        self.bias = nn.Parameter(torch.zeros(features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_norm:
            x = self.norm(x)

        if self.use_gpu:
            out = tropical_minplus_matmul_gpu(x, self.weight)
        else:
            out = tropical_minplus_matmul(x, self.weight)

        # Tropical affine: y = min(out, bias)
        return torch.minimum(out, self.bias)

    def extra_repr(self) -> str:
        return f"features={self.features}, gpu={self.use_gpu}, norm={self.use_norm}"


# Aliases for backward compatibility
MaxPlusLayer = MaxPlusAffine
MinPlusLayer = MinPlusAffine


class TropicalReLU(nn.Module):
    """
    ReLU as a special case of MaxPlus.

    ReLU(x) = max(x, 0) is equivalent to tropical addition with zero.
    This is mainly for demonstration/comparison.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x)


class TropicalLeakyReLU(nn.Module):
    """
    LeakyReLU as a max of two linear functions.

    LeakyReLU(x) = max(x, alpha*x) = max-affine with 2 pieces.
    """

    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.leaky_relu(x, self.negative_slope)


__all__ = [
    "MaxPlusAffine",
    "MinPlusAffine",
    "MaxPlusLayer",  # Alias
    "MinPlusLayer",  # Alias
    "TropicalReLU",
    "TropicalLeakyReLU",
    "TROPICAL_GEMM_AVAILABLE",
    "GPU_AVAILABLE",
]
