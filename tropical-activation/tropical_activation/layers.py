"""
Tropical Layers: MaxPlusLayer and MinPlusLayer for Min-Max-Plus Neural Networks.

These layers implement tropical matrix operations as neural network layers with
full autograd support. They form the nonlinear component of MMP-NNs.

Mathematical formulation:
- MaxPlusLayer: y_j = max_k(x_k + W_kj) + b_j
- MinPlusLayer: y_j = min_k(x_k + W_kj) + b_j

Reference: Luo & Fan 2021 - "Min-Max-Plus Neural Networks"
"""

import math
from typing import Optional

import torch
import torch.nn as nn

import tropical_gemm as tg
from tropical_gemm.pytorch import (
    TropicalMaxPlusMatmul,
    TropicalMinPlusMatmul,
    TropicalMaxPlusMatmulGPU,
    TropicalMinPlusMatmulGPU,
    GPU_AVAILABLE,
    _DLPACK_AVAILABLE,
)


class MaxPlusLayer(nn.Module):
    """
    Max-Plus tropical layer: y_j = max_k(x_k + W_kj) + b_j

    Uses tropical-gemm for high-performance SIMD (CPU) or CUDA (GPU) computation.
    Tropical GEMM: C[i,j] = max_k(A[i,k] + B[k,j])

    This layer provides the nonlinearity in MMP neural networks by taking the
    maximum over all (input + weight) combinations.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        bias: If True, adds a learnable bias. Default: True.

    Shape:
        - Input: (*, in_features) where * means any number of dimensions
        - Output: (*, out_features)

    Attributes:
        weight: Learnable weights of shape (in_features, out_features).
        bias: Learnable bias of shape (out_features) if bias=True, else None.

    Example:
        >>> layer = MaxPlusLayer(20, 30)
        >>> x = torch.randn(128, 20)
        >>> output = layer(x)  # shape: (128, 30)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Weight shape: (in_features, out_features) = (K, N) for matmul
        # This allows C[i,j] = max_k(x[i,k] + weight[k,j])
        self.weight = nn.Parameter(
            torch.empty(in_features, out_features, **factory_kwargs)
        )

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize weights using tropical-aware initialization.

        For tropical layers, we use a smaller initialization scale since
        the operation is max(x + w) instead of sum(x * w).
        """
        # Initialize with small values centered around 0
        # This ensures all inputs have a chance to "win" initially
        nn.init.uniform_(self.weight, -0.1, 0.1)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: compute tropical max-plus matmul.

        Args:
            x: Input tensor of shape (*, in_features)

        Returns:
            Output tensor of shape (*, out_features)
        """
        # Handle multi-dimensional inputs
        original_shape = x.shape[:-1]
        x_flat = x.reshape(-1, self.in_features)

        # Choose backend based on device and availability
        if x.is_cuda and GPU_AVAILABLE and _DLPACK_AVAILABLE:
            output = TropicalMaxPlusMatmulGPU.apply(
                x_flat.contiguous(), self.weight.contiguous()
            )
        else:
            output = TropicalMaxPlusMatmul.apply(
                x_flat.contiguous(), self.weight.contiguous()
            )

        # Reshape back to original batch dimensions
        output = output.reshape(*original_shape, self.out_features)

        if self.bias is not None:
            output = output + self.bias

        return output

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


class MinPlusLayer(nn.Module):
    """
    Min-Plus tropical layer: y_j = min_k(x_k + W_kj) + b_j

    Uses tropical-gemm for optimized computation.
    Tropical GEMM: C[i,j] = min_k(A[i,k] + B[k,j])

    This layer complements MaxPlusLayer for full expressiveness in MMP networks.
    Together, max-plus and min-plus layers achieve universal approximation.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        bias: If True, adds a learnable bias. Default: True.

    Shape:
        - Input: (*, in_features) where * means any number of dimensions
        - Output: (*, out_features)

    Attributes:
        weight: Learnable weights of shape (in_features, out_features).
        bias: Learnable bias of shape (out_features) if bias=True, else None.

    Example:
        >>> layer = MinPlusLayer(20, 30)
        >>> x = torch.randn(128, 20)
        >>> output = layer(x)  # shape: (128, 30)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Weight shape: (in_features, out_features) = (K, N) for matmul
        self.weight = nn.Parameter(
            torch.empty(in_features, out_features, **factory_kwargs)
        )

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize weights using tropical-aware initialization."""
        nn.init.uniform_(self.weight, -0.1, 0.1)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: compute tropical min-plus matmul.

        Args:
            x: Input tensor of shape (*, in_features)

        Returns:
            Output tensor of shape (*, out_features)
        """
        # Handle multi-dimensional inputs
        original_shape = x.shape[:-1]
        x_flat = x.reshape(-1, self.in_features)

        # Choose backend based on device and availability
        if x.is_cuda and GPU_AVAILABLE and _DLPACK_AVAILABLE:
            output = TropicalMinPlusMatmulGPU.apply(
                x_flat.contiguous(), self.weight.contiguous()
            )
        else:
            output = TropicalMinPlusMatmul.apply(
                x_flat.contiguous(), self.weight.contiguous()
            )

        # Reshape back to original batch dimensions
        output = output.reshape(*original_shape, self.out_features)

        if self.bias is not None:
            output = output + self.bias

        return output

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


class TropicalReLU(nn.Module):
    """
    ReLU as a special case of MaxPlusLayer.

    ReLU(x) = max(x, 0) can be viewed as a tropical max-plus operation
    with fixed weights: W = [[0], [-inf]]

    This is primarily for demonstration and conversion purposes.
    For training, use MaxPlusLayer with learnable weights.

    Shape:
        - Input: (*, features)
        - Output: (*, features)

    Example:
        >>> relu = TropicalReLU()
        >>> x = torch.randn(128, 64)
        >>> output = relu(x)  # Same as torch.relu(x)
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply element-wise max(x, 0) using tropical interpretation."""
        return torch.relu(x)


class TropicalLeakyReLU(nn.Module):
    """
    LeakyReLU as a max-affine function with 2 pieces.

    LeakyReLU(x, α) = max(x, αx)

    This can be implemented as a MaxPlusLayer with specific structure,
    but for efficiency we use the direct formula.

    Args:
        negative_slope: Slope for negative inputs. Default: 0.01

    Shape:
        - Input: (*, features)
        - Output: (*, features)
    """

    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply element-wise max(x, αx)."""
        return torch.nn.functional.leaky_relu(x, self.negative_slope)


__all__ = [
    "MaxPlusLayer",
    "MinPlusLayer",
    "TropicalReLU",
    "TropicalLeakyReLU",
]
