"""
Tropical Blocks: Building blocks for Tropical Neural Networks.

TropicalBlock combines Linear with MaxPlusAffine and MinPlusAffine layers.
The Linear layer handles dimension changes, tropical layers act as activations.

Architecture:
    Linear(in → out) → MaxPlusAffine(out) → MinPlusAffine(out)

Reference: Luo & Fan 2021 - "Min-Max-Plus Neural Networks"
"""

from typing import Optional

import torch
import torch.nn as nn

from .layers import MaxPlusAffine, MinPlusAffine


class TropicalBlock(nn.Module):
    """
    Tropical Block: Linear → MaxPlusAffine → MinPlusAffine

    Linear handles dimension change, tropical layers provide nonlinearity.
    This is the recommended building block for tropical networks.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        use_gpu: Use GPU acceleration for tropical layers.
        dropout: Dropout probability. Default: 0.0.

    Shape:
        - Input: (N, in_features)
        - Output: (N, out_features)

    Example:
        >>> block = TropicalBlock(784, 256)
        >>> x = torch.randn(32, 784)
        >>> output = block(x)  # shape: (32, 256)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_gpu: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Linear handles dimension change
        self.linear = nn.Linear(in_features, out_features)

        # Tropical layers act as activation (square matrices)
        self.maxplus = MaxPlusAffine(out_features, use_gpu=use_gpu)
        self.minplus = MinPlusAffine(out_features, use_gpu=use_gpu)

        # Optional dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)     # Dimension change: W @ x + b
        x = self.maxplus(x)    # Tropical activation: max(max_k(x + W), b)
        x = self.minplus(x)    # Tropical activation: min(min_k(x + W), b)
        x = self.dropout(x)
        return x


# Alias for backward compatibility
MMPBlock = TropicalBlock


class ResidualTropicalBlock(nn.Module):
    """
    Residual Tropical Block: TropicalBlock with skip connection.

    Args:
        features: Size of input/output features.
        use_gpu: Use GPU acceleration for tropical layers.
        dropout: Dropout probability. Default: 0.0.

    Shape:
        - Input: (N, features)
        - Output: (N, features)
    """

    def __init__(
        self,
        features: int,
        use_gpu: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.block = TropicalBlock(features, features, use_gpu=use_gpu, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


# Alias for backward compatibility
ResidualMMPBlock = ResidualTropicalBlock


class MaxPlusBlock(nn.Module):
    """
    MaxPlus Block: Linear → MaxPlusAffine

    Simpler block with only MaxPlus activation.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        use_gpu: Use GPU acceleration.

    Shape:
        - Input: (N, in_features)
        - Output: (N, out_features)
    """

    def __init__(self, in_features: int, out_features: int, use_gpu: bool = False):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.maxplus = MaxPlusAffine(out_features, use_gpu=use_gpu)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxplus(self.linear(x))


class MinPlusBlock(nn.Module):
    """
    MinPlus Block: Linear → MinPlusAffine

    Simpler block with only MinPlus activation.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        use_gpu: Use GPU acceleration.

    Shape:
        - Input: (N, in_features)
        - Output: (N, out_features)
    """

    def __init__(self, in_features: int, out_features: int, use_gpu: bool = False):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.minplus = MinPlusAffine(out_features, use_gpu=use_gpu)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.minplus(self.linear(x))


class TropicalMLP(nn.Module):
    """
    Tropical MLP: Multiple TropicalBlocks stacked.

    Architecture:
        Linear → MaxPlus → MinPlus → Linear → MaxPlus → MinPlus → ... → Linear

    Args:
        layer_sizes: List of layer sizes [input, hidden1, hidden2, ..., output]
        use_gpu: Use GPU acceleration for tropical layers.
        dropout: Dropout probability. Default: 0.0.

    Shape:
        - Input: (N, layer_sizes[0])
        - Output: (N, layer_sizes[-1])

    Example:
        >>> mlp = TropicalMLP([784, 256, 128, 10])
        >>> x = torch.randn(32, 784)
        >>> output = mlp(x)  # shape: (32, 10)
    """

    def __init__(
        self,
        layer_sizes: list,
        use_gpu: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert len(layer_sizes) >= 2, "Need at least input and output sizes"

        self.layer_sizes = layer_sizes
        layers = []

        for i in range(len(layer_sizes) - 1):
            in_dim = layer_sizes[i]
            out_dim = layer_sizes[i + 1]

            if i < len(layer_sizes) - 2:
                # Hidden layers: Linear → MaxPlus → MinPlus
                layers.append(TropicalBlock(in_dim, out_dim, use_gpu=use_gpu, dropout=dropout))
            else:
                # Output layer: just Linear (no tropical activation)
                layers.append(nn.Linear(in_dim, out_dim))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


__all__ = [
    "TropicalBlock",
    "ResidualTropicalBlock",
    "MaxPlusBlock",
    "MinPlusBlock",
    "TropicalMLP",
    # Aliases for backward compatibility
    "MMPBlock",
    "ResidualMMPBlock",
]
