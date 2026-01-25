"""
MMP Blocks: Building blocks for Min-Max-Plus Neural Networks.

MMPBlock combines Linear, MaxPlus, and MinPlus layers into a single unit.
This is the standard building block of MMP Neural Networks.

Architecture:
    Linear → MaxPlus → MinPlus

The linear layer provides the multiplicative transform, while MaxPlus/MinPlus
layers provide the nonlinearity through tropical operations.

Reference: Luo & Fan 2021 - "Min-Max-Plus Neural Networks"
"""

from typing import Optional

import torch
import torch.nn as nn

from .layers import MaxPlusLayer, MinPlusLayer


class MMPBlock(nn.Module):
    """
    Min-Max-Plus Block: Linear → MaxPlus → MinPlus

    The standard building block of MMP Neural Networks.
    Linear provides multiplicative transform, MaxPlus/MinPlus provide nonlinearity.

    Args:
        in_features: Size of each input sample.
        hidden_features: Size of the hidden layer (MaxPlus output).
            If None, defaults to in_features.
        out_features: Size of each output sample.
            If None, defaults to in_features.
        linear_bias: Whether the linear layer has a bias. Default: True.
        tropical_bias: Whether tropical layers have biases. Default: True.
        dropout: Dropout probability. Default: 0.0.

    Shape:
        - Input: (*, in_features)
        - Output: (*, out_features)

    Example:
        >>> block = MMPBlock(64, 128, 64)
        >>> x = torch.randn(32, 64)
        >>> output = block(x)  # shape: (32, 64)
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        linear_bias: bool = True,
        tropical_bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()

        hidden_features = hidden_features or in_features
        out_features = out_features or in_features

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

        # Standard linear layer (multiplicative)
        self.linear = nn.Linear(in_features, hidden_features, bias=linear_bias)

        # Tropical nonlinearity layers
        self.maxplus = MaxPlusLayer(hidden_features, hidden_features, bias=tropical_bias)
        self.minplus = MinPlusLayer(hidden_features, out_features, bias=tropical_bias)

        # Optional dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MMP block.

        Args:
            x: Input tensor of shape (*, in_features)

        Returns:
            Output tensor of shape (*, out_features)
        """
        x = self.linear(x)    # Standard multiplication: W @ x + b
        x = self.maxplus(x)   # Tropical nonlinearity 1: max_k(x_k + w_k)
        x = self.minplus(x)   # Tropical nonlinearity 2: min_k(x_k + w_k)
        x = self.dropout(x)
        return x


class ResidualMMPBlock(nn.Module):
    """
    Residual MMP Block: MMPBlock with skip connection.

    Similar to residual connections in standard networks, this adds
    a skip connection around the MMP block for better gradient flow.

    Args:
        features: Size of input/output features.
        hidden_features: Size of the hidden layer. If None, defaults to features * 4.
        dropout: Dropout probability. Default: 0.0.
        scale_init: Initial scale for the residual. Default: 1.0.

    Shape:
        - Input: (*, features)
        - Output: (*, features)

    Example:
        >>> block = ResidualMMPBlock(64, 256)
        >>> x = torch.randn(32, 64)
        >>> output = block(x)  # shape: (32, 64)
    """

    def __init__(
        self,
        features: int,
        hidden_features: Optional[int] = None,
        dropout: float = 0.0,
        scale_init: float = 1.0,
    ):
        super().__init__()

        hidden_features = hidden_features or features * 4

        self.block = MMPBlock(
            in_features=features,
            hidden_features=hidden_features,
            out_features=features,
            dropout=dropout,
        )

        # Learnable residual scale (for training stability)
        self.scale = nn.Parameter(torch.tensor(scale_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        return x + self.scale * self.block(x)


class MaxPlusBlock(nn.Module):
    """
    Pure MaxPlus Block: Linear → MaxPlus

    A simpler block using only max-plus nonlinearity (no min-plus).
    Useful for experiments or when min-plus isn't needed.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        linear_bias: Whether the linear layer has a bias. Default: True.
        tropical_bias: Whether the MaxPlus layer has a bias. Default: True.

    Shape:
        - Input: (*, in_features)
        - Output: (*, out_features)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        linear_bias: bool = True,
        tropical_bias: bool = True,
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=linear_bias)
        self.maxplus = MaxPlusLayer(out_features, out_features, bias=tropical_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.maxplus(x)
        return x


class MinPlusBlock(nn.Module):
    """
    Pure MinPlus Block: Linear → MinPlus

    A simpler block using only min-plus nonlinearity (no max-plus).

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        linear_bias: Whether the linear layer has a bias. Default: True.
        tropical_bias: Whether the MinPlus layer has a bias. Default: True.

    Shape:
        - Input: (*, in_features)
        - Output: (*, out_features)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        linear_bias: bool = True,
        tropical_bias: bool = True,
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=linear_bias)
        self.minplus = MinPlusLayer(out_features, out_features, bias=tropical_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.minplus(x)
        return x


class TropicalMLP(nn.Module):
    """
    Tropical MLP: Replaces standard MLP's activations with tropical layers.

    Standard MLP: Linear → ReLU → Linear → ReLU → ...
    Tropical MLP: Linear → MaxPlus → Linear → MinPlus → ...

    Alternates MaxPlus and MinPlus for full expressiveness.

    Args:
        in_features: Size of each input sample.
        hidden_features: Size of the hidden layer.
        out_features: Size of each output sample.
        num_layers: Number of linear layers. Default: 2.
        dropout: Dropout probability. Default: 0.0.

    Shape:
        - Input: (*, in_features)
        - Output: (*, out_features)

    Example:
        >>> mlp = TropicalMLP(64, 256, 10, num_layers=3)
        >>> x = torch.randn(32, 64)
        >>> output = mlp(x)  # shape: (32, 10)
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.num_layers = num_layers
        layers = []

        for i in range(num_layers):
            # Determine layer sizes
            if i == 0:
                in_dim = in_features
            else:
                in_dim = hidden_features

            if i == num_layers - 1:
                out_dim = out_features
            else:
                out_dim = hidden_features

            # Add linear layer
            layers.append(nn.Linear(in_dim, out_dim))

            # Add tropical nonlinearity (except for last layer)
            if i < num_layers - 1:
                # Alternate between MaxPlus and MinPlus
                if i % 2 == 0:
                    layers.append(MaxPlusLayer(out_dim, out_dim))
                else:
                    layers.append(MinPlusLayer(out_dim, out_dim))

                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


__all__ = [
    "MMPBlock",
    "ResidualMMPBlock",
    "MaxPlusBlock",
    "MinPlusBlock",
    "TropicalMLP",
]
