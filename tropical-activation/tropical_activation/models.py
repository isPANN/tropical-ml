"""
MMPNN: Min-Max-Plus Neural Networks.

Complete neural network architectures using tropical layers.
These networks alternate Linear, MaxPlus, and MinPlus layers
for universal approximation with reduced multiplications.

Reference: Luo & Fan 2021 - "Min-Max-Plus Neural Networks"
"""

from typing import List, Optional, Union

import torch
import torch.nn as nn

from .layers import MaxPlusLayer, MinPlusLayer
from .blocks import MMPBlock, ResidualMMPBlock


class MMPNN(nn.Module):
    """
    Min-Max-Plus Neural Network.

    Alternates Linear, MaxPlus, and MinPlus layers.
    Universal approximator with reduced multiplications.

    Architecture (with use_linear=True):
        Linear → MaxPlus → MinPlus → Linear → MaxPlus → MinPlus → ...

    Architecture (with use_linear=False, pure tropical):
        MaxPlus → MinPlus → MaxPlus → MinPlus → ...

    Args:
        layer_sizes: List of layer sizes, e.g., [784, 256, 128, 10].
        use_linear: If True, includes Linear layers. If False, pure tropical.
        dropout: Dropout probability between blocks. Default: 0.0.

    Shape:
        - Input: (batch, layer_sizes[0])
        - Output: (batch, layer_sizes[-1])

    Example:
        >>> model = MMPNN([784, 256, 128, 10])
        >>> x = torch.randn(32, 784)
        >>> output = model(x)  # shape: (32, 10)
    """

    def __init__(
        self,
        layer_sizes: List[int],
        use_linear: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()

        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes must have at least 2 elements")

        self.layer_sizes = layer_sizes
        self.use_linear = use_linear
        self.dropout_p = dropout

        layers = []

        for i in range(len(layer_sizes) - 1):
            in_features = layer_sizes[i]
            out_features = layer_sizes[i + 1]

            if use_linear:
                # Linear → MaxPlus → MinPlus pattern
                layers.append(nn.Linear(in_features, out_features))

                # Add tropical nonlinearity (except for last layer)
                if i < len(layer_sizes) - 2:
                    layers.append(MaxPlusLayer(out_features, out_features))
                    layers.append(MinPlusLayer(out_features, out_features))

                    if dropout > 0.0:
                        layers.append(nn.Dropout(dropout))
            else:
                # Pure tropical: MaxPlus → MinPlus alternation
                if i % 2 == 0:
                    layers.append(MaxPlusLayer(in_features, out_features))
                else:
                    layers.append(MinPlusLayer(in_features, out_features))

                if dropout > 0.0 and i < len(layer_sizes) - 2:
                    layers.append(nn.Dropout(dropout))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MMP network."""
        return self.layers(x)


class MMPClassifier(nn.Module):
    """
    MMP Classifier for image/vector classification tasks.

    A complete classifier using MMP blocks with optional residual connections.

    Args:
        input_dim: Input dimension (e.g., 784 for MNIST flattened).
        hidden_dims: List of hidden dimensions.
        num_classes: Number of output classes.
        use_residual: Whether to use residual connections. Default: False.
        dropout: Dropout probability. Default: 0.0.

    Shape:
        - Input: (batch, input_dim)
        - Output: (batch, num_classes)

    Example:
        >>> model = MMPClassifier(784, [256, 128], 10)
        >>> x = torch.randn(32, 784)
        >>> logits = model(x)  # shape: (32, 10)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        num_classes: int,
        use_residual: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes

        layers = []

        # Input projection
        layers.append(nn.Linear(input_dim, hidden_dims[0]))

        # Hidden MMP blocks
        for i in range(len(hidden_dims) - 1):
            if use_residual and hidden_dims[i] == hidden_dims[i + 1]:
                layers.append(
                    ResidualMMPBlock(hidden_dims[i], dropout=dropout)
                )
            else:
                layers.append(
                    MMPBlock(
                        hidden_dims[i],
                        hidden_features=hidden_dims[i],
                        out_features=hidden_dims[i + 1],
                        dropout=dropout,
                    )
                )

        # Output projection
        layers.append(nn.Linear(hidden_dims[-1], num_classes))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits."""
        return self.layers(x)


class PureTropicalNN(nn.Module):
    """
    Pure Tropical Neural Network (no standard linear layers).

    Uses only MaxPlus and MinPlus layers, eliminating multiplications
    in the nonlinear part entirely.

    This achieves the maximum reduction in multiplications but may
    have different training dynamics than hybrid approaches.

    Args:
        layer_sizes: List of layer sizes.
        bias: Whether to use biases in tropical layers. Default: True.

    Shape:
        - Input: (batch, layer_sizes[0])
        - Output: (batch, layer_sizes[-1])

    Example:
        >>> model = PureTropicalNN([784, 256, 128, 10])
        >>> x = torch.randn(32, 784)
        >>> output = model(x)  # shape: (32, 10)
    """

    def __init__(
        self,
        layer_sizes: List[int],
        bias: bool = True,
    ):
        super().__init__()

        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes must have at least 2 elements")

        self.layer_sizes = layer_sizes

        layers = []

        for i in range(len(layer_sizes) - 1):
            in_features = layer_sizes[i]
            out_features = layer_sizes[i + 1]

            # Alternate MaxPlus and MinPlus
            if i % 2 == 0:
                layers.append(MaxPlusLayer(in_features, out_features, bias=bias))
            else:
                layers.append(MinPlusLayer(in_features, out_features, bias=bias))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through pure tropical network."""
        return self.layers(x)


class MMPAutoencoder(nn.Module):
    """
    MMP Autoencoder for representation learning.

    Uses MMP blocks in both encoder and decoder for
    tropical-based representation learning.

    Args:
        input_dim: Input dimension.
        hidden_dims: List of encoder hidden dimensions.
        latent_dim: Dimension of the latent space.
        dropout: Dropout probability. Default: 0.0.

    Shape:
        - Input: (batch, input_dim)
        - Output: (batch, input_dim)

    Example:
        >>> model = MMPAutoencoder(784, [256, 128], 32)
        >>> x = torch.randn(32, 784)
        >>> reconstruction = model(x)  # shape: (32, 784)
        >>> latent = model.encode(x)   # shape: (32, 32)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        latent_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim

        # Encoder
        encoder_layers = []
        dims = [input_dim] + hidden_dims + [latent_dim]

        for i in range(len(dims) - 1):
            encoder_layers.append(
                MMPBlock(dims[i], out_features=dims[i + 1], dropout=dropout)
            )

        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder (mirror of encoder)
        decoder_layers = []
        dims_reversed = [latent_dim] + hidden_dims[::-1] + [input_dim]

        for i in range(len(dims_reversed) - 1):
            decoder_layers.append(
                MMPBlock(dims_reversed[i], out_features=dims_reversed[i + 1], dropout=dropout)
            )

        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass (encode then decode)."""
        z = self.encode(x)
        return self.decode(z)


def create_mmpnn(
    architecture: str,
    input_dim: int,
    num_classes: int,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create MMP networks.

    Args:
        architecture: One of "small", "medium", "large", "tiny".
        input_dim: Input dimension.
        num_classes: Number of output classes.
        **kwargs: Additional arguments passed to the model.

    Returns:
        An MMPNN model.

    Example:
        >>> model = create_mmpnn("medium", 784, 10)
    """
    architectures = {
        "tiny": [input_dim, 64, num_classes],
        "small": [input_dim, 128, 64, num_classes],
        "medium": [input_dim, 256, 128, 64, num_classes],
        "large": [input_dim, 512, 256, 128, 64, num_classes],
    }

    if architecture not in architectures:
        raise ValueError(f"Unknown architecture: {architecture}. Choose from {list(architectures.keys())}")

    return MMPNN(architectures[architecture], **kwargs)


__all__ = [
    "MMPNN",
    "MMPClassifier",
    "PureTropicalNN",
    "MMPAutoencoder",
    "create_mmpnn",
]
