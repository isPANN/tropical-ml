"""
Tropical Neural Networks: Complete model architectures.

Uses TropicalBlock (Linear → MaxPlusAffine → MinPlusAffine) as the building block.
Linear layers handle dimension changes, tropical layers act as activations.

Reference: Luo & Fan 2021 - "Min-Max-Plus Neural Networks"
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import MaxPlusAffine, MinPlusAffine, TropicalAffine
from .blocks import TropicalBlock, TropicalMLP, HybridBlock, HybridMLP


class TropicalNN(nn.Module):
    """
    Tropical Neural Network.

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
        >>> model = TropicalNN([784, 256, 128, 10])
        >>> x = torch.randn(32, 784)
        >>> output = model(x)  # shape: (32, 10)
    """

    def __init__(
        self,
        layer_sizes: List[int],
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
                # Hidden: Linear → MaxPlus → MinPlus
                layers.append(TropicalBlock(in_dim, out_dim, use_gpu=use_gpu, dropout=dropout))
            else:
                # Output: just Linear
                layers.append(nn.Linear(in_dim, out_dim))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


# Alias for backward compatibility
MMPNN = TropicalNN


class HybridTropicalNN(nn.Module):
    """
    Hybrid Tropical Neural Network (from mnist_tropical.py architecture).

    Uses only TropicalAffine (MaxPlusAffine) without MinPlus, which is simpler
    and often performs equally well.

    Architecture:
        Linear → TropicalAffine → Linear → TropicalAffine → ... → Linear

    This is the recommended architecture for most applications.

    Args:
        layer_sizes: List of layer sizes [input, hidden1, hidden2, ..., output]
        use_gpu: Use GPU acceleration for tropical layers.
        dropout: Dropout probability. Default: 0.0.

    Shape:
        - Input: (N, layer_sizes[0])
        - Output: (N, layer_sizes[-1])

    Example:
        >>> model = HybridTropicalNN([784, 256, 128, 10])
        >>> x = torch.randn(32, 784)
        >>> output = model(x)  # shape: (32, 10)
    """

    def __init__(
        self,
        layer_sizes: List[int],
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
                # Hidden: Linear → TropicalAffine
                layers.append(HybridBlock(in_dim, out_dim, use_gpu=use_gpu, dropout=dropout))
            else:
                # Output: just Linear
                layers.append(nn.Linear(in_dim, out_dim))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class HybridClassifier(nn.Module):
    """
    Hybrid Classifier for image/vector classification.

    Uses the hybrid architecture (Linear → TropicalAffine only).

    Args:
        input_dim: Input dimension (e.g., 784 for MNIST).
        hidden_dims: List of hidden dimensions.
        num_classes: Number of output classes.
        use_gpu: Use GPU acceleration.
        dropout: Dropout probability.

    Example:
        >>> model = HybridClassifier(784, [256, 128], 10)
        >>> x = torch.randn(32, 784)
        >>> logits = model(x)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        num_classes: int,
        use_gpu: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        layer_sizes = [input_dim] + hidden_dims + [num_classes]
        self.net = HybridTropicalNN(layer_sizes, use_gpu=use_gpu, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.net(x)


class TropicalClassifier(nn.Module):
    """
    Tropical Classifier for image/vector classification.

    Flattens input if needed, then applies TropicalNN.

    Args:
        input_dim: Input dimension (e.g., 784 for MNIST).
        hidden_dims: List of hidden dimensions.
        num_classes: Number of output classes.
        use_gpu: Use GPU acceleration.
        dropout: Dropout probability.

    Example:
        >>> model = TropicalClassifier(784, [256, 128], 10)
        >>> x = torch.randn(32, 784)
        >>> logits = model(x)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        num_classes: int,
        use_gpu: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        layer_sizes = [input_dim] + hidden_dims + [num_classes]
        self.net = TropicalNN(layer_sizes, use_gpu=use_gpu, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.net(x)


# Alias for backward compatibility
MMPClassifier = TropicalClassifier


class PureTropicalNN(nn.Module):
    """
    Pure Tropical Neural Network (minimal multiplications).

    First Linear layer, then only tropical layers.
    Architecture:
        Linear → MaxPlus → MinPlus → MaxPlus → MinPlus → ...

    Args:
        layer_sizes: List of layer sizes.
        use_gpu: Use GPU acceleration.

    Example:
        >>> model = PureTropicalNN([784, 256, 128, 10])
        >>> x = torch.randn(32, 784)
        >>> output = model(x)
    """

    def __init__(
        self,
        layer_sizes: List[int],
        use_gpu: bool = False,
    ):
        super().__init__()
        assert len(layer_sizes) >= 2

        self.layer_sizes = layer_sizes
        layers = []

        # First layer: Linear for dimension change
        layers.append(nn.Linear(layer_sizes[0], layer_sizes[1]))

        # Rest: only tropical layers (square, acting as activations)
        for i in range(1, len(layer_sizes) - 1):
            dim = layer_sizes[i]
            next_dim = layer_sizes[i + 1]

            # MaxPlus activation
            layers.append(MaxPlusAffine(dim, use_gpu=use_gpu))

            # MinPlus activation
            layers.append(MinPlusAffine(dim, use_gpu=use_gpu))

            # Linear for dimension change to next size
            if dim != next_dim:
                layers.append(nn.Linear(dim, next_dim))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class TropicalAutoencoder(nn.Module):
    """
    Tropical Autoencoder for representation learning.

    Args:
        input_dim: Input dimension.
        hidden_dims: List of encoder hidden dimensions.
        latent_dim: Latent space dimension.
        use_gpu: Use GPU acceleration.
        dropout: Dropout probability.

    Example:
        >>> model = TropicalAutoencoder(784, [256, 128], 32)
        >>> x = torch.randn(32, 784)
        >>> recon = model(x)
        >>> latent = model.encode(x)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        latent_dim: int,
        use_gpu: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Encoder
        enc_sizes = [input_dim] + hidden_dims + [latent_dim]
        self.encoder = TropicalNN(enc_sizes, use_gpu=use_gpu, dropout=dropout)

        # Decoder (mirror)
        dec_sizes = [latent_dim] + hidden_dims[::-1] + [input_dim]
        self.decoder = TropicalNN(dec_sizes, use_gpu=use_gpu, dropout=dropout)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


# Alias for backward compatibility
MMPAutoencoder = TropicalAutoencoder


def create_tropical_nn(
    architecture: str,
    input_dim: int,
    num_classes: int,
    use_gpu: bool = False,
    dropout: float = 0.0,
) -> TropicalNN:
    """
    Factory function to create Tropical NNs.

    Args:
        architecture: One of "tiny", "small", "medium", "large"
        input_dim: Input dimension
        num_classes: Number of output classes
        use_gpu: Use GPU acceleration
        dropout: Dropout probability

    Returns:
        TropicalNN model

    Example:
        >>> model = create_tropical_nn("medium", 784, 10)
    """
    architectures = {
        "tiny": [input_dim, 64, num_classes],
        "small": [input_dim, 128, 64, num_classes],
        "medium": [input_dim, 256, 128, num_classes],
        "large": [input_dim, 512, 256, 128, num_classes],
    }

    if architecture not in architectures:
        raise ValueError(f"Unknown: {architecture}. Choose from {list(architectures.keys())}")

    return TropicalNN(architectures[architecture], use_gpu=use_gpu, dropout=dropout)


# Alias for backward compatibility
create_mmpnn = create_tropical_nn


def create_hybrid_nn(
    architecture: str,
    input_dim: int,
    num_classes: int,
    use_gpu: bool = False,
    dropout: float = 0.0,
) -> HybridTropicalNN:
    """
    Factory function to create Hybrid Tropical NNs.

    Uses the simpler architecture with only TropicalAffine (no MinPlus).
    This is the recommended architecture for most applications.

    Args:
        architecture: One of "tiny", "small", "medium", "large"
        input_dim: Input dimension
        num_classes: Number of output classes
        use_gpu: Use GPU acceleration
        dropout: Dropout probability

    Returns:
        HybridTropicalNN model

    Example:
        >>> model = create_hybrid_nn("medium", 784, 10)
    """
    architectures = {
        "tiny": [input_dim, 64, num_classes],
        "small": [input_dim, 128, 64, num_classes],
        "medium": [input_dim, 256, 128, num_classes],
        "large": [input_dim, 512, 256, 128, num_classes],
    }

    if architecture not in architectures:
        raise ValueError(f"Unknown: {architecture}. Choose from {list(architectures.keys())}")

    return HybridTropicalNN(architectures[architecture], use_gpu=use_gpu, dropout=dropout)


__all__ = [
    # Recommended (Hybrid architecture)
    "HybridTropicalNN",
    "HybridClassifier",
    "create_hybrid_nn",
    # Full MMP (MaxPlus + MinPlus)
    "TropicalNN",
    "TropicalClassifier",
    "PureTropicalNN",
    "TropicalAutoencoder",
    "create_tropical_nn",
    # Aliases
    "MMPNN",
    "MMPClassifier",
    "MMPAutoencoder",
    "create_mmpnn",
]
