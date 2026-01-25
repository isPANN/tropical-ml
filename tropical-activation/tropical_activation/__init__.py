"""
Tropical Activation: Min-Max-Plus Neural Networks.

This package implements tropical algebra-based neural networks where
activation functions are replaced with tropical layers (MaxPlus/MinPlus).

Key components:
- MaxPlusLayer, MinPlusLayer: Core tropical layers with autograd support
- MMPBlock: Building block combining Linear → MaxPlus → MinPlus
- MMPNN: Full Min-Max-Plus Neural Network
- Training utilities, conversion functions, and analysis tools

Mathematical foundation:
- MaxPlus: y_j = max_k(x_k + W_kj) + b_j
- MinPlus: y_j = min_k(x_k + W_kj) + b_j

Reference: Luo & Fan 2021 - "Min-Max-Plus Neural Networks"
https://arxiv.org/abs/2102.06358

Example:
    >>> import torch
    >>> from tropical_activation import MMPNN, MaxPlusLayer
    >>>
    >>> # Create an MMP classifier
    >>> model = MMPNN([784, 256, 128, 10])
    >>>
    >>> # Forward pass
    >>> x = torch.randn(32, 784)
    >>> logits = model(x)
    >>>
    >>> # Or use individual layers
    >>> layer = MaxPlusLayer(64, 128)
    >>> output = layer(torch.randn(32, 64))
"""

__version__ = "0.1.0"

# Core layers
from .layers import (
    MaxPlusLayer,
    MinPlusLayer,
    TropicalReLU,
    TropicalLeakyReLU,
)

# Building blocks
from .blocks import (
    MMPBlock,
    ResidualMMPBlock,
    MaxPlusBlock,
    MinPlusBlock,
    TropicalMLP,
)

# Complete models
from .models import (
    MMPNN,
    MMPClassifier,
    PureTropicalNN,
    MMPAutoencoder,
    create_mmpnn,
)

# Training utilities
from .training import (
    TropicalBatchNorm,
    TropicalLayerNorm,
    tropical_weight_init,
    get_tropical_optimizer,
    train_epoch,
    evaluate,
    count_parameters,
    count_operations,
)

# Winner counting / analysis
from .counter import (
    TropicalStatistics,
    TropicalWinnerCounter,
)

# Piecewise approximations
from .approximations import (
    PiecewiseLinearActivation,
    PiecewiseSiLU,
    PiecewiseGELU,
    TropicalSiLU,
    TropicalGELU,
    AdaptivePiecewiseActivation,
    fit_piecewise_linear,
)

# Conversion utilities
from .conversion import (
    convert_relu_to_maxplus,
    convert_activation_to_tropical,
    convert_mlp_to_mmp,
    convert_to_mmp,
    convert_mmp_to_standard,
    estimate_multiplication_reduction,
    create_hybrid_model,
)

# Vision models
from .vision import (
    MMPConvClassifier,
    MMPResNet,
    CIFAR10MMP,
    ImageNetMMP,
    create_cifar10_model,
    create_imagenet_model,
)

__all__ = [
    # Version
    "__version__",
    # Layers
    "MaxPlusLayer",
    "MinPlusLayer",
    "TropicalReLU",
    "TropicalLeakyReLU",
    # Blocks
    "MMPBlock",
    "ResidualMMPBlock",
    "MaxPlusBlock",
    "MinPlusBlock",
    "TropicalMLP",
    # Models
    "MMPNN",
    "MMPClassifier",
    "PureTropicalNN",
    "MMPAutoencoder",
    "create_mmpnn",
    # Training
    "TropicalBatchNorm",
    "TropicalLayerNorm",
    "tropical_weight_init",
    "get_tropical_optimizer",
    "train_epoch",
    "evaluate",
    "count_parameters",
    "count_operations",
    # Counter
    "TropicalStatistics",
    "TropicalWinnerCounter",
    # Approximations
    "PiecewiseLinearActivation",
    "PiecewiseSiLU",
    "PiecewiseGELU",
    "TropicalSiLU",
    "TropicalGELU",
    "AdaptivePiecewiseActivation",
    "fit_piecewise_linear",
    # Conversion
    "convert_relu_to_maxplus",
    "convert_activation_to_tropical",
    "convert_mlp_to_mmp",
    "convert_to_mmp",
    "convert_mmp_to_standard",
    "estimate_multiplication_reduction",
    "create_hybrid_model",
    # Vision
    "MMPConvClassifier",
    "MMPResNet",
    "CIFAR10MMP",
    "ImageNetMMP",
    "create_cifar10_model",
    "create_imagenet_model",
]
