"""
Tropical Activation: Neural Networks with Tropical Algebra.

Replaces traditional activation functions with tropical affine layers.
Uses only additions and max/min operations instead of multiplications.

Key components:
- MaxPlusAffine, MinPlusAffine: Core tropical layers (square, with LayerNorm)
- TropicalBlock: Linear → MaxPlus → MinPlus (building block)
- TropicalNN: Full tropical neural network

Architecture:
    Linear(in→out) → MaxPlusAffine(out) → MinPlusAffine(out) → Linear → ...

Mathematical foundation:
- MaxPlusAffine: y[i] = max(max_k(LayerNorm(x)[k] + W[k,i]), b[i])
- MinPlusAffine: y[i] = min(min_k(LayerNorm(x)[k] + W[k,i]), b[i])

Reference: Luo & Fan 2021 - "Min-Max-Plus Neural Networks"
https://arxiv.org/abs/2102.06358

Example:
    >>> import torch
    >>> from tropical_activation import TropicalNN, MaxPlusAffine
    >>>
    >>> # Create a tropical classifier
    >>> model = TropicalNN([784, 256, 128, 10])
    >>> x = torch.randn(32, 784)
    >>> logits = model(x)
    >>>
    >>> # Or use individual layers
    >>> layer = MaxPlusAffine(256)  # square: 256 → 256
    >>> output = layer(torch.randn(32, 256))
"""

__version__ = "0.2.0"

# Core layers
from .layers import (
    MaxPlusAffine,
    MinPlusAffine,
    MaxPlusLayer,  # Alias
    MinPlusLayer,  # Alias
    TropicalReLU,
    TropicalLeakyReLU,
    TROPICAL_GEMM_AVAILABLE,
    GPU_AVAILABLE,
)

# Building blocks
from .blocks import (
    TropicalBlock,
    ResidualTropicalBlock,
    MaxPlusBlock,
    MinPlusBlock,
    TropicalMLP,
    MMPBlock,  # Alias
    ResidualMMPBlock,  # Alias
)

# Complete models
from .models import (
    TropicalNN,
    TropicalClassifier,
    PureTropicalNN,
    TropicalAutoencoder,
    create_tropical_nn,
    MMPNN,  # Alias
    MMPClassifier,  # Alias
    MMPAutoencoder,  # Alias
    create_mmpnn,  # Alias
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
    # Layers (new names)
    "MaxPlusAffine",
    "MinPlusAffine",
    "MaxPlusLayer",
    "MinPlusLayer",
    "TropicalReLU",
    "TropicalLeakyReLU",
    "TROPICAL_GEMM_AVAILABLE",
    "GPU_AVAILABLE",
    # Blocks (new names)
    "TropicalBlock",
    "ResidualTropicalBlock",
    "MaxPlusBlock",
    "MinPlusBlock",
    "TropicalMLP",
    "MMPBlock",
    "ResidualMMPBlock",
    # Models (new names)
    "TropicalNN",
    "TropicalClassifier",
    "PureTropicalNN",
    "TropicalAutoencoder",
    "create_tropical_nn",
    "MMPNN",
    "MMPClassifier",
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
