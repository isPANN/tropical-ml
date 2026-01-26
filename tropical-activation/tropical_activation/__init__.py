"""
Tropical Activation: Neural Networks with Tropical Algebra.

Replaces traditional activation functions with tropical affine layers.
Uses only additions and max/min operations instead of multiplications.

Key components:
- TropicalAffine (alias for MaxPlusAffine): Recommended tropical layer
- HybridTropicalNN: Recommended architecture (Linear → TropicalAffine)
- TropicalNN: Full MMP architecture (Linear → MaxPlus → MinPlus)

Recommended Architecture (Hybrid):
    Linear(in→out) → TropicalAffine(out) → Linear → TropicalAffine → ... → Linear

Full MMP Architecture:
    Linear(in→out) → MaxPlusAffine(out) → MinPlusAffine(out) → Linear → ...

Mathematical foundation:
- TropicalAffine/MaxPlusAffine: y[i] = max(max_k(LayerNorm(x)[k] + W[k,i]), b[i])
- MinPlusAffine: y[i] = min(min_k(LayerNorm(x)[k] + W[k,i]), b[i])

Reference: Luo & Fan 2021 - "Min-Max-Plus Neural Networks"
https://arxiv.org/abs/2102.06358

Example:
    >>> import torch
    >>> from tropical_activation import HybridTropicalNN, TropicalAffine
    >>>
    >>> # Create a hybrid tropical classifier (recommended)
    >>> model = HybridTropicalNN([784, 256, 128, 10])
    >>> x = torch.randn(32, 784)
    >>> logits = model(x)
    >>>
    >>> # Or use individual layers
    >>> layer = TropicalAffine(256)  # square: 256 → 256
    >>> output = layer(torch.randn(32, 256))
"""

__version__ = "0.2.0"

# Core layers
from .layers import (
    MaxPlusAffine,
    MinPlusAffine,
    TropicalAffine,  # Alias for MaxPlusAffine (recommended)
    MaxPlusLayer,  # Alias
    MinPlusLayer,  # Alias
    TropicalReLU,
    TropicalLeakyReLU,
    TROPICAL_GEMM_AVAILABLE,
    GPU_AVAILABLE,
)

# Building blocks
from .blocks import (
    HybridBlock,  # Recommended: Linear → TropicalAffine
    HybridMLP,
    TropicalBlock,  # Full MMP: Linear → MaxPlus → MinPlus
    ResidualTropicalBlock,
    MaxPlusBlock,
    MinPlusBlock,
    TropicalMLP,
    MMPBlock,  # Alias
    ResidualMMPBlock,  # Alias
)

# Complete models
from .models import (
    # Recommended (Hybrid architecture)
    HybridTropicalNN,
    HybridClassifier,
    create_hybrid_nn,
    # Full MMP (MaxPlus + MinPlus)
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
    # Layers
    "MaxPlusAffine",
    "MinPlusAffine",
    "TropicalAffine",  # Recommended (alias for MaxPlusAffine)
    "MaxPlusLayer",
    "MinPlusLayer",
    "TropicalReLU",
    "TropicalLeakyReLU",
    "TROPICAL_GEMM_AVAILABLE",
    "GPU_AVAILABLE",
    # Blocks - Hybrid (recommended)
    "HybridBlock",
    "HybridMLP",
    # Blocks - Full MMP
    "TropicalBlock",
    "ResidualTropicalBlock",
    "MaxPlusBlock",
    "MinPlusBlock",
    "TropicalMLP",
    "MMPBlock",
    "ResidualMMPBlock",
    # Models - Hybrid (recommended)
    "HybridTropicalNN",
    "HybridClassifier",
    "create_hybrid_nn",
    # Models - Full MMP
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
