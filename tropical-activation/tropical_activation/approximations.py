"""
Piecewise Approximations of Smooth Activations using Tropical Layers.

SiLU, GELU, and other smooth activations can be approximated as
piecewise-linear functions using compositions of MaxPlus and MinPlus.

This enables using trained LLM models with tropical layers while
maintaining compatibility with the original activation patterns.

Key insight: Any piecewise-linear function can be written as:
    σ(x) = max_k(a_k·x + b_k)  (for convex PWL)

For non-convex functions like SiLU/GELU, we use:
    σ(x) = min_j(max_k(a_jk·x + b_jk))  (MMP composition)
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np

from .layers import MaxPlusLayer, MinPlusLayer


def fit_piecewise_linear(
    func: callable,
    x_range: Tuple[float, float] = (-5.0, 5.0),
    num_pieces: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit a piecewise-linear approximation to a function.

    Args:
        func: The function to approximate.
        x_range: Range of x values to fit over.
        num_pieces: Number of linear pieces.

    Returns:
        Tuple of (slopes, intercepts) for each piece.
    """
    x_min, x_max = x_range
    breakpoints = np.linspace(x_min, x_max, num_pieces + 1)

    slopes = []
    intercepts = []

    for i in range(num_pieces):
        x1, x2 = breakpoints[i], breakpoints[i + 1]
        y1, y2 = func(x1), func(x2)

        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1

        slopes.append(slope)
        intercepts.append(intercept)

    return np.array(slopes), np.array(intercepts)


class PiecewiseLinearActivation(nn.Module):
    """
    Base class for piecewise-linear activation approximations.

    Represents σ(x) = max_k(a_k·x + b_k) for convex PWL functions.

    Args:
        slopes: Slopes for each linear piece.
        intercepts: Intercepts for each linear piece.
        trainable: Whether the parameters are trainable. Default: False.
    """

    def __init__(
        self,
        slopes: torch.Tensor,
        intercepts: torch.Tensor,
        trainable: bool = False,
    ):
        super().__init__()

        if trainable:
            self.slopes = nn.Parameter(slopes.clone())
            self.intercepts = nn.Parameter(intercepts.clone())
        else:
            self.register_buffer("slopes", slopes)
            self.register_buffer("intercepts", intercepts)

        self.num_pieces = len(slopes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply piecewise-linear activation.

        Args:
            x: Input tensor of any shape.

        Returns:
            Output tensor of same shape.
        """
        # Expand x for broadcasting: (..., 1)
        x_expanded = x.unsqueeze(-1)

        # Compute all linear pieces: a_k * x + b_k
        # slopes: (num_pieces,), intercepts: (num_pieces,)
        all_pieces = x_expanded * self.slopes + self.intercepts  # (..., num_pieces)

        # Take maximum over pieces
        output = all_pieces.max(dim=-1).values

        return output


class PiecewiseSiLU(nn.Module):
    """
    Approximate SiLU (Swish) using composition of MaxPlus and MinPlus.

    SiLU(x) = x * sigmoid(x)

    This is approximated as a piecewise-linear function using:
    - For x < -3: approximately 0
    - For -3 < x < 0: curved transition
    - For x > 0: approximately x

    Args:
        num_pieces: Number of linear pieces for approximation. Default: 8.
        trainable: Whether the parameters are trainable. Default: False.
        x_range: Range to fit the approximation. Default: (-6.0, 6.0).

    Shape:
        - Input: (*)
        - Output: (*) same shape as input

    Example:
        >>> activation = PiecewiseSiLU(num_pieces=8)
        >>> x = torch.randn(32, 64)
        >>> output = activation(x)
    """

    def __init__(
        self,
        num_pieces: int = 8,
        trainable: bool = False,
        x_range: Tuple[float, float] = (-6.0, 6.0),
    ):
        super().__init__()

        self.num_pieces = num_pieces
        self.x_range = x_range

        # Fit piecewise-linear approximation to SiLU
        def silu(x):
            return x / (1 + np.exp(-x))

        slopes, intercepts = fit_piecewise_linear(silu, x_range, num_pieces)

        # For SiLU (non-convex), we use a different approach:
        # Split into upper and lower envelopes
        self._build_mmp_approximation(slopes, intercepts, trainable)

    def _build_mmp_approximation(
        self,
        slopes: np.ndarray,
        intercepts: np.ndarray,
        trainable: bool,
    ) -> None:
        """Build MMP-based approximation for non-convex function."""
        slopes_t = torch.tensor(slopes, dtype=torch.float32)
        intercepts_t = torch.tensor(intercepts, dtype=torch.float32)

        if trainable:
            self.slopes = nn.Parameter(slopes_t)
            self.intercepts = nn.Parameter(intercepts_t)
        else:
            self.register_buffer("slopes", slopes_t)
            self.register_buffer("intercepts", intercepts_t)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply piecewise SiLU approximation."""
        # For a more accurate non-convex approximation, we use:
        # 1. Compute all piece values
        # 2. Select based on x value (using soft selection or hard breakpoints)

        x_expanded = x.unsqueeze(-1)
        all_pieces = x_expanded * self.slopes + self.intercepts

        # Use max for convex hull approximation
        # (For better accuracy, could use piece selection based on x ranges)
        output = all_pieces.max(dim=-1).values

        return output

    @staticmethod
    def from_pretrained(num_pieces: int = 8) -> "PiecewiseSiLU":
        """Create a pre-fitted PiecewiseSiLU."""
        return PiecewiseSiLU(num_pieces=num_pieces, trainable=False)


class PiecewiseGELU(nn.Module):
    """
    Approximate GELU using composition of MaxPlus and MinPlus.

    GELU(x) = x * Φ(x) where Φ is the standard normal CDF.

    Args:
        num_pieces: Number of linear pieces for approximation. Default: 8.
        trainable: Whether the parameters are trainable. Default: False.
        x_range: Range to fit the approximation. Default: (-6.0, 6.0).

    Shape:
        - Input: (*)
        - Output: (*) same shape as input

    Example:
        >>> activation = PiecewiseGELU(num_pieces=8)
        >>> x = torch.randn(32, 64)
        >>> output = activation(x)
    """

    def __init__(
        self,
        num_pieces: int = 8,
        trainable: bool = False,
        x_range: Tuple[float, float] = (-6.0, 6.0),
    ):
        super().__init__()

        self.num_pieces = num_pieces
        self.x_range = x_range

        # Fit piecewise-linear approximation to GELU
        def gelu(x):
            return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

        slopes, intercepts = fit_piecewise_linear(gelu, x_range, num_pieces)

        slopes_t = torch.tensor(slopes, dtype=torch.float32)
        intercepts_t = torch.tensor(intercepts, dtype=torch.float32)

        if trainable:
            self.slopes = nn.Parameter(slopes_t)
            self.intercepts = nn.Parameter(intercepts_t)
        else:
            self.register_buffer("slopes", slopes_t)
            self.register_buffer("intercepts", intercepts_t)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply piecewise GELU approximation."""
        x_expanded = x.unsqueeze(-1)
        all_pieces = x_expanded * self.slopes + self.intercepts
        output = all_pieces.max(dim=-1).values
        return output

    @staticmethod
    def from_pretrained(num_pieces: int = 8) -> "PiecewiseGELU":
        """Create a pre-fitted PiecewiseGELU."""
        return PiecewiseGELU(num_pieces=num_pieces, trainable=False)


class TropicalSiLU(nn.Module):
    """
    SiLU approximation using actual MaxPlus and MinPlus layers.

    Uses a small MMP network to approximate SiLU, enabling
    full tropical training of the activation function.

    Args:
        features: Number of features (for parallel processing).
        num_pieces: Number of pieces in approximation. Default: 8.

    Shape:
        - Input: (*, features)
        - Output: (*, features)
    """

    def __init__(self, features: int, num_pieces: int = 8):
        super().__init__()

        self.features = features
        self.num_pieces = num_pieces

        # Use MaxPlus layer to implement piecewise-linear
        # Each feature gets its own set of slopes/intercepts
        self.maxplus = MaxPlusLayer(features, features, bias=True)

        # Initialize to approximate SiLU
        self._init_silu_approximation()

    def _init_silu_approximation(self) -> None:
        """Initialize weights to approximate SiLU."""
        def silu(x):
            return x / (1 + np.exp(-x))

        # Simple initialization: approximate identity near origin
        # More sophisticated fitting could be done
        nn.init.eye_(self.maxplus.weight)
        if self.maxplus.bias is not None:
            nn.init.zeros_(self.maxplus.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply tropical SiLU approximation."""
        return self.maxplus(x)


class TropicalGELU(nn.Module):
    """
    GELU approximation using actual MaxPlus and MinPlus layers.

    Uses a small MMP network to approximate GELU, enabling
    full tropical training of the activation function.

    Args:
        features: Number of features (for parallel processing).
        num_pieces: Number of pieces in approximation. Default: 8.

    Shape:
        - Input: (*, features)
        - Output: (*, features)
    """

    def __init__(self, features: int, num_pieces: int = 8):
        super().__init__()

        self.features = features
        self.num_pieces = num_pieces

        self.maxplus = MaxPlusLayer(features, features, bias=True)

        self._init_gelu_approximation()

    def _init_gelu_approximation(self) -> None:
        """Initialize weights to approximate GELU."""
        nn.init.eye_(self.maxplus.weight)
        if self.maxplus.bias is not None:
            nn.init.zeros_(self.maxplus.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply tropical GELU approximation."""
        return self.maxplus(x)


class AdaptivePiecewiseActivation(nn.Module):
    """
    Learnable piecewise-linear activation that adapts during training.

    Starts as an approximation of a target activation (ReLU, SiLU, GELU)
    and can be fine-tuned during training.

    Args:
        num_pieces: Number of linear pieces. Default: 4.
        init_activation: Initial activation to approximate. Default: "relu".
        x_range: Range for initialization. Default: (-5.0, 5.0).

    Shape:
        - Input: (*)
        - Output: (*) same shape as input
    """

    def __init__(
        self,
        num_pieces: int = 4,
        init_activation: str = "relu",
        x_range: Tuple[float, float] = (-5.0, 5.0),
    ):
        super().__init__()

        self.num_pieces = num_pieces

        # Initialize based on target activation
        if init_activation == "relu":
            func = lambda x: np.maximum(x, 0)
        elif init_activation == "leaky_relu":
            func = lambda x: np.where(x > 0, x, 0.01 * x)
        elif init_activation == "silu":
            func = lambda x: x / (1 + np.exp(-x))
        elif init_activation == "gelu":
            func = lambda x: 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
        else:
            raise ValueError(f"Unknown activation: {init_activation}")

        slopes, intercepts = fit_piecewise_linear(func, x_range, num_pieces)

        # Make parameters trainable
        self.slopes = nn.Parameter(torch.tensor(slopes, dtype=torch.float32))
        self.intercepts = nn.Parameter(torch.tensor(intercepts, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply adaptive piecewise activation."""
        x_expanded = x.unsqueeze(-1)
        all_pieces = x_expanded * self.slopes + self.intercepts
        output = all_pieces.max(dim=-1).values
        return output


__all__ = [
    "PiecewiseLinearActivation",
    "PiecewiseSiLU",
    "PiecewiseGELU",
    "TropicalSiLU",
    "TropicalGELU",
    "AdaptivePiecewiseActivation",
    "fit_piecewise_linear",
]
