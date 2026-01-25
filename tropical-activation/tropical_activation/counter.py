"""
Winner Counter for Tropical Layers.

Tracks which inputs "win" in MaxPlus/MinPlus operations.
Useful for analyzing tropical network behavior and importance.

In tropical operations:
- MaxPlus: y_j = max_k(x_k + W_kj) - tracks which k achieves the max
- MinPlus: y_j = min_k(x_k + W_kj) - tracks which k achieves the min

Uses tropical-gemm library for high-performance argmax computation.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import tropical_gemm as tg

from .layers import MaxPlusLayer, MinPlusLayer


@dataclass
class TropicalStatistics:
    """Statistics collected from tropical forward passes for a single layer."""

    layer_name: str
    # Shape: (in_features,) - how many times each input won
    winner_count: torch.Tensor
    # Total number of output positions processed
    total_positions: int
    # Layer type: "maxplus" or "minplus"
    layer_type: str
    # Shape: (in_features,) - sum of margins when winning (optional)
    margin_sum: Optional[torch.Tensor] = None
    # Shape: (in_features,) - count for margin averaging (optional)
    margin_count: Optional[torch.Tensor] = None

    @property
    def winner_frequency(self) -> torch.Tensor:
        """Normalized winner count (count / total_positions)."""
        return self.winner_count.float() / max(self.total_positions, 1)

    @property
    def average_margin(self) -> Optional[torch.Tensor]:
        """Average margin when winning. Higher = more confident importance."""
        if self.margin_sum is None or self.margin_count is None:
            return None
        return self.margin_sum / self.margin_count.clamp(min=1)

    def to(self, device: torch.device) -> "TropicalStatistics":
        """Move statistics to specified device."""
        return TropicalStatistics(
            layer_name=self.layer_name,
            winner_count=self.winner_count.to(device),
            total_positions=self.total_positions,
            layer_type=self.layer_type,
            margin_sum=self.margin_sum.to(device) if self.margin_sum is not None else None,
            margin_count=self.margin_count.to(device) if self.margin_count is not None else None,
        )


class TropicalWinnerCounter:
    """
    Tracks which inputs win in MaxPlus/MinPlus layers.

    This class wraps a model and tracks argmax/argmin indices during
    tropical operations to identify which inputs contribute to the output.

    Example:
        >>> model = MMPNN([784, 256, 128, 10])
        >>> counter = TropicalWinnerCounter(model)
        >>> stats = counter.collect(dataloader)
        >>> print(stats['maxplus_0'].winner_frequency)
    """

    def __init__(
        self,
        model: nn.Module,
        layers: Optional[List[str]] = None,
        track_margin: bool = False,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the TropicalWinnerCounter.

        Args:
            model: The neural network model to analyze.
            layers: List of layer names to track. If None, tracks all tropical layers.
            track_margin: Whether to track winner margins (gap to 2nd place).
            device: Device for computation. If None, uses model's device.
        """
        self.model = model
        self.track_margin = track_margin

        # Try to get device from model parameters
        try:
            self.device = device or next(model.parameters()).device
        except StopIteration:
            self.device = device or torch.device("cpu")

        # Identify layers to track
        if layers is None:
            self.layer_names = self._find_tropical_layers()
        else:
            self.layer_names = layers

        # Initialize counters for each layer
        self._counters: Dict[str, _TropicalLayerCounter] = {}
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []

        self._setup_hooks()

    def _find_tropical_layers(self) -> List[str]:
        """Find all MaxPlus and MinPlus layers in the model."""
        tropical_layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, (MaxPlusLayer, MinPlusLayer)):
                tropical_layers.append(name)
        return tropical_layers

    def _setup_hooks(self) -> None:
        """Register forward hooks on target layers."""
        for name, module in self.model.named_modules():
            if name in self.layer_names:
                if isinstance(module, (MaxPlusLayer, MinPlusLayer)):
                    layer_type = "maxplus" if isinstance(module, MaxPlusLayer) else "minplus"
                    in_features = module.in_features

                    # Initialize counter for this layer
                    self._counters[name] = _TropicalLayerCounter(
                        name=name,
                        num_inputs=in_features,
                        layer_type=layer_type,
                        track_margin=self.track_margin,
                        device=self.device,
                    )

                    # Register hook
                    hook = module.register_forward_hook(
                        self._create_hook(name, module, layer_type)
                    )
                    self._hooks.append(hook)

    def _create_hook(self, layer_name: str, module: nn.Module, layer_type: str):
        """Create a forward hook for the specified layer."""

        def hook(module, input: Tuple[torch.Tensor, ...], output: torch.Tensor):
            x = input[0]
            weight = module.weight

            # Compute argmax/argmin using tropical-gemm
            argmax_indices, margin = self._compute_winners(x, weight, layer_type)

            # Update counter
            self._counters[layer_name].update(argmax_indices, margin)

        return hook

    def _compute_winners(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        layer_type: str,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute winner indices using tropical-gemm library.

        Args:
            x: Input tensor of shape (*, in_features)
            weight: Weight tensor of shape (in_features, out_features)
            layer_type: "maxplus" or "minplus"

        Returns:
            Tuple of (argmax_indices, margin)
        """
        # Flatten batch dimensions
        original_shape = x.shape[:-1]
        M = x[..., 0].numel()
        K = x.shape[-1]
        N = weight.shape[1]

        x_flat = x.reshape(-1, K)

        # Use tropical-gemm for optimized computation
        _dlpack_available = hasattr(tg, 'maxplus_matmul_dlpack')

        if x.is_cuda and _dlpack_available:
            # GPU path with DLPack
            x_contig = x_flat.contiguous()
            weight_contig = weight.contiguous()

            if layer_type == "maxplus":
                _, argmax_flat = tg.maxplus_matmul_dlpack(x_contig, weight_contig)
            else:
                _, argmax_flat = tg.minplus_matmul_dlpack(x_contig, weight_contig)
        else:
            # CPU path
            x_np = x_flat.detach().cpu().numpy().astype(np.float32)
            w_np = weight.detach().cpu().numpy().astype(np.float32)

            if not x_np.flags["C_CONTIGUOUS"]:
                x_np = np.ascontiguousarray(x_np)
            if not w_np.flags["C_CONTIGUOUS"]:
                w_np = np.ascontiguousarray(w_np)

            if layer_type == "maxplus":
                _, argmax_flat = tg.maxplus_matmul_with_argmax(x_np, w_np)
            else:
                _, argmax_flat = tg.minplus_matmul_with_argmax(x_np, w_np)

        # Reshape argmax indices
        argmax_indices = torch.from_numpy(np.array(argmax_flat).reshape(M, N))
        argmax_indices = argmax_indices.reshape(*original_shape, N).to(x.device)

        # Compute margin if needed
        margin = None
        if self.track_margin:
            margin = self._compute_margin(x, weight, argmax_indices, layer_type)

        return argmax_indices, margin

    def _compute_margin(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        argmax_indices: torch.Tensor,
        layer_type: str,
    ) -> torch.Tensor:
        """Compute the margin (gap to 2nd place) for each output position."""
        # x: (..., in_features), weight: (in_features, out_features)
        x_expanded = x.unsqueeze(-1)  # (..., in_features, 1)
        tropical_sum = x_expanded + weight  # (..., in_features, out_features)

        # Get extreme values
        if layer_type == "maxplus":
            extreme_values = tropical_sum.max(dim=-2).values
            # Mask out the max and find second max
            mask = torch.zeros_like(tropical_sum, dtype=torch.bool)
            mask.scatter_(-2, argmax_indices.unsqueeze(-2), True)
            masked = tropical_sum.masked_fill(mask, float('-inf'))
            second_extreme = masked.max(dim=-2).values
        else:
            extreme_values = tropical_sum.min(dim=-2).values
            mask = torch.zeros_like(tropical_sum, dtype=torch.bool)
            mask.scatter_(-2, argmax_indices.unsqueeze(-2), True)
            masked = tropical_sum.masked_fill(mask, float('inf'))
            second_extreme = masked.min(dim=-2).values

        margin = (extreme_values - second_extreme).abs()
        return margin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run forward pass and collect winner statistics.

        Args:
            x: Input tensor.

        Returns:
            Model output.
        """
        x = x.to(self.device)
        with torch.no_grad():
            return self.model(x)

    def collect(
        self,
        dataloader: DataLoader,
        num_batches: Optional[int] = None,
        show_progress: bool = True,
    ) -> Dict[str, TropicalStatistics]:
        """
        Collect statistics from a dataloader.

        Args:
            dataloader: DataLoader providing calibration data.
            num_batches: Maximum number of batches to process. If None, processes all.
            show_progress: Whether to show a progress bar.

        Returns:
            Dictionary mapping layer names to their TropicalStatistics.
        """
        self.model.eval()

        iterator = dataloader
        if show_progress:
            total = num_batches if num_batches else len(dataloader)
            iterator = tqdm(dataloader, total=total, desc="Collecting tropical statistics")

        for i, batch in enumerate(iterator):
            if num_batches is not None and i >= num_batches:
                break

            # Handle different batch formats
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch

            # Flatten if needed (for image data)
            if x.dim() > 2:
                x = x.view(x.size(0), -1)

            self.forward(x)

        return self.get_statistics()

    def get_statistics(self) -> Dict[str, TropicalStatistics]:
        """
        Get collected winner statistics for all layers.

        Returns:
            Dictionary mapping layer names to their TropicalStatistics.
        """
        return {name: counter.get_statistics() for name, counter in self._counters.items()}

    def reset(self) -> None:
        """Reset all counters to zero."""
        for counter in self._counters.values():
            counter.reset()

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def __del__(self):
        """Clean up hooks on deletion."""
        self.remove_hooks()


class _TropicalLayerCounter:
    """Internal counter for a single tropical layer."""

    def __init__(
        self,
        name: str,
        num_inputs: int,
        layer_type: str,
        track_margin: bool = False,
        device: Optional[torch.device] = None,
    ):
        self.name = name
        self.num_inputs = num_inputs
        self.layer_type = layer_type
        self.track_margin = track_margin
        self.device = device or torch.device("cpu")

        self.winner_count = torch.zeros(num_inputs, dtype=torch.long, device=self.device)
        self.total_positions = 0

        if track_margin:
            self.margin_sum = torch.zeros(num_inputs, dtype=torch.float32, device=self.device)
            self.margin_count = torch.zeros(num_inputs, dtype=torch.long, device=self.device)
        else:
            self.margin_sum = None
            self.margin_count = None

    def update(
        self,
        argmax_indices: torch.Tensor,
        margin: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Update counters with new argmax indices.

        Args:
            argmax_indices: Tensor of argmax indices, shape (..., out_features).
            margin: Optional tensor of margins, shape (..., out_features).
        """
        # Flatten indices
        flat_indices = argmax_indices.flatten().to(self.device)
        self.total_positions += flat_indices.numel()

        # Count winners
        ones = torch.ones_like(flat_indices, dtype=torch.long)
        self.winner_count.scatter_add_(0, flat_indices, ones)

        # Track margins if enabled
        if self.track_margin and margin is not None:
            flat_margin = margin.flatten().to(self.device)
            # Only count valid margins (not inf)
            valid_mask = torch.isfinite(flat_margin)
            if valid_mask.any():
                valid_indices = flat_indices[valid_mask]
                valid_margins = flat_margin[valid_mask]

                self.margin_sum.scatter_add_(0, valid_indices, valid_margins)
                margin_ones = torch.ones_like(valid_indices, dtype=torch.long)
                self.margin_count.scatter_add_(0, valid_indices, margin_ones)

    def get_statistics(self) -> TropicalStatistics:
        """Return TropicalStatistics for this layer."""
        return TropicalStatistics(
            layer_name=self.name,
            winner_count=self.winner_count.clone(),
            total_positions=self.total_positions,
            layer_type=self.layer_type,
            margin_sum=self.margin_sum.clone() if self.margin_sum is not None else None,
            margin_count=self.margin_count.clone() if self.margin_count is not None else None,
        )

    def reset(self) -> None:
        """Reset counters to zero."""
        self.winner_count.zero_()
        self.total_positions = 0
        if self.margin_sum is not None:
            self.margin_sum.zero_()
        if self.margin_count is not None:
            self.margin_count.zero_()


__all__ = [
    "TropicalStatistics",
    "TropicalWinnerCounter",
]
