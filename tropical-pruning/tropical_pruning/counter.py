"""
Winner Counter: Tracks which neurons "win" in tropical max-plus operations.

In tropical GEMM: C_ij = max_k(A_ik + B_kj)
The argmax indices reveal which neurons (index k) actually contribute to the output.
Neurons with low "winner count" are geometrically useless and can be pruned.

Uses tropical-gemm library for high-performance Rust/CUDA implementation.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from tqdm import tqdm
import tropical_gemm as tg


@dataclass
class WinnerStatistics:
    """Statistics collected from tropical forward passes for a single layer."""

    layer_name: str
    # Shape: (num_neurons,) - how many times each neuron achieved argmax
    winner_count: torch.Tensor
    # Total number of output positions processed
    total_positions: int
    # Shape: (num_neurons,) - sum of margins when winning (gap to 2nd place)
    margin_sum: Optional[torch.Tensor] = None
    # Shape: (num_neurons,) - count for margin averaging
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

    def to(self, device: torch.device) -> "WinnerStatistics":
        """Move statistics to specified device."""
        return WinnerStatistics(
            layer_name=self.layer_name,
            winner_count=self.winner_count.to(device),
            total_positions=self.total_positions,
            margin_sum=self.margin_sum.to(device) if self.margin_sum is not None else None,
            margin_count=self.margin_count.to(device) if self.margin_count is not None else None,
        )


class WinnerCounter:
    """
    Collects winner statistics from tropical forward passes.

    This class wraps a model and tracks argmax indices during tropical GEMM
    operations to identify which neurons contribute to the output.

    Example:
        >>> model = MyModel()
        >>> counter = WinnerCounter(model)
        >>> for batch, _ in calibration_loader:
        ...     counter.forward(batch)
        >>> stats = counter.get_statistics()
        >>> print(stats['layer1'].winner_frequency)
    """

    def __init__(
        self,
        model: nn.Module,
        layers: Optional[List[str]] = None,
        track_margin: bool = True,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the WinnerCounter.

        Args:
            model: The neural network model to analyze.
            layers: List of layer names to track. If None, tracks all Linear layers.
            track_margin: Whether to track winner margins (gap to 2nd place).
            device: Device for computation. If None, uses model's device.
        """
        self.model = model
        self.track_margin = track_margin
        self.device = device or next(model.parameters()).device

        # Identify layers to track
        if layers is None:
            self.layer_names = self._find_linear_layers()
        else:
            self.layer_names = layers

        # Initialize counters for each layer
        self._counters: Dict[str, _LayerCounter] = {}
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []

        self._setup_hooks()

    def _find_linear_layers(self) -> List[str]:
        """Find all Linear layers in the model."""
        linear_layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                linear_layers.append(name)
        return linear_layers

    def _setup_hooks(self) -> None:
        """Register forward hooks on target layers."""
        for name, module in self.model.named_modules():
            if name in self.layer_names:
                if isinstance(module, nn.Linear):
                    # Get output dimension for this layer
                    out_features = module.out_features
                    in_features = module.in_features

                    # Initialize counter for this layer
                    self._counters[name] = _LayerCounter(
                        name=name,
                        num_neurons=in_features,  # Track input neurons (the k dimension)
                        track_margin=self.track_margin,
                        device=self.device,
                    )

                    # Register hook
                    hook = module.register_forward_hook(
                        self._create_hook(name, module)
                    )
                    self._hooks.append(hook)

    def _create_hook(self, layer_name: str, module: nn.Linear):
        """Create a forward hook for the specified layer."""

        def hook(module: nn.Linear, input: Tuple[torch.Tensor, ...], output: torch.Tensor):
            # input[0] shape: (batch, ..., in_features) or (batch, seq_len, in_features)
            # weight shape: (out_features, in_features)
            x = input[0]
            weight = module.weight

            # Use tropical-gemm library for high-performance computation
            argmax_indices, margin = self._tropical_forward(x, weight)

            # Update counter
            self._counters[layer_name].update(argmax_indices, margin)

        return hook

    def _tropical_forward(
        self, x: torch.Tensor, weight: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute tropical max-plus with argmax using tropical-gemm library.

        Uses GPU implementation when available, otherwise falls back to CPU SIMD.
        """
        # tropical_gemm.maxplus_matmul_with_argmax expects:
        # A: (M, K), B: (K, N) -> C: (M, N) flattened, argmax: (M, N) flattened
        # We have x: (batch, in_features), weight: (out_features, in_features)
        # So we compute: x @ weight.T in tropical sense

        # Flatten batch dimensions for tropical_gemm
        original_shape = x.shape[:-1]
        M = x[..., 0].numel()  # batch size (flattened)
        K = x.shape[-1]  # in_features
        N = weight.shape[0]  # out_features

        x_flat = x.reshape(-1, K)  # (M, K)

        # We need x @ W^T, so B = W^T with shape (K, N)
        weight_t = weight.t().contiguous()  # (K, N)

        # Use DLPack for zero-copy GPU tensor exchange when available
        _dlpack_available = hasattr(tg, 'maxplus_matmul_dlpack')

        if x.is_cuda and _dlpack_available:
            # Zero-copy path: pass GPU tensors directly via DLPack
            x_contig = x_flat.contiguous()
            _, argmax_flat = tg.maxplus_matmul_dlpack(x_contig, weight_t)
        else:
            # Fallback: convert to numpy for CPU backend
            if x.is_cuda:
                print("Warning: Creating CPU copy for GPU tensor. This is inefficient.")
            x_np = x_flat.detach().cpu().numpy().astype('float32')
            w_np = weight_t.detach().cpu().numpy().astype('float32')
            _, argmax_flat = tg.maxplus_matmul_cpu_with_argmax(x_np, w_np)

        # Reshape from flattened (M*N,) to (M, N) then to original batch shape
        argmax_indices = torch.from_numpy(argmax_flat).reshape(M, N)
        argmax_indices = argmax_indices.reshape(*original_shape, N).to(x.device)

        # Compute margin if needed (gap to 2nd place)
        margin = None
        if self.track_margin:
            margin = self._compute_margin(x, weight, argmax_indices)

        return argmax_indices, margin

    def _compute_margin(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        argmax_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the margin (gap to 2nd place) for each output position."""
        # Reshape for broadcasting: x is (..., in_features), weight is (out, in)
        x_expanded = x.unsqueeze(-2)  # (..., 1, in_features)
        tropical_sum = x_expanded + weight  # (..., out_features, in_features)

        # Get max values
        max_values = tropical_sum.max(dim=-1).values

        # Mask out the max and find second max
        mask = torch.zeros_like(tropical_sum, dtype=torch.bool)
        mask.scatter_(-1, argmax_indices.unsqueeze(-1), True)
        masked = tropical_sum.masked_fill(mask, float('-inf'))
        second_max = masked.max(dim=-1).values

        margin = max_values - second_max
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
        dataloader: torch.utils.data.DataLoader,
        num_batches: Optional[int] = None,
        show_progress: bool = True,
    ) -> Dict[str, WinnerStatistics]:
        """
        Collect statistics from a dataloader.

        Args:
            dataloader: DataLoader providing calibration data.
            num_batches: Maximum number of batches to process. If None, processes all.
            show_progress: Whether to show a progress bar.

        Returns:
            Dictionary mapping layer names to their WinnerStatistics.
        """
        self.model.eval()

        iterator = dataloader
        if show_progress:
            total = num_batches if num_batches else len(dataloader)
            iterator = tqdm(dataloader, total=total, desc="Collecting statistics")

        for i, batch in enumerate(iterator):
            if num_batches is not None and i >= num_batches:
                break

            # Handle different batch formats
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch

            self.forward(x)

        return self.get_statistics()

    def get_statistics(self) -> Dict[str, WinnerStatistics]:
        """
        Get collected winner statistics for all layers.

        Returns:
            Dictionary mapping layer names to their WinnerStatistics.
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


class _LayerCounter:
    """Internal counter for a single layer."""

    def __init__(
        self,
        name: str,
        num_neurons: int,
        track_margin: bool = True,
        device: Optional[torch.device] = None,
    ):
        self.name = name
        self.num_neurons = num_neurons
        self.track_margin = track_margin
        self.device = device or torch.device("cpu")

        self.winner_count = torch.zeros(num_neurons, dtype=torch.long, device=self.device)
        self.total_positions = 0

        if track_margin:
            self.margin_sum = torch.zeros(num_neurons, dtype=torch.float32, device=self.device)
            self.margin_count = torch.zeros(num_neurons, dtype=torch.long, device=self.device)
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
        flat_indices = argmax_indices.flatten()
        self.total_positions += flat_indices.numel()

        # Count winners
        ones = torch.ones_like(flat_indices, dtype=torch.long)
        self.winner_count.scatter_add_(0, flat_indices, ones)

        # Track margins if enabled
        if self.track_margin and margin is not None:
            flat_margin = margin.flatten()
            # Only count valid margins (not inf)
            valid_mask = torch.isfinite(flat_margin)
            if valid_mask.any():
                valid_indices = flat_indices[valid_mask]
                valid_margins = flat_margin[valid_mask]

                self.margin_sum.scatter_add_(0, valid_indices, valid_margins)
                margin_ones = torch.ones_like(valid_indices, dtype=torch.long)
                self.margin_count.scatter_add_(0, valid_indices, margin_ones)

    def get_statistics(self) -> WinnerStatistics:
        """Return WinnerStatistics for this layer."""
        return WinnerStatistics(
            layer_name=self.name,
            winner_count=self.winner_count.clone(),
            total_positions=self.total_positions,
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


