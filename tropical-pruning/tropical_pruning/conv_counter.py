"""
Convolutional Winner Counter: Tracks which channels "win" in tropical max-plus
operations for Conv2d layers.

In standard convolution: Y[b, c_out, h, w] = sum_k(conv(X[b, k, :, :], W[c_out, k, :, :]))
In tropical convolution: Y[b, c_out, h, w] = max_k(tropical_conv(X[b, k, :, :], W[c_out, k, :, :]))

We use im2col (F.unfold) to convert convolution to matrix form, then apply tropical GEMM.
The argmax indices reveal which input channels actually contribute to each output position.
Channels with low "winner count" are geometrically useless and can be pruned.

Uses tropical-gemm library for high-performance Rust/SIMD implementation.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import tropical_gemm as tg

from tropical_pruning.counter import WinnerStatistics, _LayerCounter


@dataclass
class ConvWinnerStatistics:
    """
    Statistics collected from tropical forward passes for a Conv2d layer.

    For Conv2d, we track which INPUT CHANNELS win. The argmax in tropical GEMM
    operates over the flattened kernel dimension (C_in * kH * kW), and we map
    back to input channel indices by: channel_idx = argmax_idx // (kH * kW).
    """
    layer_name: str
    # Shape: (in_channels,) - how many times each input channel achieved argmax
    winner_count: torch.Tensor
    # Total number of output positions processed (B * C_out * H_out * W_out)
    total_positions: int
    # Shape: (in_channels,) - sum of margins when winning
    margin_sum: Optional[torch.Tensor] = None
    # Shape: (in_channels,) - count for margin averaging
    margin_count: Optional[torch.Tensor] = None
    # Kernel size for reference
    kernel_size: Tuple[int, int] = (1, 1)

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

    def to(self, device: torch.device) -> "ConvWinnerStatistics":
        """Move statistics to specified device."""
        return ConvWinnerStatistics(
            layer_name=self.layer_name,
            winner_count=self.winner_count.to(device),
            total_positions=self.total_positions,
            margin_sum=self.margin_sum.to(device) if self.margin_sum is not None else None,
            margin_count=self.margin_count.to(device) if self.margin_count is not None else None,
            kernel_size=self.kernel_size,
        )

    def to_winner_statistics(self) -> WinnerStatistics:
        """Convert to generic WinnerStatistics for compatibility with TropicalPruner."""
        return WinnerStatistics(
            layer_name=self.layer_name,
            winner_count=self.winner_count,
            total_positions=self.total_positions,
            margin_sum=self.margin_sum,
            margin_count=self.margin_count,
        )


class ConvWinnerCounter:
    """
    Collects winner statistics from tropical forward passes for Conv2d layers.

    This class wraps a model and tracks argmax indices during tropical convolution
    operations to identify which input channels contribute to the output.

    For Conv2d layers, we use F.unfold (im2col) to convert convolutions to matrix
    multiplications, enabling the use of tropical GEMM:

    1. Unfold input: X (B, C_in, H, W) -> patches (B, C_in*kH*kW, L)
       where L = H_out * W_out (number of output spatial positions)

    2. Reshape weight: W (C_out, C_in, kH, kW) -> W_flat (C_out, C_in*kH*kW)

    3. Tropical GEMM: argmax_k(patches_k + W_flat_k) for each output position

    4. Map argmax back to input channel: channel = argmax // (kH * kW)

    Example:
        >>> model = VGG16()
        >>> counter = ConvWinnerCounter(model)
        >>> for batch, _ in calibration_loader:
        ...     counter.forward(batch)
        >>> stats = counter.get_statistics()
        >>> print(stats['features.0'].winner_frequency)  # First conv layer
    """

    def __init__(
        self,
        model: nn.Module,
        layers: Optional[List[str]] = None,
        track_margin: bool = True,
        device: Optional[torch.device] = None,
        include_linear: bool = False,
    ):
        """
        Initialize the ConvWinnerCounter.

        Args:
            model: The neural network model to analyze.
            layers: List of layer names to track. If None, tracks all Conv2d layers.
            track_margin: Whether to track winner margins (gap to 2nd place).
            device: Device for computation. If None, uses model's device.
            include_linear: Whether to also track Linear layers (for hybrid models).
        """
        self.model = model
        self.track_margin = track_margin
        self.device = device or next(model.parameters()).device
        self.include_linear = include_linear

        # Identify layers to track
        if layers is None:
            self.layer_names = self._find_conv_layers()
            if include_linear:
                self.layer_names.extend(self._find_linear_layers())
        else:
            self.layer_names = layers

        # Initialize counters for each layer
        self._counters: Dict[str, _ConvLayerCounter] = {}
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []

        self._setup_hooks()

    def _find_conv_layers(self) -> List[str]:
        """Find all Conv2d layers in the model."""
        conv_layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                conv_layers.append(name)
        return conv_layers

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
                if isinstance(module, nn.Conv2d):
                    # Initialize counter for this conv layer
                    self._counters[name] = _ConvLayerCounter(
                        name=name,
                        in_channels=module.in_channels,
                        kernel_size=module.kernel_size,
                        track_margin=self.track_margin,
                        device=self.device,
                    )

                    # Register hook
                    hook = module.register_forward_hook(
                        self._create_conv_hook(name, module)
                    )
                    self._hooks.append(hook)

                elif isinstance(module, nn.Linear) and self.include_linear:
                    # Initialize counter for linear layer (same as WinnerCounter)
                    self._counters[name] = _ConvLayerCounter(
                        name=name,
                        in_channels=module.in_features,
                        kernel_size=(1, 1),  # Treat as 1x1 "kernel"
                        track_margin=self.track_margin,
                        device=self.device,
                        is_linear=True,
                    )

                    hook = module.register_forward_hook(
                        self._create_linear_hook(name, module)
                    )
                    self._hooks.append(hook)

    def _create_conv_hook(self, layer_name: str, module: nn.Conv2d):
        """Create a forward hook for a Conv2d layer."""

        def hook(module: nn.Conv2d, input: Tuple[torch.Tensor, ...], output: torch.Tensor):
            x = input[0]  # (B, C_in, H, W)
            weight = module.weight  # (C_out, C_in, kH, kW)

            # Use tropical forward to get argmax indices
            argmax_indices, margin = self._tropical_conv_forward(
                x, weight, module.kernel_size, module.stride,
                module.padding, module.dilation, module.groups
            )

            # Map argmax indices (over C_in*kH*kW) back to input channels
            kH, kW = module.kernel_size
            channel_indices = argmax_indices // (kH * kW)

            # Update counter with channel indices
            self._counters[layer_name].update(channel_indices, margin)

        return hook

    def _create_linear_hook(self, layer_name: str, module: nn.Linear):
        """Create a forward hook for a Linear layer (for hybrid models)."""

        def hook(module: nn.Linear, input: Tuple[torch.Tensor, ...], output: torch.Tensor):
            x = input[0]  # (B, ..., in_features)
            weight = module.weight  # (out_features, in_features)

            # Use tropical forward
            argmax_indices, margin = self._tropical_linear_forward(x, weight)

            # Update counter
            self._counters[layer_name].update(argmax_indices, margin)

        return hook

    def _tropical_conv_forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int],
        padding: Tuple[int, int],
        dilation: Tuple[int, int],
        groups: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute tropical max-plus convolution using im2col + tropical GEMM.

        The convolution Y = conv(X, W) is converted to matrix multiplication:
        - patches = unfold(X): (B, C_in*kH*kW, L) where L = H_out * W_out
        - W_flat = W.view(C_out, C_in*kH*kW)
        - For each batch, each spatial position l in L:
          - tropical_out[c_out, l] = max_k(patches[k, l] + W_flat[c_out, k])
          - argmax[c_out, l] = argmax_k(patches[k, l] + W_flat[c_out, k])
        """
        B, C_in, H, W = x.shape
        C_out = weight.shape[0]
        kH, kW = kernel_size

        # Handle grouped convolutions
        if groups > 1:
            # For grouped convs, process each group separately
            # Each group has C_in/groups input channels and C_out/groups output channels
            return self._tropical_grouped_conv_forward(
                x, weight, kernel_size, stride, padding, dilation, groups
            )

        # Unfold input to patches: (B, C_in*kH*kW, L)
        patches = F.unfold(
            x, kernel_size, dilation=dilation, padding=padding, stride=stride
        )
        _, K, L = patches.shape  # K = C_in * kH * kW

        # Reshape weight to (C_out, K)
        weight_flat = weight.view(C_out, -1)

        # Process each batch (tropical_gemm operates on 2D matrices)
        all_argmax = []
        all_margin = [] if self.track_margin else None

        for b in range(B):
            patch_b = patches[b]  # (K, L)

            # We want: for each output (c_out, l), find argmax_k(patch[k, l] + weight[c_out, k])
            # This is: weight_flat @ patch_b in tropical sense, with transposed convention
            # Actually: result[c_out, l] = max_k(weight_flat[c_out, k] + patch_b[k, l])

            # tropical_gemm.maxplus_matmul_with_argmax expects A @ B
            # A: (M, K), B: (K, N) -> C: (M, N)
            # We have weight_flat: (C_out, K), patch_b: (K, L)
            # So result will be (C_out, L)

            weight_np = weight_flat.detach().cpu().numpy().astype('float32')
            patch_np = patch_b.detach().cpu().numpy().astype('float32')

            result_flat, argmax_flat = tg.maxplus_matmul_with_argmax(weight_np, patch_np)

            # Reshape to (C_out, L)
            argmax_b = torch.from_numpy(argmax_flat).reshape(C_out, L).to(x.device)
            all_argmax.append(argmax_b)

            if self.track_margin:
                # Compute margin for this batch
                margin_b = self._compute_conv_margin(patch_b, weight_flat, argmax_b)
                all_margin.append(margin_b)

        # Stack batches: (B, C_out, L)
        argmax_indices = torch.stack(all_argmax, dim=0)
        margin = torch.stack(all_margin, dim=0) if self.track_margin else None

        return argmax_indices, margin

    def _tropical_grouped_conv_forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int],
        padding: Tuple[int, int],
        dilation: Tuple[int, int],
        groups: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Handle grouped convolutions (including depthwise separable)."""
        B, C_in, H, W = x.shape
        C_out = weight.shape[0]
        kH, kW = kernel_size

        c_in_per_group = C_in // groups
        c_out_per_group = C_out // groups

        # Unfold the entire input
        patches = F.unfold(
            x, kernel_size, dilation=dilation, padding=padding, stride=stride
        )  # (B, C_in*kH*kW, L)
        _, _, L = patches.shape

        # Reshape to separate groups: (B, groups, c_in_per_group*kH*kW, L)
        patches = patches.view(B, groups, c_in_per_group * kH * kW, L)

        # Process each group
        all_argmax = []
        all_margin = [] if self.track_margin else None

        for g in range(groups):
            # Get group's patches and weights
            patch_g = patches[:, g, :, :]  # (B, K_g, L) where K_g = c_in_per_group*kH*kW
            weight_g = weight[g * c_out_per_group:(g + 1) * c_out_per_group]  # (c_out_per_group, c_in_per_group, kH, kW)
            weight_g_flat = weight_g.view(c_out_per_group, -1)  # (c_out_per_group, K_g)

            # Process each batch
            group_argmax = []
            group_margin = [] if self.track_margin else None

            for b in range(B):
                patch_bg = patch_g[b]  # (K_g, L)

                weight_np = weight_g_flat.detach().cpu().numpy().astype('float32')
                patch_np = patch_bg.detach().cpu().numpy().astype('float32')

                result_flat, argmax_flat = tg.maxplus_matmul_with_argmax(weight_np, patch_np)

                argmax_bg = torch.from_numpy(argmax_flat).reshape(c_out_per_group, L).to(x.device)

                # Map argmax back to global input channel indices
                # Local argmax is in [0, K_g), convert to channel in [0, c_in_per_group)
                local_channel = argmax_bg // (kH * kW)
                # Add group offset to get global channel index
                global_channel = local_channel + g * c_in_per_group

                # We need to store indices that can be mapped back to channels
                # Store the raw argmax offset by group
                argmax_bg_global = argmax_bg + g * c_in_per_group * kH * kW

                group_argmax.append(argmax_bg_global)

                if self.track_margin:
                    margin_bg = self._compute_conv_margin(patch_bg, weight_g_flat, argmax_bg)
                    group_margin.append(margin_bg)

            # (B, c_out_per_group, L)
            group_argmax = torch.stack(group_argmax, dim=0)
            all_argmax.append(group_argmax)

            if self.track_margin:
                group_margin = torch.stack(group_margin, dim=0)
                all_margin.append(group_margin)

        # Concatenate groups: (B, C_out, L)
        argmax_indices = torch.cat(all_argmax, dim=1)
        margin = torch.cat(all_margin, dim=1) if self.track_margin else None

        return argmax_indices, margin

    def _compute_conv_margin(
        self,
        patches: torch.Tensor,
        weight_flat: torch.Tensor,
        argmax_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the margin (gap to 2nd place) for convolution.

        Args:
            patches: (K, L) - unfolded input patches
            weight_flat: (C_out, K) - flattened weights
            argmax_indices: (C_out, L) - argmax indices
        """
        C_out, L = argmax_indices.shape
        K = patches.shape[0]

        # Compute all tropical sums: (C_out, K, L)
        # tropical_sum[c, k, l] = weight_flat[c, k] + patches[k, l]
        tropical_sum = weight_flat.unsqueeze(-1) + patches.unsqueeze(0)

        # Get max values
        max_values = tropical_sum.max(dim=1).values  # (C_out, L)

        # Mask out the max and find second max
        mask = torch.zeros_like(tropical_sum, dtype=torch.bool)
        # Create indices for scatter
        mask.scatter_(1, argmax_indices.unsqueeze(1), True)
        masked = tropical_sum.masked_fill(mask, float('-inf'))
        second_max = masked.max(dim=1).values  # (C_out, L)

        margin = max_values - second_max
        return margin

    def _tropical_linear_forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Tropical forward for Linear layers (same as WinnerCounter)."""
        # Flatten batch dimensions
        original_shape = x.shape[:-1]
        M = x[..., 0].numel()
        K = x.shape[-1]
        N = weight.shape[0]

        x_flat = x.reshape(-1, K)
        weight_t = weight.t().contiguous()

        x_np = x_flat.detach().cpu().numpy().astype('float32')
        w_np = weight_t.detach().cpu().numpy().astype('float32')

        result_flat, argmax_flat = tg.maxplus_matmul_with_argmax(x_np, w_np)

        argmax_indices = torch.from_numpy(argmax_flat).reshape(M, N)
        argmax_indices = argmax_indices.reshape(*original_shape, N).to(x.device)

        margin = None
        if self.track_margin:
            margin = self._compute_linear_margin(x, weight, argmax_indices)

        return argmax_indices, margin

    def _compute_linear_margin(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        argmax_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Compute margin for Linear layers."""
        x_expanded = x.unsqueeze(-2)
        tropical_sum = x_expanded + weight

        max_values = tropical_sum.max(dim=-1).values

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
    ) -> Dict[str, Union[ConvWinnerStatistics, WinnerStatistics]]:
        """
        Collect statistics from a dataloader.

        Args:
            dataloader: DataLoader providing calibration data.
            num_batches: Maximum number of batches to process. If None, processes all.
            show_progress: Whether to show a progress bar.

        Returns:
            Dictionary mapping layer names to their statistics.
        """
        self.model.eval()

        iterator = dataloader
        if show_progress:
            total = num_batches if num_batches else len(dataloader)
            iterator = tqdm(dataloader, total=total, desc="Collecting conv statistics")

        for i, batch in enumerate(iterator):
            if num_batches is not None and i >= num_batches:
                break

            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch

            self.forward(x)

        return self.get_statistics()

    def get_statistics(self) -> Dict[str, Union[ConvWinnerStatistics, WinnerStatistics]]:
        """
        Get collected winner statistics for all layers.

        Returns:
            Dictionary mapping layer names to their statistics.
        """
        return {name: counter.get_statistics() for name, counter in self._counters.items()}

    def get_statistics_as_winner_stats(self) -> Dict[str, WinnerStatistics]:
        """
        Get statistics converted to generic WinnerStatistics format.

        Useful for compatibility with TropicalPruner.
        """
        result = {}
        for name, counter in self._counters.items():
            stats = counter.get_statistics()
            if isinstance(stats, ConvWinnerStatistics):
                result[name] = stats.to_winner_statistics()
            else:
                result[name] = stats
        return result

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


class _ConvLayerCounter:
    """Internal counter for a single Conv2d layer."""

    def __init__(
        self,
        name: str,
        in_channels: int,
        kernel_size: Tuple[int, int],
        track_margin: bool = True,
        device: Optional[torch.device] = None,
        is_linear: bool = False,
    ):
        self.name = name
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.track_margin = track_margin
        self.device = device or torch.device("cpu")
        self.is_linear = is_linear

        # Track winner counts per INPUT channel
        self.winner_count = torch.zeros(in_channels, dtype=torch.long, device=self.device)
        self.total_positions = 0

        if track_margin:
            self.margin_sum = torch.zeros(in_channels, dtype=torch.float32, device=self.device)
            self.margin_count = torch.zeros(in_channels, dtype=torch.long, device=self.device)
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

        For Conv2d: argmax_indices should already be mapped to channel indices.
        Shape: (B, C_out, L) where values are in [0, in_channels).

        For Linear: argmax_indices shape: (..., out_features)
        """
        # Flatten indices
        flat_indices = argmax_indices.flatten().to(self.device)
        self.total_positions += flat_indices.numel()

        # Clamp indices to valid range (safety check)
        flat_indices = flat_indices.clamp(0, self.in_channels - 1)

        # Count winners
        ones = torch.ones_like(flat_indices, dtype=torch.long)
        self.winner_count.scatter_add_(0, flat_indices, ones)

        # Track margins if enabled
        if self.track_margin and margin is not None:
            flat_margin = margin.flatten().to(self.device)
            valid_mask = torch.isfinite(flat_margin)
            if valid_mask.any():
                valid_indices = flat_indices[valid_mask]
                valid_margins = flat_margin[valid_mask]

                self.margin_sum.scatter_add_(0, valid_indices, valid_margins)
                margin_ones = torch.ones_like(valid_indices, dtype=torch.long)
                self.margin_count.scatter_add_(0, valid_indices, margin_ones)

    def get_statistics(self) -> Union[ConvWinnerStatistics, WinnerStatistics]:
        """Return statistics for this layer."""
        if self.is_linear:
            return WinnerStatistics(
                layer_name=self.name,
                winner_count=self.winner_count.clone(),
                total_positions=self.total_positions,
                margin_sum=self.margin_sum.clone() if self.margin_sum is not None else None,
                margin_count=self.margin_count.clone() if self.margin_count is not None else None,
            )
        else:
            return ConvWinnerStatistics(
                layer_name=self.name,
                winner_count=self.winner_count.clone(),
                total_positions=self.total_positions,
                margin_sum=self.margin_sum.clone() if self.margin_sum is not None else None,
                margin_count=self.margin_count.clone() if self.margin_count is not None else None,
                kernel_size=self.kernel_size,
            )

    def reset(self) -> None:
        """Reset counters to zero."""
        self.winner_count.zero_()
        self.total_positions = 0
        if self.margin_sum is not None:
            self.margin_sum.zero_()
        if self.margin_count is not None:
            self.margin_count.zero_()
