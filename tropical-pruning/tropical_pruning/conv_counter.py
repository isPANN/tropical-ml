"""
Convolutional Winner Counter: Tracks which output filters "win" in tropical
max-plus convolution using the Im2Col + TropicalGEMM approach.

Key insight from tropical geometry:
- Standard convolution has "cancellation effects" (positive and negative terms can cancel)
- Tropical convolution (max-plus) reveals the "geometric peak" contribution of each filter
- A filter that rarely achieves the tropical max is geometrically redundant

The approach:
1. Im2Col: Unfold input X (B, C_in, H, W) -> patches (B*H_out*W_out, C_in*k*k)
2. Reshape weights W (C_out, C_in, k, k) -> W_flat (C_out, C_in*k*k)
3. TropicalGEMM: C_trop[i,j] = max_k(patches[i,k] + W_flat[j,k])
   - i indexes spatial positions (pixels)
   - j indexes output filters
4. Winner counting: For each spatial position, which filter has the max tropical response?

This differs from standard activation-based pruning because tropical max-plus
isolates the geometric contribution of each filter without interference from
cancellation effects.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import tropical_gemm as tg

from tropical_pruning.counter import WinnerStatistics


@dataclass
class ConvWinnerStatistics:
    """
    Statistics collected from tropical convolution for a Conv2d layer.

    For Conv2d, we compute the tropical feature map using Im2Col + TropicalGEMM,
    then track which OUTPUT FILTERS win at each spatial position.

    The tropical response C_trop[i,j] = max_k(patch[i,k] + weight[j,k]) tells us
    the "geometric peak" contribution of filter j at spatial position i.
    Filters that rarely win are geometrically redundant.
    """
    layer_name: str
    # Shape: (out_channels,) - how many times each output filter achieved tropical argmax
    winner_count: torch.Tensor
    # Total number of spatial positions processed (B * H_out * W_out)
    total_positions: int
    # Shape: (out_channels,) - sum of margins when winning
    margin_sum: Optional[torch.Tensor] = None
    # Shape: (out_channels,) - count for margin averaging
    margin_count: Optional[torch.Tensor] = None
    # Number of output channels for reference
    out_channels: int = 0

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
            out_channels=self.out_channels,
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
    Collects winner statistics using Im2Col + TropicalGEMM for Conv2d layers.

    This class computes the TROPICAL feature map (not standard convolution output)
    and tracks which output filters produce the maximum tropical response at each
    spatial position.

    Why tropical instead of standard convolution?
    - Standard: Y = Σ(W * X) has cancellation effects
    - Tropical: Y_trop = max(W + X) reveals geometric peak contributions
    - Example: W=[10,-10], X=[5,5] -> Standard: 0, Tropical: 15

    The Im2Col transformation converts convolution to matrix multiplication,
    allowing us to use our SIMD-accelerated TropicalGEMM kernel.

    Example:
        >>> model = VGG16()
        >>> counter = ConvWinnerCounter(model)
        >>> stats = counter.collect(calibration_loader)
        >>> # Check which filters rarely win in tropical sense
        >>> print(stats['features.0'].winner_frequency)
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
                    # Initialize counter for this conv layer - tracks OUTPUT filters
                    self._counters[name] = _ConvLayerCounter(
                        name=name,
                        out_channels=module.out_channels,
                        track_margin=self.track_margin,
                        device=self.device,
                    )

                    # Register hook to compute tropical convolution
                    hook = module.register_forward_hook(
                        self._create_conv_hook(name, module)
                    )
                    self._hooks.append(hook)

                elif isinstance(module, nn.Linear) and self.include_linear:
                    # Initialize counter for linear layer
                    self._counters[name] = _ConvLayerCounter(
                        name=name,
                        out_channels=module.out_features,
                        track_margin=self.track_margin,
                        device=self.device,
                        is_linear=True,
                    )

                    hook = module.register_forward_hook(
                        self._create_linear_hook(name, module)
                    )
                    self._hooks.append(hook)

    def _create_conv_hook(self, layer_name: str, module: nn.Conv2d):
        """Create a forward hook for a Conv2d layer.

        This hook computes the TROPICAL feature map using Im2Col + TropicalGEMM,
        then performs pixel-wise winner counting on the tropical output.

        Steps:
        1. Unfold input X -> patches (B*L, K) where L=H_out*W_out, K=C_in*k*k
        2. Reshape weights W -> W_flat (C_out, K)
        3. TropicalGEMM: C_trop[i,j] = max_k(patches[i,k] + W_flat[j,k])
        4. Argmax over filters: which filter wins at each spatial position
        """

        def hook(module: nn.Conv2d, input: Tuple[torch.Tensor, ...], output: torch.Tensor):
            x = input[0]  # (B, C_in, H, W)
            weight = module.weight  # (C_out, C_in, kH, kW)

            # Compute tropical feature map and get argmax
            argmax_indices, margin = self._compute_tropical_conv(
                x, weight,
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
            )

            # Update counter with output filter indices
            self._counters[layer_name].update(argmax_indices, margin)

        return hook

    def _compute_tropical_conv(
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
        Compute tropical convolution using Im2Col + TropicalGEMM.

        This computes the tropical feature map:
            C_trop[b, l, c_out] = max_k(patches[b, l, k] + W_flat[c_out, k])

        where:
            - l indexes spatial positions (H_out * W_out)
            - k indexes the flattened kernel dimension (C_in * kH * kW)
            - c_out indexes output filters

        Then we find which filter wins at each spatial position.

        Returns:
            argmax_indices: (B, L) - which filter won at each spatial position
            margin: (B, L) - gap between 1st and 2nd place (optional)
        """
        B, C_in, H, W = x.shape
        C_out = weight.shape[0]
        kH, kW = kernel_size

        # Handle grouped convolutions
        if groups > 1:
            return self._compute_tropical_grouped_conv(
                x, weight, kernel_size, stride, padding, dilation, groups
            )

        # Step 1: Im2Col - Unfold input to patches
        # patches shape: (B, C_in*kH*kW, L) where L = H_out * W_out
        patches = F.unfold(
            x, kernel_size, dilation=dilation, padding=padding, stride=stride
        )
        B, K, L = patches.shape  # K = C_in * kH * kW

        # Step 2: Reshape weight to (C_out, K)
        weight_flat = weight.view(C_out, -1)

        # Step 3: TropicalGEMM for each batch
        # We want: C_trop[l, c_out] = max_k(patches[l, k] + W_flat[c_out, k])
        # This is: patches @ W_flat.T in tropical sense
        # patches: (B, K, L) -> need to transpose to (B, L, K) for matmul
        # W_flat: (C_out, K) -> W_flat.T is (K, C_out)
        # Result: (B, L, C_out)

        all_argmax = []
        all_margin = [] if self.track_margin else None

        for b in range(B):
            # patches[b]: (K, L) -> transpose to (L, K)
            patch_b = patches[b].T  # (L, K)

            # TropicalGEMM: (L, K) @ (K, C_out) -> (L, C_out)
            # Using tropical_gemm: A @ B where A is (L, K), B is (K, C_out)
            # We need B = W_flat.T = (K, C_out)
            patch_np = patch_b.detach().cpu().numpy().astype('float32')
            weight_t_np = weight_flat.T.detach().cpu().numpy().astype('float32')

            # tropical_gemm computes C[i,j] = max_k(A[i,k] + B[k,j])
            result_np, argmax_np = tg.maxplus_matmul_with_argmax(patch_np, weight_t_np)

            # result_np: (L, C_out) - tropical feature map values
            # argmax_np: (L, C_out) - which k achieved max for each (l, c_out)

            # Convert to tensors
            result_b = torch.from_numpy(result_np).to(x.device)  # (L, C_out)

            # Now find which OUTPUT FILTER wins at each spatial position
            # argmax over c_out dimension
            filter_argmax = result_b.argmax(dim=1)  # (L,) - which filter won at each position
            all_argmax.append(filter_argmax)

            if self.track_margin:
                # Compute margin: gap between 1st and 2nd place filter
                sorted_vals, _ = result_b.sort(dim=1, descending=True)
                max_vals = sorted_vals[:, 0]  # (L,)
                second_max = sorted_vals[:, 1] if C_out > 1 else max_vals
                margin_b = max_vals - second_max  # (L,)
                all_margin.append(margin_b)

        # Stack batches: (B, L)
        argmax_indices = torch.stack(all_argmax, dim=0)
        margin = torch.stack(all_margin, dim=0) if self.track_margin else None

        return argmax_indices, margin

    def _compute_tropical_grouped_conv(
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
        B, _, L = patches.shape

        # Reshape to separate groups: (B, groups, c_in_per_group*kH*kW, L)
        patches = patches.view(B, groups, c_in_per_group * kH * kW, L)

        # For grouped convs, we compute tropical response per group
        # then concatenate and find global winner
        all_results = []

        for g in range(groups):
            patch_g = patches[:, g, :, :]  # (B, K_g, L)
            weight_g = weight[g * c_out_per_group:(g + 1) * c_out_per_group]
            weight_g_flat = weight_g.view(c_out_per_group, -1)  # (c_out_per_group, K_g)

            group_results = []
            for b in range(B):
                patch_bg = patch_g[b].T  # (L, K_g)
                patch_np = patch_bg.detach().cpu().numpy().astype('float32')
                weight_t_np = weight_g_flat.T.detach().cpu().numpy().astype('float32')

                result_np, _ = tg.maxplus_matmul_with_argmax(patch_np, weight_t_np)
                result_bg = torch.from_numpy(result_np).to(x.device)  # (L, c_out_per_group)
                group_results.append(result_bg)

            group_results = torch.stack(group_results, dim=0)  # (B, L, c_out_per_group)
            all_results.append(group_results)

        # Concatenate all groups: (B, L, C_out)
        full_result = torch.cat(all_results, dim=2)

        # Find winner across all output filters
        filter_argmax = full_result.argmax(dim=2)  # (B, L)

        margin = None
        if self.track_margin:
            sorted_vals, _ = full_result.sort(dim=2, descending=True)
            max_vals = sorted_vals[:, :, 0]
            second_max = sorted_vals[:, :, 1] if C_out > 1 else max_vals
            margin = max_vals - second_max  # (B, L)

        return filter_argmax, margin

    def _create_linear_hook(self, layer_name: str, module: nn.Linear):
        """Create a forward hook for a Linear layer.

        For Linear layers, we use TropicalGEMM directly (no Im2Col needed).
        """

        def hook(module: nn.Linear, input: Tuple[torch.Tensor, ...], output: torch.Tensor):
            x = input[0]  # (B, ..., in_features)
            weight = module.weight  # (out_features, in_features)

            # Flatten batch dimensions
            original_shape = x.shape[:-1]
            x_flat = x.reshape(-1, x.shape[-1])  # (N, in_features)

            # TropicalGEMM: (N, in_features) @ (in_features, out_features) -> (N, out_features)
            x_np = x_flat.detach().cpu().numpy().astype('float32')
            weight_t_np = weight.T.detach().cpu().numpy().astype('float32')

            result_np, _ = tg.maxplus_matmul_with_argmax(x_np, weight_t_np)
            result = torch.from_numpy(result_np).to(x.device)  # (N, out_features)

            # Find winner output neuron
            argmax_indices = result.argmax(dim=1)  # (N,)

            # Compute margin
            margin = None
            if self.track_margin:
                sorted_vals, _ = result.sort(dim=1, descending=True)
                max_vals = sorted_vals[:, 0]
                second_max = sorted_vals[:, 1] if result.shape[1] > 1 else max_vals
                margin = max_vals - second_max

            # Update counter
            self._counters[layer_name].update(argmax_indices, margin)

        return hook

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
            iterator = tqdm(dataloader, total=total, desc="Collecting tropical conv statistics")

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
    """Internal counter for a single Conv2d layer.

    Tracks pixel-wise winner counts for OUTPUT filters based on tropical feature map.
    """

    def __init__(
        self,
        name: str,
        out_channels: int,
        track_margin: bool = True,
        device: Optional[torch.device] = None,
        is_linear: bool = False,
    ):
        self.name = name
        self.out_channels = out_channels
        self.track_margin = track_margin
        self.device = device or torch.device("cpu")
        self.is_linear = is_linear

        # Track winner counts per OUTPUT filter
        self.winner_count = torch.zeros(out_channels, dtype=torch.long, device=self.device)
        self.total_positions = 0

        if track_margin:
            self.margin_sum = torch.zeros(out_channels, dtype=torch.float32, device=self.device)
            self.margin_count = torch.zeros(out_channels, dtype=torch.long, device=self.device)
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

        For Conv2d: argmax_indices shape: (B, L), values in [0, out_channels)
            where L = H_out * W_out (spatial positions)
        For Linear: argmax_indices shape: (N,), values in [0, out_channels)

        Each value indicates which output filter won the tropical competition
        at that spatial position.
        """
        # Flatten indices
        flat_indices = argmax_indices.flatten().to(self.device)
        self.total_positions += flat_indices.numel()

        # Clamp indices to valid range (safety check)
        flat_indices = flat_indices.clamp(0, self.out_channels - 1)

        # Count winners using scatter_add
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
                out_channels=self.out_channels,
            )

    def reset(self) -> None:
        """Reset counters to zero."""
        self.winner_count.zero_()
        self.total_positions = 0
        if self.margin_sum is not None:
            self.margin_sum.zero_()
        if self.margin_count is not None:
            self.margin_count.zero_()
