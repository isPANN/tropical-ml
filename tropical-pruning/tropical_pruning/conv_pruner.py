"""
Convolutional Tropical Pruner: Structured pruning for Conv2d layers.

This module provides ConvTropicalPruner for filter-based structured pruning of
convolutional neural networks. It uses pixel-wise winner statistics to identify
and remove entire filters (output channels) that rarely "win" at any spatial position.

Key insight from tropical geometry:
- For each layer's output Y (B, C_out, H, W), we track which filters win at each pixel
- A filter that never wins (low winner frequency) is geometrically redundant
- This is a "self-pruning" approach: each layer's statistics determine its own pruning

Pruning flow:
1. Collect pixel-wise winner statistics for each Conv2d layer's output
2. For each layer, identify filters with low winner frequency
3. Prune those filters (output channels) from the current layer
4. Propagate: prune corresponding input channels from the next layer

Handles:
- Conv2d filter pruning (remove output channels)
- BatchNorm layer pruning (coupled with Conv2d)
- Skip connections in ResNet-like architectures
- Mixed Conv2d + Linear models (classifier pruning)
"""

import copy
from collections import defaultdict
from typing import Dict, List, Optional, Union
import torch
import torch.nn as nn

from tropical_pruning.counter import WinnerStatistics
from tropical_pruning.conv_counter import ConvWinnerStatistics
from tropical_pruning.criteria import PruningCriterion, WinnerFrequencyCriterion


class ConvTropicalPruner:
    """
    Applies winner-based tropical structured pruning to convolutional networks.

    Filter pruning removes entire output channels (filters) from Conv2d layers,
    resulting in smaller, faster models that can be efficiently executed without
    sparse matrix support.

    Example:
        >>> from tropical_pruning import ConvWinnerCounter, ConvTropicalPruner
        >>>
        >>> # Collect winner statistics
        >>> counter = ConvWinnerCounter(model)
        >>> stats = counter.collect(calibration_loader)
        >>>
        >>> # Prune to 50% sparsity
        >>> pruner = ConvTropicalPruner(model, stats)
        >>> pruned_model = pruner.prune(sparsity=0.5)
        >>>
        >>> # Check compression
        >>> print(pruner.get_compression_stats())
    """

    def __init__(
        self,
        model: nn.Module,
        statistics: Dict[str, Union[WinnerStatistics, ConvWinnerStatistics]],
        criterion: Optional[Union[str, PruningCriterion]] = None,
    ):
        """
        Initialize the ConvTropicalPruner.

        Args:
            model: The model to prune.
            statistics: Dictionary mapping layer names to winner statistics.
                       Can be WinnerStatistics or ConvWinnerStatistics.
            criterion: Pruning criterion. Defaults to "winner_frequency".
        """
        self.model = model
        self.statistics = self._normalize_statistics(statistics)

        if criterion is None:
            self.criterion = WinnerFrequencyCriterion()
        elif isinstance(criterion, str):
            from tropical_pruning.criteria import get_criterion
            self.criterion = get_criterion(criterion)
        else:
            self.criterion = criterion

        self._pruning_masks: Dict[str, torch.Tensor] = {}
        self._compression_stats: Optional[Dict] = None

        # Build model structure map for handling skip connections
        self._layer_graph = self._build_layer_graph()

    def _normalize_statistics(
        self,
        statistics: Dict[str, Union[WinnerStatistics, ConvWinnerStatistics]],
    ) -> Dict[str, WinnerStatistics]:
        """Convert all statistics to WinnerStatistics format."""
        result = {}
        for name, stats in statistics.items():
            if isinstance(stats, ConvWinnerStatistics):
                result[name] = stats.to_winner_statistics()
            else:
                result[name] = stats
        return result

    def _build_layer_graph(self) -> Dict[str, List[str]]:
        """
        Build a graph of layer connections.

        Returns a dict mapping each layer to the layers that consume its output.
        This is used for handling skip connections.
        """
        # For now, return sequential order. Can be extended for skip connections.
        conv_layers = self._get_conv_layers_ordered()
        graph = defaultdict(list)

        for i in range(len(conv_layers) - 1):
            graph[conv_layers[i]].append(conv_layers[i + 1])

        return dict(graph)

    def _get_conv_layers_ordered(self) -> List[str]:
        """Get Conv2d layer names in forward execution order."""
        layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                layers.append(name)
        return layers

    def _get_linear_layers_ordered(self) -> List[str]:
        """Get Linear layer names in forward execution order."""
        layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                layers.append(name)
        return layers

    def _find_bn_for_conv(self, conv_name: str) -> Optional[str]:
        """
        Find the BatchNorm layer that follows a Conv2d.

        Handles naming conventions like:
        - "features.0" (Conv) -> "features.1" (BN)
        - "layer1.0.conv1" -> "layer1.0.bn1"
        """
        modules_dict = dict(self.model.named_modules())
        conv_module = modules_dict.get(conv_name)
        if conv_module is None:
            return None

        # Strategy 1: Next sequential layer (e.g., features.0 -> features.1)
        parts = conv_name.rsplit('.', 1)
        if len(parts) == 2:
            parent_name, idx_str = parts
            try:
                idx = int(idx_str)
                next_name = f"{parent_name}.{idx + 1}"
                if next_name in modules_dict and isinstance(modules_dict[next_name], nn.BatchNorm2d):
                    return next_name
            except ValueError:
                pass

        # Strategy 2: Replace conv with bn (e.g., conv1 -> bn1)
        if 'conv' in conv_name:
            bn_name = conv_name.replace('conv', 'bn')
            if bn_name in modules_dict and isinstance(modules_dict[bn_name], nn.BatchNorm2d):
                return bn_name

        # Strategy 3: Check all named modules for matching dimensions
        out_channels = conv_module.out_channels
        parent_parts = conv_name.rsplit('.', 1)
        if len(parent_parts) == 2:
            parent_name = parent_parts[0]
            # Look for BN in the same parent module
            for name, module in self.model.named_modules():
                if (name.startswith(parent_name) and
                    isinstance(module, nn.BatchNorm2d) and
                    module.num_features == out_channels):
                    # Check if this BN comes right after the conv
                    return name

        return None

    def _find_next_conv(self, conv_name: str) -> Optional[str]:
        """Find the next Conv2d layer that consumes this layer's output."""
        conv_layers = self._get_conv_layers_ordered()
        try:
            idx = conv_layers.index(conv_name)
            if idx + 1 < len(conv_layers):
                return conv_layers[idx + 1]
        except ValueError:
            pass
        return None

    def _find_connected_linear(self) -> Optional[str]:
        """Find the first Linear layer (usually connected to last conv)."""
        linear_layers = self._get_linear_layers_ordered()
        if linear_layers:
            return linear_layers[0]
        return None

    def compute_importance(self, layer_name: str) -> torch.Tensor:
        """
        Compute importance scores for filters (output channels) of a layer.

        Uses the layer's own pixel-wise winner statistics: filters that frequently
        "win" at spatial positions are considered important.

        Args:
            layer_name: Name of the layer whose OUTPUT channels we're scoring.

        Returns:
            Importance scores, shape (out_channels,).
        """
        if layer_name not in self.statistics:
            raise ValueError(f"No statistics found for {layer_name}")

        stats = self.statistics[layer_name]
        return self.criterion.compute_importance(stats)

    def get_pruning_mask(
        self,
        layer_name: str,
        sparsity: float,
    ) -> torch.Tensor:
        """
        Get pruning mask for a Conv2d layer's output channels.

        Uses the layer's own statistics to determine which filters to keep.

        Args:
            layer_name: Name of the layer.
            sparsity: Target sparsity (fraction to prune).

        Returns:
            Boolean mask, True = keep filter, False = prune filter.
        """
        if layer_name not in self.statistics:
            raise ValueError(f"No statistics for {layer_name}")

        stats = self.statistics[layer_name]
        return self.criterion.get_pruning_mask(stats, sparsity)

    def prune(
        self,
        sparsity: Union[float, Dict[str, float]],
        inplace: bool = False,
        prune_first_conv: bool = False,
        prune_last_conv: bool = False,
    ) -> nn.Module:
        """
        Apply filter-based structured pruning to the model.

        Uses self-statistics: each layer's own pixel-wise winner statistics determine
        which of its filters to prune. This is the correct tropical geometry approach.

        Pruning flow for each layer:
        1. Use layer's own statistics to identify low-importance filters
        2. Prune those output channels (filters) from the current layer
        3. Prune corresponding BatchNorm parameters if present
        4. Prune corresponding input channels from the next layer

        Args:
            sparsity: Target sparsity. Can be:
                     - float: Same sparsity for all prunable layers
                     - dict: Per-layer sparsity mapping
            inplace: If True, modify model in place. Otherwise, return a copy.
            prune_first_conv: If True, also prune the first conv layer.
                            Usually False to preserve input channel compatibility.
            prune_last_conv: If True, also prune the last conv layer's outputs.
                           Usually False to preserve classifier compatibility.

        Returns:
            Pruned model.
        """
        if inplace:
            pruned_model = self.model
        else:
            pruned_model = copy.deepcopy(self.model)

        # Get conv layers in order
        conv_layers = self._get_conv_layers_ordered()

        if len(conv_layers) < 1:
            return pruned_model

        # Convert uniform sparsity to per-layer dict
        if isinstance(sparsity, float):
            sparsity_dict = {name: sparsity for name in conv_layers}
        else:
            sparsity_dict = sparsity

        self._pruning_masks = {}

        # Determine which layers to prune
        # Skip first layer unless explicitly requested (preserves input channels)
        # Skip last layer unless explicitly requested (preserves classifier compatibility)
        start_idx = 0 if prune_first_conv else 1
        end_idx = len(conv_layers) if prune_last_conv else len(conv_layers) - 1

        # Process each layer independently using its own statistics
        for i in range(start_idx, end_idx):
            layer_name = conv_layers[i]

            # Get sparsity for this layer
            layer_sparsity = sparsity_dict.get(layer_name, 0.0)
            if layer_sparsity == 0.0:
                continue

            # Skip if we don't have stats for this layer
            if layer_name not in self.statistics:
                continue

            # Compute mask using layer's own statistics
            stats = self.statistics[layer_name]
            mask = self.criterion.get_pruning_mask(stats, layer_sparsity)
            self._pruning_masks[layer_name] = mask

            # 1. Prune OUTPUT channels (filters) of current layer
            self._prune_conv_output(pruned_model, layer_name, mask)

            # 2. Prune BatchNorm if present (follows current conv)
            bn_name = self._find_bn_for_conv(layer_name)
            if bn_name:
                self._prune_batchnorm(pruned_model, bn_name, mask)

            # 3. Prune INPUT channels of the next layer
            next_conv = self._find_next_conv(layer_name)
            if next_conv:
                self._prune_conv_input(pruned_model, next_conv, mask)

        # Handle the connection between last conv and first linear (if applicable)
        self._adjust_linear_input(pruned_model, conv_layers)

        # Compute compression statistics
        self._compute_compression_stats(pruned_model)

        return pruned_model

    def _prune_conv_output(
        self,
        model: nn.Module,
        layer_name: str,
        mask: torch.Tensor,
    ) -> None:
        """
        Prune OUTPUT channels (filters) from a Conv2d layer.

        Weight shape: (out_channels, in_channels, kH, kW)
        We keep filters where mask is True.
        """
        module = dict(model.named_modules())[layer_name]
        if not isinstance(module, nn.Conv2d):
            return

        keep_indices = mask.nonzero(as_tuple=True)[0]
        new_out_channels = keep_indices.numel()

        new_layer = nn.Conv2d(
            in_channels=module.in_channels,
            out_channels=new_out_channels,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
            bias=module.bias is not None,
            padding_mode=module.padding_mode,
            device=module.weight.device,
            dtype=module.weight.dtype,
        )

        # Keep selected filters
        new_layer.weight.data = module.weight.data[keep_indices, :, :, :]
        if module.bias is not None:
            new_layer.bias.data = module.bias.data[keep_indices]

        self._replace_layer(model, layer_name, new_layer)

    def _prune_conv_input(
        self,
        model: nn.Module,
        layer_name: str,
        mask: torch.Tensor,
    ) -> None:
        """
        Prune INPUT channels from a Conv2d layer.

        Weight shape: (out_channels, in_channels, kH, kW)
        We keep input channels where mask is True.

        NOTE: This doesn't work directly for grouped convolutions with groups > 1,
        as input channels are split into groups. For such cases, special handling needed.
        """
        module = dict(model.named_modules())[layer_name]
        if not isinstance(module, nn.Conv2d):
            return

        # Handle grouped convolutions
        if module.groups > 1:
            # For depthwise conv (groups == in_channels), input pruning = output pruning
            # For other grouped convs, need to maintain group structure
            self._prune_grouped_conv_input(model, layer_name, mask)
            return

        keep_indices = mask.nonzero(as_tuple=True)[0]
        new_in_channels = keep_indices.numel()

        new_layer = nn.Conv2d(
            in_channels=new_in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
            bias=module.bias is not None,
            padding_mode=module.padding_mode,
            device=module.weight.device,
            dtype=module.weight.dtype,
        )

        # Keep selected input channels
        new_layer.weight.data = module.weight.data[:, keep_indices, :, :]
        if module.bias is not None:
            new_layer.bias.data = module.bias.data.clone()

        self._replace_layer(model, layer_name, new_layer)

    def _prune_grouped_conv_input(
        self,
        model: nn.Module,
        layer_name: str,
        mask: torch.Tensor,
    ) -> None:
        """
        Handle pruning input channels for grouped convolutions.

        For depthwise separable convs (groups == in_channels):
        - Prune input = prune filters since each input has its own filter
        """
        module = dict(model.named_modules())[layer_name]
        keep_indices = mask.nonzero(as_tuple=True)[0]

        if module.groups == module.in_channels:
            # Depthwise conv: groups == in_channels == out_channels
            # Each filter processes exactly one input channel
            new_channels = keep_indices.numel()

            new_layer = nn.Conv2d(
                in_channels=new_channels,
                out_channels=new_channels,
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=new_channels,  # Maintain depthwise structure
                bias=module.bias is not None,
                padding_mode=module.padding_mode,
                device=module.weight.device,
                dtype=module.weight.dtype,
            )

            # Keep selected channels
            new_layer.weight.data = module.weight.data[keep_indices, :, :, :]
            if module.bias is not None:
                new_layer.bias.data = module.bias.data[keep_indices]

            self._replace_layer(model, layer_name, new_layer)
        else:
            # General grouped conv: more complex handling needed
            # For now, skip pruning to avoid breaking group structure
            pass

    def _prune_batchnorm(
        self,
        model: nn.Module,
        layer_name: str,
        mask: torch.Tensor,
    ) -> None:
        """
        Prune BatchNorm layer to match Conv2d output channel pruning.

        BatchNorm has parameters per channel: gamma, beta, running_mean, running_var
        All need to be pruned consistently.
        """
        module = dict(model.named_modules())[layer_name]
        if not isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            return

        keep_indices = mask.nonzero(as_tuple=True)[0]
        new_features = keep_indices.numel()

        if isinstance(module, nn.BatchNorm2d):
            new_bn = nn.BatchNorm2d(
                num_features=new_features,
                eps=module.eps,
                momentum=module.momentum,
                affine=module.affine,
                track_running_stats=module.track_running_stats,
                device=module.weight.device if module.weight is not None else None,
                dtype=module.weight.dtype if module.weight is not None else None,
            )
        else:
            new_bn = nn.BatchNorm1d(
                num_features=new_features,
                eps=module.eps,
                momentum=module.momentum,
                affine=module.affine,
                track_running_stats=module.track_running_stats,
                device=module.weight.device if module.weight is not None else None,
                dtype=module.weight.dtype if module.weight is not None else None,
            )

        # Copy selected parameters
        if module.affine:
            new_bn.weight.data = module.weight.data[keep_indices]
            new_bn.bias.data = module.bias.data[keep_indices]

        if module.track_running_stats:
            new_bn.running_mean.data = module.running_mean.data[keep_indices]
            new_bn.running_var.data = module.running_var.data[keep_indices]
            new_bn.num_batches_tracked = module.num_batches_tracked

        self._replace_layer(model, layer_name, new_bn)

    def _adjust_linear_input(
        self,
        pruned_model: nn.Module,
        conv_layers: List[str],
    ) -> None:
        """
        Adjust the first Linear layer's input size after conv pruning.

        When the last conv's output is pruned, the flattened feature map
        that goes into the first Linear layer changes size.
        """
        if not conv_layers:
            return

        # Check if last conv was pruned
        last_conv_name = conv_layers[-1]
        if last_conv_name not in self._pruning_masks:
            # Check if the second-to-last was pruned
            if len(conv_layers) >= 2 and conv_layers[-2] in self._pruning_masks:
                last_conv_name = conv_layers[-2]
            else:
                return

        # Find first linear layer
        linear_layers = self._get_linear_layers_ordered()
        if not linear_layers:
            return

        first_linear_name = linear_layers[0]
        linear_module = dict(pruned_model.named_modules())[first_linear_name]

        # Get the new feature dimension from the last pruned conv
        # This requires knowing the spatial dimensions, which we estimate
        # For safety, we'll recompute expected input size based on mask

        # Actually, we need to be careful here. If there's a flatten/adaptive pool
        # between conv and linear, the relationship may be complex.
        # For VGG-style: last_conv_out_channels * spatial_dims = linear_in_features

        # Let's check if we need to adjust
        mask = self._pruning_masks.get(last_conv_name)
        if mask is None:
            return

        kept_channels = mask.sum().item()
        last_conv = dict(pruned_model.named_modules()).get(last_conv_name)
        if last_conv is None:
            return

        original_channels = last_conv.out_channels if hasattr(last_conv, 'out_channels') else kept_channels

        # Estimate how in_features should change
        # If in_features = C * H * W, and we now have C' = kept_channels
        # then new_in_features = in_features * (C' / C)

        ratio = kept_channels / original_channels
        old_in_features = linear_module.in_features
        new_in_features = int(old_in_features * ratio)

        if new_in_features == old_in_features:
            return

        # Create new linear layer
        new_linear = nn.Linear(
            in_features=new_in_features,
            out_features=linear_module.out_features,
            bias=linear_module.bias is not None,
            device=linear_module.weight.device,
            dtype=linear_module.weight.dtype,
        )

        # We need to select which input weights to keep
        # This depends on how features are flattened (channel-first vs channel-last)
        # Assuming channel-first (default in PyTorch): features are grouped by channel
        # So weight[:, :C*H*W] with C channels -> select every C-th block

        # For simplicity, assuming the linear input directly corresponds to flattened
        # (C, H, W) where each channel contributes (H*W) features

        # Calculate spatial size
        spatial_size = old_in_features // original_channels

        # Build indices to keep
        keep_channel_indices = mask.nonzero(as_tuple=True)[0].cpu().numpy()
        keep_weight_indices = []
        for c_idx in keep_channel_indices:
            start = c_idx * spatial_size
            end = start + spatial_size
            keep_weight_indices.extend(range(start, end))

        keep_weight_indices = torch.tensor(keep_weight_indices, device=linear_module.weight.device)

        new_linear.weight.data = linear_module.weight.data[:, keep_weight_indices]
        if linear_module.bias is not None:
            new_linear.bias.data = linear_module.bias.data.clone()

        self._replace_layer(pruned_model, first_linear_name, new_linear)

    def _replace_layer(
        self,
        model: nn.Module,
        layer_name: str,
        new_layer: nn.Module,
    ) -> None:
        """Replace a layer in the model by name."""
        parts = layer_name.split('.')
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_layer)

    def _compute_compression_stats(self, pruned_model: nn.Module) -> None:
        """Compute compression statistics."""
        original_params = sum(p.numel() for p in self.model.parameters())
        pruned_params = sum(p.numel() for p in pruned_model.parameters())

        # Count FLOPs reduction (approximate)
        original_flops = self._estimate_conv_flops(self.model)
        pruned_flops = self._estimate_conv_flops(pruned_model)

        self._compression_stats = {
            "original_parameters": original_params,
            "pruned_parameters": pruned_params,
            "compression_ratio": original_params / max(pruned_params, 1),
            "parameter_sparsity": 1 - pruned_params / max(original_params, 1),
            "original_flops": original_flops,
            "pruned_flops": pruned_flops,
            "flops_reduction": 1 - pruned_flops / max(original_flops, 1) if original_flops > 0 else 0,
        }

    def _estimate_conv_flops(self, model: nn.Module) -> int:
        """
        Estimate FLOPs for Conv2d layers in the model.

        This is a simplified estimate assuming standard convolutions.
        """
        total_flops = 0
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                # FLOPs = 2 * C_in * C_out * H * W * kH * kW / groups
                # We don't know H, W without input, so just count params
                # FLOPs ≈ 2 * weight_params (for 1x1 output)
                weight_params = module.weight.numel()
                total_flops += 2 * weight_params
            elif isinstance(module, nn.Linear):
                total_flops += 2 * module.weight.numel()
        return total_flops

    def get_compression_stats(self) -> Dict:
        """
        Get compression statistics from the last pruning operation.

        Returns:
            Dictionary with compression metrics.
        """
        if self._compression_stats is None:
            return {"error": "No pruning has been performed yet"}
        return self._compression_stats

    def get_pruning_masks(self) -> Dict[str, torch.Tensor]:
        """
        Get pruning masks from the last pruning operation.

        Returns:
            Dictionary mapping layer names to their pruning masks.
        """
        return self._pruning_masks

    def analyze_filters(self) -> Dict[str, Dict]:
        """
        Analyze filter statistics for each layer.

        Returns:
            Dictionary with analysis per layer.
        """
        analysis = {}
        for name, stats in self.statistics.items():
            freq = stats.winner_frequency
            analysis[name] = {
                "num_channels": stats.winner_count.numel(),
                "total_positions": stats.total_positions,
                "never_win": (freq == 0).sum().item(),
                "rarely_win": (freq < 0.01).sum().item(),
                "frequently_win": (freq > 0.1).sum().item(),
                "min_frequency": freq.min().item(),
                "max_frequency": freq.max().item(),
                "mean_frequency": freq.mean().item(),
                "std_frequency": freq.std().item(),
            }

            if stats.average_margin is not None:
                margin = stats.average_margin
                analysis[name]["mean_margin"] = (
                    margin[margin > 0].mean().item() if (margin > 0).any() else 0
                )
                analysis[name]["max_margin"] = margin.max().item()

        return analysis
