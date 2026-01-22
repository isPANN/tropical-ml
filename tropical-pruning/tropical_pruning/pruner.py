"""
Tropical Pruner: Main interface for winner-based tropical structured pruning.

This module provides the TropicalPruner class which takes winner statistics
and applies structured pruning to neural network models.

Key insight: Winner stats for layer[i] track which INPUT neurons "win".
These inputs are OUTPUT neurons of the previous layer, so we use stats
from layer[i+1] to decide which outputs of layer[i] to prune.
"""

import copy
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn

from tropical_pruning.counter import WinnerStatistics
from tropical_pruning.criteria import (
    PruningCriterion,
    WinnerFrequencyCriterion,
    get_criterion,
)


class TropicalPruner:
    """
    Applies winner-based tropical structured pruning to neural networks.

    Structured pruning removes entire neurons (rows/columns of weight matrices),
    resulting in smaller models that can be efficiently executed without
    sparse matrix support.

    Example:
        >>> from tropical_pruning import WinnerCounter, TropicalPruner
        >>>
        >>> # Collect winner statistics
        >>> counter = WinnerCounter(model)
        >>> stats = counter.collect(calibration_loader)
        >>>
        >>> # Prune to 50% sparsity
        >>> pruner = TropicalPruner(model, stats)
        >>> pruned_model = pruner.prune(sparsity=0.5)
        >>>
        >>> # Check compression
        >>> print(pruner.get_compression_stats())
    """

    def __init__(
        self,
        model: nn.Module,
        statistics: Dict[str, WinnerStatistics],
        criterion: Optional[Union[str, PruningCriterion]] = None,
    ):
        """
        Initialize the TropicalPruner.

        Args:
            model: The model to prune.
            statistics: Dictionary mapping layer names to WinnerStatistics.
            criterion: Pruning criterion (string name or PruningCriterion instance).
                      Defaults to "winner_frequency".
        """
        self.model = model
        self.statistics = statistics

        if criterion is None:
            self.criterion = WinnerFrequencyCriterion()
        elif isinstance(criterion, str):
            self.criterion = get_criterion(criterion)
        else:
            self.criterion = criterion

        self._pruning_masks: Dict[str, torch.Tensor] = {}
        self._compression_stats: Optional[Dict] = None

    def compute_importance(self, layer_name: str) -> torch.Tensor:
        """
        Compute importance scores for neurons in a layer.

        Args:
            layer_name: Name of the layer.

        Returns:
            Importance scores, shape (num_neurons,).
        """
        if layer_name not in self.statistics:
            raise ValueError(f"No statistics for layer: {layer_name}")

        stats = self.statistics[layer_name]
        return self.criterion.compute_importance(stats)

    def get_pruning_mask(
        self,
        layer_name: str,
        sparsity: float,
    ) -> torch.Tensor:
        """
        Get pruning mask for a layer.

        Args:
            layer_name: Name of the layer.
            sparsity: Target sparsity (fraction to prune).

        Returns:
            Boolean mask, True = keep, False = prune.
        """
        stats = self.statistics[layer_name]
        return self.criterion.get_pruning_mask(stats, sparsity)

    def prune(
        self,
        sparsity: Union[float, Dict[str, float]],
        inplace: bool = False,
    ) -> nn.Module:
        """
        Apply structured pruning to the model.

        For a sequence of Linear layers [L0, L1, L2, ...]:
        - Stats from L1 tell us which outputs of L0 are important
        - We prune OUTPUT neurons (rows) of L0 based on L1's stats
        - We also prune INPUT neurons (cols) of L1 correspondingly

        Args:
            sparsity: Target sparsity. Can be:
                     - float: Same sparsity for all prunable layers
                     - dict: Per-layer sparsity mapping
            inplace: If True, modify model in place. Otherwise, return a copy.

        Returns:
            Pruned model.
        """
        if inplace:
            pruned_model = self.model
        else:
            pruned_model = copy.deepcopy(self.model)

        # Get linear layers in order
        linear_layers = self._get_linear_layers_ordered()

        if len(linear_layers) < 2:
            return pruned_model  # Nothing to prune

        # Convert uniform sparsity to per-layer dict
        # We apply sparsity based on the stats layer (which determines importance)
        if isinstance(sparsity, float):
            sparsity_dict = {name: sparsity for name in self.statistics.keys()}
        else:
            sparsity_dict = sparsity

        # Prune hidden layers (not the first layer's input, not the last layer's output)
        # Stats from layer[i] → prune outputs of layer[i-1]
        self._pruning_masks = {}

        for i in range(1, len(linear_layers)):
            curr_layer_name = linear_layers[i]
            prev_layer_name = linear_layers[i - 1]

            # Skip if we don't have stats for this layer
            if curr_layer_name not in self.statistics:
                continue

            # Skip the last layer (don't prune classifier outputs)
            if i == len(linear_layers) - 1:
                # But we might still need to adjust input if previous was pruned
                if prev_layer_name in self._pruning_masks:
                    mask = self._pruning_masks[prev_layer_name]
                    self._prune_layer_input(pruned_model, curr_layer_name, mask)
                continue

            # Get sparsity for this pruning decision
            layer_sparsity = sparsity_dict.get(curr_layer_name, 0.0)
            if layer_sparsity == 0.0:
                continue

            # Compute mask based on winner stats of current layer
            # This tells us importance of prev layer's outputs
            mask = self.get_pruning_mask(curr_layer_name, layer_sparsity)
            self._pruning_masks[prev_layer_name] = mask

            # Prune OUTPUT neurons (rows) of previous layer
            self._prune_layer_output(pruned_model, prev_layer_name, mask)

            # Prune INPUT neurons (cols) of current layer
            self._prune_layer_input(pruned_model, curr_layer_name, mask)

        # Compute compression statistics
        self._compute_compression_stats(pruned_model)

        return pruned_model

    def _get_linear_layers_ordered(self) -> List[str]:
        """Get Linear layer names in forward execution order."""
        layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                layers.append(name)
        return layers

    def _prune_layer_output(
        self,
        model: nn.Module,
        layer_name: str,
        mask: torch.Tensor,
    ) -> None:
        """
        Prune OUTPUT neurons (rows of weight matrix) from a layer.

        Weight shape: (out_features, in_features)
        We keep rows where mask is True.
        """
        module = dict(model.named_modules())[layer_name]
        if not isinstance(module, nn.Linear):
            return

        keep_indices = mask.nonzero(as_tuple=True)[0]
        new_out_features = keep_indices.numel()

        new_layer = nn.Linear(
            module.in_features,
            new_out_features,
            bias=module.bias is not None,
            device=module.weight.device,
            dtype=module.weight.dtype,
        )

        # Keep selected rows
        new_layer.weight.data = module.weight.data[keep_indices, :]
        if module.bias is not None:
            new_layer.bias.data = module.bias.data[keep_indices]

        self._replace_layer(model, layer_name, new_layer)

    def _prune_layer_input(
        self,
        model: nn.Module,
        layer_name: str,
        mask: torch.Tensor,
    ) -> None:
        """
        Prune INPUT neurons (columns of weight matrix) from a layer.

        Weight shape: (out_features, in_features)
        We keep columns where mask is True.
        """
        module = dict(model.named_modules())[layer_name]
        if not isinstance(module, nn.Linear):
            return

        keep_indices = mask.nonzero(as_tuple=True)[0]
        new_in_features = keep_indices.numel()

        new_layer = nn.Linear(
            new_in_features,
            module.out_features,
            bias=module.bias is not None,
            device=module.weight.device,
            dtype=module.weight.dtype,
        )

        # Keep selected columns
        new_layer.weight.data = module.weight.data[:, keep_indices]
        if module.bias is not None:
            new_layer.bias.data = module.bias.data.clone()

        self._replace_layer(model, layer_name, new_layer)

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

        self._compression_stats = {
            "original_parameters": original_params,
            "pruned_parameters": pruned_params,
            "compression_ratio": original_params / max(pruned_params, 1),
            "sparsity_achieved": 1 - pruned_params / max(original_params, 1),
        }

    def get_compression_stats(self) -> Dict:
        """
        Get compression statistics from the last pruning operation.

        Returns:
            Dictionary with compression metrics.
        """
        if self._compression_stats is None:
            return {"error": "No pruning has been performed yet"}
        return self._compression_stats

    def analyze_winners(self) -> Dict[str, Dict]:
        """
        Analyze winner statistics for each layer.

        Returns:
            Dictionary with analysis per layer.
        """
        analysis = {}
        for name, stats in self.statistics.items():
            freq = stats.winner_frequency
            analysis[name] = {
                "num_neurons": stats.winner_count.numel(),
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
                analysis[name]["mean_margin"] = margin[margin > 0].mean().item() if (margin > 0).any() else 0
                analysis[name]["max_margin"] = margin.max().item()

        return analysis

    def visualize_importance(
        self,
        layer_name: Optional[str] = None,
        save_path: Optional[str] = None,
    ):
        """
        Visualize neuron importance for a layer.

        Args:
            layer_name: Layer to visualize. If None, shows all layers.
            save_path: Path to save the figure. If None, displays interactively.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for visualization. Install with: pip install matplotlib")

        layers = [layer_name] if layer_name else list(self.statistics.keys())

        fig, axes = plt.subplots(len(layers), 1, figsize=(12, 4 * len(layers)))
        if len(layers) == 1:
            axes = [axes]

        for ax, name in zip(axes, layers):
            importance = self.compute_importance(name)
            sorted_importance, _ = importance.sort(descending=True)

            ax.bar(range(len(sorted_importance)), sorted_importance.cpu().numpy())
            ax.set_xlabel("Neuron (sorted by importance)")
            ax.set_ylabel("Importance Score")
            ax.set_title(f"Layer: {name}")
            ax.axhline(y=0.01, color='r', linestyle='--', label='1% threshold')
            ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()

        return fig
