"""
FFN Tropical Pruner: Apply coupled pruning to LLM FFN layers.

SwiGLU FFN structure:
    intermediate = SiLU(gate_proj(x)) * up_proj(x)
    output = down_proj(intermediate)

Coupled pruning constraint - all three projections share the same intermediate dimension:
    gate_proj: (hidden, intermediate) → prune columns [keep_indices]
    up_proj:   (hidden, intermediate) → prune columns [keep_indices]
    down_proj: (intermediate, hidden) → prune rows [keep_indices]
"""

import copy
from typing import Dict, List, Optional, Union
import torch
import torch.nn as nn

from tropical_pruning.llm.ffn_counter import FFNStatistics
from tropical_pruning.llm.loader import (
    FFNLayerInfo,
    get_ffn_layer_names,
    count_parameters,
)


class FFNTropicalPruner:
    """
    Apply coupled tropical pruning to LLM FFN layers.

    This pruner uses winner statistics from FFNWinnerCounter to determine
    which intermediate neurons to keep in each FFN block. Pruning is applied
    consistently across gate_proj, up_proj, and down_proj to maintain dimensional
    consistency.

    Example:
        >>> counter = FFNWinnerCounter(model)
        >>> stats = counter.collect(calibration_loader)
        >>>
        >>> pruner = FFNTropicalPruner(model, stats)
        >>> pruned_model = pruner.prune(sparsity=0.3)
        >>>
        >>> print(pruner.get_compression_stats())
    """

    def __init__(
        self,
        model: nn.Module,
        statistics: Dict[int, FFNStatistics],
    ):
        """
        Initialize the FFNTropicalPruner.

        Args:
            model: The LLM model to prune.
            statistics: Dictionary mapping layer indices to FFNStatistics.
        """
        self.model = model
        self.statistics = statistics
        self.ffn_layers = get_ffn_layer_names(model)

        self._pruning_masks: Dict[int, torch.Tensor] = {}
        self._compression_stats: Optional[Dict] = None

    def compute_importance(self, layer_idx: int) -> torch.Tensor:
        """
        Compute importance scores for intermediate neurons in an FFN layer.

        Importance is based on winner frequency: how often each intermediate
        neuron "wins" the tropical argmax in down_proj.

        Args:
            layer_idx: Index of the FFN layer.

        Returns:
            Importance scores, shape (intermediate_size,).
        """
        if layer_idx not in self.statistics:
            raise ValueError(f"No statistics for layer index: {layer_idx}")

        stats = self.statistics[layer_idx]
        return stats.winner_frequency

    def get_pruning_mask(
        self,
        layer_idx: int,
        sparsity: float,
    ) -> torch.Tensor:
        """
        Get pruning mask for an FFN layer.

        Args:
            layer_idx: Index of the FFN layer.
            sparsity: Target sparsity (fraction of intermediate neurons to prune).

        Returns:
            Boolean mask, True = keep, False = prune.
            Shape: (intermediate_size,)
        """
        importance = self.compute_importance(layer_idx)
        n_neurons = importance.numel()
        n_keep = max(1, int(n_neurons * (1 - sparsity)))

        # Keep top-k by importance
        _, top_indices = importance.topk(n_keep)
        mask = torch.zeros(n_neurons, dtype=torch.bool, device=importance.device)
        mask[top_indices] = True

        return mask

    def prune(
        self,
        sparsity: Union[float, Dict[int, float]],
        inplace: bool = False,
    ) -> nn.Module:
        """
        Apply coupled structured pruning to FFN layers.

        For each FFN block, prunes the same intermediate neurons across
        gate_proj, up_proj, and down_proj to maintain consistency.

        Args:
            sparsity: Target sparsity. Can be:
                     - float: Same sparsity for all FFN layers
                     - dict: Per-layer sparsity mapping (layer_idx → sparsity)
            inplace: If True, modify model in place. Otherwise, return a copy.

        Returns:
            Pruned model.
        """
        if inplace:
            pruned_model = self.model
        else:
            pruned_model = copy.deepcopy(self.model)

        # Convert uniform sparsity to per-layer dict
        if isinstance(sparsity, float):
            sparsity_dict = {idx: sparsity for idx in self.statistics.keys()}
        else:
            sparsity_dict = sparsity

        # Get module dict for efficient lookup
        modules_dict = dict(pruned_model.named_modules())

        self._pruning_masks = {}

        for layer_info in self.ffn_layers:
            layer_idx = layer_info.layer_idx

            # Skip if no statistics for this layer
            if layer_idx not in self.statistics:
                continue

            # Get sparsity for this layer
            layer_sparsity = sparsity_dict.get(layer_idx, 0.0)
            if layer_sparsity == 0.0:
                continue

            # Compute mask
            mask = self.get_pruning_mask(layer_idx, layer_sparsity)
            self._pruning_masks[layer_idx] = mask

            # Apply coupled pruning
            self._prune_ffn_block(
                pruned_model,
                modules_dict,
                layer_info,
                mask,
            )

        # Compute compression statistics
        self._compute_compression_stats(pruned_model)

        return pruned_model

    def _prune_ffn_block(
        self,
        model: nn.Module,
        modules_dict: Dict[str, nn.Module],
        layer_info: FFNLayerInfo,
        mask: torch.Tensor,
    ) -> None:
        """
        Apply coupled pruning to a single FFN block.

        Args:
            model: Model to modify.
            modules_dict: Pre-computed module dictionary.
            layer_info: FFN layer information.
            mask: Boolean mask for intermediate neurons to keep.
        """
        keep_indices = mask.nonzero(as_tuple=True)[0]
        new_intermediate_size = keep_indices.numel()

        # Prune gate_proj: (hidden, intermediate) → prune columns
        if layer_info.gate_proj_name and layer_info.gate_proj_name in modules_dict:
            self._prune_output_features(
                model,
                modules_dict,
                layer_info.gate_proj_name,
                keep_indices,
            )

        # Prune up_proj: (hidden, intermediate) → prune columns
        if layer_info.up_proj_name and layer_info.up_proj_name in modules_dict:
            # Handle case where gate and up are the same (e.g., Phi's gate_up_proj)
            if layer_info.up_proj_name != layer_info.gate_proj_name:
                self._prune_output_features(
                    model,
                    modules_dict,
                    layer_info.up_proj_name,
                    keep_indices,
                )

        # Prune down_proj: (intermediate, hidden) → prune rows (input features)
        if layer_info.down_proj_name in modules_dict:
            self._prune_input_features(
                model,
                modules_dict,
                layer_info.down_proj_name,
                keep_indices,
            )

    def _prune_output_features(
        self,
        model: nn.Module,
        modules_dict: Dict[str, nn.Module],
        layer_name: str,
        keep_indices: torch.Tensor,
    ) -> None:
        """
        Prune OUTPUT features (rows of weight matrix) from a Linear layer.

        For gate_proj/up_proj: Weight shape is (intermediate_size, hidden_size).
        We keep rows corresponding to keep_indices.
        """
        module = modules_dict[layer_name]
        if not isinstance(module, nn.Linear):
            return

        new_out_features = keep_indices.numel()

        new_layer = nn.Linear(
            module.in_features,
            new_out_features,
            bias=module.bias is not None,
            device=module.weight.device,
            dtype=module.weight.dtype,
        )

        # Keep selected rows
        with torch.no_grad():
            new_layer.weight.copy_(module.weight[keep_indices, :])
            if module.bias is not None:
                new_layer.bias.copy_(module.bias[keep_indices])

        self._replace_layer(model, layer_name, new_layer)
        # Update modules_dict to reflect the change
        modules_dict[layer_name] = new_layer

    def _prune_input_features(
        self,
        model: nn.Module,
        modules_dict: Dict[str, nn.Module],
        layer_name: str,
        keep_indices: torch.Tensor,
    ) -> None:
        """
        Prune INPUT features (columns of weight matrix) from a Linear layer.

        For down_proj: Weight shape is (hidden_size, intermediate_size).
        We keep columns corresponding to keep_indices.
        """
        module = modules_dict[layer_name]
        if not isinstance(module, nn.Linear):
            return

        new_in_features = keep_indices.numel()

        new_layer = nn.Linear(
            new_in_features,
            module.out_features,
            bias=module.bias is not None,
            device=module.weight.device,
            dtype=module.weight.dtype,
        )

        # Keep selected columns
        with torch.no_grad():
            new_layer.weight.copy_(module.weight[:, keep_indices])
            if module.bias is not None:
                new_layer.bias.copy_(module.bias)

        self._replace_layer(model, layer_name, new_layer)
        # Update modules_dict to reflect the change
        modules_dict[layer_name] = new_layer

    def _replace_layer(
        self,
        model: nn.Module,
        layer_name: str,
        new_layer: nn.Module,
    ) -> None:
        """Replace a layer in the model by name."""
        parts = layer_name.split(".")
        parent = model
        for part in parts[:-1]:
            if part.isdigit():
                parent = parent[int(part)]
            else:
                parent = getattr(parent, part)
        setattr(parent, parts[-1], new_layer)

    def _compute_compression_stats(self, pruned_model: nn.Module) -> None:
        """Compute compression statistics."""
        original_params = count_parameters(self.model)
        pruned_params = count_parameters(pruned_model)

        # Count FFN-specific parameters
        original_ffn_params = self._count_ffn_parameters(self.model)
        pruned_ffn_params = self._count_ffn_parameters(pruned_model)

        self._compression_stats = {
            "original_parameters": original_params,
            "pruned_parameters": pruned_params,
            "compression_ratio": original_params / max(pruned_params, 1),
            "sparsity_achieved": 1 - pruned_params / max(original_params, 1),
            "original_ffn_parameters": original_ffn_params,
            "pruned_ffn_parameters": pruned_ffn_params,
            "ffn_compression_ratio": original_ffn_params / max(pruned_ffn_params, 1),
            "ffn_sparsity_achieved": 1 - pruned_ffn_params / max(original_ffn_params, 1),
        }

    def _count_ffn_parameters(self, model: nn.Module) -> int:
        """Count parameters in FFN layers only."""
        total = 0
        modules_dict = dict(model.named_modules())

        for layer_info in get_ffn_layer_names(model):
            for name in [layer_info.gate_proj_name, layer_info.up_proj_name, layer_info.down_proj_name]:
                if name and name in modules_dict:
                    module = modules_dict[name]
                    if isinstance(module, nn.Linear):
                        total += module.weight.numel()
                        if module.bias is not None:
                            total += module.bias.numel()

        return total

    def get_compression_stats(self) -> Dict:
        """
        Get compression statistics from the last pruning operation.

        Returns:
            Dictionary with compression metrics.
        """
        if self._compression_stats is None:
            return {"error": "No pruning has been performed yet"}
        return self._compression_stats

    def analyze(self) -> Dict[int, Dict]:
        """
        Analyze pruning decisions for each FFN layer.

        Returns:
            Dictionary with analysis per layer.
        """
        analysis = {}
        for layer_idx, stats in self.statistics.items():
            freq = stats.winner_frequency

            analysis[layer_idx] = {
                "intermediate_size": stats.intermediate_size,
                "neurons_pruned": (
                    (~self._pruning_masks.get(layer_idx, torch.ones(stats.intermediate_size, dtype=torch.bool)))
                    .sum()
                    .item()
                    if layer_idx in self._pruning_masks
                    else 0
                ),
                "neurons_kept": (
                    self._pruning_masks.get(layer_idx, torch.ones(stats.intermediate_size, dtype=torch.bool))
                    .sum()
                    .item()
                    if layer_idx in self._pruning_masks
                    else stats.intermediate_size
                ),
                "never_win_count": (freq == 0).sum().item(),
                "min_kept_importance": (
                    freq[self._pruning_masks[layer_idx]].min().item()
                    if layer_idx in self._pruning_masks
                    else freq.min().item()
                ),
                "max_pruned_importance": (
                    freq[~self._pruning_masks[layer_idx]].max().item()
                    if layer_idx in self._pruning_masks and (~self._pruning_masks[layer_idx]).any()
                    else 0
                ),
            }

        return analysis

    def visualize_importance(
        self,
        layer_idx: Optional[int] = None,
        save_path: Optional[str] = None,
    ):
        """
        Visualize neuron importance for FFN layers.

        Args:
            layer_idx: Layer to visualize. If None, shows all layers.
            save_path: Path to save the figure. If None, displays interactively.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for visualization. "
                "Install with: pip install matplotlib"
            )

        layers = [layer_idx] if layer_idx is not None else list(self.statistics.keys())

        n_layers = len(layers)
        fig, axes = plt.subplots(n_layers, 1, figsize=(12, 4 * n_layers))
        if n_layers == 1:
            axes = [axes]

        for ax, idx in zip(axes, layers):
            importance = self.compute_importance(idx)
            sorted_importance, _ = importance.sort(descending=True)

            ax.bar(range(len(sorted_importance)), sorted_importance.cpu().numpy(), width=1.0)
            ax.set_xlabel("Intermediate neuron (sorted by importance)")
            ax.set_ylabel("Winner frequency")
            ax.set_title(f"FFN Layer {idx}")

            # Add pruning threshold line if available
            if idx in self._pruning_masks:
                mask = self._pruning_masks[idx]
                n_keep = mask.sum().item()
                ax.axvline(x=n_keep, color="r", linestyle="--", label=f"Prune threshold ({n_keep} kept)")
                ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        else:
            plt.show()

        return fig
