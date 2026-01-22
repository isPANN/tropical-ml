"""
Baseline pruning methods for LLM FFN layers.

Implements several baseline methods for comparison with tropical pruning:
- Magnitude pruning (L1/L2 norm)
- Activation-based pruning
- Wanda-style pruning (weight * activation)
- Taylor importance (gradient-based, LLM-Pruner style)
- FLAP-style fluctuation pruning

References:
- LLM-Pruner: https://github.com/horseee/LLM-Pruner (NeurIPS 2023)
- FLAP: https://github.com/CASIA-IVA-Lab/FLAP (AAAI 2024)
- Wanda: https://github.com/locuslab/wanda (ICLR 2024)
"""

import copy
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from tqdm import tqdm

from tropical_pruning.llm.loader import get_ffn_layer_names, FFNLayerInfo


class FFNBaselinePruner:
    """Base class for FFN pruning baselines."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.ffn_layers = get_ffn_layer_names(model)
        self._compression_stats: Optional[Dict] = None

    def compute_importance(self, layer_info: FFNLayerInfo) -> torch.Tensor:
        """Compute importance scores for intermediate neurons. Override in subclass."""
        raise NotImplementedError

    def prune(self, sparsity: float, inplace: bool = False) -> nn.Module:
        """Apply structured pruning to FFN layers."""
        if inplace:
            pruned_model = self.model
        else:
            pruned_model = copy.deepcopy(self.model)

        modules_dict = dict(pruned_model.named_modules())
        original_params = sum(p.numel() for p in self.model.parameters())

        for layer_info in self.ffn_layers:
            importance = self.compute_importance(layer_info)
            n_keep = max(1, int(len(importance) * (1 - sparsity)))

            # Get indices to keep (top-k by importance)
            _, top_indices = importance.topk(n_keep)
            keep_indices = top_indices.sort().values

            self._prune_ffn_block(pruned_model, modules_dict, layer_info, keep_indices)

        pruned_params = sum(p.numel() for p in pruned_model.parameters())
        self._compression_stats = {
            "original_parameters": original_params,
            "pruned_parameters": pruned_params,
            "compression_ratio": original_params / pruned_params,
            "sparsity_achieved": 1 - pruned_params / original_params,
        }

        return pruned_model

    def _prune_ffn_block(
        self,
        model: nn.Module,
        modules_dict: Dict[str, nn.Module],
        layer_info: FFNLayerInfo,
        keep_indices: torch.Tensor,
    ) -> None:
        """Apply coupled pruning to gate/up/down projections."""
        n_keep = len(keep_indices)

        # Prune gate_proj
        gate_proj = modules_dict.get(layer_info.gate_proj_name)
        if gate_proj and isinstance(gate_proj, nn.Linear):
            new_gate = nn.Linear(
                gate_proj.in_features, n_keep,
                bias=gate_proj.bias is not None,
                device=gate_proj.weight.device,
                dtype=gate_proj.weight.dtype,
            )
            with torch.no_grad():
                new_gate.weight.copy_(gate_proj.weight[keep_indices, :])
                if gate_proj.bias is not None:
                    new_gate.bias.copy_(gate_proj.bias[keep_indices])
            self._replace_layer(model, layer_info.gate_proj_name, new_gate)
            modules_dict[layer_info.gate_proj_name] = new_gate

        # Prune up_proj
        up_proj = modules_dict.get(layer_info.up_proj_name)
        if up_proj and isinstance(up_proj, nn.Linear):
            if layer_info.up_proj_name != layer_info.gate_proj_name:
                new_up = nn.Linear(
                    up_proj.in_features, n_keep,
                    bias=up_proj.bias is not None,
                    device=up_proj.weight.device,
                    dtype=up_proj.weight.dtype,
                )
                with torch.no_grad():
                    new_up.weight.copy_(up_proj.weight[keep_indices, :])
                    if up_proj.bias is not None:
                        new_up.bias.copy_(up_proj.bias[keep_indices])
                self._replace_layer(model, layer_info.up_proj_name, new_up)
                modules_dict[layer_info.up_proj_name] = new_up

        # Prune down_proj
        down_proj = modules_dict.get(layer_info.down_proj_name)
        if down_proj and isinstance(down_proj, nn.Linear):
            new_down = nn.Linear(
                n_keep, down_proj.out_features,
                bias=down_proj.bias is not None,
                device=down_proj.weight.device,
                dtype=down_proj.weight.dtype,
            )
            with torch.no_grad():
                new_down.weight.copy_(down_proj.weight[:, keep_indices])
                if down_proj.bias is not None:
                    new_down.bias.copy_(down_proj.bias)
            self._replace_layer(model, layer_info.down_proj_name, new_down)
            modules_dict[layer_info.down_proj_name] = new_down

    def _replace_layer(self, model: nn.Module, layer_name: str, new_layer: nn.Module) -> None:
        """Replace a layer in the model by name."""
        parts = layer_name.split(".")
        parent = model
        for part in parts[:-1]:
            if part.isdigit():
                parent = parent[int(part)]
            else:
                parent = getattr(parent, part)
        setattr(parent, parts[-1], new_layer)

    def get_compression_stats(self) -> Dict:
        """Get compression statistics."""
        if self._compression_stats is None:
            return {"error": "No pruning has been performed yet"}
        return self._compression_stats


class MagnitudePruner(FFNBaselinePruner):
    """
    Magnitude-based pruning (L1 norm).

    Importance = L1 norm of weights connected to each intermediate neuron.
    This is a simple but widely-used baseline.
    """

    def __init__(self, model: nn.Module, norm: str = "l1"):
        super().__init__(model)
        self.norm = norm

    def compute_importance(self, layer_info: FFNLayerInfo) -> torch.Tensor:
        modules_dict = dict(self.model.named_modules())
        down_proj = modules_dict[layer_info.down_proj_name]

        # down_proj.weight: (hidden_size, intermediate_size)
        # Importance of neuron j = norm of column j
        if self.norm == "l1":
            return down_proj.weight.abs().sum(dim=0)
        elif self.norm == "l2":
            return down_proj.weight.pow(2).sum(dim=0).sqrt()
        else:
            raise ValueError(f"Unknown norm: {self.norm}")


class ActivationPruner(FFNBaselinePruner):
    """
    Activation-based pruning.

    Importance = mean activation magnitude for each intermediate neuron.
    Requires calibration data to compute activations.
    """

    def __init__(self, model: nn.Module, calibration_loader: torch.utils.data.DataLoader):
        super().__init__(model)
        self.calibration_loader = calibration_loader
        self._activation_stats: Dict[int, torch.Tensor] = {}
        self._collect_activations()

    def _collect_activations(self) -> None:
        """Collect activation statistics from calibration data."""
        modules_dict = dict(self.model.named_modules())
        hooks = []
        activation_sums: Dict[int, torch.Tensor] = {}
        activation_counts: Dict[int, int] = {}

        def make_hook(layer_idx: int):
            def hook(module, input, output):
                # input[0]: (batch, seq, intermediate_size) for down_proj
                x = input[0].detach()
                if layer_idx not in activation_sums:
                    activation_sums[layer_idx] = torch.zeros(
                        x.shape[-1], device=x.device, dtype=torch.float32
                    )
                    activation_counts[layer_idx] = 0
                activation_sums[layer_idx] += x.abs().sum(dim=(0, 1)).float()
                activation_counts[layer_idx] += x.shape[0] * x.shape[1]
            return hook

        # Register hooks on down_proj layers
        for layer_info in self.ffn_layers:
            down_proj = modules_dict.get(layer_info.down_proj_name)
            if down_proj and isinstance(down_proj, nn.Linear):
                hook = down_proj.register_forward_hook(make_hook(layer_info.layer_idx))
                hooks.append(hook)

        # Run calibration
        self.model.eval()
        with torch.no_grad():
            for batch in self.calibration_loader:
                if isinstance(batch, dict):
                    input_ids = batch["input_ids"]
                else:
                    input_ids = batch[0] if isinstance(batch, (list, tuple)) else batch

                device = next(self.model.parameters()).device
                input_ids = input_ids.to(device)
                self.model(input_ids)

        # Remove hooks and compute mean
        for hook in hooks:
            hook.remove()

        for layer_idx in activation_sums:
            self._activation_stats[layer_idx] = (
                activation_sums[layer_idx] / activation_counts[layer_idx]
            )

    def compute_importance(self, layer_info: FFNLayerInfo) -> torch.Tensor:
        return self._activation_stats.get(
            layer_info.layer_idx,
            torch.ones(layer_info.intermediate_size)
        )


class WandaStylePruner(FFNBaselinePruner):
    """
    Wanda-style pruning adapted for structured pruning.

    Original Wanda: importance = |weight| * ||activation||
    For structured FFN: importance[j] = sum_i |W_down[i,j]| * mean(|activation[j]|)

    Reference: https://github.com/locuslab/wanda (ICLR 2024)
    """

    def __init__(self, model: nn.Module, calibration_loader: torch.utils.data.DataLoader):
        super().__init__(model)
        self.calibration_loader = calibration_loader
        self._activation_norms: Dict[int, torch.Tensor] = {}
        self._collect_activation_norms()

    def _collect_activation_norms(self) -> None:
        """Collect activation norms from calibration data."""
        modules_dict = dict(self.model.named_modules())
        hooks = []
        activation_sums: Dict[int, torch.Tensor] = {}
        activation_counts: Dict[int, int] = {}

        def make_hook(layer_idx: int):
            def hook(module, input, output):
                x = input[0].detach()
                if layer_idx not in activation_sums:
                    activation_sums[layer_idx] = torch.zeros(
                        x.shape[-1], device=x.device, dtype=torch.float32
                    )
                    activation_counts[layer_idx] = 0
                # Wanda uses squared norm
                activation_sums[layer_idx] += x.pow(2).sum(dim=(0, 1)).float()
                activation_counts[layer_idx] += x.shape[0] * x.shape[1]
            return hook

        for layer_info in self.ffn_layers:
            down_proj = modules_dict.get(layer_info.down_proj_name)
            if down_proj and isinstance(down_proj, nn.Linear):
                hook = down_proj.register_forward_hook(make_hook(layer_info.layer_idx))
                hooks.append(hook)

        self.model.eval()
        with torch.no_grad():
            for batch in self.calibration_loader:
                if isinstance(batch, dict):
                    input_ids = batch["input_ids"]
                else:
                    input_ids = batch[0] if isinstance(batch, (list, tuple)) else batch
                device = next(self.model.parameters()).device
                input_ids = input_ids.to(device)
                self.model(input_ids)

        for hook in hooks:
            hook.remove()

        for layer_idx in activation_sums:
            # RMS of activations
            self._activation_norms[layer_idx] = (
                activation_sums[layer_idx] / activation_counts[layer_idx]
            ).sqrt()

    def compute_importance(self, layer_info: FFNLayerInfo) -> torch.Tensor:
        modules_dict = dict(self.model.named_modules())
        down_proj = modules_dict[layer_info.down_proj_name]

        # Weight magnitude
        weight_importance = down_proj.weight.abs().sum(dim=0)

        # Activation norm
        activation_norm = self._activation_norms.get(
            layer_info.layer_idx,
            torch.ones(layer_info.intermediate_size, device=weight_importance.device)
        )

        # Wanda: weight * activation
        return weight_importance * activation_norm.to(weight_importance.device)


class FLAPStylePruner(FFNBaselinePruner):
    """
    FLAP-style fluctuation-based pruning.

    Importance based on activation fluctuation (variance) weighted by weight magnitude.
    Neurons with high fluctuation and high weight magnitude are considered important.

    Reference: https://github.com/CASIA-IVA-Lab/FLAP (AAAI 2024)
    """

    def __init__(self, model: nn.Module, calibration_loader: torch.utils.data.DataLoader):
        super().__init__(model)
        self.calibration_loader = calibration_loader
        self._fluctuation_stats: Dict[int, torch.Tensor] = {}
        self._collect_fluctuation()

    def _collect_fluctuation(self) -> None:
        """Collect activation fluctuation (variance) from calibration data."""
        modules_dict = dict(self.model.named_modules())
        hooks = []
        activation_mean: Dict[int, torch.Tensor] = {}
        activation_var: Dict[int, torch.Tensor] = {}
        activation_counts: Dict[int, int] = {}

        def make_hook(layer_idx: int):
            def hook(module, input, output):
                x = input[0].detach().float()
                batch_mean = x.mean(dim=(0, 1))
                batch_var = x.var(dim=(0, 1))
                n = x.shape[0] * x.shape[1]

                if layer_idx not in activation_mean:
                    activation_mean[layer_idx] = torch.zeros_like(batch_mean)
                    activation_var[layer_idx] = torch.zeros_like(batch_var)
                    activation_counts[layer_idx] = 0

                # Welford's online algorithm for variance
                old_count = activation_counts[layer_idx]
                new_count = old_count + n
                delta = batch_mean - activation_mean[layer_idx]

                activation_mean[layer_idx] += delta * n / new_count
                activation_var[layer_idx] += batch_var * n + delta.pow(2) * old_count * n / new_count
                activation_counts[layer_idx] = new_count
            return hook

        for layer_info in self.ffn_layers:
            down_proj = modules_dict.get(layer_info.down_proj_name)
            if down_proj and isinstance(down_proj, nn.Linear):
                hook = down_proj.register_forward_hook(make_hook(layer_info.layer_idx))
                hooks.append(hook)

        self.model.eval()
        with torch.no_grad():
            for batch in self.calibration_loader:
                if isinstance(batch, dict):
                    input_ids = batch["input_ids"]
                else:
                    input_ids = batch[0] if isinstance(batch, (list, tuple)) else batch
                device = next(self.model.parameters()).device
                input_ids = input_ids.to(device)
                self.model(input_ids)

        for hook in hooks:
            hook.remove()

        for layer_idx in activation_var:
            # FLAP uses fluctuation = sqrt(variance)
            self._fluctuation_stats[layer_idx] = (
                activation_var[layer_idx] / activation_counts[layer_idx]
            ).sqrt()

    def compute_importance(self, layer_info: FFNLayerInfo) -> torch.Tensor:
        modules_dict = dict(self.model.named_modules())
        down_proj = modules_dict[layer_info.down_proj_name]

        # Weight magnitude (WIFV metric in FLAP)
        weight_importance = down_proj.weight.abs().sum(dim=0)

        # Fluctuation (input feature variance)
        fluctuation = self._fluctuation_stats.get(
            layer_info.layer_idx,
            torch.ones(layer_info.intermediate_size, device=weight_importance.device)
        )

        # FLAP: weight * fluctuation
        return weight_importance * fluctuation.to(weight_importance.device)


def get_baseline_pruner(
    method: str,
    model: nn.Module,
    calibration_loader: Optional[torch.utils.data.DataLoader] = None,
    **kwargs,
) -> FFNBaselinePruner:
    """
    Factory function to get a baseline pruner.

    Args:
        method: Pruning method name ("magnitude", "activation", "wanda", "flap")
        model: The model to prune
        calibration_loader: Calibration data loader (required for some methods)
        **kwargs: Additional arguments for the pruner

    Returns:
        FFNBaselinePruner instance
    """
    methods = {
        "magnitude": lambda: MagnitudePruner(model, **kwargs),
        "magnitude_l1": lambda: MagnitudePruner(model, norm="l1"),
        "magnitude_l2": lambda: MagnitudePruner(model, norm="l2"),
        "activation": lambda: ActivationPruner(model, calibration_loader),
        "wanda": lambda: WandaStylePruner(model, calibration_loader),
        "flap": lambda: FLAPStylePruner(model, calibration_loader),
    }

    if method not in methods:
        raise ValueError(f"Unknown method: {method}. Available: {list(methods.keys())}")

    if method in ["activation", "wanda", "flap"] and calibration_loader is None:
        raise ValueError(f"Method '{method}' requires calibration_loader")

    return methods[method]()
