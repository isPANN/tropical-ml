"""
Baseline Pruning Methods for Comparison.

This module implements standard pruning baselines to compare against
tropical winner-based pruning:
- L1/L2 Magnitude Pruning: Prune neurons with smallest weight norms
- Random Pruning: Random neuron removal (lower bound baseline)
- Activation Sparsity Pruning: Prune neurons that rarely activate
"""

import copy
from typing import Dict, List, Literal, Optional, Tuple
import torch
import torch.nn as nn


class MagnitudeStructuredPruner:
    """
    Structured pruning based on weight magnitude.

    Prunes entire output neurons (rows of weight matrix) based on
    the L1 or L2 norm of their weights.

    This is the most common baseline for structured pruning.
    """

    def __init__(
        self,
        model: nn.Module,
        norm: Literal["l1", "l2"] = "l1",
    ):
        """
        Args:
            model: The model to prune.
            norm: Norm type for computing importance ("l1" or "l2").
        """
        self.model = model
        self.norm = norm
        self._compression_stats: Optional[Dict] = None

    def _compute_importance(self, weight: torch.Tensor) -> torch.Tensor:
        """Compute importance of each output neuron (row)."""
        if self.norm == "l1":
            return weight.abs().sum(dim=1)  # Sum along input dimension
        elif self.norm == "l2":
            return weight.pow(2).sum(dim=1).sqrt()
        else:
            raise ValueError(f"Unknown norm: {self.norm}")

    def _get_pruning_mask(
        self,
        importance: torch.Tensor,
        sparsity: float,
    ) -> torch.Tensor:
        """Get mask for which neurons to keep."""
        num_neurons = importance.numel()
        num_to_prune = int(num_neurons * sparsity)

        if num_to_prune == 0:
            return torch.ones(num_neurons, dtype=torch.bool, device=importance.device)

        threshold = torch.kthvalue(importance, num_to_prune).values
        mask = importance > threshold
        return mask

    def prune(
        self,
        sparsity: float,
        inplace: bool = False,
    ) -> nn.Module:
        """
        Apply magnitude-based structured pruning.

        Args:
            sparsity: Target sparsity (fraction to prune).
            inplace: If True, modify model in place.

        Returns:
            Pruned model.
        """
        if inplace:
            pruned_model = self.model
        else:
            pruned_model = copy.deepcopy(self.model)

        # Get linear layers in order
        linear_layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                linear_layers.append(name)

        if len(linear_layers) < 2:
            return pruned_model

        # Prune hidden layers (not first input, not last output)
        for i in range(len(linear_layers) - 1):
            curr_name = linear_layers[i]
            next_name = linear_layers[i + 1]

            # Get current layer
            curr_module = dict(pruned_model.named_modules())[curr_name]

            # Compute importance of output neurons
            importance = self._compute_importance(curr_module.weight)
            mask = self._get_pruning_mask(importance, sparsity)

            # Prune output neurons (rows) of current layer
            self._prune_layer_output(pruned_model, curr_name, mask)

            # Prune input neurons (cols) of next layer
            self._prune_layer_input(pruned_model, next_name, mask)

        self._compute_compression_stats(pruned_model)
        return pruned_model

    def _prune_layer_output(
        self,
        model: nn.Module,
        layer_name: str,
        mask: torch.Tensor,
    ) -> None:
        """Prune output neurons (rows of weight matrix)."""
        module = dict(model.named_modules())[layer_name]
        keep_indices = mask.nonzero(as_tuple=True)[0]

        new_layer = nn.Linear(
            module.in_features,
            keep_indices.numel(),
            bias=module.bias is not None,
            device=module.weight.device,
            dtype=module.weight.dtype,
        )
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
        """Prune input neurons (columns of weight matrix)."""
        module = dict(model.named_modules())[layer_name]
        keep_indices = mask.nonzero(as_tuple=True)[0]

        new_layer = nn.Linear(
            keep_indices.numel(),
            module.out_features,
            bias=module.bias is not None,
            device=module.weight.device,
            dtype=module.weight.dtype,
        )
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
        """Get compression statistics."""
        if self._compression_stats is None:
            return {"error": "No pruning has been performed yet"}
        return self._compression_stats


class RandomStructuredPruner:
    """
    Random structured pruning baseline.

    Randomly removes neurons to establish a lower bound on pruning performance.
    Any reasonable pruning method should outperform random pruning.
    """

    def __init__(self, model: nn.Module, seed: Optional[int] = None):
        """
        Args:
            model: The model to prune.
            seed: Random seed for reproducibility.
        """
        self.model = model
        self.seed = seed
        self._compression_stats: Optional[Dict] = None

    def _get_random_mask(
        self,
        num_neurons: int,
        sparsity: float,
        device: torch.device,
    ) -> torch.Tensor:
        """Get random mask for which neurons to keep."""
        num_to_keep = int(num_neurons * (1 - sparsity))
        num_to_keep = max(1, num_to_keep)  # Keep at least 1 neuron

        # Random permutation and select top-k
        perm = torch.randperm(num_neurons, device=device)
        mask = torch.zeros(num_neurons, dtype=torch.bool, device=device)
        mask[perm[:num_to_keep]] = True
        return mask

    def prune(
        self,
        sparsity: float,
        inplace: bool = False,
    ) -> nn.Module:
        """
        Apply random structured pruning.

        Args:
            sparsity: Target sparsity (fraction to prune).
            inplace: If True, modify model in place.

        Returns:
            Pruned model.
        """
        if self.seed is not None:
            torch.manual_seed(self.seed)

        if inplace:
            pruned_model = self.model
        else:
            pruned_model = copy.deepcopy(self.model)

        # Get linear layers in order
        linear_layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                linear_layers.append(name)

        if len(linear_layers) < 2:
            return pruned_model

        # Prune hidden layers (not first input, not last output)
        for i in range(len(linear_layers) - 1):
            curr_name = linear_layers[i]
            next_name = linear_layers[i + 1]

            # Get current layer
            curr_module = dict(pruned_model.named_modules())[curr_name]

            # Random mask
            mask = self._get_random_mask(
                curr_module.out_features,
                sparsity,
                curr_module.weight.device,
            )

            # Prune output neurons (rows) of current layer
            self._prune_layer_output(pruned_model, curr_name, mask)

            # Prune input neurons (cols) of next layer
            self._prune_layer_input(pruned_model, next_name, mask)

        self._compute_compression_stats(pruned_model)
        return pruned_model

    def _prune_layer_output(
        self,
        model: nn.Module,
        layer_name: str,
        mask: torch.Tensor,
    ) -> None:
        """Prune output neurons (rows of weight matrix)."""
        module = dict(model.named_modules())[layer_name]
        keep_indices = mask.nonzero(as_tuple=True)[0]

        new_layer = nn.Linear(
            module.in_features,
            keep_indices.numel(),
            bias=module.bias is not None,
            device=module.weight.device,
            dtype=module.weight.dtype,
        )
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
        """Prune input neurons (columns of weight matrix)."""
        module = dict(model.named_modules())[layer_name]
        keep_indices = mask.nonzero(as_tuple=True)[0]

        new_layer = nn.Linear(
            keep_indices.numel(),
            module.out_features,
            bias=module.bias is not None,
            device=module.weight.device,
            dtype=module.weight.dtype,
        )
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
        """Get compression statistics."""
        if self._compression_stats is None:
            return {"error": "No pruning has been performed yet"}
        return self._compression_stats


class ActivationSparsityPruner:
    """
    Structured pruning based on activation sparsity.

    Prunes neurons that produce zero or near-zero activations most of the time.
    Requires collecting activations on calibration data first.
    """

    def __init__(
        self,
        model: nn.Module,
        threshold: float = 1e-6,
    ):
        """
        Args:
            model: The model to prune.
            threshold: Activation threshold (below = considered zero).
        """
        self.model = model
        self.threshold = threshold
        self._activation_stats: Dict[str, torch.Tensor] = {}
        self._hooks: List = []
        self._compression_stats: Optional[Dict] = None

    def collect_activations(
        self,
        dataloader,
        num_batches: Optional[int] = None,
    ) -> None:
        """
        Collect activation statistics from calibration data.

        Args:
            dataloader: DataLoader with calibration data.
            num_batches: Max batches to process.
        """
        self.model.eval()

        # Setup hooks to collect activations
        activation_counts: Dict[str, torch.Tensor] = {}
        total_counts: Dict[str, int] = {}

        def make_hook(name: str):
            def hook(module, input, output):
                # For Linear layers, we look at the INPUT activations (post-ReLU from prev layer)
                # input[0] shape: (batch, in_features) or (batch, seq, in_features)
                inp = input[0]
                flat_input = inp.reshape(-1, inp.shape[-1])
                non_zero = (flat_input.abs() > self.threshold).float().sum(dim=0)

                if name not in activation_counts:
                    activation_counts[name] = non_zero
                    total_counts[name] = flat_input.shape[0]
                else:
                    activation_counts[name] += non_zero
                    total_counts[name] += flat_input.shape[0]
            return hook

        # Register hooks on all linear layers except the first
        # (first layer's input is raw data, not activations)
        linear_layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                linear_layers.append(name)

        for name in linear_layers[1:]:  # Skip first layer
            module = dict(self.model.named_modules())[name]
            hook = module.register_forward_hook(make_hook(name))
            self._hooks.append(hook)

        # Run forward passes
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if num_batches is not None and i >= num_batches:
                    break
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                else:
                    x = batch
                self.model(x)

        # Remove hooks
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

        # Compute activation frequency (importance)
        for name in activation_counts:
            self._activation_stats[name] = activation_counts[name] / total_counts[name]

    def _get_pruning_mask(
        self,
        importance: torch.Tensor,
        sparsity: float,
    ) -> torch.Tensor:
        """Get mask for which neurons to keep."""
        num_neurons = importance.numel()
        num_to_prune = int(num_neurons * sparsity)

        if num_to_prune == 0:
            return torch.ones(num_neurons, dtype=torch.bool, device=importance.device)

        threshold = torch.kthvalue(importance, num_to_prune).values
        mask = importance > threshold
        return mask

    def prune(
        self,
        sparsity: float,
        inplace: bool = False,
    ) -> nn.Module:
        """
        Apply activation-based structured pruning.

        Must call collect_activations() first.

        Stats for layer[i]'s inputs = importance of layer[i-1]'s outputs.
        So we use stats from layer[i+1] to decide which outputs of layer[i] to prune.

        Args:
            sparsity: Target sparsity (fraction to prune).
            inplace: If True, modify model in place.

        Returns:
            Pruned model.
        """
        if not self._activation_stats:
            raise RuntimeError("Must call collect_activations() first")

        if inplace:
            pruned_model = self.model
        else:
            pruned_model = copy.deepcopy(self.model)

        # Get linear layers in order
        linear_layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                linear_layers.append(name)

        if len(linear_layers) < 2:
            return pruned_model

        # Prune hidden layers
        # Stats from layer[i+1] tell us importance of layer[i]'s outputs
        for i in range(len(linear_layers) - 1):
            curr_name = linear_layers[i]
            next_name = linear_layers[i + 1]

            # Use stats from next layer (its inputs = current layer's outputs)
            if next_name not in self._activation_stats:
                continue

            importance = self._activation_stats[next_name]
            mask = self._get_pruning_mask(importance, sparsity)

            # Prune output neurons (rows) of current layer
            self._prune_layer_output(pruned_model, curr_name, mask)

            # Prune input neurons (cols) of next layer
            self._prune_layer_input(pruned_model, next_name, mask)

        self._compute_compression_stats(pruned_model)
        return pruned_model

    def _prune_layer_output(
        self,
        model: nn.Module,
        layer_name: str,
        mask: torch.Tensor,
    ) -> None:
        """Prune output neurons (rows of weight matrix)."""
        module = dict(model.named_modules())[layer_name]
        keep_indices = mask.nonzero(as_tuple=True)[0]

        new_layer = nn.Linear(
            module.in_features,
            keep_indices.numel(),
            bias=module.bias is not None,
            device=module.weight.device,
            dtype=module.weight.dtype,
        )
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
        """Prune input neurons (columns of weight matrix)."""
        module = dict(model.named_modules())[layer_name]
        keep_indices = mask.nonzero(as_tuple=True)[0]

        new_layer = nn.Linear(
            keep_indices.numel(),
            module.out_features,
            bias=module.bias is not None,
            device=module.weight.device,
            dtype=module.weight.dtype,
        )
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
        """Get compression statistics."""
        if self._compression_stats is None:
            return {"error": "No pruning has been performed yet"}
        return self._compression_stats
