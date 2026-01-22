"""
Baseline Pruning Methods for Comparison.

This module implements standard pruning baselines to compare against
tropical winner-based pruning:
- L1/L2 Magnitude Pruning: Prune neurons with smallest weight norms
- Random Pruning: Random neuron removal (lower bound baseline)
- Activation Sparsity Pruning: Prune neurons that rarely activate
- Network Slimming: Prune based on BatchNorm gamma values (Liu et al. ICCV 2017)
- Taylor Pruning: Prune based on |activation * gradient| (Molchanov et al. CVPR 2019)
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


class NetworkSlimmingPruner:
    """
    Network Slimming pruning method (Liu et al. ICCV 2017).

    Prunes channels based on BatchNorm gamma (scaling factor) values.
    Channels with small gamma values are considered less important.

    Reference: "Learning Efficient Convolutional Networks through Network Slimming"
    https://arxiv.org/abs/1708.06519

    This method requires BatchNorm layers after Conv2d layers.
    """

    def __init__(self, model: nn.Module):
        """
        Args:
            model: The model to prune. Must have BatchNorm layers.
        """
        self.model = model
        self._compression_stats: Optional[Dict] = None

    def _collect_bn_gammas(self) -> Dict[str, Tuple[str, torch.Tensor]]:
        """
        Collect BatchNorm gamma values and map to corresponding conv layers.

        Returns:
            Dict mapping conv layer names to (bn_name, gamma_values).
        """
        conv_to_bn = {}
        modules_dict = dict(self.model.named_modules())

        # Find Conv2d -> BatchNorm pairs
        prev_conv_name = None
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                prev_conv_name = name
            elif isinstance(module, nn.BatchNorm2d) and prev_conv_name is not None:
                # Check if dimensions match
                conv_module = modules_dict[prev_conv_name]
                if conv_module.out_channels == module.num_features:
                    conv_to_bn[prev_conv_name] = (name, module.weight.data.abs())
                prev_conv_name = None

        return conv_to_bn

    def _get_pruning_mask(
        self,
        importance: torch.Tensor,
        sparsity: float,
    ) -> torch.Tensor:
        """Get mask for which channels to keep."""
        num_channels = importance.numel()
        num_to_prune = int(num_channels * sparsity)

        if num_to_prune == 0:
            return torch.ones(num_channels, dtype=torch.bool, device=importance.device)

        threshold = torch.kthvalue(importance, num_to_prune).values
        mask = importance > threshold
        return mask

    def prune(
        self,
        sparsity: float,
        inplace: bool = False,
    ) -> nn.Module:
        """
        Apply Network Slimming pruning.

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

        # Collect Conv -> BN mappings with gamma values
        conv_to_bn = self._collect_bn_gammas()

        if not conv_to_bn:
            print("Warning: No Conv2d -> BatchNorm pairs found. Cannot apply Network Slimming.")
            return pruned_model

        # Get conv layers in order
        conv_layers = [name for name, m in self.model.named_modules() if isinstance(m, nn.Conv2d)]

        for i, conv_name in enumerate(conv_layers[:-1]):  # Don't prune last conv
            if conv_name not in conv_to_bn:
                continue

            bn_name, gamma = conv_to_bn[conv_name]

            # Compute mask based on gamma values
            mask = self._get_pruning_mask(gamma, sparsity)

            # Prune conv output channels
            self._prune_conv_output(pruned_model, conv_name, mask)

            # Prune BatchNorm
            self._prune_batchnorm(pruned_model, bn_name, mask)

            # Prune next conv input channels (if exists)
            if i + 1 < len(conv_layers):
                next_conv = conv_layers[i + 1]
                self._prune_conv_input(pruned_model, next_conv, mask)

        self._compute_compression_stats(pruned_model)
        return pruned_model

    def _prune_conv_output(self, model: nn.Module, layer_name: str, mask: torch.Tensor) -> None:
        """Prune output channels from Conv2d."""
        module = dict(model.named_modules())[layer_name]
        if not isinstance(module, nn.Conv2d):
            return

        keep_indices = mask.nonzero(as_tuple=True)[0]
        new_out_channels = keep_indices.numel()

        new_layer = nn.Conv2d(
            module.in_channels, new_out_channels, module.kernel_size,
            module.stride, module.padding, module.dilation, module.groups,
            module.bias is not None, module.padding_mode,
            device=module.weight.device, dtype=module.weight.dtype,
        )
        new_layer.weight.data = module.weight.data[keep_indices, :, :, :]
        if module.bias is not None:
            new_layer.bias.data = module.bias.data[keep_indices]

        self._replace_layer(model, layer_name, new_layer)

    def _prune_conv_input(self, model: nn.Module, layer_name: str, mask: torch.Tensor) -> None:
        """Prune input channels from Conv2d."""
        module = dict(model.named_modules())[layer_name]
        if not isinstance(module, nn.Conv2d) or module.groups > 1:
            return

        keep_indices = mask.nonzero(as_tuple=True)[0]
        new_in_channels = keep_indices.numel()

        new_layer = nn.Conv2d(
            new_in_channels, module.out_channels, module.kernel_size,
            module.stride, module.padding, module.dilation, module.groups,
            module.bias is not None, module.padding_mode,
            device=module.weight.device, dtype=module.weight.dtype,
        )
        new_layer.weight.data = module.weight.data[:, keep_indices, :, :]
        if module.bias is not None:
            new_layer.bias.data = module.bias.data.clone()

        self._replace_layer(model, layer_name, new_layer)

    def _prune_batchnorm(self, model: nn.Module, layer_name: str, mask: torch.Tensor) -> None:
        """Prune BatchNorm layer."""
        module = dict(model.named_modules())[layer_name]
        if not isinstance(module, nn.BatchNorm2d):
            return

        keep_indices = mask.nonzero(as_tuple=True)[0]
        new_features = keep_indices.numel()

        new_bn = nn.BatchNorm2d(
            new_features, module.eps, module.momentum,
            module.affine, module.track_running_stats,
            device=module.weight.device if module.weight is not None else None,
        )

        if module.affine:
            new_bn.weight.data = module.weight.data[keep_indices]
            new_bn.bias.data = module.bias.data[keep_indices]
        if module.track_running_stats:
            new_bn.running_mean.data = module.running_mean.data[keep_indices]
            new_bn.running_var.data = module.running_var.data[keep_indices]

        self._replace_layer(model, layer_name, new_bn)

    def _replace_layer(self, model: nn.Module, layer_name: str, new_layer: nn.Module) -> None:
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


class TaylorPruner:
    """
    Taylor expansion based pruning (Molchanov et al. CVPR 2019).

    Prunes channels based on the product of activation and gradient magnitude.
    importance = |activation * gradient|

    This approximates the change in loss when removing a channel.

    Reference: "Importance Estimation for Neural Network Pruning"
    https://arxiv.org/abs/1906.10771

    Requires running forward and backward passes to collect statistics.
    """

    def __init__(self, model: nn.Module):
        """
        Args:
            model: The model to prune.
        """
        self.model = model
        self._importance: Dict[str, torch.Tensor] = {}
        self._hooks: List = []
        self._compression_stats: Optional[Dict] = None

    def collect_importance(
        self,
        dataloader,
        criterion: nn.Module = None,
        num_batches: Optional[int] = None,
    ) -> None:
        """
        Collect importance scores using Taylor approximation.

        importance(channel) = |mean(activation * gradient)|

        Args:
            dataloader: DataLoader with calibration data.
            criterion: Loss function. Defaults to CrossEntropyLoss.
            num_batches: Maximum batches to process.
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        self.model.train()  # Need gradients

        # Storage for accumulating importance
        activation_grad_product: Dict[str, torch.Tensor] = {}
        sample_count: Dict[str, int] = {}

        # Setup hooks to capture activations and gradients
        def make_hook(name: str):
            def hook(module, input, output):
                # Store activation for backward pass
                activation = output.detach()

                def grad_hook(grad):
                    # Compute |activation * gradient| summed over spatial dimensions
                    importance = (activation * grad).abs()
                    # Sum over batch and spatial, keep channel dimension
                    if len(importance.shape) == 4:  # Conv: (B, C, H, W)
                        importance = importance.sum(dim=(0, 2, 3))
                    elif len(importance.shape) == 2:  # Linear: (B, C)
                        importance = importance.sum(dim=0)
                    else:
                        importance = importance.sum(dim=0)

                    if name not in activation_grad_product:
                        activation_grad_product[name] = importance
                        sample_count[name] = 1
                    else:
                        activation_grad_product[name] += importance
                        sample_count[name] += 1

                output.register_hook(grad_hook)

            return hook

        # Register hooks on conv and linear layers
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                hook = module.register_forward_hook(make_hook(name))
                self._hooks.append(hook)

        # Forward and backward passes
        device = next(self.model.parameters()).device

        for i, batch in enumerate(dataloader):
            if num_batches is not None and i >= num_batches:
                break

            if isinstance(batch, (list, tuple)):
                inputs, targets = batch[0], batch[1]
            else:
                continue

            inputs = inputs.to(device)
            targets = targets.to(device)

            self.model.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

        # Remove hooks
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

        # Compute average importance
        for name in activation_grad_product:
            self._importance[name] = activation_grad_product[name] / sample_count[name]

        self.model.eval()

    def _get_pruning_mask(self, importance: torch.Tensor, sparsity: float) -> torch.Tensor:
        """Get mask for which channels to keep."""
        num_channels = importance.numel()
        num_to_prune = int(num_channels * sparsity)

        if num_to_prune == 0:
            return torch.ones(num_channels, dtype=torch.bool, device=importance.device)

        threshold = torch.kthvalue(importance, num_to_prune).values
        mask = importance > threshold
        return mask

    def prune(self, sparsity: float, inplace: bool = False) -> nn.Module:
        """
        Apply Taylor-based pruning.

        Must call collect_importance() first.

        Args:
            sparsity: Target sparsity (fraction to prune).
            inplace: If True, modify model in place.

        Returns:
            Pruned model.
        """
        if not self._importance:
            raise RuntimeError("Must call collect_importance() first")

        if inplace:
            pruned_model = self.model
        else:
            pruned_model = copy.deepcopy(self.model)

        # Get conv layers in order
        conv_layers = [name for name, m in self.model.named_modules() if isinstance(m, nn.Conv2d)]

        for i, conv_name in enumerate(conv_layers[:-1]):  # Don't prune last conv
            if conv_name not in self._importance:
                continue

            importance = self._importance[conv_name]
            mask = self._get_pruning_mask(importance, sparsity)

            # Prune conv output channels
            self._prune_conv_output(pruned_model, conv_name, mask)

            # Find and prune associated BatchNorm
            bn_name = self._find_bn_for_conv(conv_name)
            if bn_name:
                self._prune_batchnorm(pruned_model, bn_name, mask)

            # Prune next conv input channels
            if i + 1 < len(conv_layers):
                next_conv = conv_layers[i + 1]
                self._prune_conv_input(pruned_model, next_conv, mask)

        self._compute_compression_stats(pruned_model)
        return pruned_model

    def _find_bn_for_conv(self, conv_name: str) -> Optional[str]:
        """Find BatchNorm layer associated with a conv layer."""
        modules_dict = dict(self.model.named_modules())

        # Try common naming patterns
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

        if 'conv' in conv_name:
            bn_name = conv_name.replace('conv', 'bn')
            if bn_name in modules_dict and isinstance(modules_dict[bn_name], nn.BatchNorm2d):
                return bn_name

        return None

    def _prune_conv_output(self, model, layer_name, mask):
        """Prune output channels from Conv2d."""
        module = dict(model.named_modules())[layer_name]
        if not isinstance(module, nn.Conv2d):
            return

        keep_indices = mask.nonzero(as_tuple=True)[0]
        new_out_channels = keep_indices.numel()

        new_layer = nn.Conv2d(
            module.in_channels, new_out_channels, module.kernel_size,
            module.stride, module.padding, module.dilation, module.groups,
            module.bias is not None, device=module.weight.device,
        )
        new_layer.weight.data = module.weight.data[keep_indices, :, :, :]
        if module.bias is not None:
            new_layer.bias.data = module.bias.data[keep_indices]

        self._replace_layer(model, layer_name, new_layer)

    def _prune_conv_input(self, model, layer_name, mask):
        """Prune input channels from Conv2d."""
        module = dict(model.named_modules())[layer_name]
        if not isinstance(module, nn.Conv2d) or module.groups > 1:
            return

        keep_indices = mask.nonzero(as_tuple=True)[0]
        new_in_channels = keep_indices.numel()

        new_layer = nn.Conv2d(
            new_in_channels, module.out_channels, module.kernel_size,
            module.stride, module.padding, module.dilation, module.groups,
            module.bias is not None, device=module.weight.device,
        )
        new_layer.weight.data = module.weight.data[:, keep_indices, :, :]
        if module.bias is not None:
            new_layer.bias.data = module.bias.data.clone()

        self._replace_layer(model, layer_name, new_layer)

    def _prune_batchnorm(self, model, layer_name, mask):
        """Prune BatchNorm layer."""
        module = dict(model.named_modules())[layer_name]
        if not isinstance(module, nn.BatchNorm2d):
            return

        keep_indices = mask.nonzero(as_tuple=True)[0]
        new_features = keep_indices.numel()

        new_bn = nn.BatchNorm2d(
            new_features, module.eps, module.momentum,
            module.affine, module.track_running_stats,
            device=module.weight.device if module.weight is not None else None,
        )

        if module.affine:
            new_bn.weight.data = module.weight.data[keep_indices]
            new_bn.bias.data = module.bias.data[keep_indices]
        if module.track_running_stats:
            new_bn.running_mean.data = module.running_mean.data[keep_indices]
            new_bn.running_var.data = module.running_var.data[keep_indices]

        self._replace_layer(model, layer_name, new_bn)

    def _replace_layer(self, model, layer_name, new_layer):
        """Replace a layer in the model by name."""
        parts = layer_name.split('.')
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_layer)

    def _compute_compression_stats(self, pruned_model):
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
