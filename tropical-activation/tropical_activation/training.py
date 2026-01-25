"""
Training utilities for Min-Max-Plus Neural Networks.

Provides specialized training functions, normalization layers,
and optimization strategies for tropical networks.

Reference: Luo & Fan 2021 - "Min-Max-Plus Neural Networks"
"""

from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .layers import MaxPlusLayer, MinPlusLayer


class TropicalBatchNorm(nn.Module):
    """
    Batch Normalization adapted for tropical networks.

    Standard BatchNorm may not be optimal for tropical layers since
    the operations are max/min-based rather than sum-based.

    This implementation provides options for different normalization
    strategies suitable for tropical networks.

    Args:
        num_features: Number of features to normalize.
        eps: Small constant for numerical stability. Default: 1e-5.
        momentum: Momentum for running statistics. Default: 0.1.
        affine: Whether to learn scale and shift. Default: True.
        mode: Normalization mode - "standard", "max", or "range". Default: "standard".

    Shape:
        - Input: (N, num_features) or (N, num_features, *)
        - Output: Same as input
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        mode: str = "standard",
    ):
        super().__init__()

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.mode = mode

        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        # Running statistics
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.register_buffer("running_max", torch.zeros(num_features))
        self.register_buffer("running_min", torch.zeros(num_features))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # Compute batch statistics
            if x.dim() == 2:
                mean = x.mean(dim=0)
                var = x.var(dim=0, unbiased=False)
                max_val = x.max(dim=0).values
                min_val = x.min(dim=0).values
            else:
                # Handle higher dimensions (flatten all but feature dim)
                x_flat = x.transpose(1, -1).reshape(-1, self.num_features)
                mean = x_flat.mean(dim=0)
                var = x_flat.var(dim=0, unbiased=False)
                max_val = x_flat.max(dim=0).values
                min_val = x_flat.min(dim=0).values

            # Update running statistics
            with torch.no_grad():
                self.running_mean = (
                    1 - self.momentum
                ) * self.running_mean + self.momentum * mean
                self.running_var = (
                    1 - self.momentum
                ) * self.running_var + self.momentum * var
                self.running_max = (
                    1 - self.momentum
                ) * self.running_max + self.momentum * max_val
                self.running_min = (
                    1 - self.momentum
                ) * self.running_min + self.momentum * min_val
                self.num_batches_tracked += 1
        else:
            mean = self.running_mean
            var = self.running_var
            max_val = self.running_max
            min_val = self.running_min

        # Normalize based on mode
        if self.mode == "standard":
            x = (x - mean) / (var + self.eps).sqrt()
        elif self.mode == "max":
            # Normalize by max absolute value (useful for tropical)
            scale = torch.maximum(max_val.abs(), min_val.abs()) + self.eps
            x = x / scale
        elif self.mode == "range":
            # Normalize to [0, 1] range
            range_val = max_val - min_val + self.eps
            x = (x - min_val) / range_val
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # Apply affine transformation
        if self.affine:
            x = x * self.weight + self.bias

        return x


class TropicalLayerNorm(nn.Module):
    """
    Layer Normalization for tropical networks.

    Args:
        normalized_shape: Input shape from expected input.
        eps: Small constant for numerical stability. Default: 1e-5.
        elementwise_affine: Whether to learn scale and shift. Default: True.

    Shape:
        - Input: (*, normalized_shape)
        - Output: Same as input
    """

    def __init__(
        self,
        normalized_shape: Union[int, List[int]],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ):
        super().__init__()

        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)

        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize over the last len(normalized_shape) dimensions
        dims = tuple(range(-len(self.normalized_shape), 0))

        mean = x.mean(dim=dims, keepdim=True)
        var = x.var(dim=dims, keepdim=True, unbiased=False)

        x = (x - mean) / (var + self.eps).sqrt()

        if self.elementwise_affine:
            x = x * self.weight + self.bias

        return x


def tropical_weight_init(
    module: nn.Module,
    init_scale: float = 0.1,
    linear_init: str = "xavier",
) -> None:
    """
    Initialize weights for tropical networks.

    Tropical layers need different initialization than standard layers
    because the operations are max/min-based.

    Args:
        module: Module to initialize.
        init_scale: Scale for tropical layer initialization. Default: 0.1.
        linear_init: Initialization for linear layers - "xavier" or "kaiming".
    """
    for name, m in module.named_modules():
        if isinstance(m, (MaxPlusLayer, MinPlusLayer)):
            # Small uniform initialization for tropical layers
            nn.init.uniform_(m.weight, -init_scale, init_scale)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        elif isinstance(m, nn.Linear):
            # Standard initialization for linear layers
            if linear_init == "xavier":
                nn.init.xavier_uniform_(m.weight)
            elif linear_init == "kaiming":
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            else:
                nn.init.normal_(m.weight, std=0.02)

            if m.bias is not None:
                nn.init.zeros_(m.bias)


def get_tropical_optimizer(
    model: nn.Module,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    tropical_lr_scale: float = 1.0,
    optimizer_type: str = "adamw",
) -> optim.Optimizer:
    """
    Create an optimizer with separate learning rates for tropical and linear layers.

    Tropical layers may benefit from different learning rates than linear layers.

    Args:
        model: The model to optimize.
        lr: Base learning rate. Default: 1e-3.
        weight_decay: Weight decay (L2 regularization). Default: 0.0.
        tropical_lr_scale: Multiplier for tropical layer learning rate. Default: 1.0.
        optimizer_type: Type of optimizer - "adamw", "adam", "sgd". Default: "adamw".

    Returns:
        Configured optimizer.
    """
    # Separate parameters by layer type
    tropical_params = []
    linear_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Check if parameter belongs to tropical layer
        is_tropical = any(
            isinstance(m, (MaxPlusLayer, MinPlusLayer))
            for n, m in model.named_modules()
            if n in name
        )

        if is_tropical:
            tropical_params.append(param)
        elif "linear" in name.lower() or "weight" in name.lower():
            linear_params.append(param)
        else:
            other_params.append(param)

    # Create parameter groups
    param_groups = []

    if tropical_params:
        param_groups.append({
            "params": tropical_params,
            "lr": lr * tropical_lr_scale,
            "weight_decay": weight_decay,
        })

    if linear_params:
        param_groups.append({
            "params": linear_params,
            "lr": lr,
            "weight_decay": weight_decay,
        })

    if other_params:
        param_groups.append({
            "params": other_params,
            "lr": lr,
            "weight_decay": weight_decay,
        })

    # Create optimizer
    if optimizer_type == "adamw":
        return optim.AdamW(param_groups)
    elif optimizer_type == "adam":
        return optim.Adam(param_groups)
    elif optimizer_type == "sgd":
        return optim.SGD(param_groups, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scheduler: Optional[object] = None,
) -> Dict[str, float]:
    """
    Train for one epoch.

    Args:
        model: The model to train.
        train_loader: Training data loader.
        optimizer: The optimizer.
        criterion: Loss function.
        device: Device to train on.
        scheduler: Optional learning rate scheduler.

    Returns:
        Dictionary with training metrics.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # Flatten if needed (for image data)
        if data.dim() > 2:
            data = data.view(data.size(0), -1)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

    if scheduler is not None:
        scheduler.step()

    return {
        "loss": total_loss / len(train_loader),
        "accuracy": 100.0 * correct / total,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate the model.

    Args:
        model: The model to evaluate.
        test_loader: Test data loader.
        criterion: Loss function.
        device: Device to evaluate on.

    Returns:
        Dictionary with evaluation metrics.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)

        # Flatten if needed
        if data.dim() > 2:
            data = data.view(data.size(0), -1)

        output = model(data)
        total_loss += criterion(output, target).item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

    return {
        "loss": total_loss / len(test_loader),
        "accuracy": 100.0 * correct / total,
    }


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count parameters by layer type.

    Args:
        model: The model to analyze.

    Returns:
        Dictionary with parameter counts.
    """
    tropical_params = 0
    linear_params = 0
    other_params = 0

    for name, module in model.named_modules():
        if isinstance(module, (MaxPlusLayer, MinPlusLayer)):
            tropical_params += sum(p.numel() for p in module.parameters())
        elif isinstance(module, nn.Linear):
            linear_params += sum(p.numel() for p in module.parameters())

    total = sum(p.numel() for p in model.parameters())
    other_params = total - tropical_params - linear_params

    return {
        "tropical": tropical_params,
        "linear": linear_params,
        "other": other_params,
        "total": total,
    }


def count_operations(
    model: nn.Module,
    input_size: Tuple[int, ...],
) -> Dict[str, int]:
    """
    Estimate operation counts for the model.

    Counts multiplications and additions separately for analysis.

    Args:
        model: The model to analyze.
        input_size: Input tensor size (without batch dimension).

    Returns:
        Dictionary with operation counts.
    """
    multiplications = 0
    additions = 0
    comparisons = 0

    def hook(module, input, output):
        nonlocal multiplications, additions, comparisons

        if isinstance(module, nn.Linear):
            # Linear: out = W @ x + b
            # Multiplications: out_features * in_features * batch
            # Additions: out_features * (in_features - 1) * batch + bias
            batch = input[0].shape[0]
            in_f = module.in_features
            out_f = module.out_features
            multiplications += batch * out_f * in_f
            additions += batch * out_f * (in_f - 1)
            if module.bias is not None:
                additions += batch * out_f

        elif isinstance(module, (MaxPlusLayer, MinPlusLayer)):
            # Tropical: out_j = max/min_k(x_k + w_kj) + b_j
            # Additions: out_features * in_features * batch
            # Comparisons: out_features * (in_features - 1) * batch
            batch = input[0].shape[0]
            in_f = module.in_features
            out_f = module.out_features
            additions += batch * out_f * in_f
            comparisons += batch * out_f * (in_f - 1)
            if module.bias is not None:
                additions += batch * out_f

    hooks = []
    for module in model.modules():
        if isinstance(module, (nn.Linear, MaxPlusLayer, MinPlusLayer)):
            hooks.append(module.register_forward_hook(hook))

    # Run a forward pass
    x = torch.zeros(1, *input_size)
    model.eval()
    with torch.no_grad():
        model(x)

    # Remove hooks
    for h in hooks:
        h.remove()

    return {
        "multiplications": multiplications,
        "additions": additions,
        "comparisons": comparisons,
        "total_ops": multiplications + additions + comparisons,
    }


__all__ = [
    "TropicalBatchNorm",
    "TropicalLayerNorm",
    "tropical_weight_init",
    "get_tropical_optimizer",
    "train_epoch",
    "evaluate",
    "count_parameters",
    "count_operations",
]
