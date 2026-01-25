"""
Model Conversion Utilities for Tropical Networks.

Provides functions to convert between standard neural networks and
tropical (MMP) networks.

Conversions supported:
- ReLU networks → MaxPlus networks (exact for certain architectures)
- Standard NNs → MMP-NNs (approximate)
- MMP-NNs → Standard NNs (for inference)
"""

import copy
from typing import Callable, Dict, Optional, Type

import torch
import torch.nn as nn

from .layers import MaxPlusLayer, MinPlusLayer, TropicalReLU
from .blocks import MMPBlock
from .approximations import PiecewiseSiLU, PiecewiseGELU


def convert_relu_to_maxplus(
    model: nn.Module,
    inplace: bool = False,
) -> nn.Module:
    """
    Replace ReLU activations with MaxPlus layers.

    ReLU(x) = max(x, 0) can be viewed as a special case of MaxPlus
    with fixed weights. This conversion replaces ReLU with trainable
    MaxPlus layers initialized to behave like ReLU.

    Note: This is primarily for experimental purposes. The resulting
    network will have different training dynamics.

    Args:
        model: The model to convert.
        inplace: If True, modifies the model in place. Default: False.

    Returns:
        Converted model with MaxPlus layers instead of ReLU.

    Example:
        >>> model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 10))
        >>> mmp_model = convert_relu_to_maxplus(model)
    """
    if not inplace:
        model = copy.deepcopy(model)

    def replace_relu(module: nn.Module, name: str = "") -> None:
        for child_name, child in module.named_children():
            if isinstance(child, (nn.ReLU, nn.ReLU6)):
                # ReLU is element-wise, so we use a simple wrapper
                setattr(module, child_name, TropicalReLU())
            else:
                replace_relu(child, f"{name}.{child_name}" if name else child_name)

    replace_relu(model)
    return model


def convert_activation_to_tropical(
    model: nn.Module,
    activation_types: Optional[Dict[Type, Callable]] = None,
    inplace: bool = False,
) -> nn.Module:
    """
    Replace various activations with their tropical approximations.

    Args:
        model: The model to convert.
        activation_types: Dict mapping activation types to their replacements.
            If None, uses default mappings.
        inplace: If True, modifies the model in place. Default: False.

    Returns:
        Converted model with tropical activations.

    Example:
        >>> model = MyModel()  # Has SiLU activations
        >>> tropical_model = convert_activation_to_tropical(model)
    """
    if not inplace:
        model = copy.deepcopy(model)

    if activation_types is None:
        activation_types = {
            nn.ReLU: lambda: TropicalReLU(),
            nn.ReLU6: lambda: TropicalReLU(),
            nn.SiLU: lambda: PiecewiseSiLU(num_pieces=8),
            nn.GELU: lambda: PiecewiseGELU(num_pieces=8),
        }

    def replace_activations(module: nn.Module) -> None:
        for child_name, child in module.named_children():
            for act_type, replacement_fn in activation_types.items():
                if isinstance(child, act_type):
                    setattr(module, child_name, replacement_fn())
                    break
            else:
                replace_activations(child)

    replace_activations(model)
    return model


def convert_mlp_to_mmp(
    model: nn.Module,
    inplace: bool = False,
) -> nn.Module:
    """
    Convert Linear → Activation → Linear patterns to MMP blocks.

    Identifies sequential patterns of Linear → ReLU → Linear and
    replaces them with equivalent MMPBlock structures.

    Args:
        model: The model to convert.
        inplace: If True, modifies the model in place. Default: False.

    Returns:
        Converted model with MMP blocks.

    Example:
        >>> model = nn.Sequential(
        ...     nn.Linear(784, 256),
        ...     nn.ReLU(),
        ...     nn.Linear(256, 10)
        ... )
        >>> mmp_model = convert_mlp_to_mmp(model)
    """
    if not inplace:
        model = copy.deepcopy(model)

    # Handle Sequential models specially
    if isinstance(model, nn.Sequential):
        return _convert_sequential_to_mmp(model)

    # For other models, recursively process children
    for name, child in list(model.named_children()):
        if isinstance(child, nn.Sequential):
            setattr(model, name, _convert_sequential_to_mmp(child))
        else:
            convert_mlp_to_mmp(child, inplace=True)

    return model


def _convert_sequential_to_mmp(seq: nn.Sequential) -> nn.Sequential:
    """Convert a Sequential model's Linear → ReLU → Linear to MMP."""
    children = list(seq.children())
    new_children = []
    i = 0

    while i < len(children):
        # Check for Linear → ReLU → Linear pattern
        if (
            i + 2 < len(children)
            and isinstance(children[i], nn.Linear)
            and isinstance(children[i + 1], (nn.ReLU, nn.ReLU6, nn.LeakyReLU))
            and isinstance(children[i + 2], nn.Linear)
        ):
            linear1 = children[i]
            linear2 = children[i + 2]

            # Create MMPBlock
            block = MMPBlock(
                in_features=linear1.in_features,
                hidden_features=linear1.out_features,
                out_features=linear2.out_features,
            )

            # Copy weights from linear1
            with torch.no_grad():
                block.linear.weight.copy_(linear1.weight)
                if linear1.bias is not None:
                    block.linear.bias.copy_(linear1.bias)

            new_children.append(block)
            i += 3
        else:
            new_children.append(children[i])
            i += 1

    return nn.Sequential(*new_children)


def convert_to_mmp(
    model: nn.Module,
    convert_activations: bool = True,
    convert_mlp_blocks: bool = True,
    inplace: bool = False,
) -> nn.Module:
    """
    Full conversion of a standard neural network to MMP architecture.

    This is the main conversion function that combines all conversion
    strategies.

    Args:
        model: The model to convert.
        convert_activations: Whether to convert activations. Default: True.
        convert_mlp_blocks: Whether to convert MLP patterns. Default: True.
        inplace: If True, modifies in place. Default: False.

    Returns:
        Converted MMP model.

    Example:
        >>> model = torchvision.models.resnet18(pretrained=True)
        >>> mmp_model = convert_to_mmp(model)
    """
    if not inplace:
        model = copy.deepcopy(model)

    if convert_activations:
        model = convert_activation_to_tropical(model, inplace=True)

    if convert_mlp_blocks:
        model = convert_mlp_to_mmp(model, inplace=True)

    return model


def convert_mmp_to_standard(
    model: nn.Module,
    inplace: bool = False,
) -> nn.Module:
    """
    Convert MMP network back to standard network for inference.

    Replaces MaxPlus/MinPlus layers with their standard equivalents.
    This can be useful for deployment on hardware that doesn't support
    tropical operations efficiently.

    Note: This conversion is approximate and may lose some expressiveness.

    Args:
        model: The MMP model to convert.
        inplace: If True, modifies in place. Default: False.

    Returns:
        Standard neural network.
    """
    if not inplace:
        model = copy.deepcopy(model)

    def replace_tropical(module: nn.Module) -> None:
        for child_name, child in list(module.named_children()):
            if isinstance(child, MaxPlusLayer):
                # Replace with Linear + ReLU approximation
                linear = nn.Linear(child.in_features, child.out_features)
                with torch.no_grad():
                    # Use weight as a guide for linear weights
                    # This is an approximation
                    linear.weight.data = child.weight.t().clone()
                    if child.bias is not None:
                        linear.bias.data = child.bias.clone()
                setattr(module, child_name, nn.Sequential(linear, nn.ReLU()))

            elif isinstance(child, MinPlusLayer):
                # MinPlus is harder to approximate with standard layers
                # Use Linear + negative ReLU approximation
                linear = nn.Linear(child.in_features, child.out_features)
                with torch.no_grad():
                    linear.weight.data = child.weight.t().clone()
                    if child.bias is not None:
                        linear.bias.data = child.bias.clone()
                setattr(module, child_name, linear)

            elif isinstance(child, TropicalReLU):
                setattr(module, child_name, nn.ReLU())

            elif isinstance(child, (PiecewiseSiLU, )):
                setattr(module, child_name, nn.SiLU())

            elif isinstance(child, (PiecewiseGELU, )):
                setattr(module, child_name, nn.GELU())

            else:
                replace_tropical(child)

    replace_tropical(model)
    return model


def estimate_multiplication_reduction(
    model: nn.Module,
    input_size: tuple,
) -> Dict[str, int]:
    """
    Estimate the reduction in multiplications from using MMP layers.

    Args:
        model: The model to analyze.
        input_size: Input tensor size (without batch dimension).

    Returns:
        Dictionary with multiplication counts.
    """
    from .training import count_operations

    ops = count_operations(model, input_size)

    return {
        "multiplications": ops["multiplications"],
        "additions": ops["additions"],
        "comparisons": ops["comparisons"],
        "total_ops": ops["total_ops"],
        "mult_reduction_ratio": 1.0 - (ops["multiplications"] / max(ops["total_ops"], 1)),
    }


def create_hybrid_model(
    model: nn.Module,
    tropical_layer_names: list,
    inplace: bool = False,
) -> nn.Module:
    """
    Create a hybrid model with both standard and tropical layers.

    Allows selective conversion of specific layers to tropical.

    Args:
        model: The model to convert.
        tropical_layer_names: List of layer names to convert to tropical.
        inplace: If True, modifies in place. Default: False.

    Returns:
        Hybrid model.

    Example:
        >>> model = nn.Sequential(...)
        >>> hybrid = create_hybrid_model(model, ['layer1', 'layer3'])
    """
    if not inplace:
        model = copy.deepcopy(model)

    for name, module in model.named_modules():
        if name in tropical_layer_names:
            if isinstance(module, nn.Linear):
                # Get parent and attribute name
                parts = name.rsplit('.', 1)
                if len(parts) == 1:
                    parent = model
                    attr = parts[0]
                else:
                    parent = dict(model.named_modules())[parts[0]]
                    attr = parts[1]

                # Create MaxPlus replacement
                maxplus = MaxPlusLayer(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                )

                # Initialize from linear weights (transpose for tropical format)
                with torch.no_grad():
                    maxplus.weight.data = module.weight.t().clone()
                    if module.bias is not None:
                        maxplus.bias.data = module.bias.clone()

                setattr(parent, attr, maxplus)

    return model


__all__ = [
    "convert_relu_to_maxplus",
    "convert_activation_to_tropical",
    "convert_mlp_to_mmp",
    "convert_to_mmp",
    "convert_mmp_to_standard",
    "estimate_multiplication_reduction",
    "create_hybrid_model",
]
