"""
Tropical Layers: PyTorch layers that use tropical operations.

This module provides drop-in replacements for standard PyTorch layers
that use tropical (max-plus) algebra instead of standard linear algebra.
"""

from typing import Optional
import torch
import torch.nn as nn


class TropicalLinear(nn.Module):
    """
    Linear layer using tropical (max-plus) operations.

    Instead of: Y = XW^T + b
    Computes:   Y_ij = max_k(X_ik + W_jk) + b_j

    This is a drop-in replacement for nn.Linear that uses tropical GEMM.
    Useful for training and fine-tuning in the tropical domain.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        bias: If True, adds a learnable bias. Default: True.
        device: Device for the layer.
        dtype: Data type for the layer.

    Example:
        >>> layer = TropicalLinear(128, 64)
        >>> x = torch.randn(32, 128)
        >>> y = layer(x)  # shape: (32, 64)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        factory_kwargs = {'device': device, 'dtype': dtype}

        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, **factory_kwargs)
        )

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameters."""
        # Use log-scale initialization for tropical weights
        # Since tropical multiplication is addition, we want weights
        # that are meaningful as additive terms
        nn.init.uniform_(self.weight, -1.0, 1.0)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using tropical max-plus operation.

        Args:
            x: Input tensor of shape (*, in_features).

        Returns:
            Output tensor of shape (*, out_features).
        """
        return TropicalLinearFunction.apply(x, self.weight, self.bias)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'

    @classmethod
    def from_linear(cls, linear: nn.Linear, log_transform: bool = True) -> "TropicalLinear":
        """
        Create a TropicalLinear from a standard nn.Linear.

        Args:
            linear: Source Linear layer.
            log_transform: If True, apply log transform to weights.
                          This converts multiplicative weights to additive.

        Returns:
            TropicalLinear with converted weights.
        """
        tropical = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )

        if log_transform:
            # Convert multiplicative weights to additive (log domain)
            # W_tropical = log(|W_standard| + eps) * sign(W_standard)
            eps = 1e-8
            tropical.weight.data = torch.log(linear.weight.data.abs() + eps)
        else:
            tropical.weight.data = linear.weight.data.clone()

        if linear.bias is not None:
            if log_transform:
                tropical.bias.data = torch.log(linear.bias.data.abs() + eps)
            else:
                tropical.bias.data = linear.bias.data.clone()

        return tropical

    def to_linear(self, exp_transform: bool = True) -> nn.Linear:
        """
        Convert back to a standard nn.Linear.

        Args:
            exp_transform: If True, apply exp transform to weights.
                          This converts additive weights back to multiplicative.

        Returns:
            Standard Linear layer.
        """
        linear = nn.Linear(
            self.in_features,
            self.out_features,
            bias=self.bias is not None,
            device=self.weight.device,
            dtype=self.weight.dtype,
        )

        if exp_transform:
            linear.weight.data = torch.exp(self.weight.data)
        else:
            linear.weight.data = self.weight.data.clone()

        if self.bias is not None:
            if exp_transform:
                linear.bias.data = torch.exp(self.bias.data)
            else:
                linear.bias.data = self.bias.data.clone()

        return linear


class TropicalLinearFunction(torch.autograd.Function):
    """
    Autograd function for tropical linear operation.

    Forward:  Y_ij = max_k(X_ik + W_jk) + b_j
    Backward: Gradient flows only through the argmax path
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            ctx: Context for saving tensors.
            x: Input tensor, shape (*, in_features).
            weight: Weight tensor, shape (out_features, in_features).
            bias: Optional bias tensor, shape (out_features,).

        Returns:
            Output tensor, shape (*, out_features).
        """
        # Reshape x for broadcasting
        # x: (*, in_features) -> (*, 1, in_features)
        # weight: (out_features, in_features)
        # x + weight: (*, out_features, in_features)

        x_expanded = x.unsqueeze(-2)  # (*, 1, in_features)
        tropical_sum = x_expanded + weight  # (*, out_features, in_features)

        # Max over in_features dimension
        output, argmax_indices = tropical_sum.max(dim=-1)  # (*, out_features)

        # Add bias
        if bias is not None:
            output = output + bias

        # Save for backward
        ctx.save_for_backward(x, weight, argmax_indices)
        ctx.has_bias = bias is not None

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Backward pass.

        In tropical algebra, gradient flows only through the argmax path.
        """
        x, weight, argmax_indices = ctx.saved_tensors

        grad_x = None
        grad_weight = None
        grad_bias = None

        # grad_output shape: (*, out_features)
        # argmax_indices shape: (*, out_features)

        if ctx.needs_input_grad[0]:
            # Gradient w.r.t. x
            # For each output position, gradient flows to the input that won
            grad_x = torch.zeros_like(x)
            # Scatter gradient to the winning input positions
            grad_x.scatter_add_(
                -1,
                argmax_indices,
                grad_output,
            )

        if ctx.needs_input_grad[1]:
            # Gradient w.r.t. weight
            # grad_weight[j, k] = sum over (batch, positions) where argmax == k
            grad_weight = torch.zeros_like(weight)

            # Flatten batch dimensions
            batch_shape = grad_output.shape[:-1]
            num_batch = grad_output[..., 0].numel()
            flat_grad = grad_output.reshape(num_batch, -1)  # (batch, out_features)
            flat_argmax = argmax_indices.reshape(num_batch, -1)  # (batch, out_features)

            # For each output feature j
            for j in range(weight.shape[0]):
                # Get argmax indices for this output feature across batch
                k_indices = flat_argmax[:, j]  # (batch,)
                grads = flat_grad[:, j]  # (batch,)

                # Accumulate gradients
                grad_weight[j].scatter_add_(0, k_indices, grads)

        if ctx.has_bias and ctx.needs_input_grad[2]:
            # Gradient w.r.t. bias
            # Sum over all batch dimensions
            grad_bias = grad_output.sum(dim=tuple(range(grad_output.dim() - 1)))

        return grad_x, grad_weight, grad_bias


def convert_to_tropical(
    model: nn.Module,
    layers: Optional[list] = None,
    log_transform: bool = True,
    inplace: bool = False,
) -> nn.Module:
    """
    Convert Linear layers to TropicalLinear.

    Args:
        model: Model to convert.
        layers: List of layer names to convert. If None, converts all Linear layers.
        log_transform: Whether to apply log transform to weights.
        inplace: If True, modify model in place.

    Returns:
        Model with TropicalLinear layers.
    """
    import copy

    if not inplace:
        model = copy.deepcopy(model)

    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            if layers is None or name in layers:
                tropical_layer = TropicalLinear.from_linear(module, log_transform)
                # Replace the layer
                parts = name.split('.')
                parent = model
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                setattr(parent, parts[-1], tropical_layer)

    return model


def convert_to_standard(
    model: nn.Module,
    exp_transform: bool = True,
    inplace: bool = False,
) -> nn.Module:
    """
    Convert TropicalLinear layers back to standard Linear.

    Args:
        model: Model to convert.
        exp_transform: Whether to apply exp transform to weights.
        inplace: If True, modify model in place.

    Returns:
        Model with standard Linear layers.
    """
    import copy

    if not inplace:
        model = copy.deepcopy(model)

    for name, module in list(model.named_modules()):
        if isinstance(module, TropicalLinear):
            linear_layer = module.to_linear(exp_transform)
            # Replace the layer
            parts = name.split('.')
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], linear_layer)

    return model
