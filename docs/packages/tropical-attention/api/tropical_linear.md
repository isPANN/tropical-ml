# TropicalLinear

Linear projection using tropical (max-plus) matrix multiplication.

## Overview

`TropicalLinear` is the tropical analog of `nn.Linear`. Instead of:

$$y = Wx + b \quad \text{(standard)}$$

It computes:

$$y[i] = \max_k(x[k] + W[i,k]) + b[i] \quad \text{(tropical)}$$

Where:

- Standard addition becomes **max**
- Standard multiplication becomes **addition**

## Implementation Explained

```python
class TropicalLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))  # (1)!
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None  # (2)!
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))  # (3)!

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle batched input
        orig_shape = x.shape[:-1]  # (4)!
        x_2d = x.reshape(-1, x.shape[-1])

        # Tropical matmul: (B*N, in) ⊙ (in, out) -> (B*N, out)
        if x.is_cuda and GPU_AVAILABLE:
            y_2d = tropical_maxplus_matmul_gpu(x_2d, self.weight.t())  # (5)!
        else:
            y_2d = tropical_maxplus_matmul(x_2d, self.weight.t())

        # Reshape back
        y = y_2d.reshape(*orig_shape, -1)  # (6)!

        if self.bias is not None:
            y = y + self.bias  # (7)!

        return y
```

1. **Weight matrix**: Shape `(out_features, in_features)` - same as `nn.Linear`. Stored as a learnable `nn.Parameter`.

2. **Optional bias**: Unlike standard linear layers, bias defaults to `False` since tropical operations often don't need it.

3. **Kaiming initialization**: Uses the same initialization as `nn.Linear` with `a=sqrt(5)`. Works well even for tropical layers.

4. **Flatten batch dimensions**: Save original shape `(B, N, ...)` and flatten to 2D `(B*N, in_features)` for matrix multiplication.

5. **Tropical matrix multiplication**: Computes $y[i,j] = \max_k(x[i,k] + W^T[k,j])$. The `.t()` transposes weight from `(out, in)` to `(in, out)`.

6. **Restore shape**: Unflatten back to original batch dimensions with new feature size.

7. **Standard bias addition**: Uses regular addition (not tropical max) for gradient flow compatibility.

## Mathematical Details

### Max-Plus Semiring

In the max-plus semiring:

| Operation | Standard | Max-Plus |
|-----------|----------|----------|
| Addition  | $a + b$  | $\max(a, b)$ |
| Multiplication | $a \times b$ | $a + b$ |

### Matrix Multiplication

For matrices $A$ (M×K) and $B$ (K×N):

$$C[i,j] = \bigoplus_{k=1}^{K} A[i,k] \otimes B[k,j] = \max_{k=1}^{K}(A[i,k] + B[k,j])$$

### Gradient Flow

The gradient is **sparse**: only the argmax index contributes to each output.

```python
# For C[i,j] = max_k(A[i,k] + B[k,j])
# Let k* = argmax_k(A[i,k] + B[k,j])

# Then:
# ∂C[i,j]/∂A[i,k*] = 1
# ∂C[i,j]/∂A[i,k] = 0  for k ≠ k*
```

## API Reference

::: tropical_attention.layers.tropical_linear.TropicalLinear
    options:
      show_root_heading: true
      show_source: true
      members:
        - __init__
        - forward

## Usage Examples

### Basic Usage

```python
import torch
from tropical_attention import TropicalLinear

linear = TropicalLinear(
    in_features=512,   # (1)!
    out_features=256,  # (2)!
    bias=True,         # (3)!
)

x = torch.randn(32, 100, 512)  # (4)!
output = linear(x)  # (5)!
```

1. **in_features**: Input dimension (last axis of input tensor).

2. **out_features**: Output dimension (last axis of output tensor).

3. **bias**: Optional bias term added after tropical matmul.

4. **Input**: Any shape `(..., in_features)` - handles arbitrary batch dimensions.

5. **Output**: Shape `(32, 100, 256)` - last dimension changed to out_features.

### Without Bias

```python
linear = TropicalLinear(512, 256, bias=False)  # (1)!
```

1. **No bias**: Pure tropical linear transformation without additive bias term.

### GPU Acceleration

```python
from tropical_gemm.pytorch import GPU_AVAILABLE

print(f"GPU available: {GPU_AVAILABLE}")  # (1)!

linear = TropicalLinear(512, 256).cuda()  # (2)!
x = torch.randn(32, 100, 512).cuda()
output = linear(x)  # (3)!
```

1. **Check availability**: GPU kernels require CUDA-enabled `tropical-gemm` build.

2. **Move to GPU**: Standard PyTorch `.cuda()` call.

3. **Automatic dispatch**: Forward pass automatically uses `tropical_maxplus_matmul_gpu`.

## Comparison with nn.Linear

| Aspect | nn.Linear | TropicalLinear |
|--------|-----------|----------------|
| Operation | Sum of products | Max of sums |
| Formula | $y = Wx + b$ | $y[i] = \max_k(x[k] + W[i,k]) + b[i]$ |
| Gradient | Dense (all weights contribute) | Sparse (only argmax) |
| Bias default | `True` | `False` |
| Initialization | Kaiming uniform | Kaiming uniform |
