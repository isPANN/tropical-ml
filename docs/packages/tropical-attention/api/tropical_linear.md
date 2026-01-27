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

# Create layer
linear = TropicalLinear(
    in_features=512,
    out_features=256,
    bias=True,
)

# Forward pass
x = torch.randn(32, 100, 512)
output = linear(x)  # (32, 100, 256)
```

### Without Bias

```python
linear = TropicalLinear(512, 256, bias=False)
```

### GPU Acceleration

The layer automatically uses GPU kernels when available:

```python
from tropical_gemm.pytorch import GPU_AVAILABLE

print(f"GPU available: {GPU_AVAILABLE}")

linear = TropicalLinear(512, 256).cuda()
x = torch.randn(32, 100, 512).cuda()
output = linear(x)  # Uses CUDA kernels
```

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

For output $C[i,j] = \max_k(A[i,k] + B[k,j])$:

- $\frac{\partial C[i,j]}{\partial A[i,k^*]} = 1$ where $k^* = \arg\max_k(A[i,k] + B[k,j])$
- All other gradients are 0

## Comparison with nn.Linear

| Aspect | nn.Linear | TropicalLinear |
|--------|-----------|----------------|
| Operation | Sum of products | Max of sums |
| Gradient | Dense (all weights contribute) | Sparse (only argmax) |
| Bias | Added | Added (standard, not tropical) |
| Initialization | Kaiming uniform | Kaiming uniform |
