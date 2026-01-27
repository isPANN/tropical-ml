# Tropical Attention

**Multi-Head Attention using Tropical Geometry and Hilbert Projective Metric**

## Overview

Tropical Attention replaces softmax-based attention with operations from tropical geometry. Instead of computing attention weights via softmax, we use the **Hilbert projective metric** to measure similarity in tropical projective space.

## Installation

```bash
pip install tropical-attention
```

## Key Features

- **Drop-in replacement** for `nn.MultiheadAttention`
- **Tropical projections** using max-plus matrix multiplication
- **Hilbert distance** for attention scoring
- **GPU-accelerated** via optimized CUDA kernels

## Architecture

```
Input X
    │
    ▼
Tropicalize (log + normalize)
    │
    ├──► Q = TropicalLinear(X)
    ├──► K = TropicalLinear(X)
    └──► V = TropicalLinear(X)
           │
           ▼
    Scores = -hilbert_distance(Q, K)
           │
           ▼
    C = tropical_matmul(Scores, V)
           │
           ▼
    Detropicalize (exp)
           │
           ▼
    Output = Linear(C)
```

## Quick Example

```python
import torch
from tropical_attention import TropicalMultiheadAttention

attn = TropicalMultiheadAttention(
    d_model=512,
    num_heads=8,
    dropout=0.1,
    batch_first=True,
)

x = torch.randn(32, 100, 512)
output, attn_weights = attn(x, x, x)
```

## API Reference

| Component | Description |
|-----------|-------------|
| [`TropicalMultiheadAttention`](api/attention.md) | Multi-head tropical attention |
| [`TropicalLinear`](api/tropical_linear.md) | Linear layer using max-plus matmul |
| [`Tropicalize`](api/tropicalize.md) | Convert Euclidean to tropical space |
| [`Detropicalize`](api/tropicalize.md) | Convert tropical to Euclidean space |
| [`hilbert_distance`](api/hilbert.md) | Hilbert projective metric |
| [`TropicalTransformerEncoderLayer`](api/transformer.md) | Complete transformer encoder |

## Mathematical Foundation

### Tropical Semiring

| Standard | Tropical (Max-Plus) |
|----------|---------------------|
| $a + b$  | $\max(a, b)$        |
| $a \times b$ | $a + b$         |

### Hilbert Projective Metric

$$d_H(x, y) = \max_i(x_i - y_i) - \min_i(x_i - y_i)$$

This metric measures angular distance in tropical projective space, making it ideal for attention scoring.

## Comparison with Standard Attention

| Aspect | Standard Attention | Tropical Attention |
|--------|-------------------|-------------------|
| Scoring | $\text{softmax}(QK^T / \sqrt{d})$ | $-d_H(Q, K)$ |
| Aggregation | Weighted sum | Max-plus matmul |
| Normalization | Softmax (sum to 1) | Projective (max = 0) |
| Gradients | Dense | Sparse (argmax only) |
