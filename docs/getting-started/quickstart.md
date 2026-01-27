# Quickstart

This guide covers the essential usage patterns for Tropical ML packages.

## Tropical Matrix Multiplication

The foundation of all tropical operations:

```python
import numpy as np
from tropical_gemm import maxplus_matmul, minplus_matmul

a = np.random.randn(100, 50).astype(np.float32)
b = np.random.randn(50, 80).astype(np.float32)

# Max-plus: c[i,j] = max_k(a[i,k] + b[k,j])
c_maxplus = maxplus_matmul(a, b)

# Min-plus: c[i,j] = min_k(a[i,k] + b[k,j])
c_minplus = minplus_matmul(a, b)
```

### With PyTorch (Autograd Support)

```python
import torch
from tropical_gemm.pytorch import tropical_maxplus_matmul

a = torch.randn(100, 50, requires_grad=True)
b = torch.randn(50, 80, requires_grad=True)

c = tropical_maxplus_matmul(a, b)
loss = c.sum()
loss.backward()  # Gradients flow through argmax
```

## Tropical Attention

Drop-in replacement for `nn.MultiheadAttention`:

```python
import torch
from tropical_attention import TropicalMultiheadAttention

# Create layer
attn = TropicalMultiheadAttention(
    d_model=512,
    num_heads=8,
    dropout=0.1,
    batch_first=True,
)

# Self-attention
x = torch.randn(32, 100, 512)
output, weights = attn(x, x, x)

# Cross-attention
query = torch.randn(32, 50, 512)
key = torch.randn(32, 100, 512)
value = torch.randn(32, 100, 512)
output, _ = attn(query, key, value)
```

### With Masking

```python
# Padding mask (True = ignore)
padding_mask = torch.zeros(32, 100, dtype=torch.bool)
padding_mask[:, 80:] = True

output, _ = attn(x, x, x, key_padding_mask=padding_mask)

# Causal mask for autoregressive
causal_mask = torch.triu(torch.ones(100, 100), diagonal=1).bool()
output, _ = attn(x, x, x, attn_mask=causal_mask)
```

## Transformer Encoder

Complete transformer layer with tropical attention:

```python
from tropical_attention import TropicalTransformerEncoderLayer

# Single layer
layer = TropicalTransformerEncoderLayer(
    d_model=512,
    nhead=8,
    dim_feedforward=2048,
    dropout=0.1,
)

x = torch.randn(32, 100, 512)
output = layer(x)

# Stack multiple layers
import torch.nn as nn

encoder = nn.Sequential(*[
    TropicalTransformerEncoderLayer(d_model=512, nhead=8)
    for _ in range(6)
])
output = encoder(x)
```

## Tropical Linear Layer

Linear projection using max-plus matmul:

```python
from tropical_attention import TropicalLinear

linear = TropicalLinear(in_features=512, out_features=256)
x = torch.randn(32, 100, 512)
output = linear(x)  # (32, 100, 256)
```

## Hilbert Distance

The metric used for attention scores:

```python
from tropical_attention import hilbert_distance

# Compute pairwise distances
q = torch.randn(32, 8, 50, 64)   # (batch, heads, seq_q, d_k)
k = torch.randn(32, 8, 100, 64)  # (batch, heads, seq_k, d_k)

distances = hilbert_distance(q, k)  # (32, 8, 50, 100)

# For attention: closer = higher score
scores = -distances
```

## GPU Acceleration

All operations automatically use GPU when available:

```python
from tropical_gemm.pytorch import GPU_AVAILABLE

print(f"GPU available: {GPU_AVAILABLE}")

# Move to GPU
model = TropicalMultiheadAttention(d_model=512, num_heads=8).cuda()
x = torch.randn(32, 100, 512).cuda()
output, _ = model(x, x, x)  # Uses CUDA kernels
```

## Next Steps

- [Tropical Attention API](../packages/tropical-attention/index.md)
- [Tropical GEMM API](../packages/tropical-gemm/index.md)
- [Theory: Tropical Geometry](../theory/tropical-geometry.md)
