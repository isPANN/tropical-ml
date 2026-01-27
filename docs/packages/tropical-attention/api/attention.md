# TropicalMultiheadAttention

Multi-Head Tropical Attention (MHTA) - a drop-in replacement for `nn.MultiheadAttention`.

## Overview

`TropicalMultiheadAttention` replaces softmax-based attention with tropical geometry:

1. **Tropicalize** inputs to log-space
2. **Project** Q, K, V using tropical (max-plus) linear layers
3. **Score** using negative Hilbert distance (closer = higher score)
4. **Aggregate** using tropical matmul
5. **De-tropicalize** back to Euclidean space

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

## API Reference

::: tropical_attention.layers.attention.TropicalMultiheadAttention
    options:
      show_root_heading: true
      show_source: true
      members:
        - __init__
        - forward

## Usage Examples

### Basic Self-Attention

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
output, weights = attn(x, x, x)
# output: (32, 100, 512)
# weights: (32, 8, 100, 100) if need_weights=True
```

### Cross-Attention

```python
query = torch.randn(32, 50, 512)   # Target sequence
key = torch.randn(32, 100, 512)    # Source sequence
value = torch.randn(32, 100, 512)

output, _ = attn(query, key, value)
# output: (32, 50, 512)
```

### With Masking

```python
# Padding mask
padding_mask = torch.zeros(32, 100, dtype=torch.bool)
padding_mask[:, 80:] = True  # Ignore positions 80-99

output, _ = attn(x, x, x, key_padding_mask=padding_mask)

# Causal mask for autoregressive
causal = torch.triu(torch.ones(100, 100), diagonal=1).bool()
output, _ = attn(x, x, x, attn_mask=causal)
```

### Sequence-First Format

```python
attn = TropicalMultiheadAttention(
    d_model=512,
    num_heads=8,
    batch_first=False,  # (seq, batch, d_model)
)

x = torch.randn(100, 32, 512)  # (seq_len, batch, d_model)
output, _ = attn(x, x, x)
```

## Comparison with nn.MultiheadAttention

| Feature | nn.MultiheadAttention | TropicalMultiheadAttention |
|---------|----------------------|---------------------------|
| Attention scoring | Softmax(QK^T / √d) | -hilbert_distance(Q, K) |
| Q, K, V projection | Standard linear | Tropical linear (max-plus) |
| Aggregation | Weighted sum | Tropical matmul (max-plus) |
| Normalization | Softmax (sum to 1) | Projective (max coord = 0) |
