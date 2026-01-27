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

## Forward Pass Explained

Here's the complete forward pass with line-by-line annotations:

```python
def forward(self, query, key, value, key_padding_mask=None,
            need_weights=False, attn_mask=None):

    # Handle batch_first format
    if not self.batch_first:  # (1)!
        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)

    B, N, _ = query.shape  # (2)!
    S = key.shape[1]
    H = self.num_heads
    d_k = self.head_dim

    # Step 1: Tropicalize inputs
    q_trop = self.tropicalize(query)  # (3)!
    k_trop = self.tropicalize(key)
    v_trop = self.tropicalize(value)

    # Step 2: Tropical projections
    Q = self.q_proj(q_trop)  # (4)!
    K = self.k_proj(k_trop)
    V = self.v_proj(v_trop)

    # Step 3: Reshape for multi-head
    Q = Q.view(B, N, H, d_k).transpose(1, 2)  # (5)!
    K = K.view(B, S, H, d_k).transpose(1, 2)
    V = V.view(B, S, H, d_k).transpose(1, 2)

    # Step 4: Compute attention scores via Hilbert distance
    scores = -hilbert_distance(Q, K)  # (6)!

    # Step 5: Apply masks
    if attn_mask is not None:  # (7)!
        scores = scores.masked_fill(attn_mask, float("-inf"))
    if key_padding_mask is not None:
        mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
        scores = scores.masked_fill(mask, float("-inf"))

    # Step 6: Tropical aggregation (max-plus matmul)
    scores_flat = scores.reshape(B * H, N, S)  # (8)!
    V_flat = V.reshape(B * H, S, d_k)

    if query.is_cuda and GPU_AVAILABLE:
        C_flat = self._batched_tropical_matmul_gpu(scores_flat, V_flat)  # (9)!
    else:
        C_flat = self._batched_tropical_matmul(scores_flat, V_flat)

    C = C_flat.view(B, H, N, d_k)

    # Step 7: De-tropicalize and reshape
    C = self.detropicalize(C)  # (10)!
    C = C.transpose(1, 2).reshape(B, N, self.d_model)

    # Step 8: Output projection (standard, not tropical)
    output = self.out_proj(C)  # (11)!
    output = self.dropout(output)

    return output, scores if need_weights else None
```

1. **Format handling**: Convert from `(seq, batch, d_model)` to `(batch, seq, d_model)` if needed. This ensures consistent internal processing regardless of input format.

2. **Extract dimensions**: `B`=batch size, `N`=query length, `S`=key/value length, `H`=number of heads, `d_k`=head dimension (d_model / num_heads).

3. **Tropicalize**: Convert to tropical projective space via $z = \log(\text{clamp}(x)) - \max$. This maps positive values to log-space and normalizes so max coordinate = 0.

4. **Tropical Q, K, V projections**: Unlike standard linear layers that compute $y = Wx + b$, these use max-plus matmul: $y[i] = \max_k(x[k] + W[i,k]) + b[i]$.

5. **Multi-head reshape**: Split the d_model dimension into H heads of size d_k. Shape goes from `(B, N, d_model)` to `(B, H, N, d_k)`.

6. **Hilbert distance scoring**: Compute $S[i,j] = -d_H(Q[i], K[j])$ where $d_H(x,y) = \max(x-y) - \min(x-y)$. Negative distance means closer vectors get higher scores.

7. **Masking**: Set masked positions to $-\infty$ so they contribute nothing in the subsequent max-plus aggregation (since $\max(x, -\infty) = x$).

8. **Flatten for batched matmul**: Reshape from `(B, H, N, S)` to `(B*H, N, S)` to process all batch-head combinations in parallel.

9. **Tropical aggregation**: Compute $C[i,:] = \max_j(S[i,j] + V[j,:])$ using max-plus matmul. This replaces the weighted sum $\sum_j \text{softmax}(S)_{ij} \cdot V[j,:]$ in standard attention.

10. **De-tropicalize**: Convert back to Euclidean space via $x = \exp(z)$. This undoes the log transform from tropicalization.

11. **Output projection**: A standard (non-tropical) linear layer to mix information across heads and project back to d_model dimensions.

## Tropicalize Step Detail

```python
class Tropicalize(nn.Module):
    def forward(self, x):
        u = torch.log(torch.clamp(x, min=self.eps))  # (1)!
        z = u - u.max(dim=-1, keepdim=True).values   # (2)!
        return z
```

1. **Log transform**: Clamp values to avoid log(0), then take logarithm. This converts multiplicative relationships to additive (tropical) ones.

2. **Projective normalization**: Subtract the max so the largest coordinate becomes 0. This places vectors on the tropical simplex where they are projectively normalized.

## Hilbert Distance Detail

```python
def hilbert_distance(q, k):
    q_exp = q.unsqueeze(-2)  # (1)!
    k_exp = k.unsqueeze(-3)

    diff = q_exp - k_exp     # (2)!

    d_h = diff.max(dim=-1).values - diff.min(dim=-1).values  # (3)!
    return d_h
```

1. **Expand for broadcasting**: Add dimensions so `q` becomes `(B, H, Nq, 1, d_k)` and `k` becomes `(B, H, 1, Nk, d_k)`.

2. **Pairwise difference**: Broadcasting computes all `Nq × Nk` pairwise differences, resulting in shape `(B, H, Nq, Nk, d_k)`.

3. **Hilbert metric**: $d_H(x, y) = \max_i(x_i - y_i) - \min_i(x_i - y_i)$. Measures the "oscillation" of the difference vector - if constant (projectively equivalent), distance is 0.

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
    d_model=512,   # (1)!
    num_heads=8,   # (2)!
    dropout=0.1,
    batch_first=True,
)

x = torch.randn(32, 100, 512)  # (3)!
output, weights = attn(x, x, x)  # (4)!
```

1. **d_model**: Total embedding dimension. Must be divisible by num_heads.

2. **num_heads**: Number of attention heads. Each head has dimension `d_model // num_heads = 64`.

3. **Input shape**: `(batch=32, seq_len=100, d_model=512)`.

4. **Self-attention**: Query, Key, and Value are all the same tensor. Output has same shape as input.

### Cross-Attention

```python
query = torch.randn(32, 50, 512)   # (1)!
key = torch.randn(32, 100, 512)    # (2)!
value = torch.randn(32, 100, 512)

output, _ = attn(query, key, value)  # (3)!
```

1. **Query**: Target sequence with 50 positions.

2. **Key/Value**: Source sequence with 100 positions. Key and Value must have the same sequence length.

3. **Output**: Shape `(32, 50, 512)` - same sequence length as query.

### With Masking

```python
# Padding mask: True = ignore this position
padding_mask = torch.zeros(32, 100, dtype=torch.bool)
padding_mask[:, 80:] = True  # (1)!

output, _ = attn(x, x, x, key_padding_mask=padding_mask)

# Causal mask for autoregressive models
causal = torch.triu(torch.ones(100, 100), diagonal=1).bool()  # (2)!
output, _ = attn(x, x, x, attn_mask=causal)
```

1. **Padding mask**: Positions 80-99 are padding tokens and will be ignored (scores set to $-\infty$).

2. **Causal mask**: Upper triangular matrix prevents attending to future positions. Position $i$ can only attend to positions $\leq i$.

## Comparison with nn.MultiheadAttention

| Feature | nn.MultiheadAttention | TropicalMultiheadAttention |
|---------|----------------------|---------------------------|
| Attention scoring | Softmax(QK^T / √d) | -hilbert_distance(Q, K) |
| Q, K, V projection | Standard linear | Tropical linear (max-plus) |
| Aggregation | Weighted sum | Tropical matmul (max-plus) |
| Normalization | Softmax (sum to 1) | Projective (max coord = 0) |
