# Tropical Attention

Multi-Head Tropical Attention using tropical geometry and the Hilbert projective metric.

## Installation

```bash
pip install -e .
```

## Usage

```python
from tropical_attention import TropicalMultiheadAttention, TropicalTransformerEncoderLayer

# Drop-in replacement for nn.MultiheadAttention
attn = TropicalMultiheadAttention(d_model=64, num_heads=8)
x = torch.randn(4, 16, 64)  # (batch, seq, d_model)
out, weights = attn(x, x, x)

# Transformer encoder layer with tropical attention
layer = TropicalTransformerEncoderLayer(d_model=64, nhead=8)
out = layer(x)
```

## Architecture

```
Input X ∈ ℝ^(B, N, d)
    │
    ▼ Tropicalize: Z = log(clamp(X)) - max
    │
    ├─► Q = tropical_maxplus_matmul(Z, W_Q)
    ├─► K = tropical_maxplus_matmul(Z, W_K)
    └─► V = tropical_maxplus_matmul(Z, W_V)
    │
    ▼ Scores: S[i,j] = -hilbert_distance(Q[i], K[j])
    │
    ▼ Context: C = tropical_maxplus_matmul(S, V)
    │
    ▼ De-tropicalize: H = exp(C)
    │
    ▼ Output projection: out = H @ W_O
    │
Output ∈ ℝ^(B, N, d)
```
