# hilbert_distance

Compute pairwise Hilbert projective distances.

## Overview

The Hilbert projective metric is a natural distance measure in tropical projective space. It quantifies how "different" two vectors are, ignoring projective equivalence (constant shifts).

$$d_H(x, y) = \max_i(x_i - y_i) - \min_i(x_i - y_i)$$

## Implementation Explained

```python
def hilbert_distance(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise Hilbert projective distances.

    Args:
        q: (batch, heads, seq_q, d_k) - query vectors
        k: (batch, heads, seq_k, d_k) - key vectors

    Returns:
        distances: (batch, heads, seq_q, seq_k)
    """
    # Expand for pairwise computation
    q_exp = q.unsqueeze(-2)  # (1)!
    k_exp = k.unsqueeze(-3)  # (2)!

    # Difference: (B, H, Nq, Nk, d_k)
    diff = q_exp - k_exp  # (3)!

    # Hilbert distance = max - min along last dim
    d_h = diff.max(dim=-1).values - diff.min(dim=-1).values  # (4)!

    return d_h
```

1. **Expand query**: Shape `(B, H, Nq, d_k)` → `(B, H, Nq, 1, d_k)`. The new dimension at position -2 will broadcast with keys.

2. **Expand key**: Shape `(B, H, Nk, d_k)` → `(B, H, 1, Nk, d_k)`. The new dimension at position -3 will broadcast with queries.

3. **Pairwise difference**: Broadcasting computes `diff[b,h,i,j,:] = q[b,h,i,:] - k[b,h,j,:]` for all `i,j` pairs. Result shape: `(B, H, Nq, Nk, d_k)`.

4. **Hilbert formula**: For each pair `(i,j)`, compute `max(diff) - min(diff)` over the d_k dimension. This gives the Hilbert distance $d_H(q_i, k_j)$.

## API Reference

::: tropical_attention.layers.hilbert.hilbert_distance
    options:
      show_root_heading: true
      show_source: true

## Usage Examples

### Basic Usage

```python
import torch
from tropical_attention import hilbert_distance

# Query and Key tensors in tropical space
q = torch.randn(32, 8, 50, 64)   # (1)!
k = torch.randn(32, 8, 100, 64)  # (2)!

distances = hilbert_distance(q, k)  # (3)!
```

1. **Query shape**: `(batch=32, heads=8, seq_q=50, d_k=64)`.

2. **Key shape**: `(batch=32, heads=8, seq_k=100, d_k=64)`.

3. **Output shape**: `(32, 8, 50, 100)` - distance from each of 50 queries to each of 100 keys.

### For Attention Scores

```python
# In tropical attention, we use NEGATIVE Hilbert distance
scores = -hilbert_distance(q, k)  # (1)!

# Now: higher score = closer = more attention
```

1. **Negative distance**: Smaller distance means vectors are more similar, so we negate to get higher scores for more similar pairs.

### Single Vector Pair

```python
x = torch.tensor([1.0, 2.0, 3.0, 4.0])  # (1)!
y = torch.tensor([2.0, 1.0, 4.0, 3.0])

# Reshape for the function
x = x.view(1, 1, 1, 4)
y = y.view(1, 1, 1, 4)

d = hilbert_distance(x, y)
print(f"d_H(x, y) = {d.item():.4f}")  # (2)!
```

1. **Two vectors**: We'll compute the Hilbert distance between these 4-dimensional vectors.

2. **Result**: `d_H = max([−1, 1, −1, 1]) − min([−1, 1, −1, 1]) = 1 − (−1) = 2.0`

### Projective Equivalence

```python
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([2.0, 3.0, 4.0])  # (1)!

x = x.view(1, 1, 1, 3)
y = y.view(1, 1, 1, 3)

d = hilbert_distance(x, y)
print(f"d_H(x, y) = {d.item():.4f}")  # (2)!
```

1. **Projectively equivalent**: `y = x + 1`, so these vectors represent the same direction in tropical projective space.

2. **Zero distance**: `d_H = max([−1, −1, −1]) − min([−1, −1, −1]) = −1 − (−1) = 0.0`

## Mathematical Properties

### Definition

For vectors $x, y \in \mathbb{R}^n$:

$$d_H(x, y) = \max_i(x_i - y_i) - \min_i(x_i - y_i)$$

Equivalently:

$$d_H(x, y) = \max_{i,j}[(x_i - y_i) - (x_j - y_j)]$$

### Key Properties

| Property | Formula | Meaning |
|----------|---------|---------|
| **Non-negative** | $d_H(x, y) \geq 0$ | Distances are never negative |
| **Symmetric** | $d_H(x, y) = d_H(y, x)$ | Order doesn't matter |
| **Projective** | $d_H(x + c, y + c) = d_H(x, y)$ | Invariant to constant shifts |
| **Identity** | $d_H(x, y) = 0 \Leftrightarrow x = y + c$ | Zero iff projectively equivalent |

### Projective Invariance Proof

```python
# For any scalar c:
# d_H(x + c, y + c)
#   = max_i((x_i + c) - (y_i + c)) - min_i((x_i + c) - (y_i + c))
#   = max_i(x_i - y_i) - min_i(x_i - y_i)
#   = d_H(x, y)
```

## Geometric Interpretation

### Oscillation View

$d_H(x, y)$ measures the **oscillation** (spread) of the difference $x - y$:

```python
diff = x - y
oscillation = diff.max() - diff.min()  # (1)!
```

1. **Oscillation**: If `x - y` is constant (projectively equivalent), oscillation = 0. If `x - y` varies a lot, oscillation is large.

### Tropical Projective Space

In tropical projective space $\mathbb{TP}^{n-1}$:

- Vectors are equivalence classes under constant shifts
- The Hilbert metric measures "angular distance" between directions
- Zero distance means same direction (projectively equivalent)

## Gradient Properties

The gradient of Hilbert distance is sparse:

```python
# For d_H(x, y) = max_i(x_i - y_i) - min_j(x_j - y_j)
# Let i* = argmax, j* = argmin

# ∂d_H/∂x_i = +1 if i = i* (the argmax)
# ∂d_H/∂x_j = -1 if j = j* (the argmin)
# ∂d_H/∂x_k =  0 otherwise
```

This sparsity propagates through tropical attention, leading to efficient gradients.

## Why Hilbert for Attention?

| Advantage | Explanation |
|-----------|-------------|
| **Projective invariance** | Matches the normalization in Tropicalize |
| **Interpretable** | Measures angular separation in tropical space |
| **Efficient** | Only max/min operations, fully vectorizable |
| **Bounded** | Unlike dot product, doesn't grow with dimension |
| **Sparse gradients** | Only 2 coordinates contribute to gradient |
