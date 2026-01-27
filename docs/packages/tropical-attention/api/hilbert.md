# hilbert_distance

Compute pairwise Hilbert projective distances.

## Overview

The Hilbert projective metric is a natural distance measure in tropical projective space. It quantifies how "different" two vectors are, ignoring projective equivalence (constant shifts).

$$d_H(x, y) = \max_i(x_i - y_i) - \min_i(x_i - y_i)$$

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
q = torch.randn(32, 8, 50, 64)   # (batch, heads, seq_q, d_k)
k = torch.randn(32, 8, 100, 64)  # (batch, heads, seq_k, d_k)

# Compute pairwise distances
distances = hilbert_distance(q, k)
# Shape: (32, 8, 50, 100)
```

### For Attention Scores

In tropical attention, we use **negative** Hilbert distance as scores:

```python
scores = -hilbert_distance(q, k)
# Higher score = closer = more attention
```

### Single Vectors

```python
# Two individual vectors
x = torch.randn(64)
y = torch.randn(64)

# Reshape for the function
x = x.view(1, 1, 1, 64)
y = y.view(1, 1, 1, 64)

d = hilbert_distance(x, y)
print(f"Distance: {d.item():.4f}")
```

## Mathematical Properties

### Definition

For vectors $x, y \in \mathbb{R}^n$:

$$d_H(x, y) = \max_i(x_i - y_i) - \min_i(x_i - y_i)$$

### Key Properties

| Property | Description |
|----------|-------------|
| **Non-negative** | $d_H(x, y) \geq 0$ |
| **Symmetric** | $d_H(x, y) = d_H(y, x)$ |
| **Projective** | $d_H(x + c, y + c) = d_H(x, y)$ for any scalar $c$ |
| **Identity** | $d_H(x, y) = 0 \Leftrightarrow x = y + c$ for some $c$ |

### Projective Invariance

The distance is invariant to adding constants:

```python
x = torch.randn(64)
y = torch.randn(64)
c = 5.0

d1 = hilbert_distance(x.view(1,1,1,-1), y.view(1,1,1,-1))
d2 = hilbert_distance((x+c).view(1,1,1,-1), (y+c).view(1,1,1,-1))

print(f"d(x, y) = {d1.item():.4f}")
print(f"d(x+c, y+c) = {d2.item():.4f}")
# These are equal
```

## Geometric Interpretation

The Hilbert metric measures the "spread" of coordinate-wise differences:

- If $x - y$ has the same value everywhere: $d_H = 0$ (projectively equivalent)
- If $x - y$ varies a lot: $d_H$ is large (very different directions)

### Example

```python
# Projectively equivalent (differ by constant)
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([2.0, 3.0, 4.0])
# d_H = max([-1,-1,-1]) - min([-1,-1,-1]) = -1 - (-1) = 0

# Very different
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([3.0, 2.0, 1.0])
# d_H = max([-2,0,2]) - min([-2,0,2]) = 2 - (-2) = 4
```

## Why Hilbert Distance for Attention?

1. **Projective invariance**: Matches the normalization in tropical attention
2. **Interpretable**: Measures angular difference in tropical geometry
3. **Efficient**: Simple max/min operations, vectorizable
4. **Gradient-friendly**: Differentiable (subgradient at ties)
