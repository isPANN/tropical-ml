# Hilbert Projective Metric

The Hilbert projective metric is the distance measure used in Tropical Attention for computing attention scores.

## Definition

For vectors $x, y \in \mathbb{R}^n$:

$$d_H(x, y) = \max_i(x_i - y_i) - \min_i(x_i - y_i)$$

Equivalently:

$$d_H(x, y) = \max_{i,j}[(x_i - y_i) - (x_j - y_j)]$$

## Properties

### 1. Non-Negativity

$$d_H(x, y) \geq 0$$

with equality if and only if $x - y$ is constant (i.e., $x$ and $y$ are projectively equivalent).

### 2. Symmetry

$$d_H(x, y) = d_H(y, x)$$

### 3. Projective Invariance

$$d_H(x + c\mathbf{1}, y + c\mathbf{1}) = d_H(x, y)$$

for any scalar $c$. This means the metric is well-defined on tropical projective space.

### 4. Triangle Inequality

$$d_H(x, z) \leq d_H(x, y) + d_H(y, z)$$

## Geometric Interpretation

### Projective Space View

In tropical projective space $\mathbb{TP}^{n-1}$, vectors are equivalence classes under constant shifts. The Hilbert metric measures the "angular distance" between two directions.

### Oscillation View

$d_H(x, y)$ measures the **oscillation** of the difference $x - y$:

$$d_H(x, y) = \text{oscillation}(x - y) = \max(x-y) - \min(x-y)$$

If $x - y$ is constant, the oscillation is 0 (projectively equivalent).

## Connection to Other Metrics

### Relation to $\ell^\infty$ Norm

$$d_H(x, y) \leq 2 \|x - y\|_\infty$$

### Relation to Variation

For normalized vectors (max = 0):

$$d_H(x, y) = -\min_i(x_i - y_i) - (-\min_j(y_j - x_j))$$

## Use in Tropical Attention

In Tropical Attention, we use **negative Hilbert distance** as attention scores:

$$\text{score}(q, k) = -d_H(q, k)$$

This means:

- **Closer vectors** (smaller $d_H$) get **higher scores**
- **Projectively equivalent** vectors get score 0 (maximum)

### Why Hilbert Distance?

1. **Projective invariance**: Matches the normalization in tropicalization
2. **Bounded**: Unlike softmax scores, Hilbert distances are bounded
3. **Interpretable**: Measures angular separation in tropical space
4. **Efficient**: Only requires max and min operations

## Implementation

```python
def hilbert_distance(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise Hilbert distances.

    Args:
        q: (batch, heads, seq_q, d_k)
        k: (batch, heads, seq_k, d_k)

    Returns:
        distances: (batch, heads, seq_q, seq_k)
    """
    # Expand for pairwise computation
    q_exp = q.unsqueeze(-2)  # (B, H, Nq, 1, d_k)
    k_exp = k.unsqueeze(-3)  # (B, H, 1, Nk, d_k)

    # Difference
    diff = q_exp - k_exp  # (B, H, Nq, Nk, d_k)

    # Hilbert distance = max - min
    d_h = diff.max(dim=-1).values - diff.min(dim=-1).values

    return d_h
```

## Gradient Properties

The gradient of Hilbert distance is sparse:

$$\frac{\partial d_H}{\partial x_i} = \begin{cases}
1 & \text{if } i = \arg\max_j(x_j - y_j) \\
-1 & \text{if } i = \arg\min_j(x_j - y_j) \\
0 & \text{otherwise}
\end{cases}$$

This sparsity propagates through the attention mechanism, leading to efficient gradient computation.

## References

- Hilbert, D. (1895). "Über die gerade Linie als kürzeste Verbindung zweier Punkte"
- Birkhoff, G. (1957). "Extensions of Jentzsch's theorem"
- Lemmens & Nussbaum, *Nonlinear Perron-Frobenius Theory*
