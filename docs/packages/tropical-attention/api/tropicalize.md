# Tropicalize / Detropicalize

Convert between Euclidean and tropical projective space.

## Overview

These layers handle the conversion between standard neural network representations and the tropical projective space used by tropical attention.

**Tropicalize**: Maps positive values to the tropical simplex

$$z = \log(\text{clamp}(x, \epsilon)) - \max(\log(\text{clamp}(x, \epsilon)))$$

**Detropicalize**: Maps back to positive reals

$$x = \exp(z)$$

## Tropicalize Implementation

```python
class Tropicalize(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps  # (1)!

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Clamp to positive, take log
        u = torch.log(torch.clamp(x, min=self.eps))  # (2)!

        # Normalize: subtract max so max coord = 0
        z = u - u.max(dim=-1, keepdim=True).values  # (3)!

        return z
```

1. **Epsilon**: Small constant to prevent `log(0)`. Default `1e-8` works for most cases.

2. **Log transform**: First clamp values to be at least `eps`, then take logarithm. This maps positive reals to all of $\mathbb{R}$.

3. **Projective normalization**: Subtract the maximum value along the last dimension. Result: max coordinate = 0, all others ≤ 0.

## Detropicalize Implementation

```python
class Detropicalize(nn.Module):
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.exp(z)  # (1)!
```

1. **Exponential**: Inverse of log transform. Maps tropical simplex (max = 0) back to positive reals where max = 1.

## Why This Works

### Tropical Projective Space

In tropical geometry, vectors that differ by a constant are **projectively equivalent**:

```python
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([2.0, 3.0, 4.0])  # (1)!

# These represent the same "direction" in tropical projective space
```

1. **Same direction**: `y = x + 1`, so `x` and `y` are projectively equivalent. They represent the same point in $\mathbb{TP}^2$.

### The Tropical Simplex

The tropical simplex is the set of representatives where max = 0:

```python
x = torch.tensor([1.0, 2.0, 3.0])
z = tropicalize(x)  # (1)!
# z = [-2.0, -1.0, 0.0]  (after log and normalization)
```

1. **Normalized form**: After tropicalization, max(z) = 0. This is the canonical representative of the equivalence class.

## API Reference

### Tropicalize

::: tropical_attention.layers.tropicalize.Tropicalize
    options:
      show_root_heading: true
      show_source: true
      members:
        - __init__
        - forward

### Detropicalize

::: tropical_attention.layers.tropicalize.Detropicalize
    options:
      show_root_heading: true
      show_source: true
      members:
        - forward

## Usage Examples

### Basic Conversion

```python
import torch
from tropical_attention import Tropicalize, Detropicalize

tropicalize = Tropicalize(eps=1e-8)
detropicalize = Detropicalize()

# Positive input (e.g., after ReLU or from embeddings)
x = torch.rand(32, 100, 512) + 0.1  # (1)!

# Convert to tropical space
z = tropicalize(x)  # (2)!

# Convert back
x_recovered = detropicalize(z)  # (3)!
```

1. **Ensure positive**: Add 0.1 to guarantee all values are positive before log transform.

2. **To tropical**: `z` is now in the tropical simplex with `max(z, dim=-1) = 0` for each position.

3. **Back to Euclidean**: Note that `x_recovered ≠ x` exactly due to projective normalization, but they're proportional along each position.

### Verifying Properties

```python
x = torch.rand(32, 100, 512) + 0.1
z = tropicalize(x)

# Tropical simplex property
print(z.max(dim=-1).values)  # (1)!
print(z.min(dim=-1).values)  # (2)!
print((z <= 0).all())        # (3)!
```

1. **Max is zero**: All entries in the tensor are 0.0 (within floating point precision).

2. **Min is negative**: All entries are negative (log of values < max).

3. **All non-positive**: By construction, all coordinates are ≤ 0.

### Custom Epsilon

```python
# For very small values, use larger epsilon
tropicalize = Tropicalize(eps=1e-6)  # (1)!

# For values known to be > 0.01
tropicalize = Tropicalize(eps=1e-10)  # (2)!
```

1. **Larger eps**: Use when input might contain very small values near zero.

2. **Smaller eps**: Use when confident inputs are well above zero.

## Mathematical Details

### Why Log Transform?

The log transform converts **multiplicative** relationships to **additive**:

| Euclidean | Tropical (after log) |
|-----------|---------------------|
| $a \times b$ | $\log(a) + \log(b)$ |
| $\max(a, b)$ | $\max(\log(a), \log(b))$ |

This is why tropical matmul uses max-plus: it's equivalent to max-product in the original space.

### Numerical Stability

The `eps` parameter prevents `log(0) = -inf`:

```python
# Without eps:
torch.log(torch.tensor(0.0))  # tensor(-inf)

# With eps:
torch.log(torch.clamp(torch.tensor(0.0), min=1e-8))  # tensor(-18.42)
```

| Input Range | Recommended eps |
|-------------|-----------------|
| > 0.01 | `1e-8` (default) |
| > 0.0001 | `1e-6` |
| Very small | `1e-4` |

## Composition Pattern

Tropicalize and Detropicalize are typically used as a pair:

```python
class TropicalBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.tropicalize = Tropicalize()
        self.tropical_linear = TropicalLinear(d_model, d_model)  # (1)!
        self.detropicalize = Detropicalize()

    def forward(self, x):
        z = self.tropicalize(x)           # (2)!
        z = self.tropical_linear(z)        # (3)!
        return self.detropicalize(z)       # (4)!
```

1. **Tropical operation**: Any tropical layer (linear, attention, etc.)

2. **Enter tropical space**: Convert to log-normalized representation.

3. **Operate tropically**: Max-plus operations work correctly in this space.

4. **Exit tropical space**: Convert back for compatibility with standard layers.
