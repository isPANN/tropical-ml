# Tropicalize / Detropicalize

Convert between Euclidean and tropical projective space.

## Overview

These layers handle the conversion between standard neural network representations and the tropical projective space used by tropical attention.

**Tropicalize**: Maps positive values to the tropical simplex

$$z = \log(\text{clamp}(x, \epsilon)) - \max(\log(\text{clamp}(x, \epsilon)))$$

**Detropicalize**: Maps back to positive reals

$$x = \exp(z)$$

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
x = torch.rand(32, 100, 512) + 0.1  # Ensure positive

# Convert to tropical space
z = tropicalize(x)
# z is in tropical simplex: max(z, dim=-1) = 0

# Convert back
x_recovered = detropicalize(z)
```

### Custom Epsilon

```python
# For very small values, use larger epsilon
tropicalize = Tropicalize(eps=1e-6)
```

### Checking Properties

```python
x = torch.rand(32, 100, 512) + 0.1
z = tropicalize(x)

# Tropical simplex property: max coordinate = 0
print(z.max(dim=-1).values)  # All zeros
print(z.min(dim=-1).values)  # All negative
```

## Mathematical Details

### Tropical Projective Space

In tropical geometry, vectors that differ by a constant are equivalent:

$$x \sim x + c \quad \text{for any scalar } c$$

This is called **projective equivalence**. The tropical simplex is the set of representatives where the maximum coordinate is 0.

### Why Log Transform?

The log transform converts multiplicative relationships to additive:

- Standard attention: $\text{softmax}(QK^T/\sqrt{d})$
- Tropical attention: Uses log-space where operations become max-plus

### Numerical Stability

The `eps` parameter prevents log(0):

```python
u = torch.log(torch.clamp(x, min=eps))
```

Choose `eps` based on your input range:

| Input Range | Recommended eps |
|-------------|-----------------|
| > 0.01 | 1e-8 (default) |
| > 0.0001 | 1e-6 |
| Very small | 1e-4 |

## Composition

Tropicalize and Detropicalize are typically used together:

```python
class TropicalBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.tropicalize = Tropicalize()
        self.tropical_op = TropicalLinear(d_model, d_model)
        self.detropicalize = Detropicalize()

    def forward(self, x):
        z = self.tropicalize(x)      # To tropical space
        z = self.tropical_op(z)       # Tropical operations
        return self.detropicalize(z)  # Back to Euclidean
```
