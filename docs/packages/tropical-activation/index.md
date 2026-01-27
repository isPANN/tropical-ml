# Tropical Activation

**Activation Functions Based on Tropical Operations**

!!! warning "Experimental"
    This package is experimental and under active development.

## Overview

Tropical Activation provides activation functions derived from tropical geometry operations.

## Installation

```bash
pip install tropical-activation
```

## Concepts

### Tropical ReLU

Traditional ReLU: $\text{ReLU}(x) = \max(0, x)$

This is already a tropical operation (max in the max-plus semiring).

### Tropical Softmax

Instead of standard softmax:

$$\text{softmax}(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

Tropical softmax uses:

$$\text{trop-softmax}(x)_i = x_i - \max_j(x_j)$$

This normalizes so the maximum value is 0 (projective normalization).

## API Reference

*Documentation coming soon.*

## Related Work

- [Tropical Attention](../tropical-attention/index.md) uses tropical operations for attention mechanisms
- [Tropical GEMM](../tropical-gemm/index.md) provides the underlying matrix operations
