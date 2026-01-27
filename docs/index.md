# Tropical ML

**Tropical Geometry for Machine Learning**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

Tropical ML is a collection of libraries that bring **tropical geometry** and **max-plus algebra** to machine learning. These mathematical structures provide alternative computational primitives that can be more efficient and interpretable for certain tasks.

## Packages

| Package | Description | Status |
|---------|-------------|--------|
| [**tropical-gemm**](packages/tropical-gemm/index.md) | High-performance tropical matrix multiplication (Rust + CUDA) | Stable |
| [**tropical-attention**](packages/tropical-attention/index.md) | Multi-head attention using Hilbert projective metric | Stable |
| [**tropical-activation**](packages/tropical-activation/index.md) | Tropical activation functions | Experimental |
| [**tropical-pruning**](packages/tropical-pruning/index.md) | Neural network pruning via tropical geometry | Experimental |

## What is Tropical Geometry?

Tropical geometry replaces standard arithmetic with the **max-plus semiring**:

| Standard | Tropical (Max-Plus) |
|----------|---------------------|
| $a + b$  | $\max(a, b)$        |
| $a \times b$ | $a + b$         |

This seemingly simple change has profound implications:

- **Matrix multiplication** becomes finding longest/shortest paths
- **Polynomials** become piecewise-linear functions
- **Algebraic varieties** become polyhedral complexes

### Why Tropical ML?

1. **Efficiency**: Max and addition are simpler than multiply-accumulate
2. **Interpretability**: Tropical operations have geometric meaning
3. **Sparsity**: Gradients are naturally sparse (only argmax contributes)
4. **Robustness**: Max operations are less sensitive to outliers

## Quick Start

=== "Tropical Attention"

    ```python
    import torch
    from tropical_attention import TropicalMultiheadAttention

    attn = TropicalMultiheadAttention(d_model=512, num_heads=8)
    x = torch.randn(32, 100, 512)
    output, _ = attn(x, x, x)
    ```

=== "Tropical GEMM"

    ```python
    import numpy as np
    from tropical_gemm import maxplus_matmul

    a = np.random.randn(100, 50).astype(np.float32)
    b = np.random.randn(50, 80).astype(np.float32)
    c = maxplus_matmul(a, b)  # c[i,j] = max_k(a[i,k] + b[k,j])
    ```

## Installation

```bash
# Core packages
pip install tropical-gemm
pip install tropical-attention

# With GPU support
pip install tropical-gemm[cuda]
```

See [Installation Guide](getting-started/installation.md) for detailed instructions.

## License

MIT License - see individual packages for details.
