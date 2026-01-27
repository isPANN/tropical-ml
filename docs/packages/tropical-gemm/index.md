# Tropical GEMM

**High-Performance Tropical Matrix Multiplication**

## Overview

Tropical GEMM provides optimized implementations of tropical (max-plus and min-plus) matrix multiplication in Rust, with Python bindings and PyTorch integration.

## Installation

```bash
# Basic installation
pip install tropical-gemm

# With PyTorch support
pip install tropical-gemm[torch]

# With CUDA support
pip install tropical-gemm[cuda]
```

## Operations

### Max-Plus Matmul

$$C[i,j] = \max_k(A[i,k] + B[k,j])$$

```python
import numpy as np
from tropical_gemm import maxplus_matmul

a = np.random.randn(100, 50).astype(np.float32)
b = np.random.randn(50, 80).astype(np.float32)
c = maxplus_matmul(a, b)
```

### Min-Plus Matmul

$$C[i,j] = \min_k(A[i,k] + B[k,j])$$

```python
from tropical_gemm import minplus_matmul

c = minplus_matmul(a, b)
```

### Max-Mul Matmul

$$C[i,j] = \max_k(A[i,k] \times B[k,j])$$

```python
from tropical_gemm import maxmul_matmul

c = maxmul_matmul(a, b)
```

## PyTorch Integration

Full autograd support for gradient-based learning:

```python
import torch
from tropical_gemm.pytorch import (
    tropical_maxplus_matmul,
    tropical_minplus_matmul,
    tropical_maxmul_matmul,
    GPU_AVAILABLE,
)

a = torch.randn(100, 50, requires_grad=True)
b = torch.randn(50, 80, requires_grad=True)

c = tropical_maxplus_matmul(a, b)
loss = c.sum()
loss.backward()  # Gradients computed via argmax
```

### GPU Acceleration

```python
if GPU_AVAILABLE:
    from tropical_gemm.pytorch import tropical_maxplus_matmul_gpu

    a = torch.randn(100, 50, device='cuda')
    b = torch.randn(50, 80, device='cuda')
    c = tropical_maxplus_matmul_gpu(a, b)
```

## Performance

The Rust backend uses:

- **SIMD vectorization** for CPU (AVX2/AVX-512)
- **CUDA kernels** for GPU
- **Cache-friendly tiling** for large matrices

## API Reference

See [API Reference](api.md) for detailed documentation.
