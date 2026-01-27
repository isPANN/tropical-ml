# Packages

Tropical ML consists of several independent but interrelated packages:

## Core Libraries

### [Tropical GEMM](tropical-gemm/index.md)

High-performance tropical matrix multiplication implemented in Rust with CUDA support.

- **Max-plus matmul**: $C[i,j] = \max_k(A[i,k] + B[k,j])$
- **Min-plus matmul**: $C[i,j] = \min_k(A[i,k] + B[k,j])$
- **Max-mul matmul**: $C[i,j] = \max_k(A[i,k] \times B[k,j])$
- SIMD-optimized CPU backend
- CUDA kernels for GPU acceleration
- PyTorch autograd integration

```bash
pip install tropical-gemm[torch]
```

---

### [Tropical Attention](tropical-attention/index.md)

Multi-head attention using tropical geometry and Hilbert projective metric.

- Drop-in replacement for `nn.MultiheadAttention`
- Tropical Q, K, V projections
- Hilbert distance for attention scoring
- Full transformer encoder layer

```bash
pip install tropical-attention
```

---

## Experimental Libraries

### [Tropical Activation](tropical-activation/index.md)

Activation functions based on tropical operations.

- Tropical ReLU variants
- Max-plus nonlinearities

```bash
pip install tropical-activation
```

---

### [Tropical Pruning](tropical-pruning/index.md)

Neural network pruning using tropical geometry analysis.

- Tropical rank analysis
- Structured pruning via max-plus algebra

```bash
pip install tropical-pruning
```

---

## Package Dependencies

```
tropical-gemm (core)
    │
    ├── tropical-attention
    │       └── tropical-gemm[torch]
    │
    ├── tropical-activation
    │       └── tropical-gemm
    │
    └── tropical-pruning
            └── tropical-gemm[torch]
```
