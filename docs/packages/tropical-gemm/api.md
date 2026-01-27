# Tropical GEMM API Reference

## NumPy Functions

### maxplus_matmul

```python
def maxplus_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Max-plus tropical matrix multiplication.

    C[i,j] = max_k(A[i,k] + B[k,j])

    Args:
        a: Input array of shape (M, K), float32
        b: Input array of shape (K, N), float32

    Returns:
        Output array of shape (M, N)
    """
```

### minplus_matmul

```python
def minplus_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Min-plus tropical matrix multiplication.

    C[i,j] = min_k(A[i,k] + B[k,j])

    Args:
        a: Input array of shape (M, K), float32
        b: Input array of shape (K, N), float32

    Returns:
        Output array of shape (M, N)
    """
```

### maxmul_matmul

```python
def maxmul_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Max-mul matrix multiplication.

    C[i,j] = max_k(A[i,k] * B[k,j])

    Args:
        a: Input array of shape (M, K), float32
        b: Input array of shape (K, N), float32

    Returns:
        Output array of shape (M, N)
    """
```

### maxplus_matmul_with_argmax

```python
def maxplus_matmul_with_argmax(
    a: np.ndarray, b: np.ndarray
) -> tuple[list[float], list[int]]:
    """
    Max-plus matmul returning both result and argmax indices.

    Args:
        a: Input array of shape (M, K), float32
        b: Input array of shape (K, N), float32

    Returns:
        Tuple of (result, argmax) as flattened lists
    """
```

---

## PyTorch Functions

### tropical_maxplus_matmul

```python
def tropical_maxplus_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    MaxPlus tropical matrix multiplication with autograd support.

    C[i,j] = max_k(A[i,k] + B[k,j])

    Gradients are sparse: only the argmax index contributes.

    Args:
        a: Input tensor of shape (M, K)
        b: Input tensor of shape (K, N)

    Returns:
        Output tensor of shape (M, N)
    """
```

### tropical_minplus_matmul

```python
def tropical_minplus_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    MinPlus tropical matrix multiplication with autograd support.

    C[i,j] = min_k(A[i,k] + B[k,j])
    """
```

### tropical_maxmul_matmul

```python
def tropical_maxmul_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    MaxMul matrix multiplication with autograd support.

    C[i,j] = max_k(A[i,k] * B[k,j])
    """
```

### GPU Variants

```python
def tropical_maxplus_matmul_gpu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """GPU-accelerated max-plus matmul using CUDA kernels."""

def tropical_minplus_matmul_gpu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """GPU-accelerated min-plus matmul using CUDA kernels."""

def tropical_maxmul_matmul_gpu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """GPU-accelerated max-mul matmul using CUDA kernels."""
```

---

## Constants

### GPU_AVAILABLE

```python
GPU_AVAILABLE: bool
```

`True` if CUDA kernels are available (library compiled with CUDA support and GPU detected).

---

## Gradient Behavior

Tropical operations have **sparse gradients** because only the argmax index contributes:

For $C[i,j] = \max_k(A[i,k] + B[k,j])$:

$$\frac{\partial C[i,j]}{\partial A[i,k]} = \begin{cases} 1 & \text{if } k = \arg\max_k(A[i,k] + B[k,j]) \\ 0 & \text{otherwise} \end{cases}$$

This sparsity can be beneficial for:

- Memory efficiency (fewer gradient updates)
- Implicit feature selection
- Robustness to noise
