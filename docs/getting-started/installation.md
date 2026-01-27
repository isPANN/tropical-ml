# Installation

## Requirements

- Python 3.9 or higher
- NumPy 1.20 or higher

Additional requirements vary by package.

## Package Installation

### Tropical GEMM

The core library for tropical matrix operations:

```bash
pip install tropical-gemm
```

With PyTorch integration:

```bash
pip install tropical-gemm[torch]
```

With CUDA support (requires CUDA toolkit):

```bash
pip install tropical-gemm[cuda]
```

### Tropical Attention

Attention mechanisms using tropical geometry:

```bash
pip install tropical-attention
```

This automatically installs `tropical-gemm[torch]` as a dependency.

### Development Installation

For development, clone the monorepo:

```bash
git clone https://github.com/isPANN/tropical-ml.git
cd tropical-ml

# Install specific packages in development mode
pip install -e ./tropical-gemm[torch]
pip install -e ./tropical-attention[dev]
```

## Verifying Installation

```python
# Check tropical-gemm
import tropical_gemm
print(f"tropical-gemm version: {tropical_gemm.__version__}")
print(f"CUDA available: {tropical_gemm.cuda_available()}")

# Check tropical-attention
import tropical_attention
print(f"tropical-attention version: {tropical_attention.__version__}")

# Quick test
import torch
from tropical_attention import TropicalMultiheadAttention

attn = TropicalMultiheadAttention(d_model=64, num_heads=4)
x = torch.randn(2, 10, 64)
output, _ = attn(x, x, x)
print(f"Output shape: {output.shape}")  # (2, 10, 64)
```

## GPU Support

### Check GPU Availability

```python
from tropical_gemm.pytorch import GPU_AVAILABLE
print(f"GPU kernels available: {GPU_AVAILABLE}")
```

### CUDA Requirements

For GPU acceleration:

1. CUDA toolkit 11.0+
2. Compatible NVIDIA GPU
3. `tropical-gemm` built with CUDA support

### Troubleshooting GPU Issues

If `GPU_AVAILABLE` is `False`:

1. Verify CUDA installation: `nvcc --version`
2. Check PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
3. Reinstall tropical-gemm with CUDA: `pip install --force-reinstall tropical-gemm[cuda]`

## Building Documentation

To build docs locally:

```bash
pip install mkdocs-material mkdocstrings[python]
mkdocs serve
```

Docs will be available at `http://127.0.0.1:8000/`.
