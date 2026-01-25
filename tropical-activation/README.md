# Tropical Activation

Min-Max-Plus Neural Networks using Tropical Algebra.

## Installation

```bash
pip install tropical-activation
```

## Quick Start

```python
import torch
from tropical_activation import MMPNN, MaxPlusLayer

# Create an MMP classifier
model = MMPNN([784, 256, 128, 10])

# Forward pass
x = torch.randn(32, 784)
logits = model(x)

# Or use individual layers
layer = MaxPlusLayer(64, 128)
output = layer(torch.randn(32, 64))
```

## Model Architectures

### Baseline (Standard ReLU Network)

```
┌───────┐     ┌───────┐     ┌───────┐     ┌───────┐     ┌───────┐
│ Input │────▶│Linear │────▶│ ReLU  │────▶│Linear │────▶│ ReLU  │────▶ Output
└───────┘     └───────┘     └───────┘     └───────┘     └───────┘
                 W×x           max(x,0)       W×x          max(x,0)

              ╔═══════════════════════════════════════════════════╗
              ║  Multiplications: Many (every Linear layer)       ║
              ╚═══════════════════════════════════════════════════╝
```

### Hybrid (Linear + MaxPlus + MinPlus) - Recommended

```
┌───────┐     ┌───────┐     ┌─────────┐     ┌─────────┐     ┌───────┐
│ Input │────▶│Linear │────▶│ MaxPlus │────▶│ MinPlus │────▶│Linear │────▶ ...
└───────┘     └───────┘     └─────────┘     └─────────┘     └───────┘
                 W×x         max(x+W)         min(x+W)         W×x

              ╔═══════════════════════════════════════════════════╗
              ║  Multiplications: Some (Linear layers only)       ║
              ║  Tropical layers use addition + max/min           ║
              ╚═══════════════════════════════════════════════════╝
```

### Tropical (Minimal Multiplications)

```
┌───────┐     ┌───────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
│ Input │────▶│Linear │────▶│ MaxPlus │────▶│ MinPlus │────▶│ MaxPlus │────▶ ...
└───────┘     └───────┘     └─────────┘     └─────────┘     └─────────┘
              (first only)   max(x+W)         min(x+W)        max(x+W)

              ╔═══════════════════════════════════════════════════╗
              ║  Multiplications: Minimal (first layer only)      ║
              ║  Rest of network is multiplication-free           ║
              ╚═══════════════════════════════════════════════════╝
```

### Pure (Zero Multiplications)

```
┌───────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
│ Input │────▶│ MaxPlus │────▶│ MinPlus │────▶│ MaxPlus │────▶│ MinPlus │────▶ Output
└───────┘     └─────────┘     └─────────┘     └─────────┘     └─────────┘
               max(x+W)         min(x+W)        max(x+W)         min(x+W)

              ╔═══════════════════════════════════════════════════╗
              ║  Multiplications: ZERO                            ║
              ║  Only additions and max/min comparisons           ║
              ╚═══════════════════════════════════════════════════╝
```

### Architecture Comparison

| Model | Linear Layers | Tropical Layers | Multiplications | Use Case |
|-------|---------------|-----------------|-----------------|----------|
| `baseline` | All | None | Many | Standard baseline |
| `hybrid` | Alternating | Alternating | Some | Best accuracy (recommended) |
| `tropical` | First only | Rest | Minimal | Efficiency + accuracy |
| `pure` | None | All | **Zero** | Maximum efficiency |

## Mathematical Foundation

**Standard Linear Layer:**
```
y = W × x + b       (uses multiplication)
```

**MaxPlus Layer (Tropical):**
```
y_j = max_k(x_k + W_kj) + b_j    (addition + max only)
```

**MinPlus Layer (Tropical):**
```
y_j = min_k(x_k + W_kj) + b_j    (addition + min only)
```

## Training

```bash
# Train on MNIST
python examples/train_mnist.py --model hybrid --epochs 20

# Compare architectures
python examples/train_mnist.py --model baseline --epochs 20
python examples/train_mnist.py --model tropical --epochs 20
python examples/train_mnist.py --model pure --epochs 20

# Train on CIFAR-10
python examples/train_cifar10.py --model mmp --epochs 200

# Train on ImageNet (distributed)
torchrun --nproc_per_node=4 examples/train_imagenet.py --data /path/to/imagenet --amp
```

## Reference

Luo & Fan 2021 - "Min-Max-Plus Neural Networks"
https://arxiv.org/abs/2102.06358

## License

MIT
