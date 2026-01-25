# Tropical Activation

Neural Networks with Tropical Algebra.

Replaces ReLU/activations with tropical affine layers (max/min + addition only).

## Installation

```bash
pip install tropical-activation
```

## Quick Start

```python
import torch
from tropical_activation import TropicalNN, MaxPlusAffine

# Tropical classifier
model = TropicalNN([784, 256, 128, 10])
x = torch.randn(32, 784)
logits = model(x)

# Individual layer (square matrix, acts as activation)
layer = MaxPlusAffine(256)  # 256 → 256
output = layer(torch.randn(32, 256))
```

## Architecture

```
┌───────┐     ┌───────┐     ┌─────────┐     ┌─────────┐     ┌───────┐
│ Input │────▶│Linear │────▶│ MaxPlus │────▶│ MinPlus │────▶│Linear │────▶ Output
└───────┘     └───────┘     └─────────┘     └─────────┘     └───────┘
                 W×x        max(x+W,b)       min(x+W,b)         W×x
```

**Key design:**
- Linear layers handle dimension changes
- Tropical layers are **square** (activation replacement)
- LayerNorm before tropical operation (stabilizes training)
- Bias as **threshold** via max/min (true tropical affine)

## Layers

**MaxPlusAffine:**
```
y[i] = max(max_k(LayerNorm(x)[k] + W[k,i]), b[i])
```

**MinPlusAffine:**
```
y[i] = min(min_k(LayerNorm(x)[k] + W[k,i]), b[i])
```

## Why Tropical?

| Operation | ReLU | Tropical |
|-----------|------|----------|
| Forward | multiply + max | **add + max/min** |
| Multiplications | Many | Fewer |
| Hardware | Standard | Efficient (add/max only) |

## Training

```bash
# MNIST
python examples/train_mnist.py --model tropical --epochs 15
python examples/train_mnist.py --model baseline --epochs 15

# CIFAR-10
python examples/train_cifar10.py --model tropical --epochs 200
```

## Reference

Luo & Fan 2021 - "Min-Max-Plus Neural Networks"
https://arxiv.org/abs/2102.06358

## License

MIT
