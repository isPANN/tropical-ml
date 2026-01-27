# TropicalTransformerEncoderLayer

Complete transformer encoder layer using Tropical Attention.

## Overview

`TropicalTransformerEncoderLayer` follows the standard Transformer architecture but replaces the self-attention mechanism with `TropicalMultiheadAttention`.

```
┌─────────────────────────────────────┐
│           Input (src)               │
└───────────────┬─────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│    TropicalMultiheadAttention       │
│         (self-attention)            │
└───────────────┬─────────────────────┘
                │
        ┌───────┴───────┐
        │   Add & Norm  │ ← Residual connection
        └───────┬───────┘
                │
                ▼
┌─────────────────────────────────────┐
│      Feedforward Network            │
│   Linear → Activation → Linear      │
└───────────────┬─────────────────────┘
                │
        ┌───────┴───────┐
        │   Add & Norm  │ ← Residual connection
        └───────┬───────┘
                │
                ▼
┌─────────────────────────────────────┐
│            Output                   │
└─────────────────────────────────────┘
```

## API Reference

::: tropical_attention.models.transformer.TropicalTransformerEncoderLayer
    options:
      show_root_heading: true
      show_source: true
      members:
        - __init__
        - forward

## Usage Examples

### Single Layer

```python
import torch
from tropical_attention import TropicalTransformerEncoderLayer

layer = TropicalTransformerEncoderLayer(
    d_model=512,
    nhead=8,
    dim_feedforward=2048,
    dropout=0.1,
    activation="relu",
    batch_first=True,
)

x = torch.randn(32, 100, 512)
output = layer(x)  # (32, 100, 512)
```

### Stacked Encoder

```python
import torch.nn as nn

class TropicalEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward=2048):
        super().__init__()
        self.layers = nn.ModuleList([
            TropicalTransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
            )
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=mask)
        return x

encoder = TropicalEncoder(d_model=512, nhead=8, num_layers=6)
output = encoder(x)
```

### With Masking

```python
# Padding mask
padding_mask = torch.zeros(32, 100, dtype=torch.bool)
padding_mask[:, 80:] = True

output = layer(x, src_key_padding_mask=padding_mask)

# Causal mask for autoregressive
causal_mask = torch.triu(torch.ones(100, 100), diagonal=1).bool()
output = layer(x, src_mask=causal_mask)
```

### Different Activations

```python
# ReLU (default)
layer_relu = TropicalTransformerEncoderLayer(
    d_model=512, nhead=8, activation="relu"
)

# GELU (often better for NLP)
layer_gelu = TropicalTransformerEncoderLayer(
    d_model=512, nhead=8, activation="gelu"
)
```

## Complete Model Example

```python
import torch
import torch.nn as nn
from tropical_attention import TropicalTransformerEncoderLayer

class TropicalClassifier(nn.Module):
    """Text classifier using Tropical Transformer."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        num_classes: int = 10,
        max_seq_len: int = 512,
    ):
        super().__init__()

        # Embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_seq_len, d_model) * 0.02
        )

        # Tropical encoder
        self.encoder_layers = nn.ModuleList([
            TropicalTransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=0.1,
            )
            for _ in range(num_layers)
        ])

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x, padding_mask=None):
        # x: (batch, seq_len) token ids
        seq_len = x.size(1)

        # Embed
        h = self.embedding(x) + self.pos_embedding[:, :seq_len]

        # Encode
        for layer in self.encoder_layers:
            h = layer(h, src_key_padding_mask=padding_mask)

        # Pool and classify
        h = h.mean(dim=1)  # Global average pooling
        return self.classifier(h)

# Usage
model = TropicalClassifier(vocab_size=30000, num_classes=5)
tokens = torch.randint(0, 30000, (32, 128))
logits = model(tokens)  # (32, 5)
```

## Comparison with nn.TransformerEncoderLayer

| Component | nn.TransformerEncoderLayer | TropicalTransformerEncoderLayer |
|-----------|---------------------------|--------------------------------|
| Self-attention | nn.MultiheadAttention | TropicalMultiheadAttention |
| Feedforward | Same | Same |
| LayerNorm | Same | Same |
| Residual | Same | Same |

The only difference is the attention mechanism - all other components are identical.
