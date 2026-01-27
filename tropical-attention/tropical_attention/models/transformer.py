"""
Tropical Transformer Encoder Layer.

Transformer encoder layer using Tropical Attention instead of standard attention.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers.attention import TropicalMultiheadAttention


class TropicalTransformerEncoderLayer(nn.Module):
    """
    Transformer encoder layer using Tropical Attention.

    Architecture follows standard Transformer but replaces self-attention
    with TropicalMultiheadAttention:
        1. Tropical self-attention with residual + LayerNorm
        2. Standard feedforward with residual + LayerNorm

    Args:
        d_model: Total dimension of the model
        nhead: Number of attention heads
        dim_feedforward: Dimension of feedforward network (default: 2048)
        dropout: Dropout probability (default: 0.1)
        activation: Activation function ("relu" or "gelu", default: "relu")
        batch_first: If True, input/output is (batch, seq, d_model)

    Shape:
        - src: (B, N, d_model) if batch_first else (N, B, d_model)
        - output: same shape as src
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        batch_first: bool = True,
    ):
        super().__init__()

        # Tropical attention instead of standard
        self.self_attn = TropicalMultiheadAttention(
            d_model=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=batch_first,
        )

        # Standard feedforward (NOT tropical)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            src: Source sequence (B, N, d_model) if batch_first
            src_mask: Attention mask (N, N) or (B*H, N, N)
            src_key_padding_mask: Padding mask (B, N), True = ignore

        Returns:
            output: Transformed sequence, same shape as src
        """
        # Self-attention with residual
        src2, _ = self.self_attn(
            src,
            src,
            src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )
        src = self.norm1(src + self.dropout1(src2))

        # Feedforward with residual
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = self.norm2(src + self.dropout2(src2))

        return src

    def extra_repr(self) -> str:
        return f"d_model={self.self_attn.d_model}, nhead={self.self_attn.num_heads}"


__all__ = ["TropicalTransformerEncoderLayer"]
