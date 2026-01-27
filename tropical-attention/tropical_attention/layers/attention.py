"""
Multi-Head Tropical Attention (MHTA).

Attention mechanism using tropical geometry and Hilbert projective metric.
Drop-in replacement for nn.MultiheadAttention.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .tropicalize import Tropicalize, Detropicalize
from .hilbert import hilbert_distance
from .tropical_linear import TropicalLinear

# Import tropical-gemm backend
try:
    from tropical_gemm.pytorch import (
        tropical_maxplus_matmul,
        tropical_maxplus_matmul_gpu,
        GPU_AVAILABLE,
    )

    TROPICAL_GEMM_AVAILABLE = True
except ImportError:
    TROPICAL_GEMM_AVAILABLE = False
    GPU_AVAILABLE = False


class TropicalMultiheadAttention(nn.Module):
    """
    Multi-Head Tropical Attention (MHTA).

    Drop-in replacement for nn.MultiheadAttention using tropical geometry.

    Architecture:
        1. Tropicalize inputs: Z = log(clamp(X)) - max
        2. Tropical projections: Q, K, V via max-plus matmul
        3. Attention scores: S[i,j] = -hilbert_distance(Q[i], K[j])
        4. Tropical aggregation: C = max-plus matmul(S, V)
        5. De-tropicalize: H = exp(C)
        6. Output projection: out = H @ W_O (standard matmul)

    Args:
        d_model: Total dimension of the model
        num_heads: Number of attention heads
        dropout: Dropout probability
        bias: If True, add bias to projections
        batch_first: If True, input/output is (batch, seq, d_model)

    Shape:
        - query: (B, N, d_model) if batch_first else (N, B, d_model)
        - key: (B, S, d_model) if batch_first else (S, B, d_model)
        - value: (B, S, d_model) if batch_first else (S, B, d_model)
        - output: same shape as query
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        batch_first: bool = True,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.batch_first = batch_first

        # Tropicalization
        self.tropicalize = Tropicalize()
        self.detropicalize = Detropicalize()

        # Tropical Q, K, V projections
        self.q_proj = TropicalLinear(d_model, d_model, bias=bias)
        self.k_proj = TropicalLinear(d_model, d_model, bias=bias)
        self.v_proj = TropicalLinear(d_model, d_model, bias=bias)

        # Standard output projection (NOT tropical)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            query: (B, N, d_model) if batch_first else (N, B, d_model)
            key: (B, S, d_model) if batch_first else (S, B, d_model)
            value: (B, S, d_model) if batch_first else (S, B, d_model)
            key_padding_mask: (B, S) True = ignore
            attn_mask: (N, S) or (B*H, N, S)
            need_weights: If True, return attention weights

        Returns:
            output: same shape as query
            attn_weights: (B, H, N, S) if need_weights else None
        """
        # Handle batch_first
        if not self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        B, N, _ = query.shape
        S = key.shape[1]
        H = self.num_heads
        d_k = self.head_dim

        # Step 1: Tropicalize inputs
        q_trop = self.tropicalize(query)
        k_trop = self.tropicalize(key)
        v_trop = self.tropicalize(value)

        # Step 2: Tropical projections
        Q = self.q_proj(q_trop)  # (B, N, d_model)
        K = self.k_proj(k_trop)  # (B, S, d_model)
        V = self.v_proj(v_trop)  # (B, S, d_model)

        # Step 3: Reshape for multi-head
        Q = Q.view(B, N, H, d_k).transpose(1, 2)  # (B, H, N, d_k)
        K = K.view(B, S, H, d_k).transpose(1, 2)  # (B, H, S, d_k)
        V = V.view(B, S, H, d_k).transpose(1, 2)  # (B, H, S, d_k)

        # Step 4: Compute attention scores via Hilbert distance
        # Scores = -d_H(Q, K), so closer = higher score
        scores = -hilbert_distance(Q, K)  # (B, H, N, S)

        # Step 5: Apply masks
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, float("-inf"))
        if key_padding_mask is not None:
            # (B, S) -> (B, 1, 1, S)
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, float("-inf"))

        # Step 6: Tropical aggregation (max-plus matmul)
        # C[b,h,i,:] = max_j(S[b,h,i,j] + V[b,h,j,:])
        # Reshape for batched tropical matmul
        scores_flat = scores.reshape(B * H, N, S)  # (B*H, N, S)
        V_flat = V.reshape(B * H, S, d_k)  # (B*H, S, d_k)

        if query.is_cuda and GPU_AVAILABLE:
            C_flat = self._batched_tropical_matmul_gpu(scores_flat, V_flat)
        else:
            C_flat = self._batched_tropical_matmul(scores_flat, V_flat)

        C = C_flat.view(B, H, N, d_k)  # (B, H, N, d_k)

        # Step 7: De-tropicalize and reshape
        C = self.detropicalize(C)
        C = C.transpose(1, 2).reshape(B, N, self.d_model)  # (B, N, d_model)

        # Step 8: Output projection (standard, not tropical)
        output = self.out_proj(C)
        output = self.dropout(output)

        # Handle batch_first for output
        if not self.batch_first:
            output = output.transpose(0, 1)

        attn_weights = scores if need_weights else None
        return output, attn_weights

    def _batched_tropical_matmul(
        self, a: torch.Tensor, b: torch.Tensor
    ) -> torch.Tensor:
        """Batched tropical matmul for CPU."""
        batch_size = a.shape[0]
        results = []
        for i in range(batch_size):
            results.append(tropical_maxplus_matmul(a[i], b[i]))
        return torch.stack(results)

    def _batched_tropical_matmul_gpu(
        self, a: torch.Tensor, b: torch.Tensor
    ) -> torch.Tensor:
        """Batched tropical matmul for GPU."""
        batch_size = a.shape[0]
        results = []
        for i in range(batch_size):
            results.append(tropical_maxplus_matmul_gpu(a[i], b[i]))
        return torch.stack(results)

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, num_heads={self.num_heads}, batch_first={self.batch_first}"


__all__ = ["TropicalMultiheadAttention"]
