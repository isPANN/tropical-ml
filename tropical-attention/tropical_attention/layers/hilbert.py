"""
Hilbert Projective Metric.

The Hilbert projective metric is a natural distance on tropical projective space.
It measures how "different" two tropical vectors are up to scaling.
"""

import torch


def hilbert_distance(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise Hilbert projective distances.

    The Hilbert projective distance is defined as:
        d_H(x, y) = max_i(x_i - y_i) - min_i(x_i - y_i)

    This metric is:
    - Projective: d_H(x + c, y + c) = d_H(x, y) for any scalar c
    - Symmetric: d_H(x, y) = d_H(y, x)
    - Non-negative: d_H(x, y) >= 0, with equality iff x = y + c

    Args:
        q: (batch, heads, seq_q, d_k) - query vectors in tropical space
        k: (batch, heads, seq_k, d_k) - key vectors in tropical space

    Returns:
        distances: (batch, heads, seq_q, seq_k) - pairwise Hilbert distances
    """
    # Expand for pairwise computation
    # q: (B, H, Nq, 1, d_k)
    # k: (B, H, 1, Nk, d_k)
    q_exp = q.unsqueeze(-2)
    k_exp = k.unsqueeze(-3)

    # Difference: (B, H, Nq, Nk, d_k)
    diff = q_exp - k_exp

    # Hilbert distance = max - min along last dim
    d_h = diff.max(dim=-1).values - diff.min(dim=-1).values

    return d_h


__all__ = ["hilbert_distance"]
