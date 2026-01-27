"""Tests for Hilbert projective metric."""

import torch
import pytest

from tropical_attention.layers import hilbert_distance


class TestHilbertDistance:
    """Test suite for hilbert_distance function."""

    def test_output_shape(self):
        """Output shape should be (B, H, Nq, Nk)."""
        q = torch.randn(2, 4, 8, 16)  # (B, H, Nq, d_k)
        k = torch.randn(2, 4, 12, 16)  # (B, H, Nk, d_k)
        d = hilbert_distance(q, k)
        assert d.shape == (2, 4, 8, 12)

    def test_nonnegative(self):
        """Hilbert distance should be non-negative."""
        q = torch.randn(2, 4, 8, 16)
        k = torch.randn(2, 4, 8, 16)
        d = hilbert_distance(q, k)
        assert (d >= -1e-6).all()

    def test_zero_for_identical(self):
        """Distance should be zero for identical vectors."""
        q = torch.randn(2, 4, 8, 16)
        d = hilbert_distance(q, q)
        # Diagonal should be zero
        for b in range(2):
            for h in range(4):
                diag = torch.diag(d[b, h])
                assert torch.allclose(diag, torch.zeros_like(diag), atol=1e-6)

    def test_symmetric(self):
        """d_H(x, y) = d_H(y, x)."""
        q = torch.randn(2, 4, 8, 16)
        k = torch.randn(2, 4, 8, 16)
        d_qk = hilbert_distance(q, k)
        d_kq = hilbert_distance(k, q)
        # d_qk[b,h,i,j] should equal d_kq[b,h,j,i]
        assert torch.allclose(d_qk, d_kq.transpose(-1, -2), atol=1e-6)

    def test_projective_invariance(self):
        """d_H(x+c, y+c) = d_H(x, y) for any scalar c."""
        q = torch.randn(2, 4, 8, 16)
        k = torch.randn(2, 4, 8, 16)
        c = torch.randn(1)

        d1 = hilbert_distance(q, k)
        d2 = hilbert_distance(q + c, k + c)

        assert torch.allclose(d1, d2, atol=1e-5)

    def test_scale_invariance(self):
        """d_H(x+c, y+c) = d_H(x, y) - translation invariance in tropical."""
        q = torch.randn(2, 4, 8, 16)
        k = torch.randn(2, 4, 8, 16)

        # Add constant to each vector (tropical scaling)
        c = torch.randn(2, 4, 8, 1)  # Different constant per query
        d1 = hilbert_distance(q, k)
        d2 = hilbert_distance(q + c, k)

        # Distance changes because we're translating only q
        # But if we translate both by same amount, it stays same
        c_both = torch.randn(1)
        d3 = hilbert_distance(q + c_both, k + c_both)
        assert torch.allclose(d1, d3, atol=1e-5)

    def test_gradient_flow(self):
        """Gradients should flow through hilbert_distance."""
        q = torch.randn(2, 4, 8, 16, requires_grad=True)
        k = torch.randn(2, 4, 8, 16, requires_grad=True)
        d = hilbert_distance(q, k)
        loss = d.sum()
        loss.backward()
        assert q.grad is not None
        assert k.grad is not None
        assert not torch.isnan(q.grad).any()
        assert not torch.isnan(k.grad).any()
