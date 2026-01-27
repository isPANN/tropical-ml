"""Tests for TropicalMultiheadAttention."""

import torch
import pytest

from tropical_attention.layers import TropicalMultiheadAttention


class TestTropicalMultiheadAttention:
    """Test suite for TropicalMultiheadAttention."""

    def test_output_shape(self):
        """Output shape should match query shape."""
        attn = TropicalMultiheadAttention(d_model=64, num_heads=8)
        x = torch.randn(4, 16, 64)  # (batch, seq, d_model)
        out, _ = attn(x, x, x)
        assert out.shape == x.shape

    def test_output_shape_different_kv_len(self):
        """Output shape should match query shape even with different key/value length."""
        attn = TropicalMultiheadAttention(d_model=64, num_heads=8)
        q = torch.randn(4, 16, 64)
        kv = torch.randn(4, 32, 64)
        out, _ = attn(q, kv, kv)
        assert out.shape == q.shape

    def test_gradient_flow(self):
        """Gradients should flow correctly through the attention."""
        attn = TropicalMultiheadAttention(d_model=64, num_heads=8)
        x = torch.randn(4, 16, 64, requires_grad=True)
        out, _ = attn(x, x, x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_no_nan_output(self):
        """Output should not contain NaN values."""
        attn = TropicalMultiheadAttention(d_model=64, num_heads=8)
        x = torch.randn(4, 16, 64)
        out, _ = attn(x, x, x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_batch_first_false(self):
        """Should work with batch_first=False."""
        attn = TropicalMultiheadAttention(d_model=64, num_heads=8, batch_first=False)
        x = torch.randn(16, 4, 64)  # (seq, batch, d_model)
        out, _ = attn(x, x, x)
        assert out.shape == x.shape

    def test_need_weights(self):
        """Should return attention weights when requested."""
        attn = TropicalMultiheadAttention(d_model=64, num_heads=8)
        x = torch.randn(4, 16, 64)
        out, weights = attn(x, x, x, need_weights=True)
        assert weights is not None
        assert weights.shape == (4, 8, 16, 16)  # (B, H, N, S)

    def test_key_padding_mask(self):
        """Should apply key padding mask correctly."""
        attn = TropicalMultiheadAttention(d_model=64, num_heads=8)
        x = torch.randn(4, 16, 64)
        # Mask last 4 positions in each sequence
        mask = torch.zeros(4, 16, dtype=torch.bool)
        mask[:, -4:] = True
        out, weights = attn(x, x, x, key_padding_mask=mask, need_weights=True)
        # Masked positions should have -inf scores
        assert (weights[:, :, :, -4:] == float("-inf")).all()

    def test_dropout_during_training(self):
        """Dropout should be applied during training."""
        attn = TropicalMultiheadAttention(d_model=64, num_heads=8, dropout=0.5)
        attn.train()
        x = torch.randn(4, 16, 64)
        out1, _ = attn(x, x, x)
        out2, _ = attn(x, x, x)
        # Outputs should differ due to dropout
        assert not torch.allclose(out1, out2)

    def test_no_dropout_during_eval(self):
        """Dropout should not be applied during eval."""
        attn = TropicalMultiheadAttention(d_model=64, num_heads=8, dropout=0.5)
        attn.eval()
        x = torch.randn(4, 16, 64)
        out1, _ = attn(x, x, x)
        out2, _ = attn(x, x, x)
        # Outputs should be identical
        assert torch.allclose(out1, out2)

    def test_d_model_num_heads_compatibility(self):
        """Should raise error if d_model not divisible by num_heads."""
        with pytest.raises(AssertionError):
            TropicalMultiheadAttention(d_model=64, num_heads=7)

    def test_parameter_count(self):
        """Should have expected number of parameters."""
        d_model = 64
        attn = TropicalMultiheadAttention(d_model=d_model, num_heads=8, bias=True)
        # Q, K, V projections: 3 * (d_model * d_model + d_model for bias)
        # Output projection: d_model * d_model + d_model for bias
        expected_params = 3 * (d_model * d_model + d_model) + (d_model * d_model + d_model)
        actual_params = sum(p.numel() for p in attn.parameters())
        assert actual_params == expected_params
