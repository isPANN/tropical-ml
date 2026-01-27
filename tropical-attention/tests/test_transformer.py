"""Tests for TropicalTransformerEncoderLayer."""

import torch
import pytest

from tropical_attention.models import TropicalTransformerEncoderLayer


class TestTropicalTransformerEncoderLayer:
    """Test suite for TropicalTransformerEncoderLayer."""

    def test_output_shape(self):
        """Output shape should match input shape."""
        layer = TropicalTransformerEncoderLayer(d_model=64, nhead=8)
        x = torch.randn(4, 16, 64)  # (batch, seq, d_model)
        out = layer(x)
        assert out.shape == x.shape

    def test_gradient_flow(self):
        """Gradients should flow correctly through the layer."""
        layer = TropicalTransformerEncoderLayer(d_model=64, nhead=8)
        x = torch.randn(4, 16, 64, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_no_nan_output(self):
        """Output should not contain NaN values."""
        layer = TropicalTransformerEncoderLayer(d_model=64, nhead=8)
        x = torch.randn(4, 16, 64)
        out = layer(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_with_mask(self):
        """Should work with attention mask."""
        layer = TropicalTransformerEncoderLayer(d_model=64, nhead=8)
        x = torch.randn(4, 16, 64)
        # Causal mask
        mask = torch.triu(torch.ones(16, 16, dtype=torch.bool), diagonal=1)
        out = layer(x, src_mask=mask)
        assert out.shape == x.shape
        assert not torch.isnan(out).any()

    def test_with_padding_mask(self):
        """Should work with key padding mask."""
        layer = TropicalTransformerEncoderLayer(d_model=64, nhead=8)
        x = torch.randn(4, 16, 64)
        # Mask last 4 positions
        padding_mask = torch.zeros(4, 16, dtype=torch.bool)
        padding_mask[:, -4:] = True
        out = layer(x, src_key_padding_mask=padding_mask)
        assert out.shape == x.shape
        assert not torch.isnan(out).any()

    def test_gelu_activation(self):
        """Should work with GELU activation."""
        layer = TropicalTransformerEncoderLayer(d_model=64, nhead=8, activation="gelu")
        x = torch.randn(4, 16, 64)
        out = layer(x)
        assert out.shape == x.shape

    def test_invalid_activation(self):
        """Should raise error for invalid activation."""
        with pytest.raises(ValueError):
            TropicalTransformerEncoderLayer(d_model=64, nhead=8, activation="invalid")

    def test_custom_feedforward_dim(self):
        """Should work with custom feedforward dimension."""
        layer = TropicalTransformerEncoderLayer(
            d_model=64, nhead=8, dim_feedforward=1024
        )
        x = torch.randn(4, 16, 64)
        out = layer(x)
        assert out.shape == x.shape

    def test_stacked_layers(self):
        """Multiple layers should stack correctly."""
        layers = torch.nn.ModuleList([
            TropicalTransformerEncoderLayer(d_model=64, nhead=8)
            for _ in range(3)
        ])
        x = torch.randn(4, 16, 64)
        for layer in layers:
            x = layer(x)
        assert x.shape == (4, 16, 64)
        assert not torch.isnan(x).any()

    def test_dropout_during_training(self):
        """Dropout should be applied during training."""
        layer = TropicalTransformerEncoderLayer(d_model=64, nhead=8, dropout=0.5)
        layer.train()
        x = torch.randn(4, 16, 64)
        out1 = layer(x)
        out2 = layer(x)
        # Outputs should differ due to dropout
        assert not torch.allclose(out1, out2)

    def test_deterministic_during_eval(self):
        """Output should be deterministic during eval."""
        layer = TropicalTransformerEncoderLayer(d_model=64, nhead=8, dropout=0.5)
        layer.eval()
        x = torch.randn(4, 16, 64)
        out1 = layer(x)
        out2 = layer(x)
        # Outputs should be identical
        assert torch.allclose(out1, out2)
