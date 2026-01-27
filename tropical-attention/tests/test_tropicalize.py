"""Tests for Tropicalize and Detropicalize layers."""

import torch
import pytest

from tropical_attention.layers import Tropicalize, Detropicalize


class TestTropicalize:
    """Test suite for Tropicalize layer."""

    def test_output_shape(self):
        """Output shape should match input shape."""
        layer = Tropicalize()
        x = torch.randn(4, 16, 64).abs() + 0.1  # Positive values
        z = layer(x)
        assert z.shape == x.shape

    def test_max_is_zero(self):
        """Max coordinate should be zero (tropical simplex)."""
        layer = Tropicalize()
        x = torch.randn(4, 16, 64).abs() + 0.1
        z = layer(x)
        max_vals = z.max(dim=-1).values
        assert torch.allclose(max_vals, torch.zeros_like(max_vals), atol=1e-6)

    def test_all_nonpositive(self):
        """All values should be <= 0 after normalization."""
        layer = Tropicalize()
        x = torch.randn(4, 16, 64).abs() + 0.1
        z = layer(x)
        assert (z <= 1e-6).all()

    def test_handles_negative_inputs(self):
        """Should clamp negative inputs to eps."""
        layer = Tropicalize(eps=1e-8)
        x = torch.randn(4, 16, 64)  # May contain negatives
        z = layer(x)
        assert not torch.isnan(z).any()
        assert not torch.isinf(z).any()


class TestDetropicalize:
    """Test suite for Detropicalize layer."""

    def test_output_shape(self):
        """Output shape should match input shape."""
        layer = Detropicalize()
        z = torch.randn(4, 16, 64)
        x = layer(z)
        assert x.shape == z.shape

    def test_output_positive(self):
        """Output should be positive (exp of any real is positive)."""
        layer = Detropicalize()
        z = torch.randn(4, 16, 64)
        x = layer(z)
        assert (x > 0).all()

    def test_normalized_input(self):
        """For normalized tropical input (max=0), max output should be 1."""
        layer = Detropicalize()
        z = torch.randn(4, 16, 64)
        z = z - z.max(dim=-1, keepdim=True).values  # Normalize
        x = layer(z)
        max_vals = x.max(dim=-1).values
        assert torch.allclose(max_vals, torch.ones_like(max_vals), atol=1e-6)


class TestRoundTrip:
    """Test tropicalize -> detropicalize round trip."""

    def test_roundtrip_preserves_ratios(self):
        """Round trip should preserve ratios between coordinates."""
        tropicalize = Tropicalize()
        detropicalize = Detropicalize()

        x = torch.randn(4, 16, 64).abs() + 0.1
        z = tropicalize(x)
        x_recovered = detropicalize(z)

        # After round trip, values are normalized (max = 1)
        # Check that ratios are preserved
        x_normalized = x / x.max(dim=-1, keepdim=True).values
        assert torch.allclose(x_recovered, x_normalized, atol=1e-5)
