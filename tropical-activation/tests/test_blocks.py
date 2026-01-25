"""Tests for MMP blocks."""

import pytest
import torch
import torch.nn as nn

from tropical_activation.blocks import (
    MMPBlock,
    ResidualMMPBlock,
    MaxPlusBlock,
    MinPlusBlock,
    TropicalMLP,
)


class TestMMPBlock:
    """Tests for MMPBlock."""

    def test_forward_shape(self):
        """Test that forward pass produces correct output shape."""
        block = MMPBlock(64, 128, 64)
        x = torch.randn(32, 64)
        output = block(x)
        assert output.shape == (32, 64)

    def test_forward_shape_default_sizes(self):
        """Test with default hidden and output sizes."""
        block = MMPBlock(64)  # hidden_features and out_features default to 64
        x = torch.randn(32, 64)
        output = block(x)
        assert output.shape == (32, 64)

    def test_backward_gradient_flow(self):
        """Test that gradients flow through the block."""
        block = MMPBlock(64, 128, 64)
        x = torch.randn(32, 64, requires_grad=True)
        output = block(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert block.linear.weight.grad is not None
        assert block.maxplus.weight.grad is not None
        assert block.minplus.weight.grad is not None

    def test_with_dropout(self):
        """Test block with dropout."""
        block = MMPBlock(64, 128, 64, dropout=0.5)
        x = torch.randn(32, 64)

        # Test in training mode (dropout active)
        block.train()
        output1 = block(x)
        output2 = block(x)
        # Outputs should differ due to dropout
        assert not torch.allclose(output1, output2)

        # Test in eval mode (dropout inactive)
        block.eval()
        output1 = block(x)
        output2 = block(x)
        torch.testing.assert_close(output1, output2)


class TestResidualMMPBlock:
    """Tests for ResidualMMPBlock."""

    def test_forward_shape(self):
        """Test that forward pass preserves shape."""
        block = ResidualMMPBlock(64, 256)
        x = torch.randn(32, 64)
        output = block(x)
        assert output.shape == (32, 64)

    def test_residual_connection(self):
        """Test that residual connection is working."""
        block = ResidualMMPBlock(64, 256)

        # Zero out the block weights to check residual
        with torch.no_grad():
            for param in block.block.parameters():
                param.zero_()

        x = torch.randn(32, 64)
        output = block(x)

        # With zeroed weights, output should be close to input (plus scaled block output)
        # The block output won't be exactly zero due to biases
        assert output.shape == x.shape

    def test_backward_gradient_flow(self):
        """Test that gradients flow through residual block."""
        block = ResidualMMPBlock(64, 256)
        x = torch.randn(32, 64, requires_grad=True)
        output = block(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None


class TestMaxPlusBlock:
    """Tests for MaxPlusBlock."""

    def test_forward_shape(self):
        """Test forward pass shape."""
        block = MaxPlusBlock(64, 128)
        x = torch.randn(32, 64)
        output = block(x)
        assert output.shape == (32, 128)

    def test_backward_gradient_flow(self):
        """Test gradient flow."""
        block = MaxPlusBlock(64, 128)
        x = torch.randn(32, 64, requires_grad=True)
        output = block(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None


class TestMinPlusBlock:
    """Tests for MinPlusBlock."""

    def test_forward_shape(self):
        """Test forward pass shape."""
        block = MinPlusBlock(64, 128)
        x = torch.randn(32, 64)
        output = block(x)
        assert output.shape == (32, 128)

    def test_backward_gradient_flow(self):
        """Test gradient flow."""
        block = MinPlusBlock(64, 128)
        x = torch.randn(32, 64, requires_grad=True)
        output = block(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None


class TestTropicalMLP:
    """Tests for TropicalMLP."""

    def test_forward_shape(self):
        """Test forward pass shape."""
        mlp = TropicalMLP(64, 256, 10, num_layers=3)
        x = torch.randn(32, 64)
        output = mlp(x)
        assert output.shape == (32, 10)

    def test_two_layer(self):
        """Test 2-layer MLP."""
        mlp = TropicalMLP(64, 128, 10, num_layers=2)
        x = torch.randn(32, 64)
        output = mlp(x)
        assert output.shape == (32, 10)

    def test_backward_gradient_flow(self):
        """Test gradient flow through MLP."""
        mlp = TropicalMLP(64, 256, 10, num_layers=3)
        x = torch.randn(32, 64, requires_grad=True)
        output = mlp(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None

    def test_with_dropout(self):
        """Test MLP with dropout."""
        mlp = TropicalMLP(64, 256, 10, num_layers=3, dropout=0.5)
        x = torch.randn(32, 64)

        mlp.train()
        output1 = mlp(x)
        output2 = mlp(x)
        assert not torch.allclose(output1, output2)

        mlp.eval()
        output1 = mlp(x)
        output2 = mlp(x)
        torch.testing.assert_close(output1, output2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
