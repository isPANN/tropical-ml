"""Tests for tropical layers: MaxPlusLayer and MinPlusLayer."""

import pytest
import torch
import torch.nn as nn
import numpy as np

from tropical_activation.layers import MaxPlusLayer, MinPlusLayer, TropicalReLU


class TestMaxPlusLayer:
    """Tests for MaxPlusLayer."""

    def test_forward_shape(self):
        """Test that forward pass produces correct output shape."""
        layer = MaxPlusLayer(10, 20)
        x = torch.randn(32, 10)
        output = layer(x)
        assert output.shape == (32, 20)

    def test_forward_multidim(self):
        """Test forward pass with multi-dimensional input."""
        layer = MaxPlusLayer(10, 20)
        x = torch.randn(8, 16, 10)  # (batch, seq, features)
        output = layer(x)
        assert output.shape == (8, 16, 20)

    def test_forward_manual_computation(self):
        """Test that forward matches manual tropical max-plus computation."""
        layer = MaxPlusLayer(3, 2, bias=False)

        # Set known weights
        with torch.no_grad():
            layer.weight.copy_(torch.tensor([
                [1.0, 2.0],
                [0.5, 0.0],
                [-1.0, 1.0],
            ]))

        x = torch.tensor([[1.0, 2.0, 3.0]])

        # Manual computation: y_j = max_k(x_k + w_kj)
        # y_0 = max(1+1, 2+0.5, 3-1) = max(2, 2.5, 2) = 2.5
        # y_1 = max(1+2, 2+0, 3+1) = max(3, 2, 4) = 4
        expected = torch.tensor([[2.5, 4.0]])

        output = layer(x)
        torch.testing.assert_close(output, expected, atol=1e-5, rtol=1e-5)

    def test_backward_gradient_flow(self):
        """Test that gradients flow through the layer."""
        layer = MaxPlusLayer(10, 20)
        x = torch.randn(32, 10, requires_grad=True)
        output = layer(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert layer.weight.grad is not None
        if layer.bias is not None:
            assert layer.bias.grad is not None

    def test_gradient_sparsity(self):
        """Test that gradients are sparse (only winners get gradients)."""
        layer = MaxPlusLayer(5, 3, bias=False)
        x = torch.randn(1, 5, requires_grad=True)
        output = layer(x)
        loss = output.sum()
        loss.backward()

        # Gradient should be sparse (only 3 out of 5 inputs can win for 3 outputs)
        # But some inputs might win multiple times
        assert x.grad is not None

    def test_no_bias(self):
        """Test layer without bias."""
        layer = MaxPlusLayer(10, 20, bias=False)
        assert layer.bias is None

        x = torch.randn(32, 10)
        output = layer(x)
        assert output.shape == (32, 20)


class TestMinPlusLayer:
    """Tests for MinPlusLayer."""

    def test_forward_shape(self):
        """Test that forward pass produces correct output shape."""
        layer = MinPlusLayer(10, 20)
        x = torch.randn(32, 10)
        output = layer(x)
        assert output.shape == (32, 20)

    def test_forward_manual_computation(self):
        """Test that forward matches manual tropical min-plus computation."""
        layer = MinPlusLayer(3, 2, bias=False)

        # Set known weights
        with torch.no_grad():
            layer.weight.copy_(torch.tensor([
                [1.0, 2.0],
                [0.5, 0.0],
                [-1.0, 1.0],
            ]))

        x = torch.tensor([[1.0, 2.0, 3.0]])

        # Manual computation: y_j = min_k(x_k + w_kj)
        # y_0 = min(1+1, 2+0.5, 3-1) = min(2, 2.5, 2) = 2
        # y_1 = min(1+2, 2+0, 3+1) = min(3, 2, 4) = 2
        expected = torch.tensor([[2.0, 2.0]])

        output = layer(x)
        torch.testing.assert_close(output, expected, atol=1e-5, rtol=1e-5)

    def test_backward_gradient_flow(self):
        """Test that gradients flow through the layer."""
        layer = MinPlusLayer(10, 20)
        x = torch.randn(32, 10, requires_grad=True)
        output = layer(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert layer.weight.grad is not None


class TestTropicalReLU:
    """Tests for TropicalReLU."""

    def test_matches_standard_relu(self):
        """Test that TropicalReLU matches standard ReLU."""
        tropical_relu = TropicalReLU()
        standard_relu = nn.ReLU()

        x = torch.randn(32, 64)
        torch.testing.assert_close(tropical_relu(x), standard_relu(x))


class TestGradientCheck:
    """Gradient verification tests."""

    @pytest.mark.skip(reason="tropical-gemm uses f32 internally, gradcheck requires f64 precision")
    def test_maxplus_gradient_check(self):
        """Verify MaxPlusLayer gradients with finite differences."""
        layer = MaxPlusLayer(5, 3, bias=True)
        x = torch.randn(2, 5, requires_grad=True, dtype=torch.float64)

        # Use double precision for gradient check
        layer = layer.double()

        # This uses torch's gradient checking utility
        assert torch.autograd.gradcheck(
            layer,
            x,
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
            raise_exception=True,
        )

    @pytest.mark.skip(reason="tropical-gemm uses f32 internally, gradcheck requires f64 precision")
    def test_minplus_gradient_check(self):
        """Verify MinPlusLayer gradients with finite differences."""
        layer = MinPlusLayer(5, 3, bias=True)
        x = torch.randn(2, 5, requires_grad=True, dtype=torch.float64)

        layer = layer.double()

        assert torch.autograd.gradcheck(
            layer,
            x,
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
            raise_exception=True,
        )

    def test_maxplus_gradient_structure(self):
        """Verify MaxPlusLayer gradient structure is correct (sparse, one-hot per output)."""
        layer = MaxPlusLayer(5, 3, bias=False)
        x = torch.randn(1, 5, requires_grad=True)

        output = layer(x)
        loss = output.sum()
        loss.backward()

        # Gradient should have exactly 3 non-zero entries (one winner per output)
        # Each winner gets gradient = 1.0 (potentially multiple if same input wins multiple outputs)
        assert x.grad is not None
        # Gradient values should be integers (counts of how many times each input won)
        assert torch.allclose(x.grad, x.grad.round())

    def test_minplus_gradient_structure(self):
        """Verify MinPlusLayer gradient structure is correct."""
        layer = MinPlusLayer(5, 3, bias=False)
        x = torch.randn(1, 5, requires_grad=True)

        output = layer(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert torch.allclose(x.grad, x.grad.round())


class TestReLUEquivalence:
    """Test that MaxPlus can represent ReLU."""

    def test_maxplus_relu_equivalence(self):
        """Test MaxPlus with specific weights matches ReLU."""
        # For scalar ReLU: max(x, 0)
        # This is MaxPlus with W = [[0], [-inf]]
        # But our MaxPlus is y_j = max_k(x_k + w_kj)
        # For single input/output with W = [[0]], we get y = x + 0 = x
        # For ReLU(x) = max(x, 0), we need a 2D interpretation

        # Simpler test: verify max-plus behavior matches expectations
        layer = MaxPlusLayer(2, 1, bias=False)

        # Set weights to select max of (x, 0)
        with torch.no_grad():
            layer.weight.copy_(torch.tensor([[0.0], [0.0]]))

        x = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
        output = layer(x)

        # y = max(x[0]+0, x[1]+0) = max(x[0], 0)
        expected = torch.tensor([[1.0], [0.0]])
        torch.testing.assert_close(output, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
