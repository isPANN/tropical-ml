"""Tests for MMP models."""

import pytest
import torch
import torch.nn as nn

from tropical_activation.models import (
    MMPNN,
    MMPClassifier,
    PureTropicalNN,
    MMPAutoencoder,
    create_mmpnn,
)


class TestMMPNN:
    """Tests for MMPNN."""

    def test_forward_shape(self):
        """Test that forward pass produces correct output shape."""
        model = MMPNN([784, 256, 128, 10])
        x = torch.randn(32, 784)
        output = model(x)
        assert output.shape == (32, 10)

    def test_two_layer(self):
        """Test 2-layer network."""
        model = MMPNN([784, 10])
        x = torch.randn(32, 784)
        output = model(x)
        assert output.shape == (32, 10)

    def test_with_dropout(self):
        """Test network with dropout."""
        model = MMPNN([784, 256, 128, 10], dropout=0.5)
        x = torch.randn(32, 784)

        model.train()
        output1 = model(x)
        output2 = model(x)
        assert not torch.allclose(output1, output2)

        model.eval()
        output1 = model(x)
        output2 = model(x)
        torch.testing.assert_close(output1, output2)

    def test_pure_tropical(self):
        """Test pure tropical network (no linear layers)."""
        model = MMPNN([784, 256, 128, 10], use_linear=False)
        x = torch.randn(32, 784)
        output = model(x)
        assert output.shape == (32, 10)

    def test_backward_gradient_flow(self):
        """Test that gradients flow through the network."""
        model = MMPNN([784, 256, 10])
        x = torch.randn(32, 784, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        # Check that at least some parameters have gradients
        grad_count = sum(1 for p in model.parameters() if p.grad is not None)
        assert grad_count > 0

    def test_invalid_layer_sizes(self):
        """Test that invalid layer sizes raise an error."""
        with pytest.raises(ValueError):
            MMPNN([784])  # Need at least 2 layers


class TestMMPClassifier:
    """Tests for MMPClassifier."""

    def test_forward_shape(self):
        """Test forward pass shape."""
        model = MMPClassifier(784, [256, 128], 10)
        x = torch.randn(32, 784)
        output = model(x)
        assert output.shape == (32, 10)

    def test_with_residual(self):
        """Test classifier with residual connections."""
        # Use same hidden dim for residual to work
        model = MMPClassifier(784, [256, 256], 10, use_residual=True)
        x = torch.randn(32, 784)
        output = model(x)
        assert output.shape == (32, 10)

    def test_backward_gradient_flow(self):
        """Test gradient flow."""
        model = MMPClassifier(784, [256, 128], 10)
        x = torch.randn(32, 784, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None


class TestPureTropicalNN:
    """Tests for PureTropicalNN."""

    def test_forward_shape(self):
        """Test forward pass shape."""
        model = PureTropicalNN([784, 256, 128, 10])
        x = torch.randn(32, 784)
        output = model(x)
        assert output.shape == (32, 10)

    def test_no_linear_layers(self):
        """Verify no Linear layers in the model."""
        model = PureTropicalNN([784, 256, 10])

        for module in model.modules():
            assert not isinstance(module, nn.Linear)

    def test_backward_gradient_flow(self):
        """Test gradient flow."""
        model = PureTropicalNN([784, 256, 10])
        x = torch.randn(32, 784, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None


class TestMMPAutoencoder:
    """Tests for MMPAutoencoder."""

    def test_forward_shape(self):
        """Test forward pass (reconstruction) shape."""
        model = MMPAutoencoder(784, [256, 128], 32)
        x = torch.randn(32, 784)
        output = model(x)
        assert output.shape == (32, 784)

    def test_encode_shape(self):
        """Test encode output shape."""
        model = MMPAutoencoder(784, [256, 128], 32)
        x = torch.randn(32, 784)
        latent = model.encode(x)
        assert latent.shape == (32, 32)

    def test_decode_shape(self):
        """Test decode output shape."""
        model = MMPAutoencoder(784, [256, 128], 32)
        z = torch.randn(32, 32)
        output = model.decode(z)
        assert output.shape == (32, 784)

    def test_backward_gradient_flow(self):
        """Test gradient flow."""
        model = MMPAutoencoder(784, [256], 32)
        x = torch.randn(32, 784, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None


class TestCreateMmpnn:
    """Tests for create_mmpnn factory function."""

    def test_tiny_architecture(self):
        """Test tiny architecture."""
        model = create_mmpnn("tiny", 784, 10)
        x = torch.randn(32, 784)
        output = model(x)
        assert output.shape == (32, 10)

    def test_small_architecture(self):
        """Test small architecture."""
        model = create_mmpnn("small", 784, 10)
        x = torch.randn(32, 784)
        output = model(x)
        assert output.shape == (32, 10)

    def test_medium_architecture(self):
        """Test medium architecture."""
        model = create_mmpnn("medium", 784, 10)
        x = torch.randn(32, 784)
        output = model(x)
        assert output.shape == (32, 10)

    def test_large_architecture(self):
        """Test large architecture."""
        model = create_mmpnn("large", 784, 10)
        x = torch.randn(32, 784)
        output = model(x)
        assert output.shape == (32, 10)

    def test_invalid_architecture(self):
        """Test invalid architecture name raises error."""
        with pytest.raises(ValueError):
            create_mmpnn("unknown", 784, 10)


class TestModelTraining:
    """Integration tests for training models."""

    def test_simple_training_step(self):
        """Test a simple training step."""
        model = MMPNN([64, 32, 10])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        # Single batch
        x = torch.randn(16, 64)
        y = torch.randint(0, 10, (16,))

        # Forward
        output = model(x)
        loss = criterion(output, y)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Verify loss is a valid number
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_convergence_on_simple_task(self):
        """Test that model can learn a simple pattern."""
        torch.manual_seed(42)

        model = MMPNN([2, 16, 2])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        # Simple XOR-like pattern
        x = torch.tensor([
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ])
        y = torch.tensor([0, 1, 1, 0])

        # Train for a few steps
        initial_loss = None
        for _ in range(100):
            output = model(x)
            loss = criterion(output, y)

            if initial_loss is None:
                initial_loss = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Loss should decrease
        final_loss = loss.item()
        assert final_loss < initial_loss


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
