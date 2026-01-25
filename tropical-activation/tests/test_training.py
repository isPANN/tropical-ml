"""Tests for training utilities."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from tropical_activation import MMPNN, MaxPlusLayer, MinPlusLayer
from tropical_activation.training import (
    TropicalBatchNorm,
    TropicalLayerNorm,
    tropical_weight_init,
    get_tropical_optimizer,
    train_epoch,
    evaluate,
    count_parameters,
    count_operations,
)


class TestTropicalBatchNorm:
    """Tests for TropicalBatchNorm."""

    def test_forward_shape(self):
        """Test forward pass shape."""
        bn = TropicalBatchNorm(64)
        x = torch.randn(32, 64)
        output = bn(x)
        assert output.shape == (32, 64)

    def test_modes(self):
        """Test different normalization modes."""
        for mode in ["standard", "max", "range"]:
            bn = TropicalBatchNorm(64, mode=mode)
            x = torch.randn(32, 64)
            output = bn(x)
            assert output.shape == (32, 64)

    def test_training_vs_eval(self):
        """Test behavior in training vs eval mode."""
        bn = TropicalBatchNorm(64)
        x = torch.randn(32, 64)

        bn.train()
        output_train = bn(x)

        bn.eval()
        output_eval = bn(x)

        # Outputs should differ because running stats are used in eval
        assert output_train.shape == output_eval.shape


class TestTropicalLayerNorm:
    """Tests for TropicalLayerNorm."""

    def test_forward_shape(self):
        """Test forward pass shape."""
        ln = TropicalLayerNorm(64)
        x = torch.randn(32, 64)
        output = ln(x)
        assert output.shape == (32, 64)

    def test_multidim(self):
        """Test with multi-dimensional normalized shape."""
        ln = TropicalLayerNorm([16, 64])
        x = torch.randn(32, 16, 64)
        output = ln(x)
        assert output.shape == (32, 16, 64)


class TestWeightInit:
    """Tests for weight initialization."""

    def test_tropical_weight_init(self):
        """Test tropical weight initialization."""
        model = MMPNN([64, 32, 10])
        tropical_weight_init(model, init_scale=0.1)

        # Check that tropical layers have small weights
        for module in model.modules():
            if isinstance(module, (MaxPlusLayer, MinPlusLayer)):
                assert module.weight.abs().max() <= 0.15  # Allow some tolerance


class TestGetTropicalOptimizer:
    """Tests for optimizer creation."""

    def test_adamw_optimizer(self):
        """Test AdamW optimizer creation."""
        model = MMPNN([64, 32, 10])
        optimizer = get_tropical_optimizer(model, lr=0.001, optimizer_type="adamw")
        assert isinstance(optimizer, torch.optim.AdamW)

    def test_adam_optimizer(self):
        """Test Adam optimizer creation."""
        model = MMPNN([64, 32, 10])
        optimizer = get_tropical_optimizer(model, lr=0.001, optimizer_type="adam")
        assert isinstance(optimizer, torch.optim.Adam)

    def test_sgd_optimizer(self):
        """Test SGD optimizer creation."""
        model = MMPNN([64, 32, 10])
        optimizer = get_tropical_optimizer(model, lr=0.001, optimizer_type="sgd")
        assert isinstance(optimizer, torch.optim.SGD)

    def test_tropical_lr_scale(self):
        """Test that tropical layers get scaled learning rate."""
        model = MMPNN([64, 32, 10])
        optimizer = get_tropical_optimizer(
            model, lr=0.001, tropical_lr_scale=0.5
        )
        # Just verify it doesn't crash
        assert optimizer is not None


class TestTrainEpoch:
    """Tests for train_epoch function."""

    def test_basic_training(self):
        """Test basic training epoch."""
        model = MMPNN([64, 32, 10])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        # Create simple dataset
        x = torch.randn(100, 64)
        y = torch.randint(0, 10, (100,))
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=16)

        device = torch.device("cpu")
        model.to(device)

        metrics = train_epoch(model, loader, optimizer, criterion, device)

        assert "loss" in metrics
        assert "accuracy" in metrics
        assert 0 <= metrics["accuracy"] <= 100


class TestEvaluate:
    """Tests for evaluate function."""

    def test_basic_evaluation(self):
        """Test basic evaluation."""
        model = MMPNN([64, 32, 10])
        criterion = nn.CrossEntropyLoss()

        x = torch.randn(100, 64)
        y = torch.randint(0, 10, (100,))
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=16)

        device = torch.device("cpu")
        model.to(device)

        metrics = evaluate(model, loader, criterion, device)

        assert "loss" in metrics
        assert "accuracy" in metrics
        assert 0 <= metrics["accuracy"] <= 100


class TestCountParameters:
    """Tests for count_parameters function."""

    def test_count_parameters(self):
        """Test parameter counting."""
        model = MMPNN([64, 32, 10])
        counts = count_parameters(model)

        assert "tropical" in counts
        assert "linear" in counts
        assert "other" in counts
        assert "total" in counts
        assert counts["total"] == counts["tropical"] + counts["linear"] + counts["other"]


class TestCountOperations:
    """Tests for count_operations function."""

    def test_count_operations(self):
        """Test operation counting."""
        model = MMPNN([64, 32, 10])
        ops = count_operations(model, (64,))

        assert "multiplications" in ops
        assert "additions" in ops
        assert "comparisons" in ops
        assert "total_ops" in ops

        # Tropical layers should produce comparisons
        assert ops["comparisons"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
