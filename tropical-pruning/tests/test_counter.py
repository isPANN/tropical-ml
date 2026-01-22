"""Tests for WinnerCounter and WinnerStatistics."""

import pytest
import torch
import torch.nn as nn

from tropical_pruning.counter import WinnerCounter, WinnerStatistics


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 4)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)


class TestWinnerStatistics:
    """Tests for WinnerStatistics dataclass."""

    def test_winner_frequency(self):
        """Test winner frequency computation."""
        stats = WinnerStatistics(
            layer_name="test",
            winner_count=torch.tensor([10, 20, 30, 40]),
            total_positions=100,
        )

        freq = stats.winner_frequency
        assert freq.shape == (4,)
        assert torch.allclose(freq, torch.tensor([0.1, 0.2, 0.3, 0.4]))

    def test_average_margin(self):
        """Test average margin computation."""
        stats = WinnerStatistics(
            layer_name="test",
            winner_count=torch.tensor([10, 20]),
            total_positions=30,
            margin_sum=torch.tensor([5.0, 10.0]),
            margin_count=torch.tensor([10, 20]),
        )

        margin = stats.average_margin
        assert margin is not None
        assert torch.allclose(margin, torch.tensor([0.5, 0.5]))

    def test_average_margin_none(self):
        """Test average margin is None when not tracked."""
        stats = WinnerStatistics(
            layer_name="test",
            winner_count=torch.tensor([10, 20]),
            total_positions=30,
        )

        assert stats.average_margin is None


class TestWinnerCounter:
    """Tests for WinnerCounter class."""

    @pytest.fixture
    def model(self):
        """Create a simple model for testing."""
        return SimpleModel()

    @pytest.fixture
    def sample_data(self):
        """Create sample input data."""
        return torch.randn(32, 10)

    def test_init(self, model):
        """Test WinnerCounter initialization."""
        counter = WinnerCounter(model)

        # Should find both linear layers
        assert len(counter.layer_names) == 2
        assert "fc1" in counter.layer_names
        assert "fc2" in counter.layer_names

    def test_init_with_specific_layers(self, model):
        """Test initialization with specific layers."""
        counter = WinnerCounter(model, layers=["fc1"])

        assert len(counter.layer_names) == 1
        assert "fc1" in counter.layer_names

    def test_forward(self, model, sample_data):
        """Test forward pass updates counters."""
        counter = WinnerCounter(model)
        counter.forward(sample_data)

        stats = counter.get_statistics()
        assert "fc1" in stats
        assert stats["fc1"].total_positions > 0

    def test_collect(self, model):
        """Test collecting statistics from dataloader."""
        # Create a simple dataloader
        data = [torch.randn(16, 10) for _ in range(5)]
        dataloader = [(d,) for d in data]

        counter = WinnerCounter(model)
        stats = counter.collect(dataloader, show_progress=False)

        assert "fc1" in stats
        assert "fc2" in stats
        assert stats["fc1"].total_positions > 0

    def test_reset(self, model, sample_data):
        """Test resetting counters."""
        counter = WinnerCounter(model)
        counter.forward(sample_data)

        # Should have non-zero counts
        stats_before = counter.get_statistics()
        assert stats_before["fc1"].total_positions > 0

        # Reset
        counter.reset()
        stats_after = counter.get_statistics()
        assert stats_after["fc1"].total_positions == 0

    def test_track_margin(self, model, sample_data):
        """Test margin tracking."""
        counter = WinnerCounter(model, track_margin=True)
        counter.forward(sample_data)

        stats = counter.get_statistics()
        assert stats["fc1"].margin_sum is not None
        assert stats["fc1"].margin_count is not None

    def test_no_margin_tracking(self, model, sample_data):
        """Test without margin tracking."""
        counter = WinnerCounter(model, track_margin=False)
        counter.forward(sample_data)

        stats = counter.get_statistics()
        assert stats["fc1"].margin_sum is None

    def test_remove_hooks(self, model):
        """Test hook removal."""
        counter = WinnerCounter(model)
        assert len(counter._hooks) > 0

        counter.remove_hooks()
        assert len(counter._hooks) == 0


class TestWinnerCounterIntegration:
    """Integration tests for winner counting."""

    def test_winner_count_consistency(self):
        """Test that winner counts are consistent across multiple passes."""
        torch.manual_seed(42)
        model = SimpleModel()
        data = torch.randn(100, 10)

        counter = WinnerCounter(model)
        counter.forward(data)
        stats1 = counter.get_statistics()

        counter.reset()
        counter.forward(data)
        stats2 = counter.get_statistics()

        # Same data should give same winner counts
        assert torch.equal(stats1["fc1"].winner_count, stats2["fc1"].winner_count)

    def test_all_neurons_covered(self):
        """Test that all neuron indices are valid."""
        model = SimpleModel()
        data = torch.randn(1000, 10)

        counter = WinnerCounter(model)
        counter.forward(data)
        stats = counter.get_statistics()

        # Winner count should have size equal to input features
        assert stats["fc1"].winner_count.shape[0] == 10  # in_features of fc1
        assert stats["fc2"].winner_count.shape[0] == 8   # in_features of fc2
