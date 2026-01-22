"""Tests for ConvWinnerCounter and ConvWinnerStatistics."""

import pytest
import torch
import torch.nn as nn

from tropical_pruning.conv_counter import ConvWinnerCounter, ConvWinnerStatistics
from tropical_pruning.counter import WinnerStatistics


class SimpleCNN(nn.Module):
    """Simple CNN model for testing."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, 10)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)


class GroupedConvModel(nn.Module):
    """Model with grouped convolution for testing."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 8, kernel_size=3, padding=1, groups=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, 4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


class DepthwiseSeparableModel(nn.Module):
    """Model with depthwise separable convolution."""

    def __init__(self):
        super().__init__()
        # Depthwise: groups = in_channels = out_channels
        self.depthwise = nn.Conv2d(8, 8, kernel_size=3, padding=1, groups=8)
        # Pointwise: 1x1 conv
        self.pointwise = nn.Conv2d(8, 16, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 4)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


class TestConvWinnerStatistics:
    """Tests for ConvWinnerStatistics dataclass."""

    def test_winner_frequency(self):
        """Test winner frequency computation."""
        stats = ConvWinnerStatistics(
            layer_name="test_conv",
            winner_count=torch.tensor([10, 20, 30, 40]),
            total_positions=100,
            kernel_size=(3, 3),
        )

        freq = stats.winner_frequency
        assert freq.shape == (4,)
        assert torch.allclose(freq, torch.tensor([0.1, 0.2, 0.3, 0.4]))

    def test_average_margin(self):
        """Test average margin computation."""
        stats = ConvWinnerStatistics(
            layer_name="test_conv",
            winner_count=torch.tensor([10, 20]),
            total_positions=30,
            margin_sum=torch.tensor([5.0, 10.0]),
            margin_count=torch.tensor([10, 20]),
            kernel_size=(3, 3),
        )

        margin = stats.average_margin
        assert margin is not None
        assert torch.allclose(margin, torch.tensor([0.5, 0.5]))

    def test_average_margin_none(self):
        """Test average margin is None when not tracked."""
        stats = ConvWinnerStatistics(
            layer_name="test_conv",
            winner_count=torch.tensor([10, 20]),
            total_positions=30,
            kernel_size=(3, 3),
        )

        assert stats.average_margin is None

    def test_to_device(self):
        """Test moving statistics to a different device."""
        stats = ConvWinnerStatistics(
            layer_name="test_conv",
            winner_count=torch.tensor([10, 20]),
            total_positions=30,
            margin_sum=torch.tensor([5.0, 10.0]),
            margin_count=torch.tensor([10, 20]),
            kernel_size=(3, 3),
        )

        # Move to same device (CPU) as a basic test
        new_stats = stats.to(torch.device("cpu"))
        assert new_stats.winner_count.device.type == "cpu"
        assert new_stats.margin_sum.device.type == "cpu"

    def test_to_winner_statistics(self):
        """Test conversion to WinnerStatistics."""
        conv_stats = ConvWinnerStatistics(
            layer_name="test_conv",
            winner_count=torch.tensor([10, 20, 30]),
            total_positions=60,
            margin_sum=torch.tensor([1.0, 2.0, 3.0]),
            margin_count=torch.tensor([10, 20, 30]),
            kernel_size=(3, 3),
        )

        winner_stats = conv_stats.to_winner_statistics()

        assert isinstance(winner_stats, WinnerStatistics)
        assert winner_stats.layer_name == "test_conv"
        assert torch.equal(winner_stats.winner_count, conv_stats.winner_count)
        assert winner_stats.total_positions == conv_stats.total_positions


class TestConvWinnerCounter:
    """Tests for ConvWinnerCounter class."""

    @pytest.fixture
    def model(self):
        """Create a simple CNN for testing."""
        return SimpleCNN()

    @pytest.fixture
    def sample_data(self):
        """Create sample input data."""
        return torch.randn(4, 3, 32, 32)

    def test_init(self, model):
        """Test ConvWinnerCounter initialization."""
        counter = ConvWinnerCounter(model)

        # Should find both conv layers
        assert len(counter.layer_names) == 2
        assert "conv1" in counter.layer_names
        assert "conv2" in counter.layer_names

    def test_init_with_specific_layers(self, model):
        """Test initialization with specific layers."""
        counter = ConvWinnerCounter(model, layers=["conv1"])

        assert len(counter.layer_names) == 1
        assert "conv1" in counter.layer_names

    def test_init_with_include_linear(self, model):
        """Test initialization including linear layers."""
        counter = ConvWinnerCounter(model, include_linear=True)

        # Should find conv + linear layers
        assert "conv1" in counter.layer_names
        assert "conv2" in counter.layer_names
        assert "fc" in counter.layer_names

    def test_forward(self, model, sample_data):
        """Test forward pass updates counters."""
        counter = ConvWinnerCounter(model)
        counter.forward(sample_data)

        stats = counter.get_statistics()
        assert "conv1" in stats
        assert stats["conv1"].total_positions > 0
        assert isinstance(stats["conv1"], ConvWinnerStatistics)

    def test_collect(self, model):
        """Test collecting statistics from dataloader."""
        # Create a simple dataloader
        data = [torch.randn(2, 3, 32, 32) for _ in range(3)]
        dataloader = [(d,) for d in data]

        counter = ConvWinnerCounter(model)
        stats = counter.collect(dataloader, show_progress=False)

        assert "conv1" in stats
        assert "conv2" in stats
        assert stats["conv1"].total_positions > 0
        assert stats["conv2"].total_positions > 0

    def test_collect_with_num_batches(self, model):
        """Test collecting with limited batches."""
        data = [torch.randn(2, 3, 32, 32) for _ in range(10)]
        dataloader = [(d,) for d in data]

        counter = ConvWinnerCounter(model)
        counter.collect(dataloader, num_batches=3, show_progress=False)

        stats = counter.get_statistics()
        # With 2 samples per batch, 3 batches = 6 samples processed
        # Exact total_positions depends on spatial dimensions
        assert stats["conv1"].total_positions > 0

    def test_reset(self, model, sample_data):
        """Test resetting counters."""
        counter = ConvWinnerCounter(model)
        counter.forward(sample_data)

        # Should have non-zero counts
        stats_before = counter.get_statistics()
        assert stats_before["conv1"].total_positions > 0

        # Reset
        counter.reset()
        stats_after = counter.get_statistics()
        assert stats_after["conv1"].total_positions == 0

    def test_track_margin(self, model, sample_data):
        """Test margin tracking."""
        counter = ConvWinnerCounter(model, track_margin=True)
        counter.forward(sample_data)

        stats = counter.get_statistics()
        assert stats["conv1"].margin_sum is not None
        assert stats["conv1"].margin_count is not None

    def test_no_margin_tracking(self, model, sample_data):
        """Test without margin tracking."""
        counter = ConvWinnerCounter(model, track_margin=False)
        counter.forward(sample_data)

        stats = counter.get_statistics()
        assert stats["conv1"].margin_sum is None

    def test_remove_hooks(self, model):
        """Test hook removal."""
        counter = ConvWinnerCounter(model)
        assert len(counter._hooks) > 0

        counter.remove_hooks()
        assert len(counter._hooks) == 0

    def test_get_statistics_as_winner_stats(self, model, sample_data):
        """Test conversion to WinnerStatistics format."""
        counter = ConvWinnerCounter(model)
        counter.forward(sample_data)

        stats = counter.get_statistics_as_winner_stats()
        assert "conv1" in stats
        assert isinstance(stats["conv1"], WinnerStatistics)


class TestGroupedConvolutions:
    """Tests for grouped convolution handling."""

    def test_grouped_conv_basic(self):
        """Test basic grouped convolution."""
        model = GroupedConvModel()
        data = torch.randn(2, 4, 16, 16)

        counter = ConvWinnerCounter(model)
        counter.forward(data)

        stats = counter.get_statistics()
        assert "conv1" in stats
        assert stats["conv1"].total_positions > 0
        # Winner count should have 4 channels (in_channels)
        assert stats["conv1"].winner_count.shape[0] == 4

    def test_depthwise_separable(self):
        """Test depthwise separable convolution."""
        model = DepthwiseSeparableModel()
        data = torch.randn(2, 8, 16, 16)

        counter = ConvWinnerCounter(model)
        counter.forward(data)

        stats = counter.get_statistics()

        # Depthwise conv: 8 input channels
        assert "depthwise" in stats
        assert stats["depthwise"].winner_count.shape[0] == 8

        # Pointwise conv: 8 input channels
        assert "pointwise" in stats
        assert stats["pointwise"].winner_count.shape[0] == 8


class TestConvWinnerCounterIntegration:
    """Integration tests for conv winner counting."""

    def test_winner_count_consistency(self):
        """Test that winner counts are consistent across multiple passes."""
        torch.manual_seed(42)
        model = SimpleCNN()
        data = torch.randn(4, 3, 32, 32)

        counter = ConvWinnerCounter(model)
        counter.forward(data)
        stats1 = counter.get_statistics()

        counter.reset()
        counter.forward(data)
        stats2 = counter.get_statistics()

        # Same data should give same winner counts
        assert torch.equal(stats1["conv1"].winner_count, stats2["conv1"].winner_count)
        assert stats1["conv1"].total_positions == stats2["conv1"].total_positions

    def test_all_channels_have_valid_counts(self):
        """Test that all channel counts are non-negative."""
        model = SimpleCNN()
        data = torch.randn(8, 3, 32, 32)

        counter = ConvWinnerCounter(model)
        counter.forward(data)
        stats = counter.get_statistics()

        # All winner counts should be non-negative
        assert (stats["conv1"].winner_count >= 0).all()
        assert (stats["conv2"].winner_count >= 0).all()

    def test_winner_count_correct_shape(self):
        """Test winner count has correct shape matching input channels."""
        model = SimpleCNN()
        data = torch.randn(4, 3, 32, 32)

        counter = ConvWinnerCounter(model)
        counter.forward(data)
        stats = counter.get_statistics()

        # conv1: in_channels=3
        assert stats["conv1"].winner_count.shape[0] == 3
        # conv2: in_channels=16
        assert stats["conv2"].winner_count.shape[0] == 16

    def test_total_positions_reasonable(self):
        """Test total positions matches expected value."""
        model = SimpleCNN()
        batch_size = 4
        spatial = 32
        data = torch.randn(batch_size, 3, spatial, spatial)

        counter = ConvWinnerCounter(model)
        counter.forward(data)
        stats = counter.get_statistics()

        # conv1 output: (batch, 16, 32, 32) -> 16 * 32 * 32 positions per sample
        # Total: 4 * 16 * 32 * 32 = 65536
        expected_conv1 = batch_size * 16 * spatial * spatial
        assert stats["conv1"].total_positions == expected_conv1

    def test_frequency_sums_to_approximately_one(self):
        """Test that winner frequencies sum to approximately 1 per output channel."""
        model = SimpleCNN()
        data = torch.randn(8, 3, 32, 32)

        counter = ConvWinnerCounter(model)
        counter.forward(data)
        stats = counter.get_statistics()

        # Winner frequency should sum to approximately 1
        # (each output position has exactly one winner)
        freq_sum = stats["conv1"].winner_frequency.sum().item()
        # Allow some tolerance due to how we count (may be slightly > 1 due to normalization)
        assert 0.9 < freq_sum < 1.1, f"Frequency sum: {freq_sum}"

    def test_kernel_size_stored(self):
        """Test that kernel size is correctly stored in statistics."""
        model = SimpleCNN()
        data = torch.randn(2, 3, 32, 32)

        counter = ConvWinnerCounter(model)
        counter.forward(data)
        stats = counter.get_statistics()

        # Both conv layers have 3x3 kernels
        assert stats["conv1"].kernel_size == (3, 3)
        assert stats["conv2"].kernel_size == (3, 3)


class TestMixedModel:
    """Tests for models with both Conv2d and Linear layers."""

    def test_hybrid_model_conv_only(self):
        """Test that by default only conv layers are tracked."""
        model = SimpleCNN()
        data = torch.randn(2, 3, 32, 32)

        counter = ConvWinnerCounter(model, include_linear=False)
        counter.forward(data)
        stats = counter.get_statistics()

        assert "conv1" in stats
        assert "conv2" in stats
        assert "fc" not in stats

    def test_hybrid_model_with_linear(self):
        """Test tracking both conv and linear layers."""
        model = SimpleCNN()
        data = torch.randn(2, 3, 32, 32)

        counter = ConvWinnerCounter(model, include_linear=True)
        counter.forward(data)
        stats = counter.get_statistics()

        assert "conv1" in stats
        assert "conv2" in stats
        assert "fc" in stats

        # Conv stats should be ConvWinnerStatistics
        assert isinstance(stats["conv1"], ConvWinnerStatistics)
        # Linear stats should be WinnerStatistics
        assert isinstance(stats["fc"], WinnerStatistics)
