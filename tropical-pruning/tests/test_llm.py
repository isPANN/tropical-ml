"""
Tests for LLM FFN Tropical Pruning.

These tests use a mock LLM-like model to test the FFN pruning components
without requiring actual HuggingFace model downloads.
"""

import pytest
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional
from torch.utils.data import DataLoader, TensorDataset


# Mock model configuration
@dataclass
class MockLLMConfig:
    """Mock configuration for testing."""
    model_type: str = "llama"
    num_hidden_layers: int = 4
    hidden_size: int = 64
    intermediate_size: int = 128
    vocab_size: int = 1000


class MockSwiGLUMLP(nn.Module):
    """Mock SwiGLU FFN block similar to LLaMA."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.nn.functional.silu(self.gate_proj(x))
        up = self.up_proj(x)
        intermediate = gate * up
        return self.down_proj(intermediate)


class MockTransformerBlock(nn.Module):
    """Mock transformer block with FFN."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.mlp = MockSwiGLUMLP(hidden_size, intermediate_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(x)


class MockLLM(nn.Module):
    """Mock LLM model for testing FFN pruning."""

    def __init__(self, config: MockLLMConfig):
        super().__init__()
        self.config = config

        # Embedding layer
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Transformer layers
        self.model.layers = nn.ModuleList([
            MockTransformerBlock(config.hidden_size, config.intermediate_size)
            for _ in range(config.num_hidden_layers)
        ])

        # LM head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        x = self.model.embed_tokens(input_ids)

        for layer in self.model.layers:
            x = layer(x)

        logits = self.lm_head(x)

        # Return object with logits attribute to match HuggingFace format
        class Output:
            pass
        out = Output()
        out.logits = logits
        return out


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"

    def __call__(self, text, return_tensors=None, truncation=False, add_special_tokens=True):
        # Simple tokenization: random integers
        if isinstance(text, str):
            num_tokens = min(len(text.split()), 100)
        else:
            num_tokens = 100

        torch.manual_seed(42)
        input_ids = torch.randint(2, self.vocab_size, (1, num_tokens))

        class Output:
            pass
        out = Output()
        out.input_ids = input_ids
        return {"input_ids": input_ids}


@pytest.fixture
def mock_model():
    """Create a mock LLM model."""
    config = MockLLMConfig()
    return MockLLM(config)


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""
    return MockTokenizer()


@pytest.fixture
def calibration_dataloader():
    """Create a simple calibration dataloader."""
    torch.manual_seed(42)
    input_ids = torch.randint(2, 1000, (8, 32))  # 8 samples, 32 tokens each
    attention_mask = torch.ones_like(input_ids)
    dataset = TensorDataset(input_ids, attention_mask)
    return DataLoader(dataset, batch_size=2)


class TestFFNLayerDetection:
    """Test FFN layer detection and architecture identification."""

    def test_detect_ffn_layers(self, mock_model):
        """Test that FFN layers are correctly detected."""
        from tropical_pruning.llm.loader import get_ffn_layer_names

        layers = get_ffn_layer_names(mock_model)

        assert len(layers) == 4  # 4 transformer blocks
        assert layers[0].intermediate_size == 128
        assert layers[0].hidden_size == 64

    def test_ffn_layer_names(self, mock_model):
        """Test that layer names are correct."""
        from tropical_pruning.llm.loader import get_ffn_layer_names

        layers = get_ffn_layer_names(mock_model)

        # Check layer naming pattern
        for i, layer in enumerate(layers):
            assert layer.layer_idx == i
            assert "down_proj" in layer.down_proj_name


class TestFFNStatistics:
    """Test FFNStatistics dataclass."""

    def test_winner_frequency(self):
        """Test winner frequency computation."""
        from tropical_pruning.llm.ffn_counter import FFNStatistics

        winner_count = torch.tensor([10, 20, 30, 40])
        stats = FFNStatistics(
            layer_idx=0,
            intermediate_winner_count=winner_count,
            intermediate_total_positions=100,
        )

        freq = stats.winner_frequency
        assert freq.shape == (4,)
        assert torch.allclose(freq, torch.tensor([0.1, 0.2, 0.3, 0.4]))

    def test_average_margin(self):
        """Test average margin computation."""
        from tropical_pruning.llm.ffn_counter import FFNStatistics

        stats = FFNStatistics(
            layer_idx=0,
            intermediate_winner_count=torch.tensor([10, 20]),
            intermediate_total_positions=30,
            intermediate_margin_sum=torch.tensor([5.0, 10.0]),
            intermediate_margin_count=torch.tensor([10, 20]),
        )

        margin = stats.average_margin
        assert margin is not None
        assert torch.allclose(margin, torch.tensor([0.5, 0.5]))

    def test_to_device(self):
        """Test moving statistics to device."""
        from tropical_pruning.llm.ffn_counter import FFNStatistics

        stats = FFNStatistics(
            layer_idx=0,
            intermediate_winner_count=torch.tensor([10, 20]),
            intermediate_total_positions=30,
        )

        # Move to CPU (should work regardless of CUDA availability)
        moved = stats.to(torch.device("cpu"))
        assert moved.intermediate_winner_count.device == torch.device("cpu")


class TestFFNWinnerCounter:
    """Test FFNWinnerCounter class."""

    def test_init(self, mock_model):
        """Test counter initialization."""
        from tropical_pruning.llm.ffn_counter import FFNWinnerCounter

        counter = FFNWinnerCounter(mock_model)

        # Should have counters for all FFN layers
        assert len(counter._counters) == 4
        counter.remove_hooks()

    def test_forward_updates_counters(self, mock_model):
        """Test that forward pass updates counters."""
        from tropical_pruning.llm.ffn_counter import FFNWinnerCounter

        counter = FFNWinnerCounter(mock_model, track_margin=False)

        # Initial state
        for c in counter._counters.values():
            assert c.total_positions == 0

        # Run forward
        input_ids = torch.randint(2, 1000, (2, 16))
        counter.forward(input_ids)

        # Counters should be updated
        for c in counter._counters.values():
            assert c.total_positions > 0
            assert c.winner_count.sum().item() == c.total_positions

        counter.remove_hooks()

    def test_collect(self, mock_model, calibration_dataloader):
        """Test collecting statistics from dataloader."""
        from tropical_pruning.llm.ffn_counter import FFNWinnerCounter

        counter = FFNWinnerCounter(mock_model, track_margin=True)

        # Adapt dataloader format to return dicts
        def dict_collate(batch):
            input_ids = torch.stack([b[0] for b in batch])
            attention_mask = torch.stack([b[1] for b in batch])
            return {"input_ids": input_ids, "attention_mask": attention_mask}

        # Recreate dataloader with dict format
        torch.manual_seed(42)
        input_ids = torch.randint(2, 1000, (8, 32))
        attention_mask = torch.ones_like(input_ids)
        dataset = TensorDataset(input_ids, attention_mask)
        loader = DataLoader(dataset, batch_size=2, collate_fn=dict_collate)

        stats = counter.collect(loader, show_progress=False)

        assert len(stats) == 4
        for layer_idx, s in stats.items():
            assert s.intermediate_total_positions > 0
            assert s.intermediate_winner_count.sum().item() == s.intermediate_total_positions

        counter.remove_hooks()

    def test_reset(self, mock_model):
        """Test counter reset."""
        from tropical_pruning.llm.ffn_counter import FFNWinnerCounter

        counter = FFNWinnerCounter(mock_model)

        # Run forward
        input_ids = torch.randint(2, 1000, (2, 16))
        counter.forward(input_ids)

        # Reset
        counter.reset()

        for c in counter._counters.values():
            assert c.total_positions == 0
            assert c.winner_count.sum().item() == 0

        counter.remove_hooks()

    def test_analyze(self, mock_model):
        """Test statistics analysis."""
        from tropical_pruning.llm.ffn_counter import FFNWinnerCounter

        counter = FFNWinnerCounter(mock_model)
        input_ids = torch.randint(2, 1000, (4, 32))
        counter.forward(input_ids)

        analysis = counter.analyze()

        assert len(analysis) == 4
        for layer_idx, a in analysis.items():
            assert "intermediate_size" in a
            assert "total_positions" in a
            assert "never_win" in a
            assert "mean_frequency" in a

        counter.remove_hooks()


class TestFFNTropicalPruner:
    """Test FFNTropicalPruner class."""

    def test_compute_importance(self, mock_model):
        """Test importance computation."""
        from tropical_pruning.llm.ffn_counter import FFNWinnerCounter, FFNStatistics
        from tropical_pruning.llm.ffn_pruner import FFNTropicalPruner

        # Create mock statistics
        stats = {
            0: FFNStatistics(
                layer_idx=0,
                intermediate_winner_count=torch.arange(128),  # Increasing importance
                intermediate_total_positions=1000,
            ),
        }

        pruner = FFNTropicalPruner(mock_model, stats)
        importance = pruner.compute_importance(0)

        assert importance.shape == (128,)
        # Last neuron should have highest importance
        assert importance[-1] > importance[0]

    def test_get_pruning_mask(self, mock_model):
        """Test pruning mask generation."""
        from tropical_pruning.llm.ffn_counter import FFNStatistics
        from tropical_pruning.llm.ffn_pruner import FFNTropicalPruner

        stats = {
            0: FFNStatistics(
                layer_idx=0,
                intermediate_winner_count=torch.arange(128),
                intermediate_total_positions=1000,
            ),
        }

        pruner = FFNTropicalPruner(mock_model, stats)
        mask = pruner.get_pruning_mask(0, sparsity=0.5)

        assert mask.shape == (128,)
        assert mask.sum().item() == 64  # 50% kept
        # High-importance neurons should be kept
        assert mask[127].item() == True
        assert mask[0].item() == False

    def test_prune(self, mock_model):
        """Test pruning operation."""
        from tropical_pruning.llm.ffn_counter import FFNWinnerCounter
        from tropical_pruning.llm.ffn_pruner import FFNTropicalPruner

        # Collect real statistics
        counter = FFNWinnerCounter(mock_model, track_margin=False)
        input_ids = torch.randint(2, 1000, (8, 32))
        counter.forward(input_ids)
        stats = counter.get_statistics()
        counter.remove_hooks()

        # Prune
        pruner = FFNTropicalPruner(mock_model, stats)
        pruned_model = pruner.prune(sparsity=0.3, inplace=False)

        # Check dimensions changed
        for layer in pruned_model.model.layers:
            # After 30% pruning, should have ~90 neurons (70% of 128)
            expected_intermediate = int(128 * 0.7)
            assert layer.mlp.gate_proj.out_features == expected_intermediate
            assert layer.mlp.up_proj.out_features == expected_intermediate
            assert layer.mlp.down_proj.in_features == expected_intermediate

    def test_prune_forward_pass(self, mock_model):
        """Test that pruned model can do forward pass."""
        from tropical_pruning.llm.ffn_counter import FFNWinnerCounter
        from tropical_pruning.llm.ffn_pruner import FFNTropicalPruner

        counter = FFNWinnerCounter(mock_model, track_margin=False)
        input_ids = torch.randint(2, 1000, (4, 32))
        counter.forward(input_ids)
        stats = counter.get_statistics()
        counter.remove_hooks()

        pruner = FFNTropicalPruner(mock_model, stats)
        pruned_model = pruner.prune(sparsity=0.3, inplace=False)

        # Forward pass should work
        output = pruned_model(input_ids)
        assert output.logits.shape == (4, 32, 1000)

    def test_compression_stats(self, mock_model):
        """Test compression statistics."""
        from tropical_pruning.llm.ffn_counter import FFNWinnerCounter
        from tropical_pruning.llm.ffn_pruner import FFNTropicalPruner

        counter = FFNWinnerCounter(mock_model, track_margin=False)
        input_ids = torch.randint(2, 1000, (4, 32))
        counter.forward(input_ids)
        stats = counter.get_statistics()
        counter.remove_hooks()

        pruner = FFNTropicalPruner(mock_model, stats)
        pruned_model = pruner.prune(sparsity=0.3, inplace=False)

        compression = pruner.get_compression_stats()

        assert "original_parameters" in compression
        assert "pruned_parameters" in compression
        assert "compression_ratio" in compression
        assert "ffn_sparsity_achieved" in compression
        assert compression["compression_ratio"] > 1.0

    def test_per_layer_sparsity(self, mock_model):
        """Test per-layer sparsity setting."""
        from tropical_pruning.llm.ffn_counter import FFNWinnerCounter
        from tropical_pruning.llm.ffn_pruner import FFNTropicalPruner

        counter = FFNWinnerCounter(mock_model, track_margin=False)
        input_ids = torch.randint(2, 1000, (4, 32))
        counter.forward(input_ids)
        stats = counter.get_statistics()
        counter.remove_hooks()

        # Different sparsity per layer
        sparsity_dict = {0: 0.2, 1: 0.4, 2: 0.0, 3: 0.5}

        pruner = FFNTropicalPruner(mock_model, stats)
        pruned_model = pruner.prune(sparsity=sparsity_dict, inplace=False)

        layers = list(pruned_model.model.layers)

        # Layer 0: 20% pruned -> 80% of 128 = ~102
        assert layers[0].mlp.down_proj.in_features == int(128 * 0.8)

        # Layer 1: 40% pruned -> 60% of 128 = ~77
        assert layers[1].mlp.down_proj.in_features == int(128 * 0.6)

        # Layer 2: no pruning -> 128
        assert layers[2].mlp.down_proj.in_features == 128

        # Layer 3: 50% pruned -> 50% of 128 = 64
        assert layers[3].mlp.down_proj.in_features == int(128 * 0.5)


class TestCalibrationDataset:
    """Test CalibrationDataset utilities."""

    def test_from_random(self, mock_tokenizer):
        """Test random calibration data generation."""
        from tropical_pruning.llm.calibration import CalibrationDataset

        loader = CalibrationDataset.from_random(
            mock_tokenizer,
            num_samples=8,
            seq_length=32,
            batch_size=2,
        )

        batches = list(loader)
        assert len(batches) == 4  # 8 samples / batch size 2

        batch = batches[0]
        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert batch["input_ids"].shape == (2, 32)

    def test_from_text(self, mock_tokenizer):
        """Test calibration from custom text."""
        from tropical_pruning.llm.calibration import CalibrationDataset

        text = "This is a test. " * 1000

        loader = CalibrationDataset.from_text(
            text,
            mock_tokenizer,
            num_samples=4,
            seq_length=16,
            batch_size=2,
        )

        batches = list(loader)
        assert len(batches) >= 1


class TestIntegration:
    """Integration tests for the full pipeline."""

    def test_full_pipeline(self, mock_model, mock_tokenizer):
        """Test full pruning pipeline."""
        from tropical_pruning.llm.ffn_counter import FFNWinnerCounter
        from tropical_pruning.llm.ffn_pruner import FFNTropicalPruner
        from tropical_pruning.llm.calibration import CalibrationDataset

        # Create calibration data
        loader = CalibrationDataset.from_random(
            mock_tokenizer,
            num_samples=8,
            seq_length=32,
            batch_size=2,
        )

        # Collect statistics
        counter = FFNWinnerCounter(mock_model, track_margin=True)
        stats = counter.collect(loader, show_progress=False)
        counter.remove_hooks()

        # Prune
        pruner = FFNTropicalPruner(mock_model, stats)
        pruned_model = pruner.prune(sparsity=0.3, inplace=False)

        # Verify forward pass works
        input_ids = torch.randint(2, 1000, (2, 16))
        output = pruned_model(input_ids)
        assert output.logits.shape[0] == 2

        # Verify compression
        compression = pruner.get_compression_stats()
        assert compression["compression_ratio"] > 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
