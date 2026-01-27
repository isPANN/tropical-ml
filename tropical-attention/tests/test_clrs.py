"""Tests for CLRS benchmark integration."""

import pytest
import torch
import sys
sys.path.insert(0, '.')

from benchmarks.clrs.processor import TropicalProcessor
from benchmarks.clrs.model import TropicalCLRSModel


class TestTropicalProcessor:
    """Tests for TropicalProcessor."""

    def test_initialization(self):
        """Test processor initialization."""
        processor = TropicalProcessor(
            node_dim=64,
            edge_dim=64,
            hidden_dim=64,
            num_heads=4,
            num_layers=2,
        )
        assert processor.hidden_dim == 64
        assert processor.num_heads == 4
        assert processor.num_layers == 2

    def test_forward_shape(self):
        """Test output shapes."""
        processor = TropicalProcessor(
            node_dim=64,
            edge_dim=64,
            hidden_dim=64,
            num_heads=4,
        )

        batch_size, num_nodes, hidden_dim = 4, 16, 64
        node_fts = torch.randn(batch_size, num_nodes, hidden_dim)
        edge_fts = torch.randn(batch_size, num_nodes, num_nodes, hidden_dim)
        graph_fts = torch.randn(batch_size, hidden_dim)
        adj_mat = torch.ones(batch_size, num_nodes, num_nodes)
        hidden = processor.get_initial_hidden(batch_size, num_nodes, node_fts.device)

        output, new_hidden = processor(node_fts, edge_fts, graph_fts, adj_mat, hidden)

        assert output.shape == (batch_size, num_nodes, hidden_dim)
        assert new_hidden.shape == (batch_size, num_nodes, hidden_dim)

    def test_gradient_flow(self):
        """Test that gradients flow through the processor."""
        hidden_dim = 32
        processor = TropicalProcessor(
            node_dim=hidden_dim,
            edge_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_heads=4,
        )

        batch_size, num_nodes = 2, 8
        node_fts = torch.randn(batch_size, num_nodes, hidden_dim, requires_grad=True)
        edge_fts = torch.randn(batch_size, num_nodes, num_nodes, hidden_dim)
        graph_fts = torch.randn(batch_size, hidden_dim)
        adj_mat = torch.ones(batch_size, num_nodes, num_nodes)
        hidden = processor.get_initial_hidden(batch_size, num_nodes, node_fts.device)

        output, _ = processor(node_fts, edge_fts, graph_fts, adj_mat, hidden)
        loss = output.sum()
        loss.backward()

        assert node_fts.grad is not None
        assert not torch.isnan(node_fts.grad).any()

    def test_initial_hidden(self):
        """Test initial hidden state."""
        processor = TropicalProcessor(hidden_dim=64)
        hidden = processor.get_initial_hidden(4, 16, torch.device('cpu'))

        assert hidden.shape == (4, 16, 64)
        assert (hidden == 0).all()

    def test_sparse_adjacency(self):
        """Test with sparse adjacency matrix."""
        hidden_dim = 32
        processor = TropicalProcessor(
            node_dim=hidden_dim,
            edge_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_heads=4,
        )

        batch_size, num_nodes = 2, 8
        node_fts = torch.randn(batch_size, num_nodes, hidden_dim)
        edge_fts = torch.randn(batch_size, num_nodes, num_nodes, hidden_dim)
        graph_fts = torch.randn(batch_size, hidden_dim)
        # Sparse adjacency (only some edges)
        adj_mat = torch.eye(num_nodes).unsqueeze(0).expand(batch_size, -1, -1)
        hidden = processor.get_initial_hidden(batch_size, num_nodes, node_fts.device)

        output, new_hidden = processor(node_fts, edge_fts, graph_fts, adj_mat, hidden)

        assert output.shape == (batch_size, num_nodes, hidden_dim)
        assert not torch.isnan(output).any()


class TestTropicalCLRSModel:
    """Tests for TropicalCLRSModel."""

    def test_initialization(self):
        """Test model initialization."""
        model = TropicalCLRSModel(
            hidden_dim=64,
            num_heads=4,
            num_layers=2,
        )
        assert model.hidden_dim == 64

    def test_add_encoder_decoder(self):
        """Test adding encoders and decoders."""
        model = TropicalCLRSModel(hidden_dim=64)

        model.add_encoder("node_input", input_dim=1, location="node")
        model.add_encoder("edge_input", input_dim=1, location="edge")
        model.add_decoder("node_output", output_dim=1, location="node")
        model.add_decoder("pointer_output", output_dim=1, location="pointer")

        assert "node_input" in model.node_encoders
        assert "edge_input" in model.edge_encoders
        assert "node_output" in model.node_decoders
        assert "pointer_output" in model.pointer_decoders


try:
    import clrs
    CLRS_AVAILABLE = True
except ImportError:
    CLRS_AVAILABLE = False


@pytest.mark.skipif(not CLRS_AVAILABLE, reason="CLRS not installed")
class TestCLRSIntegration:
    """Integration tests with CLRS library."""

    def test_clrs_sampler(self):
        """Test CLRS sampler works."""
        sampler, spec = clrs.build_sampler('bfs', num_samples=10, length=8, seed=42)
        feedback = sampler.next(batch_size=2)

        assert len(feedback.features.inputs) > 0
        assert len(feedback.outputs) > 0

    def test_clrs_data_to_pytorch(self):
        """Test converting CLRS data to PyTorch tensors."""
        import numpy as np

        sampler, spec = clrs.build_sampler('bfs', num_samples=10, length=8, seed=42)
        feedback = sampler.next(batch_size=2)

        # Convert inputs to PyTorch
        for inp in feedback.features.inputs:
            data = np.array(inp.data)
            tensor = torch.from_numpy(data).float()
            assert tensor.dim() >= 2

    def test_full_training_step(self):
        """Test a full training step."""
        import numpy as np

        # Create sampler
        sampler, spec = clrs.build_sampler('bfs', num_samples=10, length=8, seed=42)

        # Import training components
        from benchmarks.clrs.train import TropicalCLRSNet

        # Create model
        model = TropicalCLRSNet(
            spec=spec,
            hidden_dim=32,
            num_heads=4,
        )

        # Get a batch
        feedback = sampler.next(batch_size=2)

        # Convert to PyTorch
        inputs = {}
        outputs = {}
        for inp in feedback.features.inputs:
            inputs[inp.name] = torch.from_numpy(np.array(inp.data)).float()
        for out in feedback.outputs:
            outputs[out.name] = torch.from_numpy(np.array(out.data)).float()

        lengths = torch.tensor([6, 6])

        # Forward pass
        predictions = model(inputs, {}, lengths)

        assert len(predictions) > 0
        for name, pred in predictions.items():
            assert not torch.isnan(pred).any()
