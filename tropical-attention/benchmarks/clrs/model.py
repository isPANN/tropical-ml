"""
Tropical CLRS Model.

Full model for CLRS benchmark with encoder, processor, and decoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .processor import TropicalProcessor


class TropicalCLRSModel(nn.Module):
    """
    Complete model for CLRS algorithmic reasoning benchmark.

    Architecture:
        1. Encoder: Maps algorithm inputs/hints to hidden representations
        2. Processor: Tropical attention-based message passing
        3. Decoder: Maps hidden representations to outputs

    Args:
        hidden_dim: Hidden dimension throughout the model
        num_heads: Number of attention heads
        num_layers: Number of processor layers
        encode_hints: Whether to encode intermediate hints
        dropout: Dropout probability
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 1,
        encode_hints: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.encode_hints = encode_hints

        # Processor
        self.processor = TropicalProcessor(
            node_dim=hidden_dim,
            edge_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )

        # Encoders for different data types (will be populated per algorithm)
        self.node_encoders = nn.ModuleDict()
        self.edge_encoders = nn.ModuleDict()
        self.graph_encoders = nn.ModuleDict()

        # Decoders for outputs
        self.node_decoders = nn.ModuleDict()
        self.edge_decoders = nn.ModuleDict()
        self.graph_decoders = nn.ModuleDict()
        self.pointer_decoders = nn.ModuleDict()

    def encode_inputs(
        self,
        inputs: dict,
        batch_size: int,
        num_nodes: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode algorithm inputs to hidden representations.

        Args:
            inputs: Dictionary of input tensors
            batch_size: Batch size
            num_nodes: Number of nodes

        Returns:
            node_fts: Encoded node features
            edge_fts: Encoded edge features
            graph_fts: Encoded graph features
        """
        device = next(iter(inputs.values())).device

        # Initialize feature tensors
        node_fts = torch.zeros(batch_size, num_nodes, self.hidden_dim, device=device)
        edge_fts = torch.zeros(batch_size, num_nodes, num_nodes, self.hidden_dim, device=device)
        graph_fts = torch.zeros(batch_size, self.hidden_dim, device=device)

        for name, tensor in inputs.items():
            if name in self.node_encoders:
                node_fts = node_fts + self.node_encoders[name](tensor)
            elif name in self.edge_encoders:
                edge_fts = edge_fts + self.edge_encoders[name](tensor)
            elif name in self.graph_encoders:
                graph_fts = graph_fts + self.graph_encoders[name](tensor)

        return node_fts, edge_fts, graph_fts

    def decode_outputs(
        self,
        node_fts: torch.Tensor,
        edge_fts: torch.Tensor,
        graph_fts: torch.Tensor,
        output_types: dict,
    ) -> dict:
        """
        Decode hidden representations to algorithm outputs.

        Args:
            node_fts: Node features (batch, num_nodes, hidden_dim)
            edge_fts: Edge features (batch, num_nodes, num_nodes, hidden_dim)
            graph_fts: Graph features (batch, hidden_dim)
            output_types: Dictionary specifying output names and types

        Returns:
            outputs: Dictionary of decoded outputs
        """
        outputs = {}

        for name, out_type in output_types.items():
            if out_type == "node" and name in self.node_decoders:
                outputs[name] = self.node_decoders[name](node_fts)
            elif out_type == "edge" and name in self.edge_decoders:
                outputs[name] = self.edge_decoders[name](edge_fts)
            elif out_type == "graph" and name in self.graph_decoders:
                outputs[name] = self.graph_decoders[name](graph_fts)
            elif out_type == "pointer" and name in self.pointer_decoders:
                # Pointer: compute attention scores between nodes
                ptr_query = self.pointer_decoders[name](node_fts)  # (B, N, H)
                ptr_key = node_fts  # (B, N, H)
                outputs[name] = torch.bmm(ptr_query, ptr_key.transpose(1, 2))  # (B, N, N)

        return outputs

    def forward(
        self,
        inputs: dict,
        hints: dict | None = None,
        adj_mat: torch.Tensor | None = None,
        num_steps: int = 1,
    ) -> dict:
        """
        Forward pass through the model.

        Args:
            inputs: Dictionary of input tensors
            hints: Optional dictionary of hint tensors (for training)
            adj_mat: Adjacency matrix (batch, num_nodes, num_nodes)
            num_steps: Number of processing steps

        Returns:
            outputs: Dictionary of output tensors
        """
        # Get dimensions from inputs
        sample_tensor = next(iter(inputs.values()))
        if sample_tensor.dim() == 2:
            batch_size, num_nodes = sample_tensor.shape
        else:
            batch_size = sample_tensor.shape[0]
            num_nodes = sample_tensor.shape[1]

        device = sample_tensor.device

        # Default adjacency: fully connected
        if adj_mat is None:
            adj_mat = torch.ones(batch_size, num_nodes, num_nodes, device=device)

        # Encode inputs
        node_fts, edge_fts, graph_fts = self.encode_inputs(inputs, batch_size, num_nodes)

        # Initialize hidden state
        hidden = self.processor.get_initial_hidden(batch_size, num_nodes, device)

        # Process for num_steps
        for step in range(num_steps):
            # Optionally encode hints
            if self.encode_hints and hints is not None:
                hint_node, hint_edge, hint_graph = self.encode_inputs(
                    {k: v[:, step] for k, v in hints.items() if v.dim() > 2},
                    batch_size,
                    num_nodes,
                )
                node_fts = node_fts + hint_node
                edge_fts = edge_fts + hint_edge

            # Process one step
            output, hidden = self.processor(
                node_fts, edge_fts, graph_fts, adj_mat, hidden
            )

        # For now, return the final hidden state
        # In practice, you'd decode this based on the algorithm's output spec
        return {
            "node_fts": output,
            "hidden": hidden,
        }

    def add_encoder(
        self,
        name: str,
        input_dim: int,
        location: str = "node",
    ):
        """Add an encoder for a specific input."""
        encoder = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        if location == "node":
            self.node_encoders[name] = encoder
        elif location == "edge":
            self.edge_encoders[name] = encoder
        elif location == "graph":
            self.graph_encoders[name] = encoder

    def add_decoder(
        self,
        name: str,
        output_dim: int,
        location: str = "node",
    ):
        """Add a decoder for a specific output."""
        if location == "pointer":
            decoder = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.pointer_decoders[name] = decoder
        else:
            decoder = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, output_dim),
            )
            if location == "node":
                self.node_decoders[name] = decoder
            elif location == "edge":
                self.edge_decoders[name] = decoder
            elif location == "graph":
                self.graph_decoders[name] = decoder


__all__ = ["TropicalCLRSModel"]
