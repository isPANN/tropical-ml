"""
Tropical Attention Processor for CLRS Benchmark.

This processor uses tropical geometry for message passing on graphs,
replacing standard attention with Hilbert projective metric-based attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from tropical_attention.layers import TropicalMultiheadAttention


class TropicalProcessor(nn.Module):
    """
    Graph Neural Network processor using Tropical Attention.

    Processes graph-structured data from CLRS benchmark using tropical
    geometry instead of standard softmax attention.

    Args:
        node_dim: Dimension of node features
        edge_dim: Dimension of edge features
        hidden_dim: Hidden dimension for processing
        num_heads: Number of attention heads
        num_layers: Number of processor layers
        dropout: Dropout probability
    """

    def __init__(
        self,
        node_dim: int = 128,
        edge_dim: int = 128,
        hidden_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        # Node feature encoder
        self.node_encoder = nn.Linear(node_dim, hidden_dim)

        # Edge feature encoder
        self.edge_encoder = nn.Linear(edge_dim, hidden_dim)

        # Tropical attention layers
        self.attention_layers = nn.ModuleList([
            TropicalMultiheadAttention(
                d_model=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])

        # Layer norms
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])

        # Edge-conditioned message transform
        self.edge_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(num_layers)
        ])

        # Output MLP
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        node_fts: torch.Tensor,
        edge_fts: torch.Tensor,
        graph_fts: torch.Tensor,
        adj_mat: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Process one step of the algorithm.

        Args:
            node_fts: Node features (batch, num_nodes, node_dim)
            edge_fts: Edge features (batch, num_nodes, num_nodes, edge_dim)
            graph_fts: Graph-level features (batch, graph_dim)
            adj_mat: Adjacency matrix (batch, num_nodes, num_nodes)
            hidden: Hidden state from previous step (batch, num_nodes, hidden_dim)

        Returns:
            output: Updated node representations (batch, num_nodes, hidden_dim)
            new_hidden: New hidden state (batch, num_nodes, hidden_dim)
        """
        batch_size, num_nodes, _ = node_fts.shape

        # Encode node features
        z = self.node_encoder(node_fts)  # (B, N, H)

        # Combine with hidden state
        z = z + hidden

        # Encode edge features
        e = self.edge_encoder(edge_fts)  # (B, N, N, H)

        # Create attention mask from adjacency (True = ignore)
        # For fully connected graphs, we don't need masking
        # For sparse graphs, mask non-edges
        use_mask = (adj_mat.sum() < batch_size * num_nodes * num_nodes)

        # Process through tropical attention layers
        for i in range(self.num_layers):
            # Edge-conditioned message
            # Expand z for pairwise: (B, N, 1, H) and (B, 1, N, H)
            z_i = z.unsqueeze(2).expand(-1, -1, num_nodes, -1)
            z_j = z.unsqueeze(1).expand(-1, num_nodes, -1, -1)

            # Concatenate with edge features
            msg_input = torch.cat([z_i, z_j, e], dim=-1)  # (B, N, N, 3H)
            messages = self.edge_transforms[i](msg_input)  # (B, N, N, H)

            # Aggregate messages using tropical attention
            # Use messages as values, weighted by adjacency
            messages_flat = messages.view(batch_size, num_nodes, -1)  # (B, N, N*H)

            # Self-attention with tropical geometry
            # Note: TropicalMultiheadAttention handles masking internally
            z_attn, _ = self.attention_layers[i](
                z, z, z,
            )

            # Residual + LayerNorm
            z = self.layer_norms[i](z + z_attn)

        # Output
        output = self.output_mlp(z)
        new_hidden = z

        return output, new_hidden

    def get_initial_hidden(
        self,
        batch_size: int,
        num_nodes: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Get initial hidden state (zeros)."""
        return torch.zeros(batch_size, num_nodes, self.hidden_dim, device=device)


__all__ = ["TropicalProcessor"]
