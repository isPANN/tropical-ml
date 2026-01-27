#!/usr/bin/env python3
"""
Training script for CLRS Benchmark with Tropical Attention.

Usage:
    tropical-clrs --algorithm bfs --epochs 100
    tropical-clrs --algorithm dijkstra --hidden_dim 256 --num_heads 8
    tropical-clrs --algorithm bfs --device cuda  # GPU training
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    import clrs
except ImportError:
    raise ImportError(
        "CLRS not installed. Run: pip install dm-clrs\n"
        "Or: uv add dm-clrs"
    )

try:
    from .processor import TropicalProcessor
except ImportError:
    from processor import TropicalProcessor


# Available algorithms in CLRS
ALGORITHMS = [
    # Sorting
    "insertion_sort", "bubble_sort", "heapsort", "quicksort",
    # Searching
    "minimum", "binary_search", "find_maximum_subarray",
    # Divide and conquer
    "maximum_subarray",
    # Greedy
    "activity_selector", "task_scheduling",
    # Dynamic Programming
    "matrix_chain_order", "lcs_length", "optimal_bst",
    # Graph - BFS/DFS
    "bfs", "dfs", "topological_sort", "strongly_connected_components",
    # Graph - Shortest paths
    "dijkstra", "bellman_ford", "dag_shortest_paths", "floyd_warshall",
    # Graph - MST
    "prim", "kruskal",
    # Strings
    "naive_string_matcher", "kmp_matcher",
    # Geometry
    "segments_intersect", "graham_scan", "jarvis_march",
]


class CLRSDataset(Dataset):
    """PyTorch Dataset wrapper for CLRS data."""

    def __init__(
        self,
        algorithm: str,
        split: str = "train",
        data_dir: str = "/tmp/CLRS30",
    ):
        self.algorithm = algorithm
        self.split = split
        self.data_dir = data_dir

        # Determine number of samples and length based on split
        if split == "train":
            self.num_samples = 1000
            self.length = 16
        elif split == "val":
            self.num_samples = 32
            self.length = 16
        else:  # test
            self.num_samples = 32
            self.length = 64

        # Create sampler with correct parameters
        self.sampler, self.spec = clrs.build_sampler(
            algorithm,
            num_samples=self.num_samples,
            length=self.length,
            seed=42 if split == "train" else 123,
        )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Sample a new problem instance
        feedback = self.sampler.next(batch_size=1)

        # Convert JAX arrays to PyTorch tensors
        inputs = {}
        outputs = {}
        hints = {}

        # Process features
        for inp in feedback.features.inputs:
            name = inp.name
            data = np.array(inp.data)
            inputs[name] = torch.from_numpy(data).float().squeeze(0)

        # Process outputs
        for out in feedback.outputs:
            name = out.name
            data = np.array(out.data)
            outputs[name] = torch.from_numpy(data).float().squeeze(0)

        # Process hints (intermediate states)
        for hint in feedback.features.hints:
            name = hint.name
            data = np.array(hint.data)
            hints[name] = torch.from_numpy(data).float().squeeze(0)

        return {
            "inputs": inputs,
            "outputs": outputs,
            "hints": hints,
            "lengths": int(feedback.features.lengths[0]),
        }


def collate_fn(batch):
    """Custom collate function for variable-sized graphs."""
    # For now, assume fixed size within batch
    inputs = {}
    outputs = {}
    hints = {}

    # Collect all keys
    input_keys = batch[0]["inputs"].keys()
    output_keys = batch[0]["outputs"].keys()
    hint_keys = batch[0]["hints"].keys()

    # Stack tensors
    for key in input_keys:
        inputs[key] = torch.stack([item["inputs"][key] for item in batch])
    for key in output_keys:
        outputs[key] = torch.stack([item["outputs"][key] for item in batch])
    for key in hint_keys:
        hints[key] = torch.stack([item["hints"][key] for item in batch])

    lengths = torch.tensor([item["lengths"] for item in batch])

    return {
        "inputs": inputs,
        "outputs": outputs,
        "hints": hints,
        "lengths": lengths,
    }


class TropicalCLRSNet(nn.Module):
    """
    Neural network for CLRS using Tropical Attention processor.
    """

    def __init__(
        self,
        spec,
        hidden_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.spec = spec

        # Processor
        self.processor = TropicalProcessor(
            node_dim=hidden_dim,
            edge_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )

        # Build encoders based on spec (location-aware)
        self.node_encoders = nn.ModuleDict()
        self.edge_encoders = nn.ModuleDict()
        self.decoders = nn.ModuleDict()

        for name, (stage_spec, loc, dtype) in spec.items():
            if stage_spec in [clrs.Stage.INPUT, clrs.Stage.HINT]:
                # Determine input dimension based on type
                in_dim = 1  # Most types are scalar per element

                if loc == clrs.Location.NODE:
                    self.node_encoders[name] = nn.Linear(in_dim, hidden_dim)
                elif loc == clrs.Location.EDGE:
                    self.edge_encoders[name] = nn.Linear(in_dim, hidden_dim)

        # Build decoders for outputs
        for name, (stage_spec, loc, dtype) in spec.items():
            if stage_spec == clrs.Stage.OUTPUT:
                if dtype == clrs.Type.POINTER:
                    # Pointer network: predict which node to point to
                    self.decoders[name] = nn.Linear(hidden_dim, hidden_dim)
                else:
                    self.decoders[name] = nn.Linear(hidden_dim, 1)

    def forward(self, inputs, hints, lengths):
        """
        Forward pass.

        Args:
            inputs: Dictionary of input tensors
            hints: Dictionary of hint tensors
            lengths: Number of algorithm steps

        Returns:
            outputs: Dictionary of output predictions
        """
        # Get dimensions from a node input
        for name, (stage, loc, dtype) in self.spec.items():
            if stage == clrs.Stage.INPUT and loc == clrs.Location.NODE and name in inputs:
                sample = inputs[name]
                break

        batch_size, num_nodes = sample.shape[:2]
        device = sample.device

        # Encode node features
        node_fts = torch.zeros(batch_size, num_nodes, self.hidden_dim, device=device)
        for name, tensor in inputs.items():
            if name in self.node_encoders:
                # Node inputs: (B, N) -> (B, N, 1) -> (B, N, H)
                if tensor.dim() == 2:
                    tensor = tensor.unsqueeze(-1)
                encoded = self.node_encoders[name](tensor.float())
                node_fts = node_fts + encoded

        # Encode edge features
        edge_fts = torch.zeros(batch_size, num_nodes, num_nodes, self.hidden_dim, device=device)
        adj_mat = torch.ones(batch_size, num_nodes, num_nodes, device=device)

        for name, tensor in inputs.items():
            if name in self.edge_encoders:
                # Edge inputs: (B, N, N) -> (B, N, N, 1) -> (B, N, N, H)
                if tensor.dim() == 3:
                    tensor = tensor.unsqueeze(-1)
                encoded = self.edge_encoders[name](tensor.float())
                edge_fts = edge_fts + encoded

                # Use adjacency info if available
                if name in ['adj', 'A']:
                    adj_mat = (inputs[name].float() > 0).float()

        # Graph features
        graph_fts = torch.zeros(batch_size, self.hidden_dim, device=device)

        # Initialize hidden
        hidden = self.processor.get_initial_hidden(batch_size, num_nodes, device)

        # Process for max_length steps
        max_steps = lengths.max().item() if lengths.numel() > 0 else 1
        for step in range(max_steps):
            output, hidden = self.processor(
                node_fts, edge_fts, graph_fts, adj_mat, hidden
            )

        # Decode outputs
        outputs = {}
        for name, decoder in self.decoders.items():
            if self.spec[name][2] == clrs.Type.POINTER:
                # Pointer: compute attention scores
                query = decoder(output)  # (B, N, H)
                key = output  # (B, N, H)
                outputs[name] = torch.bmm(query, key.transpose(1, 2))  # (B, N, N)
            else:
                outputs[name] = decoder(output).squeeze(-1)  # (B, N)

        return outputs


def compute_loss(predictions, targets, spec):
    """Compute loss based on output types."""
    total_loss = 0.0
    num_outputs = 0

    for name, pred in predictions.items():
        if name not in targets:
            continue

        target = targets[name]
        output_type = spec[name][2]

        if output_type == clrs.Type.POINTER:
            # Cross entropy for pointer
            # pred: (B, N, N), target: (B, N) indices
            pred_flat = pred.view(-1, pred.shape[-1])
            target_flat = target.long().view(-1)
            loss = F.cross_entropy(pred_flat, target_flat)
        elif output_type == clrs.Type.MASK:
            # Binary cross entropy for mask
            loss = F.binary_cross_entropy_with_logits(pred, target.float())
        else:
            # MSE for scalars
            loss = F.mse_loss(pred, target.float())

        total_loss += loss
        num_outputs += 1

    return total_loss / max(num_outputs, 1)


def compute_accuracy(predictions, targets, spec):
    """Compute accuracy for pointer/mask outputs."""
    correct = 0
    total = 0

    for name, pred in predictions.items():
        if name not in targets:
            continue

        target = targets[name]
        output_type = spec[name][2]

        if output_type == clrs.Type.POINTER:
            # Argmax accuracy
            pred_idx = pred.argmax(dim=-1)  # (B, N)
            correct += (pred_idx == target.long()).float().mean().item()
            total += 1
        elif output_type == clrs.Type.MASK:
            pred_mask = (pred > 0).float()
            correct += (pred_mask == target).float().mean().item()
            total += 1

    return correct / max(total, 1)


def train_epoch(model, dataloader, optimizer, spec, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0
    use_cuda = device.type == "cuda"

    for batch in dataloader:
        # Async data transfer for GPU
        inputs = {k: v.to(device, non_blocking=use_cuda) for k, v in batch["inputs"].items()}
        outputs = {k: v.to(device, non_blocking=use_cuda) for k, v in batch["outputs"].items()}
        hints = {k: v.to(device, non_blocking=use_cuda) for k, v in batch["hints"].items()}
        lengths = batch["lengths"].to(device, non_blocking=use_cuda)

        optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()

        predictions = model(inputs, hints, lengths)
        loss = compute_loss(predictions, outputs, spec)
        acc = compute_accuracy(predictions, outputs, spec)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += acc
        num_batches += 1

    return total_loss / num_batches, total_acc / num_batches


@torch.no_grad()
def evaluate(model, dataloader, spec, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0
    use_cuda = device.type == "cuda"

    for batch in dataloader:
        inputs = {k: v.to(device, non_blocking=use_cuda) for k, v in batch["inputs"].items()}
        outputs = {k: v.to(device, non_blocking=use_cuda) for k, v in batch["outputs"].items()}
        hints = {k: v.to(device, non_blocking=use_cuda) for k, v in batch["hints"].items()}
        lengths = batch["lengths"].to(device, non_blocking=use_cuda)

        predictions = model(inputs, hints, lengths)
        loss = compute_loss(predictions, outputs, spec)
        acc = compute_accuracy(predictions, outputs, spec)

        total_loss += loss.item()
        total_acc += acc
        num_batches += 1

    return total_loss / num_batches, total_acc / num_batches


def main():
    parser = argparse.ArgumentParser(description="Train Tropical Attention on CLRS")
    parser.add_argument("--algorithm", type=str, default="bfs", choices=ALGORITHMS)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers (0 for main process)")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    # Setup device
    device = torch.device(args.device)
    use_cuda = device.type == "cuda"

    print(f"Training on {args.algorithm} with Tropical Attention")
    print(f"Device: {device}")

    # Show GPU info if using CUDA
    if use_cuda:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        # Enable TF32 for faster matmuls on Ampere+ GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Create datasets
    train_dataset = CLRSDataset(args.algorithm, split="train")
    val_dataset = CLRSDataset(args.algorithm, split="val")
    test_dataset = CLRSDataset(args.algorithm, split="test")

    # Get spec
    _, spec = clrs.build_sampler(args.algorithm, num_samples=1, length=16, seed=42)

    # DataLoader settings optimized for GPU
    loader_kwargs = {
        "batch_size": args.batch_size,
        "collate_fn": collate_fn,
        "num_workers": args.num_workers,
        "pin_memory": use_cuda,  # Faster CPU->GPU transfer
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, **loader_kwargs)
    test_loader = DataLoader(test_dataset, **loader_kwargs)

    # Create model
    model = TropicalCLRSNet(
        spec=spec,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    best_val_acc = 0.0
    save_dir = Path(args.save_dir) / args.algorithm
    save_dir.mkdir(parents=True, exist_ok=True)

    total_start = time.time()
    for epoch in range(args.epochs):
        epoch_start = time.time()

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, spec, device)
        val_loss, val_acc = evaluate(model, val_loader, spec, device)

        epoch_time = time.time() - epoch_start

        print(f"Epoch {epoch+1}/{args.epochs} ({epoch_time:.1f}s)")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_dir / "best_model.pt")
            print(f"  Saved best model (val_acc: {val_acc:.4f})")

    total_time = time.time() - total_start
    print(f"\nTraining completed in {total_time:.1f}s")

    # Test evaluation
    model.load_state_dict(torch.load(save_dir / "best_model.pt", weights_only=True))
    test_loss, test_acc = evaluate(model, test_loader, spec, device)
    print(f"\nTest Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
