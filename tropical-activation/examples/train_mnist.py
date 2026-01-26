#!/usr/bin/env python3
"""
Train Tropical Neural Networks on MNIST.

Compares three architectures:
1. Standard ReLU Network (baseline)
2. Hybrid Tropical NN (Linear -> TropicalAffine) - RECOMMENDED
3. Full MMP Tropical NN (Linear -> MaxPlus -> MinPlus)

Usage:
    python train_mnist.py --model baseline --epochs 15
    python train_mnist.py --model hybrid --epochs 15    # Recommended
    python train_mnist.py --model mmp --epochs 15

Architecture Details:
    Baseline:  Linear -> ReLU -> Linear -> ReLU -> Linear
    Hybrid:    Linear -> TropicalAffine -> Linear -> TropicalAffine -> Linear
    MMP:       Linear -> MaxPlus -> MinPlus -> Linear -> MaxPlus -> MinPlus -> Linear

TropicalAffine/MaxPlusAffine: y[i] = max(max_k(LayerNorm(x)[k] + W[k,i]), b[i])
MinPlusAffine: y[i] = min(min_k(LayerNorm(x)[k] + W[k,i]), b[i])

GPU Acceleration:
    When --use-gpu is enabled, tropical layers use optimized CUDA kernels from
    tropical-gemm for both forward and backward passes.

Requirements:
    pip install tropical-activation torchvision
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from tropical_activation import HybridTropicalNN, TropicalNN, GPU_AVAILABLE
from tropical_activation.training import tropical_weight_init, count_parameters


class BaselineMLP(nn.Module):
    """Standard MLP with ReLU activations for comparison."""

    def __init__(self, layer_sizes: list, dropout: float = 0.0):
        super().__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x.view(x.size(0), -1))


def create_model(model_type: str, layer_sizes: list, dropout: float = 0.0, use_gpu: bool = False):
    """
    Create model based on type.

    Args:
        model_type: One of "baseline", "hybrid", "mmp"
        layer_sizes: List of layer sizes [input, hidden1, hidden2, ..., output]
        dropout: Dropout probability
        use_gpu: Use tropical-gemm GPU kernels for tropical layers

    Returns:
        Model instance
    """
    if model_type == "baseline":
        return BaselineMLP(layer_sizes, dropout=dropout)
    elif model_type == "hybrid":
        # Linear -> TropicalAffine (RECOMMENDED)
        return HybridTropicalNN(layer_sizes, dropout=dropout, use_gpu=use_gpu)
    elif model_type == "mmp":
        # Linear -> MaxPlus -> MinPlus
        return TropicalNN(layer_sizes, dropout=dropout, use_gpu=use_gpu)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose from: baseline, hybrid, mmp")


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for data, target in loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data.view(data.size(0), -1))
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += output.argmax(1).eq(target).sum().item()
        total += target.size(0)

    return total_loss / len(loader), 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for data, target in loader:
        data, target = data.to(device), target.to(device)

        output = model(data.view(data.size(0), -1))
        total_loss += criterion(output, target).item()
        correct += output.argmax(1).eq(target).sum().item()
        total += target.size(0)

    return total_loss / len(loader), 100.0 * correct / total


def main():
    parser = argparse.ArgumentParser(description="Train Tropical NN on MNIST")
    parser.add_argument("--model", type=str, default="hybrid",
                       choices=["baseline", "hybrid", "mmp"],
                       help="Model type: baseline (ReLU), hybrid (recommended), mmp (full)")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--init-scale", type=float, default=0.5,
                       help="Initialization scale for tropical weights")
    parser.add_argument("--use-gpu", action="store_true",
                       help="Use tropical-gemm GPU kernels (requires CUDA)")
    parser.add_argument("--data-dir", type=str, default="./data")
    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Determine if we should use tropical-gemm GPU kernels
    use_tropical_gpu = args.use_gpu and GPU_AVAILABLE and torch.cuda.is_available()

    print("=" * 60)
    print(f"MNIST Training: {args.model.upper()}")
    print("=" * 60)
    print(f"Device: {device}")
    if args.model in ["hybrid", "mmp"]:
        print(f"Tropical GPU kernels: {'enabled' if use_tropical_gpu else 'disabled'}")
        if args.use_gpu and not use_tropical_gpu:
            if not GPU_AVAILABLE:
                print("  (tropical-gemm GPU not available)")
            elif not torch.cuda.is_available():
                print("  (CUDA not available)")

    # Print architecture description
    if args.model == "baseline":
        print("\nArchitecture: Linear -> ReLU -> Linear -> ReLU -> Linear")
    elif args.model == "hybrid":
        print("\nArchitecture: Linear -> TropicalAffine -> Linear -> TropicalAffine -> Linear")
        print("TropicalAffine: y[i] = max(max_k(LayerNorm(x)[k] + W[k,i]), b[i])")
        print("(RECOMMENDED architecture)")
    else:  # mmp
        print("\nArchitecture: Linear -> MaxPlus -> MinPlus -> Linear -> MaxPlus -> MinPlus -> Linear")
        print("MaxPlusAffine: y[i] = max(max_k(LayerNorm(x)[k] + W[k,i]), b[i])")
        print("MinPlusAffine: y[i] = min(min_k(LayerNorm(x)[k] + W[k,i]), b[i])")

    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_data = datasets.MNIST(args.data_dir, train=True, download=True, transform=transform)
    test_data = datasets.MNIST(args.data_dir, train=False, transform=transform)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size)

    print(f"\nTrain: {len(train_data)}, Test: {len(test_data)}")

    # Model
    layer_sizes = [784, 256, 128, 10]
    model = create_model(args.model, layer_sizes, dropout=args.dropout, use_gpu=use_tropical_gpu)
    model.to(device)

    # Initialize tropical weights
    if args.model in ["hybrid", "mmp"]:
        tropical_weight_init(model, init_scale=args.init_scale)

    # Print model info
    param_counts = count_parameters(model)
    print(f"\nParameters: {param_counts['total']:,}")
    if args.model in ["hybrid", "mmp"]:
        print(f"  Tropical: {param_counts['tropical']:,}")
        print(f"  Linear: {param_counts['linear']:,}")

    # Training
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    print(f"\nTraining for {args.epochs} epochs...")
    print("-" * 55)

    best_acc = 0
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        marker = " *" if test_acc > best_acc else ""
        if test_acc > best_acc:
            best_acc = test_acc

        print(f"Epoch {epoch:2d}/{args.epochs}: "
              f"Loss={train_loss:.4f}, Train={train_acc:.1f}%, Test={test_acc:.1f}%{marker}")

    print("-" * 55)
    print(f"Time: {time.time() - start:.1f}s, Best Test Accuracy: {best_acc:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
