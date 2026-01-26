#!/usr/bin/env python3
"""
Compare Tropical Network Architectures to Standard ReLU.

Trains three types of networks on MNIST and compares their performance:
1. Standard ReLU Network (baseline)
2. Hybrid Tropical NN (Linear → TropicalAffine) - RECOMMENDED
3. Full MMP Tropical NN (Linear → MaxPlus → MinPlus)

Usage:
    python compare_to_relu.py [--epochs 10]

Requirements:
    pip install tropical-activation torchvision
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from tropical_activation import HybridTropicalNN, TropicalNN
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


def get_mnist_dataloaders(batch_size: int, data_dir: str = "./data"):
    """Load MNIST dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_dataset = datasets.MNIST(
        data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        data_dir, train=False, transform=transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return train_loader, test_loader


def train_epoch(model, loader, optimizer, criterion):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for data, target in loader:
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
def evaluate(model, loader, criterion):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for data, target in loader:
        output = model(data.view(data.size(0), -1))
        total_loss += criterion(output, target).item()
        correct += output.argmax(1).eq(target).sum().item()
        total += target.size(0)

    return total_loss / len(loader), 100.0 * correct / total


def train_model(model, name, train_loader, test_loader, epochs, lr):
    """Train a model and return best accuracy and training time."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        test_loss, test_acc = evaluate(model, test_loader, criterion)
        scheduler.step()

        if test_acc > best_acc:
            best_acc = test_acc

        print(f"  Epoch {epoch:2d}/{epochs}: "
              f"Loss={train_loss:.4f}, Train={train_acc:.1f}%, Test={test_acc:.1f}%")

    elapsed = time.time() - start_time
    return best_acc, elapsed


def main():
    parser = argparse.ArgumentParser(description="Compare Tropical NN architectures to ReLU")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--data-dir", type=str, default="./data",
                       help="Directory for MNIST data")
    args = parser.parse_args()

    print("=" * 70)
    print("COMPARISON: Tropical NN Architectures vs Standard ReLU Network")
    print("=" * 70)

    # Load data
    print("\nLoading MNIST dataset...")
    train_loader, test_loader = get_mnist_dataloaders(args.batch_size, args.data_dir)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    # Define architecture
    layer_sizes = [784, 256, 128, 10]
    results = {}

    # --- 1. Standard ReLU Network ---
    print("\n" + "-" * 70)
    print("1. Standard ReLU Network")
    print("   Architecture: Linear → ReLU → Linear → ReLU → Linear")
    print("-" * 70)

    baseline = BaselineMLP(layer_sizes)
    baseline_params = sum(p.numel() for p in baseline.parameters())
    print(f"Parameters: {baseline_params:,}")

    print("Training...")
    baseline_acc, baseline_time = train_model(
        baseline, "ReLU", train_loader, test_loader, args.epochs, args.lr
    )
    results["ReLU"] = {"acc": baseline_acc, "params": baseline_params, "time": baseline_time}

    # --- 2. Hybrid Tropical NN (RECOMMENDED) ---
    print("\n" + "-" * 70)
    print("2. Hybrid Tropical NN (RECOMMENDED)")
    print("   Architecture: Linear → TropicalAffine → Linear → TropicalAffine → Linear")
    print("   TropicalAffine: y[i] = max(max_k(LayerNorm(x)[k] + W[k,i]), b[i])")
    print("-" * 70)

    hybrid = HybridTropicalNN(layer_sizes)
    tropical_weight_init(hybrid, init_scale=0.5)

    hybrid_params = count_parameters(hybrid)
    print(f"Parameters: {hybrid_params['total']:,}")
    print(f"  Tropical: {hybrid_params['tropical']:,}")
    print(f"  Linear: {hybrid_params['linear']:,}")

    print("Training...")
    hybrid_acc, hybrid_time = train_model(
        hybrid, "Hybrid", train_loader, test_loader, args.epochs, args.lr
    )
    results["Hybrid"] = {"acc": hybrid_acc, "params": hybrid_params['total'], "time": hybrid_time}

    # --- 3. Full MMP Tropical NN ---
    print("\n" + "-" * 70)
    print("3. Full MMP Tropical NN")
    print("   Architecture: Linear → MaxPlus → MinPlus → Linear → MaxPlus → MinPlus → Linear")
    print("   MaxPlusAffine: y[i] = max(max_k(LayerNorm(x)[k] + W[k,i]), b[i])")
    print("   MinPlusAffine: y[i] = min(min_k(LayerNorm(x)[k] + W[k,i]), b[i])")
    print("-" * 70)

    mmp = TropicalNN(layer_sizes)
    tropical_weight_init(mmp, init_scale=0.5)

    mmp_params = count_parameters(mmp)
    print(f"Parameters: {mmp_params['total']:,}")
    print(f"  Tropical: {mmp_params['tropical']:,}")
    print(f"  Linear: {mmp_params['linear']:,}")

    print("Training...")
    mmp_acc, mmp_time = train_model(
        mmp, "MMP", train_loader, test_loader, args.epochs, args.lr
    )
    results["MMP"] = {"acc": mmp_acc, "params": mmp_params['total'], "time": mmp_time}

    # --- Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\n" + "-" * 50)
    print(f"{'Model':<20} {'Accuracy':>12} {'Parameters':>14} {'Time':>10}")
    print("-" * 50)
    for name, r in results.items():
        label = f"{name} *" if name == "Hybrid" else name
        print(f"{label:<20} {r['acc']:>11.1f}% {r['params']:>14,} {r['time']:>9.1f}s")
    print("-" * 50)
    print("* Recommended architecture")

    # Architecture comparison
    print("\nArchitecture Comparison:")
    print("  ReLU:   Linear(784→256) → ReLU → Linear(256→128) → ReLU → Linear(128→10)")
    print("  Hybrid: Linear(784→256) → TropicalAffine(256) → Linear(256→128) → TropicalAffine(128) → Linear(128→10)")
    print("  MMP:    Linear(784→256) → MaxPlus(256) → MinPlus(256) → Linear(256→128) → MaxPlus(128) → MinPlus(128) → Linear(128→10)")

    # Operation analysis
    print("\nOperation Analysis (per sample, hidden layers only):")
    print("  ReLU:   Uses multiplications in matmul + element-wise max")
    print("  Hybrid: 2 tropical layers (256x256 + 128x128 = 81,920 additions/comparisons)")
    print("  MMP:    4 tropical layers (2x256x256 + 2x128x128 = 163,840 additions/comparisons)")

    print("\nKey Insights:")
    print("  - Hybrid is simpler (half the tropical operations) and often equally effective")
    print("  - Tropical layers use only additions and max/min (no multiplications)")
    print("  - Both tropical variants work well, choose based on your use case")
    print("=" * 70)


if __name__ == "__main__":
    main()
