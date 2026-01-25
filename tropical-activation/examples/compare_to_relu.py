#!/usr/bin/env python3
"""
Compare Tropical Networks to Standard ReLU Networks.

Trains both a tropical network and a standard ReLU network on MNIST
and compares their performance and operation counts.

Usage:
    python compare_to_relu.py [--epochs 15]

Requirements:
    pip install tropical-activation torchvision
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from tropical_activation import TropicalNN
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
    """Train a model and return best accuracy."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        test_loss, test_acc = evaluate(model, test_loader, criterion)
        scheduler.step()

        if test_acc > best_acc:
            best_acc = test_acc

        if epoch % 5 == 0 or epoch == epochs:
            print(f"  Epoch {epoch:2d}: Train={train_acc:.1f}%, Test={test_acc:.1f}%")

    return best_acc


def main():
    parser = argparse.ArgumentParser(description="Compare Tropical NN to ReLU")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--data-dir", type=str, default="./data",
                       help="Directory for MNIST data")
    args = parser.parse_args()

    print("=" * 60)
    print("COMPARISON: Tropical NN vs Standard ReLU Network")
    print("=" * 60)

    # Load data
    print("\nLoading MNIST dataset...")
    train_loader, test_loader = get_mnist_dataloaders(args.batch_size, args.data_dir)

    # Define architecture
    layer_sizes = [784, 256, 128, 10]

    # --- Standard ReLU Network ---
    print("\n" + "-" * 40)
    print("1. Standard ReLU Network")
    print("-" * 40)

    baseline = BaselineMLP(layer_sizes)
    baseline_params = sum(p.numel() for p in baseline.parameters())
    print(f"Parameters: {baseline_params:,}")

    print("Training...")
    baseline_acc = train_model(baseline, "ReLU", train_loader, test_loader, args.epochs, args.lr)

    # --- Tropical Neural Network ---
    print("\n" + "-" * 40)
    print("2. Tropical Neural Network")
    print("-" * 40)

    tropical = TropicalNN(layer_sizes)
    tropical_weight_init(tropical, init_scale=0.1)

    tropical_params = count_parameters(tropical)
    print(f"Parameters: {tropical_params['total']:,}")
    print(f"  Tropical: {tropical_params['tropical']:,}")
    print(f"  Linear: {tropical_params['linear']:,}")

    print("Training...")
    tropical_acc = train_model(tropical, "Tropical", train_loader, test_loader, args.epochs, args.lr)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("\nAccuracy:")
    print(f"  Standard ReLU:  {baseline_acc:.1f}%")
    print(f"  Tropical NN:    {tropical_acc:.1f}%")

    print("\nParameters:")
    print(f"  Standard ReLU:  {baseline_params:,}")
    print(f"  Tropical NN:    {tropical_params['total']:,}")

    # Operation analysis
    print("\nOperation Analysis (per sample):")

    # ReLU network: multiplications in matrix multiply
    relu_muls = 0
    relu_adds = 0
    for i in range(len(layer_sizes) - 1):
        relu_muls += layer_sizes[i] * layer_sizes[i + 1]
        relu_adds += layer_sizes[i] * layer_sizes[i + 1] + layer_sizes[i + 1]

    print(f"  ReLU Network:")
    print(f"    Multiplications: {relu_muls:,}")
    print(f"    Additions: {relu_adds:,}")

    # Tropical network: Linear layers have muls, tropical layers only have adds/comparisons
    # Linear: 784→256, 256→128, 128→10
    # Tropical: MaxPlusAffine(256), MinPlusAffine(256), MaxPlusAffine(128), MinPlusAffine(128)
    trop_muls = layer_sizes[0] * layer_sizes[1]  # First linear
    for i in range(1, len(layer_sizes) - 1):
        trop_muls += layer_sizes[i] * layer_sizes[i + 1]  # Hidden/output linear

    # Tropical layers: n*n additions + n*n comparisons (for each MaxPlus and MinPlus)
    trop_adds = 0
    trop_comps = 0
    for i in range(1, len(layer_sizes) - 1):
        dim = layer_sizes[i]
        # MaxPlusAffine + MinPlusAffine
        trop_adds += 2 * dim * dim  # x[k] + W[k,j]
        trop_comps += 2 * dim * dim  # max/min comparisons

    print(f"  Tropical Network:")
    print(f"    Multiplications: {trop_muls:,}")
    print(f"    Additions: {trop_adds:,} (tropical layers)")
    print(f"    Comparisons: {trop_comps:,} (max/min)")

    # Note about tropical advantage
    print("\nNote: Tropical layers use only additions and max/min operations,")
    print("which are more efficient than multiplications on specialized hardware.")


if __name__ == "__main__":
    main()
