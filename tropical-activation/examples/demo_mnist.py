#!/usr/bin/env python3
"""
Demo: Train Hybrid Tropical Neural Network on MNIST.

This demonstrates the recommended hybrid architecture that works very well:
    Linear → TropicalAffine → Linear → TropicalAffine → ... → Linear

TropicalAffine: y[i] = max(max_k(LayerNorm(x)[k] + W[k,i]), b[i])

Usage:
    python demo_mnist.py [--epochs 10]

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

from tropical_activation import HybridTropicalNN
from tropical_activation.training import tropical_weight_init, count_parameters


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


def main():
    parser = argparse.ArgumentParser(description="Demo: Hybrid Tropical NN on MNIST")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--data-dir", type=str, default="./data",
                       help="Directory for MNIST data")
    args = parser.parse_args()

    print("=" * 60)
    print("Hybrid Tropical Neural Network Demo - MNIST")
    print("=" * 60)
    print()
    print("Architecture: Linear → TropicalAffine → Linear → TropicalAffine → Linear")
    print("TropicalAffine: y[i] = max(max_k(LayerNorm(x)[k] + W[k,i]), b[i])")

    # Load data
    print("\nLoading MNIST dataset...")
    train_loader, test_loader = get_mnist_dataloaders(args.batch_size, args.data_dir)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    # Create model: Linear → TropicalAffine → Linear → TropicalAffine → Linear
    print("\nCreating Hybrid Tropical NN...")
    model = HybridTropicalNN([784, 256, 128, 10])

    # Initialize weights
    tropical_weight_init(model, init_scale=0.5)

    # Print model info
    param_counts = count_parameters(model)
    print(f"Parameters: {param_counts['total']:,}")
    print(f"  Tropical: {param_counts['tropical']:,}")
    print(f"  Linear: {param_counts['linear']:,}")

    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    print("\nTraining...")
    print("-" * 55)
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        test_loss, test_acc = evaluate(model, test_loader, criterion)
        scheduler.step()

        marker = " *" if test_acc > best_acc else ""
        if test_acc > best_acc:
            best_acc = test_acc

        print(f"Epoch {epoch:2d}/{args.epochs}: "
              f"Loss={train_loss:.4f}, Train={train_acc:.1f}%, Test={test_acc:.1f}%{marker}")

    print("-" * 55)
    print(f"Best test accuracy: {best_acc:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
