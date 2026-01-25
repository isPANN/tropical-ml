#!/usr/bin/env python3
"""
Train Tropical Neural Networks on MNIST.

Compares tropical networks to standard ReLU baselines.

Usage:
    python train_mnist.py --model tropical --epochs 15
    python train_mnist.py --model baseline --epochs 15

Architecture:
    Tropical:  Linear → MaxPlusAffine → MinPlusAffine → Linear → ...
    Baseline:  Linear → ReLU → Linear → ReLU → ...
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from tropical_activation import TropicalNN, MaxPlusAffine, MinPlusAffine


class BaselineMLP(nn.Module):
    """Standard ReLU MLP for comparison."""

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


def create_model(model_type: str, dropout: float = 0.0):
    """Create model."""
    if model_type == "tropical":
        # Linear → MaxPlus → MinPlus → Linear → MaxPlus → MinPlus → Linear
        return TropicalNN([784, 256, 128, 10], dropout=dropout)
    elif model_type == "baseline":
        return BaselineMLP([784, 256, 128, 10], dropout=dropout)
    else:
        raise ValueError(f"Unknown: {model_type}")


def train_epoch(model, loader, optimizer, criterion):
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


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def main():
    parser = argparse.ArgumentParser(description="Train on MNIST")
    parser.add_argument("--model", type=str, default="tropical",
                       choices=["tropical", "baseline"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.0)
    args = parser.parse_args()

    print("=" * 50)
    print(f"MNIST Training: {args.model}")
    print("=" * 50)

    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_data = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_data = datasets.MNIST("./data", train=False, transform=transform)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size)

    print(f"Train: {len(train_data)}, Test: {len(test_data)}")

    # Model
    model = create_model(args.model, args.dropout)
    print(f"Parameters: {count_params(model):,}")

    # Training
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    best_acc = 0
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        test_loss, test_acc = evaluate(model, test_loader, criterion)
        scheduler.step()

        marker = " *" if test_acc > best_acc else ""
        if test_acc > best_acc:
            best_acc = test_acc

        print(f"Epoch {epoch:2d}/{args.epochs}: "
              f"Loss={train_loss:.4f}, Train={train_acc:.1f}%, Test={test_acc:.1f}%{marker}")

    print(f"\nTime: {time.time() - start:.1f}s, Best: {best_acc:.1f}%")


if __name__ == "__main__":
    main()
