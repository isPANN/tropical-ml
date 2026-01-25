#!/usr/bin/env python3
"""
Compare MMP Networks to Standard ReLU Networks.

This example trains both an MMP-NN and a standard ReLU network on MNIST
and compares their performance, parameter counts, and operation counts.

Usage:
    python compare_to_relu.py [--epochs 10] [--batch-size 128]

Requirements:
    pip install tropical-activation[experiment]
"""

import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from tropical_activation import MMPNN
from tropical_activation.training import (
    tropical_weight_init,
    train_epoch,
    evaluate,
    count_parameters,
    count_operations,
)


class StandardMLP(nn.Module):
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
        return self.layers(x)


def get_mnist_dataloaders(batch_size: int, data_dir: str = "./data"):
    """Load MNIST dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(-1)),
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


def train_model(model, train_loader, test_loader, epochs, lr, device):
    """Train a model and return final metrics."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.to(device)

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        test_metrics = evaluate(model, test_loader, criterion, device)

        if test_metrics['accuracy'] > best_acc:
            best_acc = test_metrics['accuracy']

        if epoch % 5 == 0 or epoch == epochs:
            print(f"  Epoch {epoch:2d}: "
                  f"Train Acc: {train_metrics['accuracy']:.2f}%, "
                  f"Test Acc: {test_metrics['accuracy']:.2f}%")

    return best_acc, test_metrics


def main():
    parser = argparse.ArgumentParser(description="Compare MMP-NN to ReLU network")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--data-dir", type=str, default="./data",
                       help="Directory for MNIST data")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print("Loading MNIST dataset...")
    train_loader, test_loader = get_mnist_dataloaders(args.batch_size, args.data_dir)

    # Define architecture
    layer_sizes = [784, 256, 128, 10]

    print("\n" + "=" * 60)
    print("COMPARISON: MMP-NN vs Standard ReLU Network")
    print("=" * 60)

    # --- Standard ReLU Network ---
    print("\n1. Standard ReLU Network")
    print("-" * 40)

    relu_model = StandardMLP(layer_sizes)

    relu_params = sum(p.numel() for p in relu_model.parameters())
    print(f"Parameters: {relu_params:,}")

    print("Training...")
    relu_best_acc, relu_final = train_model(
        relu_model, train_loader, test_loader, args.epochs, args.lr, device
    )

    # --- MMP Neural Network (with Linear layers) ---
    print("\n2. MMP Neural Network (Hybrid)")
    print("-" * 40)

    mmp_model = MMPNN(layer_sizes, use_linear=True)
    tropical_weight_init(mmp_model)

    mmp_param_counts = count_parameters(mmp_model)
    print(f"Parameters: {mmp_param_counts['total']:,} "
          f"(Tropical: {mmp_param_counts['tropical']:,}, "
          f"Linear: {mmp_param_counts['linear']:,})")

    print("Training...")
    mmp_best_acc, mmp_final = train_model(
        mmp_model, train_loader, test_loader, args.epochs, args.lr, device
    )

    # --- Pure Tropical Network ---
    print("\n3. Pure Tropical Network (No Linear)")
    print("-" * 40)

    pure_tropical = MMPNN(layer_sizes, use_linear=False)
    tropical_weight_init(pure_tropical)

    pure_param_counts = count_parameters(pure_tropical)
    print(f"Parameters: {pure_param_counts['total']:,} "
          f"(All tropical)")

    print("Training...")
    pure_best_acc, pure_final = train_model(
        pure_tropical, train_loader, test_loader, args.epochs, args.lr, device
    )

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("\nAccuracy Comparison:")
    print(f"  Standard ReLU:     {relu_best_acc:.2f}%")
    print(f"  MMP-NN (Hybrid):   {mmp_best_acc:.2f}%")
    print(f"  Pure Tropical:     {pure_best_acc:.2f}%")

    print("\nParameter Counts:")
    print(f"  Standard ReLU:     {relu_params:,}")
    print(f"  MMP-NN (Hybrid):   {mmp_param_counts['total']:,}")
    print(f"  Pure Tropical:     {pure_param_counts['total']:,}")

    print("\nOperation Counts (per sample):")

    # For ReLU model, we estimate operations
    relu_ops = {"multiplications": 0, "additions": 0, "comparisons": 0}
    for i in range(len(layer_sizes) - 1):
        in_f, out_f = layer_sizes[i], layer_sizes[i + 1]
        relu_ops["multiplications"] += out_f * in_f
        relu_ops["additions"] += out_f * (in_f - 1) + out_f  # matmul + bias
        if i < len(layer_sizes) - 2:
            relu_ops["comparisons"] += out_f  # ReLU comparisons

    mmp_ops = count_operations(mmp_model, (784,))
    pure_ops = count_operations(pure_tropical, (784,))

    print(f"\n  Standard ReLU:")
    print(f"    Multiplications: {relu_ops['multiplications']:,}")
    print(f"    Additions:       {relu_ops['additions']:,}")
    print(f"    Comparisons:     {relu_ops['comparisons']:,}")

    print(f"\n  MMP-NN (Hybrid):")
    print(f"    Multiplications: {mmp_ops['multiplications']:,}")
    print(f"    Additions:       {mmp_ops['additions']:,}")
    print(f"    Comparisons:     {mmp_ops['comparisons']:,}")

    print(f"\n  Pure Tropical:")
    print(f"    Multiplications: {pure_ops['multiplications']:,}")
    print(f"    Additions:       {pure_ops['additions']:,}")
    print(f"    Comparisons:     {pure_ops['comparisons']:,}")

    # Calculate reduction
    if relu_ops['multiplications'] > 0:
        mmp_reduction = 1 - mmp_ops['multiplications'] / relu_ops['multiplications']
        pure_reduction = 1 - pure_ops['multiplications'] / relu_ops['multiplications']
        print(f"\nMultiplication Reduction:")
        print(f"  MMP-NN (Hybrid):   {mmp_reduction:.1%}")
        print(f"  Pure Tropical:     {pure_reduction:.1%}")


if __name__ == "__main__":
    main()
