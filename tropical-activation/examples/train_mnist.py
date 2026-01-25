#!/usr/bin/env python3
"""
Train MMP Neural Networks on MNIST.

This script trains various MMP-NN architectures on MNIST and compares them
to standard ReLU baselines.

Usage:
    python train_mnist.py --model hybrid --epochs 20      # Linear + MaxPlus + MinPlus (recommended)
    python train_mnist.py --model tropical --epochs 20    # Minimal multiplications
    python train_mnist.py --model pure --epochs 20        # Zero multiplications
    python train_mnist.py --model baseline --epochs 20    # ReLU baseline

Requirements:
    pip install tropical-activation torchvision
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from tropical_activation import (
    MMPNN,
    MMPClassifier,
    PureTropicalNN,
    MaxPlusLayer,
    MinPlusLayer,
)
from tropical_activation.training import (
    tropical_weight_init,
    count_parameters,
    count_operations,
)


class BaselineMLP(nn.Module):
    """Standard MLP with ReLU for comparison."""

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


def get_mnist_loaders(batch_size: int, data_dir: str = "./data", num_workers: int = 4):
    """Get MNIST data loaders with standard preprocessing."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_dataset = datasets.MNIST(
        data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        data_dir, train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader


def create_model(model_type: str, dropout: float = 0.0):
    """Create model based on type."""
    input_dim = 784
    num_classes = 10

    if model_type == "hybrid":
        # Linear → MaxPlus → MinPlus → Linear → ... (has multiplications, most expressive)
        model = MMPNN([input_dim, 512, 256, 128, num_classes], use_linear=True, dropout=dropout)
    elif model_type == "hybrid_small":
        model = MMPNN([input_dim, 256, 128, num_classes], use_linear=True, dropout=dropout)
    elif model_type == "tropical":
        # Linear (first only) → MaxPlus → MinPlus → ... (minimal multiplications)
        model = MMPNN([input_dim, 512, 256, 128, num_classes], use_linear=False, dropout=dropout)
    elif model_type == "pure":
        # MaxPlus → MinPlus → ... (zero multiplications)
        model = PureTropicalNN([input_dim, 512, 256, 128, num_classes])
    elif model_type == "classifier":
        model = MMPClassifier(input_dim, [512, 256, 128], num_classes, dropout=dropout)
    elif model_type == "baseline":
        # Standard ReLU network for comparison
        model = BaselineMLP([input_dim, 512, 256, 128, num_classes], dropout=dropout)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model


def train_epoch(model, loader, optimizer, criterion, device, epoch, log_interval=100):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        data = data.view(data.size(0), -1)  # Flatten

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

        if batch_idx % log_interval == 0 and batch_idx > 0:
            print(f"  Batch {batch_idx}/{len(loader)}: "
                  f"Loss: {loss.item():.4f}, "
                  f"Acc: {100. * correct / total:.2f}%")

    return {
        "loss": total_loss / len(loader),
        "accuracy": 100. * correct / total,
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for data, target in loader:
        data, target = data.to(device), target.to(device)
        data = data.view(data.size(0), -1)

        output = model(data)
        total_loss += criterion(output, target).item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

    return {
        "loss": total_loss / len(loader),
        "accuracy": 100. * correct / total,
    }


def main():
    parser = argparse.ArgumentParser(description="Train MMP-NN on MNIST")
    parser.add_argument("--model", type=str, default="hybrid",
                       choices=["hybrid", "hybrid_small", "tropical", "pure",
                               "classifier", "baseline"],
                       help="Model architecture")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--save-dir", type=str, default="./checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    print("\nLoading MNIST...")
    train_loader, test_loader = get_mnist_loaders(args.batch_size, args.data_dir)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    # Model
    print(f"\nCreating {args.model} model...")
    model = create_model(args.model, args.dropout)
    model.to(device)

    # Initialize tropical weights
    if args.model != "baseline":
        tropical_weight_init(model, init_scale=0.1)

    # Model info
    param_counts = count_parameters(model)
    print(f"\nModel parameters:")
    print(f"  Tropical: {param_counts['tropical']:,}")
    print(f"  Linear: {param_counts['linear']:,}")
    print(f"  Total: {param_counts['total']:,}")

    if args.model != "baseline":
        ops = count_operations(model, (784,))
        print(f"\nOperations per sample:")
        print(f"  Multiplications: {ops['multiplications']:,}")
        print(f"  Additions: {ops['additions']:,}")
        print(f"  Comparisons: {ops['comparisons']:,}")

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    print(f"\n{'='*60}")
    print("Training")
    print(f"{'='*60}")

    best_acc = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs} (lr={scheduler.get_last_lr()[0]:.6f})")
        print("-" * 40)

        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        test_metrics = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}%")
        print(f"Test  - Loss: {test_metrics['loss']:.4f}, Acc: {test_metrics['accuracy']:.2f}%")

        history.append({
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["accuracy"],
            "test_loss": test_metrics["loss"],
            "test_acc": test_metrics["accuracy"],
        })

        if test_metrics["accuracy"] > best_acc:
            best_acc = test_metrics["accuracy"]
            # Save best model
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_acc": best_acc,
                "args": vars(args),
            }, os.path.join(args.save_dir, f"mnist_{args.model}_best.pt"))

    # Final results
    print(f"\n{'='*60}")
    print("Results")
    print(f"{'='*60}")
    print(f"Best test accuracy: {best_acc:.2f}%")

    # Save history
    with open(os.path.join(args.save_dir, f"mnist_{args.model}_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nCheckpoints saved to {args.save_dir}")


if __name__ == "__main__":
    main()
