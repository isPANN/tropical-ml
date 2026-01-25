#!/usr/bin/env python3
"""
Train an MMP Neural Network on MNIST.

This example demonstrates how to train a Min-Max-Plus Neural Network
on the MNIST dataset for handwritten digit classification.

Usage:
    python train_mnist_mmp.py [--epochs 10] [--batch-size 128] [--lr 0.001]

Requirements:
    pip install tropical-activation[experiment]
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from tropical_activation import MMPNN, create_mmpnn
from tropical_activation.training import (
    tropical_weight_init,
    get_tropical_optimizer,
    train_epoch,
    evaluate,
    count_parameters,
    count_operations,
)


def get_mnist_dataloaders(batch_size: int, data_dir: str = "./data"):
    """Load MNIST dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(-1)),  # Flatten to 784
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


def main():
    parser = argparse.ArgumentParser(description="Train MMP-NN on MNIST")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--architecture", type=str, default="medium",
                       choices=["tiny", "small", "medium", "large"],
                       help="Model architecture")
    parser.add_argument("--use-linear", action="store_true", default=True,
                       help="Use linear layers (hybrid MMP)")
    parser.add_argument("--pure-tropical", action="store_true",
                       help="Use pure tropical network (no linear layers)")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")
    parser.add_argument("--data-dir", type=str, default="./data",
                       help="Directory for MNIST data")
    parser.add_argument("--save-model", action="store_true",
                       help="Save the trained model")
    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print("Loading MNIST dataset...")
    train_loader, test_loader = get_mnist_dataloaders(args.batch_size, args.data_dir)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    # Create model
    print(f"\nCreating {args.architecture} MMP-NN...")
    use_linear = not args.pure_tropical
    model = create_mmpnn(
        args.architecture,
        input_dim=784,
        num_classes=10,
        use_linear=use_linear,
        dropout=args.dropout,
    )
    model.to(device)

    # Initialize weights
    tropical_weight_init(model)

    # Print model info
    param_counts = count_parameters(model)
    print(f"Model parameters:")
    print(f"  Tropical: {param_counts['tropical']:,}")
    print(f"  Linear: {param_counts['linear']:,}")
    print(f"  Total: {param_counts['total']:,}")

    ops = count_operations(model, (784,))
    print(f"\nOperation counts (per sample):")
    print(f"  Multiplications: {ops['multiplications']:,}")
    print(f"  Additions: {ops['additions']:,}")
    print(f"  Comparisons: {ops['comparisons']:,}")
    print(f"  Total: {ops['total_ops']:,}")

    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = get_tropical_optimizer(model, lr=args.lr, tropical_lr_scale=1.0)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    print("\nTraining...")
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, scheduler
        )
        test_metrics = evaluate(model, test_loader, criterion, device)

        print(f"Epoch {epoch:2d}/{args.epochs}: "
              f"Train Loss: {train_metrics['loss']:.4f}, "
              f"Train Acc: {train_metrics['accuracy']:.2f}%, "
              f"Test Loss: {test_metrics['loss']:.4f}, "
              f"Test Acc: {test_metrics['accuracy']:.2f}%")

        if test_metrics['accuracy'] > best_acc:
            best_acc = test_metrics['accuracy']

    print(f"\nBest test accuracy: {best_acc:.2f}%")

    # Save model if requested
    if args.save_model:
        save_path = Path("mmpnn_mnist.pt")
        torch.save({
            'model_state_dict': model.state_dict(),
            'architecture': args.architecture,
            'use_linear': use_linear,
            'best_accuracy': best_acc,
        }, save_path)
        print(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()
