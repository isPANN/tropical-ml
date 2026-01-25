#!/usr/bin/env python3
"""
Train Tropical Neural Networks on CIFAR-10.

Conv backbone + tropical classifier head.

Usage:
    python train_cifar10.py --model tropical --epochs 200
    python train_cifar10.py --model baseline --epochs 200

Requirements:
    pip install tropical-activation torchvision
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from tropical_activation.vision import create_cifar10_model
from tropical_activation.training import tropical_weight_init, count_parameters


def get_cifar10_loaders(
    batch_size: int,
    data_dir: str = "./data",
    num_workers: int = 4,
    augment: bool = True,
):
    """Get CIFAR-10 data loaders with augmentation."""

    # Normalization values for CIFAR-10
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_dataset = datasets.CIFAR10(
        data_dir, train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.CIFAR10(
        data_dir, train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def train_epoch(model, loader, optimizer, criterion, device, epoch):
    """Train for one epoch."""
    model.train()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Measure accuracy
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), data.size(0))
        top1.update(acc1[0].item(), data.size(0))
        top5.update(acc5[0].item(), data.size(0))

    return {
        "loss": losses.avg,
        "top1": top1.avg,
        "top5": top5.avg,
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate the model."""
    model.eval()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for data, target in loader:
        data, target = data.to(device), target.to(device)

        output = model(data)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), data.size(0))
        top1.update(acc1[0].item(), data.size(0))
        top5.update(acc5[0].item(), data.size(0))

    return {
        "loss": losses.avg,
        "top1": top1.avg,
        "top5": top5.avg,
    }


def main():
    parser = argparse.ArgumentParser(description="Train MMP-NN on CIFAR-10")
    parser.add_argument("--model", type=str, default="tropical",
                       choices=["tropical", "baseline", "resnet_tropical", "resnet_baseline"],
                       help="Model architecture")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--save-dir", type=str, default="./checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    print("\nLoading CIFAR-10...")
    train_loader, test_loader = get_cifar10_loaders(
        args.batch_size, args.data_dir, augment=not args.no_augment
    )
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    # Model
    print(f"\nCreating {args.model} model...")
    model = create_cifar10_model(args.model, dropout=args.dropout)
    model.to(device)

    # Initialize tropical weights
    if "tropical" in args.model:
        tropical_weight_init(model, init_scale=0.1)

    # Model info
    param_counts = count_parameters(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")
    print(f"  Tropical: {param_counts['tropical']:,}")
    print(f"  Linear: {param_counts['linear']:,}")

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # Learning rate scheduler (cosine annealing)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Resume from checkpoint
    start_epoch = 1
    best_acc = 0.0

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint["epoch"] + 1
            best_acc = checkpoint["best_acc"]
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            print(f"Resumed from epoch {start_epoch - 1}, best_acc: {best_acc:.2f}%")

    # Training loop
    print(f"\n{'='*60}")
    print("Training")
    print(f"{'='*60}")

    history = []
    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(start_epoch, args.epochs + 1):
        start_time = time.time()

        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        test_metrics = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        epoch_time = time.time() - start_time

        print(f"Epoch {epoch:3d}/{args.epochs} ({epoch_time:.1f}s) | "
              f"LR: {scheduler.get_last_lr()[0]:.4f} | "
              f"Train: {train_metrics['top1']:.2f}% | "
              f"Test: {test_metrics['top1']:.2f}% (Top5: {test_metrics['top5']:.2f}%)")

        history.append({
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_top1": train_metrics["top1"],
            "train_top5": train_metrics["top5"],
            "test_loss": test_metrics["loss"],
            "test_top1": test_metrics["top1"],
            "test_top5": test_metrics["top5"],
            "lr": scheduler.get_last_lr()[0],
        })

        # Save checkpoint
        is_best = test_metrics["top1"] > best_acc
        if is_best:
            best_acc = test_metrics["top1"]

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_acc": best_acc,
            "args": vars(args),
        }

        torch.save(checkpoint, os.path.join(args.save_dir, f"cifar10_{args.model}_last.pt"))

        if is_best:
            torch.save(checkpoint, os.path.join(args.save_dir, f"cifar10_{args.model}_best.pt"))
            print(f"  -> New best accuracy: {best_acc:.2f}%")

    # Final results
    print(f"\n{'='*60}")
    print("Results")
    print(f"{'='*60}")
    print(f"Best test accuracy: {best_acc:.2f}%")

    # Save history
    with open(os.path.join(args.save_dir, f"cifar10_{args.model}_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nCheckpoints saved to {args.save_dir}")


if __name__ == "__main__":
    main()
