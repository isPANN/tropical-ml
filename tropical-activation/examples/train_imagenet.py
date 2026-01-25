#!/usr/bin/env python3
"""
Train MMP Neural Networks on ImageNet.

This script trains MMP networks on ImageNet with distributed training support,
mixed precision, and proper data augmentation.

Usage:
    # Single GPU
    python train_imagenet.py --data /path/to/imagenet --model mmp

    # Multi-GPU (DDP)
    torchrun --nproc_per_node=4 train_imagenet.py --data /path/to/imagenet --model mmp

    # Baseline comparison
    python train_imagenet.py --data /path/to/imagenet --model baseline

Requirements:
    pip install tropical-activation torchvision timm
"""

import argparse
import json
import os
import random
import time
import warnings
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler, autocast
from torchvision import datasets, transforms

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from tropical_activation.vision import create_imagenet_model, ImageNetMMP
from tropical_activation.training import tropical_weight_init, count_parameters


def setup_distributed():
    """Initialize distributed training."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])

        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

        return rank, world_size, local_rank
    else:
        return 0, 1, 0


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """Check if this is the main process."""
    return not dist.is_initialized() or dist.get_rank() == 0


def get_imagenet_loaders(
    data_dir: str,
    batch_size: int,
    num_workers: int = 8,
    distributed: bool = False,
):
    """Get ImageNet data loaders with standard augmentation."""

    # ImageNet normalization
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    if distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, train_sampler


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


def train_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    epoch,
    scaler=None,
    use_amp=False,
):
    """Train for one epoch with optional mixed precision."""
    model.train()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    batch_time = AverageMeter()

    end = time.time()

    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)

        optimizer.zero_grad()

        if use_amp and scaler is not None:
            with autocast():
                output = model(data)
                loss = criterion(output, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        # Measure accuracy
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), data.size(0))
        top1.update(acc1[0].item(), data.size(0))
        top5.update(acc5[0].item(), data.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % 100 == 0 and is_main_process():
            print(f"  Batch {batch_idx:4d}/{len(loader)} | "
                  f"Loss: {losses.avg:.4f} | "
                  f"Top1: {top1.avg:.2f}% | "
                  f"Top5: {top5.avg:.2f}% | "
                  f"Time: {batch_time.avg:.3f}s")

    return {
        "loss": losses.avg,
        "top1": top1.avg,
        "top5": top5.avg,
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp=False):
    """Evaluate the model."""
    model.eval()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for data, target in loader:
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)

        if use_amp:
            with autocast():
                output = model(data)
                loss = criterion(output, target)
        else:
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
    parser = argparse.ArgumentParser(description="Train MMP-NN on ImageNet")
    parser.add_argument("--data", type=str, required=True, help="Path to ImageNet dataset")
    parser.add_argument("--model", type=str, default="mmp",
                       choices=["mmp", "baseline"],
                       help="Model architecture")
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--save-dir", type=str, default="./checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", help="Use mixed precision training")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate only")
    args = parser.parse_args()

    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    distributed = world_size > 1

    # Set seed
    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)

    # Device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    if is_main_process():
        print(f"Using device: {device}")
        print(f"Distributed: {distributed} (world_size={world_size})")

    # Data
    if is_main_process():
        print("\nLoading ImageNet...")

    # Adjust batch size for distributed training
    batch_size_per_gpu = args.batch_size // world_size

    train_loader, val_loader, train_sampler = get_imagenet_loaders(
        args.data,
        batch_size_per_gpu,
        num_workers=args.workers,
        distributed=distributed,
    )

    if is_main_process():
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        print(f"Batch size per GPU: {batch_size_per_gpu}")

    # Model
    if is_main_process():
        print(f"\nCreating {args.model} model...")

    model = create_imagenet_model(args.model, dropout=args.dropout)
    model.to(device)

    # Initialize tropical weights
    if "mmp" in args.model:
        tropical_weight_init(model, init_scale=0.1)

    # Wrap with DDP
    if distributed:
        model = DDP(model, device_ids=[local_rank])

    # Model info
    if is_main_process():
        param_counts = count_parameters(model.module if distributed else model)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\nModel parameters: {total_params:,}")
        print(f"  Tropical: {param_counts['tropical']:,}")
        print(f"  Linear: {param_counts['linear']:,}")

    # Training setup
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # Learning rate scheduler (step decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # Mixed precision
    scaler = GradScaler() if args.amp else None

    # Resume from checkpoint
    start_epoch = 1
    best_acc = 0.0

    if args.resume:
        if os.path.isfile(args.resume):
            if is_main_process():
                print(f"Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint["epoch"] + 1
            best_acc = checkpoint["best_acc"]
            model_to_load = model.module if distributed else model
            model_to_load.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            if scaler is not None and "scaler_state_dict" in checkpoint:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
            if is_main_process():
                print(f"Resumed from epoch {start_epoch - 1}, best_acc: {best_acc:.2f}%")

    # Evaluate only
    if args.evaluate:
        val_metrics = evaluate(model, val_loader, criterion, device, use_amp=args.amp)
        if is_main_process():
            print(f"Validation: Top1: {val_metrics['top1']:.2f}%, Top5: {val_metrics['top5']:.2f}%")
        cleanup_distributed()
        return

    # Training loop
    if is_main_process():
        print(f"\n{'='*60}")
        print("Training")
        print(f"{'='*60}")
        os.makedirs(args.save_dir, exist_ok=True)

    history = []

    for epoch in range(start_epoch, args.epochs + 1):
        if distributed:
            train_sampler.set_epoch(epoch)

        start_time = time.time()

        if is_main_process():
            print(f"\nEpoch {epoch}/{args.epochs} (lr={scheduler.get_last_lr()[0]:.6f})")

        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch,
            scaler=scaler, use_amp=args.amp
        )
        val_metrics = evaluate(model, val_loader, criterion, device, use_amp=args.amp)
        scheduler.step()

        epoch_time = time.time() - start_time

        if is_main_process():
            print(f"Epoch {epoch:3d} ({epoch_time:.1f}s) | "
                  f"Train: {train_metrics['top1']:.2f}% | "
                  f"Val: {val_metrics['top1']:.2f}% (Top5: {val_metrics['top5']:.2f}%)")

            history.append({
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_top1": train_metrics["top1"],
                "train_top5": train_metrics["top5"],
                "val_loss": val_metrics["loss"],
                "val_top1": val_metrics["top1"],
                "val_top5": val_metrics["top5"],
                "lr": scheduler.get_last_lr()[0],
            })

            # Save checkpoint
            is_best = val_metrics["top1"] > best_acc
            if is_best:
                best_acc = val_metrics["top1"]

            model_to_save = model.module if distributed else model
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model_to_save.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_acc": best_acc,
                "args": vars(args),
            }
            if scaler is not None:
                checkpoint["scaler_state_dict"] = scaler.state_dict()

            torch.save(checkpoint, os.path.join(args.save_dir, f"imagenet_{args.model}_last.pt"))

            if is_best:
                torch.save(checkpoint, os.path.join(args.save_dir, f"imagenet_{args.model}_best.pt"))
                print(f"  -> New best accuracy: {best_acc:.2f}%")

    # Final results
    if is_main_process():
        print(f"\n{'='*60}")
        print("Results")
        print(f"{'='*60}")
        print(f"Best validation accuracy: {best_acc:.2f}%")

        # Save history
        with open(os.path.join(args.save_dir, f"imagenet_{args.model}_history.json"), "w") as f:
            json.dump(history, f, indent=2)

        print(f"\nCheckpoints saved to {args.save_dir}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
