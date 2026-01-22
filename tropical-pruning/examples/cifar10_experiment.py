#!/usr/bin/env python3
"""
CIFAR-10 Pruning Experiments for Tropical Pruning Paper.

This script runs comprehensive experiments comparing tropical pruning against
baselines on CIFAR-10 with VGG16-BN and ResNet18 architectures.

Usage:
    python cifar10_experiment.py --model vgg16 --sparsity 0.5 --seeds 5
    python cifar10_experiment.py --model resnet18 --full-sweep
    python cifar10_experiment.py --calibration-ablation

Results are saved to results/cifar10/ directory.
"""

import argparse
import copy
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
import torchvision
import torchvision.transforms as transforms
from torchvision.models import vgg16_bn, resnet18

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tropical_pruning import (
    ConvWinnerCounter,
    ConvTropicalPruner,
    create_winner_counter,
)
from tropical_pruning.baselines import (
    MagnitudeStructuredPruner,
    RandomStructuredPruner,
)
from tropical_pruning.experiment import (
    ExperimentConfig,
    ExperimentRunner,
    ResultStats,
    run_calibration_ablation,
)
from tropical_pruning.finetuning import (
    FinetuneConfig,
    finetune_pruned_model,
    quick_finetune,
)


# Default configuration
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"Using CUDA backend: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  Device count: {torch.cuda.device_count()}")
    print(f"  Current device: {torch.cuda.current_device()}")
    print(f"  Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    print(f"  Memory reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Using Apple MPS backend")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU backend")

DATA_DIR = Path("./data")
RESULTS_DIR = Path("./results/cifar10")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_cifar10_loaders(
    batch_size: int = 128,
    val_split: float = 0.1,
    calibration_samples: int = 1000,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """
    Get CIFAR-10 data loaders.

    Returns:
        Tuple of (train_loader, val_loader, test_loader, calibration_loader)
    """
    # Data augmentation and normalization
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Load datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root=str(DATA_DIR), train=True, download=True, transform=transform_train
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=str(DATA_DIR), train=False, download=True, transform=transform_test
    )

    # Split training into train and validation
    train_size = int((1 - val_split) * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_data, val_data = random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Create calibration subset (without augmentation for stability)
    cal_dataset = torchvision.datasets.CIFAR10(
        root=str(DATA_DIR), train=True, download=True, transform=transform_test
    )
    cal_indices = torch.randperm(len(cal_dataset))[:calibration_samples].tolist()
    cal_data = Subset(cal_dataset, cal_indices)

    # Create loaders
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    calibration_loader = DataLoader(
        cal_data, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader, calibration_loader


def create_vgg16_cifar10(pretrained: bool = True) -> nn.Module:
    """
    Create VGG16-BN adapted for CIFAR-10.

    - Uses pretrained ImageNet weights
    - Replaces classifier for 10 classes
    - Optionally fine-tunes on CIFAR-10
    """
    model = vgg16_bn(weights="IMAGENET1K_V1" if pretrained else None)

    # Replace classifier for CIFAR-10 (10 classes instead of 1000)
    # VGG16 expects 224x224, CIFAR-10 is 32x32
    # After all convs, feature map is 1x1 for 32x32 input
    model.classifier = nn.Sequential(
        nn.Linear(512, 512),
        nn.ReLU(True),
        nn.Dropout(0.5),
        nn.Linear(512, 512),
        nn.ReLU(True),
        nn.Dropout(0.5),
        nn.Linear(512, 10),
    )

    # Adapt avgpool for 32x32 input
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    return model


def create_resnet18_cifar10(pretrained: bool = True) -> nn.Module:
    """
    Create ResNet18 adapted for CIFAR-10.

    - Uses pretrained ImageNet weights
    - Replaces first conv for smaller input
    - Replaces fc for 10 classes
    """
    model = resnet18(weights="IMAGENET1K_V1" if pretrained else None)

    # Adapt for CIFAR-10 (32x32 instead of 224x224)
    # Replace first conv: 7x7, stride 2 -> 3x3, stride 1
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

    # Remove maxpool (not needed for small images)
    model.maxpool = nn.Identity()

    # Replace fc for 10 classes
    model.fc = nn.Linear(model.fc.in_features, 10)

    return model


def get_model_fn(model_name: str, pretrained: bool = True):
    """Get model creation function by name."""
    if model_name.lower() == "vgg16":
        return lambda: create_vgg16_cifar10(pretrained)
    elif model_name.lower() == "resnet18":
        return lambda: create_resnet18_cifar10(pretrained)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device = DEVICE,
) -> float:
    """Evaluate model accuracy on test set."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return 100.0 * correct / total


def pretrain_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    lr: float = 0.01,
    device: torch.device = DEVICE,
) -> nn.Module:
    """
    Fine-tune pretrained model on CIFAR-10.

    Args:
        model: Model to fine-tune.
        train_loader: Training data.
        val_loader: Validation data.
        epochs: Number of training epochs.
        lr: Learning rate.
        device: Device to use.

    Returns:
        Fine-tuned model.
    """
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        # Train
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        scheduler.step()

        # Validate
        acc = evaluate_model(model, val_loader, device)
        if acc > best_acc:
            best_acc = acc
            best_state = copy.deepcopy(model.state_dict())

        print(f"Epoch {epoch+1}/{epochs}: Val Acc = {acc:.2f}% (Best: {best_acc:.2f}%)")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def run_pruning_experiment(
    model_name: str,
    sparsity_levels: List[float],
    seeds: List[int],
    methods: List[str],
    finetune_epochs: int = 10,
    calibration_samples: int = 1000,
    pretrained: bool = True,
    pretrain_epochs: int = 0,
) -> Dict:
    """
    Run full pruning experiment.

    Args:
        model_name: Model architecture name.
        sparsity_levels: List of sparsity levels to test.
        seeds: List of random seeds.
        methods: List of pruning methods.
        finetune_epochs: Epochs for post-pruning fine-tuning.
        calibration_samples: Number of calibration samples.
        pretrained: Whether to use pretrained weights.
        pretrain_epochs: Epochs to pretrain on CIFAR-10 (0 = skip).

    Returns:
        Dictionary with results.
    """
    print(f"\n{'='*60}")
    print(f"Running experiment: {model_name}")
    print(f"Sparsity levels: {sparsity_levels}")
    print(f"Seeds: {seeds}")
    print(f"Methods: {methods}")
    print(f"{'='*60}\n")

    # Get data loaders
    train_loader, val_loader, test_loader, cal_loader = get_cifar10_loaders(
        calibration_samples=calibration_samples
    )

    # Results storage
    results = {
        method: {sparsity: [] for sparsity in sparsity_levels}
        for method in methods
    }
    results["baseline"] = []

    for seed_idx, seed in enumerate(seeds):
        print(f"\n--- Seed {seed} ({seed_idx+1}/{len(seeds)}) ---")
        set_seed(seed)

        # Create and optionally pretrain model
        model_fn = get_model_fn(model_name, pretrained)
        base_model = model_fn().to(DEVICE)

        if pretrain_epochs > 0:
            print(f"Pretraining for {pretrain_epochs} epochs...")
            base_model = pretrain_model(
                base_model, train_loader, val_loader, epochs=pretrain_epochs
            )

        # Evaluate baseline
        baseline_acc = evaluate_model(base_model, test_loader)
        results["baseline"].append(baseline_acc)
        print(f"Baseline accuracy: {baseline_acc:.2f}%")

        # Collect tropical statistics once per seed
        print("Collecting tropical statistics...")
        counter = ConvWinnerCounter(base_model, device=DEVICE)
        stats = counter.collect(cal_loader, show_progress=True)
        counter.remove_hooks()

        for method in methods:
            print(f"\n  Method: {method}")

            for sparsity in sparsity_levels:
                # Create fresh copy
                model = copy.deepcopy(base_model)

                # Prune
                if method == "tropical":
                    pruner = ConvTropicalPruner(model, stats)
                    pruned_model = pruner.prune(sparsity, inplace=True)
                elif method == "magnitude_l1":
                    pruner = MagnitudeStructuredPruner(model, norm="l1")
                    pruned_model = pruner.prune(sparsity, inplace=True)
                elif method == "magnitude_l2":
                    pruner = MagnitudeStructuredPruner(model, norm="l2")
                    pruned_model = pruner.prune(sparsity, inplace=True)
                elif method == "random":
                    pruner = RandomStructuredPruner(model, seed=seed)
                    pruned_model = pruner.prune(sparsity, inplace=True)
                else:
                    raise ValueError(f"Unknown method: {method}")

                # Evaluate before fine-tuning
                pre_ft_acc = evaluate_model(pruned_model, test_loader)

                # Fine-tune
                if finetune_epochs > 0:
                    pruned_model, ft_result = quick_finetune(
                        pruned_model, train_loader, val_loader,
                        epochs=finetune_epochs, lr=0.001, device=DEVICE
                    )
                    post_ft_acc = ft_result["best_val_acc"]
                else:
                    post_ft_acc = pre_ft_acc

                results[method][sparsity].append(post_ft_acc)
                print(f"    Sparsity {sparsity:.0%}: {pre_ft_acc:.2f}% -> {post_ft_acc:.2f}% (finetuned)")

        del base_model, stats
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return results


def print_results_table(results: Dict, methods: List[str], sparsity_levels: List[float]) -> str:
    """Generate and print results table."""
    lines = []

    # Header
    header = "| Method | " + " | ".join(f"{s*100:.0f}%" for s in sparsity_levels) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(sparsity_levels) + 1))

    # Baseline
    baseline_mean = np.mean(results["baseline"])
    baseline_std = np.std(results["baseline"]) if len(results["baseline"]) > 1 else 0
    baseline_row = f"| baseline | " + " | ".join(
        f"{baseline_mean:.2f}±{baseline_std:.2f}" for _ in sparsity_levels
    ) + " |"
    lines.append(baseline_row)

    # Methods
    for method in methods:
        row = f"| {method} |"
        for sparsity in sparsity_levels:
            accs = results[method][sparsity]
            mean = np.mean(accs)
            std = np.std(accs) if len(accs) > 1 else 0
            row += f" {mean:.2f}±{std:.2f} |"
        lines.append(row)

    table = "\n".join(lines)
    print("\nResults Table:")
    print(table)
    return table


def run_calibration_ablation_experiment(
    model_name: str,
    sample_sizes: List[int] = [100, 500, 1000, 2000, 5000],
    sparsity: float = 0.5,
    seeds: List[int] = [42, 123, 456],
) -> Dict:
    """
    Run calibration sample ablation study.

    Tests how the number of calibration samples affects pruning quality.
    """
    print(f"\n{'='*60}")
    print(f"Calibration Ablation: {model_name}")
    print(f"Sample sizes: {sample_sizes}")
    print(f"Sparsity: {sparsity}")
    print(f"{'='*60}\n")

    # Get base loaders
    train_loader, val_loader, test_loader, _ = get_cifar10_loaders()

    # Get full training dataset for calibration subsets
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_dataset = torchvision.datasets.CIFAR10(
        root=str(DATA_DIR), train=True, download=True, transform=transform_test
    )

    results = {n: [] for n in sample_sizes}

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        set_seed(seed)

        # Create model
        model_fn = get_model_fn(model_name, pretrained=True)
        base_model = model_fn().to(DEVICE)

        for n_samples in sample_sizes:
            # Create calibration subset
            indices = torch.randperm(len(train_dataset))[:n_samples].tolist()
            cal_subset = Subset(train_dataset, indices)
            cal_loader = DataLoader(cal_subset, batch_size=64, shuffle=False)

            # Create fresh model copy
            model = copy.deepcopy(base_model)

            # Collect statistics
            counter = ConvWinnerCounter(model, device=DEVICE)
            stats = counter.collect(cal_loader, show_progress=False)
            counter.remove_hooks()

            # Prune
            pruner = ConvTropicalPruner(model, stats)
            pruned_model = pruner.prune(sparsity, inplace=True)

            # Evaluate
            acc = evaluate_model(pruned_model, test_loader)
            results[n_samples].append(acc)
            print(f"  Samples={n_samples}: {acc:.2f}%")

    # Print table
    print("\nCalibration Ablation Results:")
    print("| Samples | Accuracy |")
    print("|---------|----------|")
    for n in sample_sizes:
        mean = np.mean(results[n])
        std = np.std(results[n]) if len(results[n]) > 1 else 0
        print(f"| {n} | {mean:.2f}±{std:.2f} |")

    return results


def main():
    parser = argparse.ArgumentParser(description="CIFAR-10 Pruning Experiments")
    parser.add_argument("--model", type=str, default="vgg16",
                       choices=["vgg16", "resnet18"],
                       help="Model architecture")
    parser.add_argument("--sparsity", type=float, nargs="+",
                       default=[0.3, 0.5, 0.7],
                       help="Sparsity levels to test")
    parser.add_argument("--seeds", type=int, default=5,
                       help="Number of random seeds")
    parser.add_argument("--methods", type=str, nargs="+",
                       default=["tropical", "magnitude_l1", "magnitude_l2", "random"],
                       help="Pruning methods to compare")
    parser.add_argument("--finetune-epochs", type=int, default=10,
                       help="Epochs for post-pruning fine-tuning")
    parser.add_argument("--calibration-samples", type=int, default=1000,
                       help="Number of calibration samples")
    parser.add_argument("--pretrain-epochs", type=int, default=0,
                       help="Epochs to pretrain on CIFAR-10 (0=use ImageNet weights)")
    parser.add_argument("--full-sweep", action="store_true",
                       help="Run full sparsity sweep (10% to 90%)")
    parser.add_argument("--calibration-ablation", action="store_true",
                       help="Run calibration sample ablation study")
    parser.add_argument("--output", type=str, default=None,
                       help="Output file path")

    args = parser.parse_args()

    # Set sparsity levels
    if args.full_sweep:
        sparsity_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    else:
        sparsity_levels = args.sparsity

    # Set seeds
    seeds = [42, 123, 456, 789, 1234][:args.seeds]

    # Run experiment
    if args.calibration_ablation:
        results = run_calibration_ablation_experiment(
            model_name=args.model,
            sparsity=0.5,
            seeds=seeds,
        )
        output_name = f"{args.model}_calibration_ablation"
    else:
        results = run_pruning_experiment(
            model_name=args.model,
            sparsity_levels=sparsity_levels,
            seeds=seeds,
            methods=args.methods,
            finetune_epochs=args.finetune_epochs,
            calibration_samples=args.calibration_samples,
            pretrain_epochs=args.pretrain_epochs,
        )
        output_name = f"{args.model}_experiment"

        # Print results table
        print_results_table(results, args.methods, sparsity_levels)

    # Save results
    output_path = args.output or (RESULTS_DIR / f"{output_name}_{datetime.now():%Y%m%d_%H%M%S}.json")
    with open(output_path, "w") as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                json_results[key] = {str(k): list(v) for k, v in value.items()}
            else:
                json_results[key] = list(value) if isinstance(value, (list, np.ndarray)) else value

        json.dump({
            "model": args.model,
            "sparsity_levels": sparsity_levels,
            "seeds": seeds,
            "methods": args.methods,
            "finetune_epochs": args.finetune_epochs,
            "calibration_samples": args.calibration_samples,
            "results": json_results,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
