#!/usr/bin/env python3
"""
ImageNet experiments for tropical pruning.

This script evaluates tropical pruning on ImageNet-scale models including
MobileNetV2 and EfficientNet-B0, which use depthwise separable convolutions.

Models:
- MobileNetV2: Lightweight mobile architecture with depthwise separable convs
- EfficientNet-B0: Efficient scaling with compound coefficients

Usage:
    # Basic experiment
    python imagenet_experiment.py --data-dir /path/to/imagenet

    # Specific model
    python imagenet_experiment.py --model mobilenetv2 --sparsity 0.5

    # Full evaluation with fine-tuning
    python imagenet_experiment.py --model efficientnet_b0 --finetune --epochs 10

Requirements:
    - ImageNet dataset (ILSVRC2012) organized as:
        imagenet/train/n01440764/*.JPEG
        imagenet/val/n01440764/*.JPEG
"""

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import (
    mobilenet_v2,
    efficientnet_b0,
    MobileNet_V2_Weights,
    EfficientNet_B0_Weights,
)
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tropical_pruning import (
    ConvWinnerCounter,
    ConvTropicalPruner,
    ExperimentConfig,
    ExperimentRunner,
    FinetuneConfig,
    finetune_pruned_model,
    MagnitudeStructuredPruner,
    NetworkSlimmingPruner,
    TaylorPruner,
)
from tropical_pruning.benchmarks import PruningBenchmark


# ImageNet normalization
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_imagenet_transforms(train: bool = True, input_size: int = 224):
    """Get ImageNet transforms for training or validation."""
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        # Validation: resize to 256 then center crop to 224
        resize_size = int(input_size / 0.875)  # 224 / 0.875 = 256
        return transforms.Compose([
            transforms.Resize(resize_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])


def load_imagenet(
    data_dir: str,
    batch_size: int = 64,
    num_workers: int = 4,
    input_size: int = 224,
    subset_size: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Load ImageNet dataset.

    Args:
        data_dir: Path to ImageNet root directory.
        batch_size: Batch size for data loaders.
        num_workers: Number of data loading workers.
        input_size: Input image size (224 for most models).
        subset_size: If provided, use only this many samples for quick testing.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    train_transform = get_imagenet_transforms(train=True, input_size=input_size)
    val_transform = get_imagenet_transforms(train=False, input_size=input_size)

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    if subset_size is not None:
        # Use random subset for quick testing
        train_indices = torch.randperm(len(train_dataset))[:subset_size].tolist()
        val_indices = torch.randperm(len(val_dataset))[:min(subset_size // 5, 10000)].tolist()
        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_model(model_name: str, pretrained: bool = True) -> nn.Module:
    """
    Get a pretrained ImageNet model.

    Args:
        model_name: One of 'mobilenetv2', 'efficientnet_b0'.
        pretrained: Whether to load pretrained weights.

    Returns:
        Pretrained model.
    """
    model_name = model_name.lower()

    if model_name in ["mobilenetv2", "mobilenet_v2"]:
        weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        return mobilenet_v2(weights=weights)
    elif model_name in ["efficientnet_b0", "efficientnetb0"]:
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        return efficientnet_b0(weights=weights)
    else:
        raise ValueError(f"Unknown model: {model_name}. "
                        f"Supported: mobilenetv2, efficientnet_b0")


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    topk: Tuple[int, ...] = (1, 5),
) -> Dict[str, float]:
    """
    Evaluate model on validation set.

    Args:
        model: Model to evaluate.
        val_loader: Validation data loader.
        device: Computation device.
        topk: Top-k accuracies to compute.

    Returns:
        Dictionary with top-k accuracies.
    """
    model.eval()

    correct = {k: 0 for k in topk}
    total = 0

    for inputs, targets in tqdm(val_loader, desc="Evaluating"):
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)

        maxk = max(topk)
        _, pred = outputs.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct_k = pred.eq(targets.view(1, -1).expand_as(pred))

        for k in topk:
            correct[k] += correct_k[:k].reshape(-1).float().sum().item()

        total += targets.size(0)

    return {f"top{k}": 100.0 * correct[k] / total for k in topk}


def run_pruning_experiment(
    model_name: str,
    data_dir: str,
    sparsity_levels: List[float],
    methods: List[str],
    device: torch.device,
    num_calibration_samples: int = 2048,
    batch_size: int = 64,
    num_workers: int = 4,
    finetune: bool = False,
    finetune_epochs: int = 5,
    seed: int = 42,
) -> Dict:
    """
    Run pruning experiment on an ImageNet model.

    Args:
        model_name: Model architecture name.
        data_dir: Path to ImageNet data.
        sparsity_levels: List of sparsity levels to test.
        methods: List of pruning methods.
        device: Computation device.
        num_calibration_samples: Number of samples for calibration.
        batch_size: Batch size.
        num_workers: Data loading workers.
        finetune: Whether to fine-tune after pruning.
        finetune_epochs: Number of fine-tuning epochs.
        seed: Random seed.

    Returns:
        Dictionary with experiment results.
    """
    torch.manual_seed(seed)

    print(f"\n{'='*60}")
    print(f"ImageNet Pruning Experiment: {model_name}")
    print(f"{'='*60}")

    # Load data
    print("\nLoading ImageNet dataset...")
    train_loader, val_loader = load_imagenet(
        data_dir, batch_size=batch_size, num_workers=num_workers
    )

    # Create calibration loader
    cal_subset = Subset(
        train_loader.dataset,
        torch.randperm(len(train_loader.dataset))[:num_calibration_samples].tolist()
    )
    cal_loader = DataLoader(
        cal_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    # Load model
    print(f"Loading {model_name}...")
    model = get_model(model_name, pretrained=True).to(device)

    # Evaluate baseline
    print("\nEvaluating baseline model...")
    baseline_metrics = evaluate_model(model, val_loader, device)
    print(f"Baseline - Top-1: {baseline_metrics['top1']:.2f}%, "
          f"Top-5: {baseline_metrics['top5']:.2f}%")

    results = {
        "model": model_name,
        "baseline": baseline_metrics,
        "sparsity_results": {},
    }

    for sparsity in sparsity_levels:
        print(f"\n{'='*40}")
        print(f"Sparsity: {sparsity*100:.0f}%")
        print(f"{'='*40}")

        sparsity_results = {}

        for method in methods:
            print(f"\n  Method: {method}")

            # Reload fresh model
            model = get_model(model_name, pretrained=True).to(device)

            # Prune model
            if method == "tropical":
                # Collect statistics
                counter = ConvWinnerCounter(model, device=device)
                counter.collect(cal_loader, show_progress=True)
                stats = counter.get_statistics_as_winner_stats()

                # Prune
                pruner = ConvTropicalPruner(model, stats)
                pruned_model = pruner.prune(sparsity=sparsity)

            elif method == "magnitude":
                pruner = MagnitudeStructuredPruner(model)
                pruned_model = pruner.prune(sparsity=sparsity)

            elif method == "slimming":
                pruner = NetworkSlimmingPruner(model)
                pruned_model = pruner.prune(sparsity=sparsity)

            elif method == "taylor":
                pruner = TaylorPruner(model)
                pruned_model = pruner.prune(
                    sparsity=sparsity,
                    dataloader=cal_loader,
                    device=device,
                )

            else:
                raise ValueError(f"Unknown method: {method}")

            pruned_model = pruned_model.to(device)

            # Evaluate before fine-tuning
            metrics_before = evaluate_model(pruned_model, val_loader, device)
            print(f"    Before FT - Top-1: {metrics_before['top1']:.2f}%, "
                  f"Top-5: {metrics_before['top5']:.2f}%")

            method_results = {
                "before_finetune": metrics_before,
            }

            # Fine-tune if requested
            if finetune:
                print(f"    Fine-tuning for {finetune_epochs} epochs...")
                config = FinetuneConfig(
                    epochs=finetune_epochs,
                    lr=0.001,
                    lr_schedule="cosine",
                    optimizer="sgd",
                    weight_decay=1e-4,
                )
                pruned_model, ft_result = finetune_pruned_model(
                    pruned_model, train_loader, val_loader,
                    config=config, device=device, show_progress=True
                )
                metrics_after = evaluate_model(pruned_model, val_loader, device)
                print(f"    After FT  - Top-1: {metrics_after['top1']:.2f}%, "
                      f"Top-5: {metrics_after['top5']:.2f}%")
                method_results["after_finetune"] = metrics_after
                method_results["finetune_result"] = ft_result.to_dict()

            sparsity_results[method] = method_results

        results["sparsity_results"][f"{sparsity*100:.0f}%"] = sparsity_results

    return results


def run_efficiency_benchmark(
    model_name: str,
    data_dir: str,
    device: torch.device,
    num_calibration_samples: int = 1024,
    batch_size: int = 32,
) -> Dict:
    """
    Benchmark efficiency metrics for different pruning methods.

    Args:
        model_name: Model architecture name.
        data_dir: Path to ImageNet data.
        device: Computation device.
        num_calibration_samples: Number of calibration samples.
        batch_size: Batch size.

    Returns:
        Benchmark results.
    """
    print(f"\n{'='*60}")
    print(f"Efficiency Benchmark: {model_name}")
    print(f"{'='*60}")

    # Load a small subset for timing
    train_loader, val_loader = load_imagenet(
        data_dir, batch_size=batch_size, subset_size=num_calibration_samples * 2
    )

    cal_subset = Subset(
        train_loader.dataset,
        torch.randperm(len(train_loader.dataset))[:num_calibration_samples].tolist()
    )
    cal_loader = DataLoader(cal_subset, batch_size=batch_size, shuffle=False)

    model = get_model(model_name, pretrained=True).to(device)

    benchmark = PruningBenchmark(model, device=device)

    # Measure statistics collection time
    print("\nMeasuring statistics collection time...")
    stats_time = benchmark.measure_statistics_collection_time(cal_loader)
    print(f"  Statistics collection: {stats_time.elapsed_time*1000:.2f} ms")

    # Compare inference times at different sparsity levels
    print("\nMeasuring inference times...")
    results = {
        "model": model_name,
        "stats_collection_ms": stats_time.elapsed_time * 1000,
        "inference_times": {},
    }

    for sparsity in [0.3, 0.5, 0.7]:
        model = get_model(model_name, pretrained=True).to(device)

        # Prune with tropical method
        counter = ConvWinnerCounter(model, device=device)
        counter.collect(cal_loader, show_progress=False)
        stats = counter.get_statistics_as_winner_stats()
        pruner = ConvTropicalPruner(model, stats)
        pruned_model = pruner.prune(sparsity=sparsity).to(device)

        # Measure inference time
        sample_input = torch.randn(1, 3, 224, 224, device=device)
        inf_time = benchmark.measure_inference_time(pruned_model, sample_input)
        results["inference_times"][f"{sparsity*100:.0f}%"] = {
            "mean_ms": inf_time.elapsed_time * 1000,
        }
        print(f"  Sparsity {sparsity*100:.0f}%: {inf_time.elapsed_time*1000:.2f} ms")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="ImageNet pruning experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to ImageNet dataset",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mobilenetv2",
        choices=["mobilenetv2", "efficientnet_b0"],
        help="Model architecture",
    )
    parser.add_argument(
        "--sparsity",
        type=float,
        nargs="+",
        default=[0.3, 0.5, 0.7],
        help="Sparsity levels to test",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["tropical", "magnitude", "slimming"],
        help="Pruning methods to compare",
    )
    parser.add_argument(
        "--finetune",
        action="store_true",
        help="Fine-tune after pruning",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of fine-tuning epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=2048,
        help="Number of calibration samples",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--benchmark-only",
        action="store_true",
        help="Only run efficiency benchmark",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results",
    )

    args = parser.parse_args()

    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"\nUsing CUDA backend: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  Device count: {torch.cuda.device_count()}")
        print(f"  Current device: {torch.cuda.current_device()}")
        print(f"  Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"  Memory reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
        print(f"  Memory capacity: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB\n")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("\nUsing Apple MPS backend\n")
    else:
        device = torch.device("cpu")
        print("\nUsing CPU backend\n")

    # Run experiments
    if args.benchmark_only:
        results = run_efficiency_benchmark(
            model_name=args.model,
            data_dir=args.data_dir,
            device=device,
            num_calibration_samples=args.calibration_samples,
            batch_size=args.batch_size,
        )
    else:
        results = run_pruning_experiment(
            model_name=args.model,
            data_dir=args.data_dir,
            sparsity_levels=args.sparsity,
            methods=args.methods,
            device=device,
            num_calibration_samples=args.calibration_samples,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            finetune=args.finetune,
            finetune_epochs=args.epochs,
            seed=args.seed,
        )

    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")

    return results


if __name__ == "__main__":
    main()
