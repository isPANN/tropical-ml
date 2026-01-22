"""
Benchmarking utilities for pruning methods.

This module provides tools for measuring and comparing:
- Statistics collection time
- Inference time/latency
- Memory usage
- FLOPs/parameter counts
- Method comparisons

Key Classes:
    - PruningBenchmark: Main benchmarking interface
    - TimingResult: Container for timing measurements
"""

import copy
import gc
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


@dataclass
class TimingResult:
    """Container for timing measurements."""

    times: List[float]
    unit: str = "ms"

    @property
    def mean(self) -> float:
        """Mean time."""
        return np.mean(self.times)

    @property
    def std(self) -> float:
        """Standard deviation."""
        return np.std(self.times) if len(self.times) > 1 else 0.0

    @property
    def min(self) -> float:
        """Minimum time."""
        return np.min(self.times)

    @property
    def max(self) -> float:
        """Maximum time."""
        return np.max(self.times)

    @property
    def median(self) -> float:
        """Median time."""
        return np.median(self.times)

    def __str__(self) -> str:
        return f"{self.mean:.2f} ± {self.std:.2f} {self.unit}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "max": self.max,
            "median": self.median,
            "unit": self.unit,
            "n_runs": len(self.times),
        }


@dataclass
class ModelStats:
    """Statistics about a model."""

    total_params: int
    trainable_params: int
    conv_params: int
    linear_params: int
    bn_params: int
    num_conv_layers: int
    num_linear_layers: int
    num_bn_layers: int
    estimated_flops: int  # Simplified estimate
    memory_mb: float  # Approximate memory in MB

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_params": self.total_params,
            "trainable_params": self.trainable_params,
            "conv_params": self.conv_params,
            "linear_params": self.linear_params,
            "bn_params": self.bn_params,
            "num_conv_layers": self.num_conv_layers,
            "num_linear_layers": self.num_linear_layers,
            "num_bn_layers": self.num_bn_layers,
            "estimated_flops": self.estimated_flops,
            "memory_mb": self.memory_mb,
        }


@dataclass
class BenchmarkResult:
    """Complete benchmark results for a method."""

    method_name: str
    sparsity: float
    accuracy: float
    statistics_time: Optional[TimingResult] = None
    pruning_time: Optional[TimingResult] = None
    inference_time: Optional[TimingResult] = None
    model_stats: Optional[ModelStats] = None
    compression_ratio: float = 1.0
    speedup: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "method_name": self.method_name,
            "sparsity": self.sparsity,
            "accuracy": self.accuracy,
            "compression_ratio": self.compression_ratio,
            "speedup": self.speedup,
        }
        if self.statistics_time:
            result["statistics_time"] = self.statistics_time.to_dict()
        if self.pruning_time:
            result["pruning_time"] = self.pruning_time.to_dict()
        if self.inference_time:
            result["inference_time"] = self.inference_time.to_dict()
        if self.model_stats:
            result["model_stats"] = self.model_stats.to_dict()
        return result


class PruningBenchmark:
    """
    Benchmarking suite for pruning methods.

    Measures and compares:
    - Statistics collection overhead
    - Pruning execution time
    - Inference latency
    - Memory usage
    - Compression ratios

    Example:
        >>> benchmark = PruningBenchmark(device="cuda")
        >>> results = benchmark.compare_methods(
        ...     model_fn=lambda: vgg16(pretrained=True),
        ...     calibration_loader=cal_loader,
        ...     test_loader=test_loader,
        ...     methods=["tropical", "magnitude_l1", "random"],
        ...     sparsity=0.5,
        ... )
        >>> print(results.to_table())
    """

    def __init__(
        self,
        device: Optional[Union[str, torch.device]] = None,
        warmup_runs: int = 5,
        timing_runs: int = 20,
    ):
        """
        Initialize benchmark.

        Args:
            device: Computation device. Auto-detects if None.
            warmup_runs: Number of warmup runs before timing.
            timing_runs: Number of runs for timing measurements.
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device) if isinstance(device, str) else device

        self.warmup_runs = warmup_runs
        self.timing_runs = timing_runs

    def measure_statistics_collection_time(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        num_batches: Optional[int] = None,
        n_runs: int = 3,
    ) -> TimingResult:
        """
        Measure time to collect winner statistics.

        Args:
            model: Model to analyze.
            dataloader: Calibration data.
            num_batches: Number of batches to process.
            n_runs: Number of timing runs.

        Returns:
            TimingResult with measurements.
        """
        from tropical_pruning.counter import create_winner_counter

        times = []

        for _ in range(n_runs):
            model_copy = copy.deepcopy(model).to(self.device)
            counter = create_winner_counter(model_copy, device=self.device)

            # Warmup
            batch = next(iter(dataloader))
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            batch = batch.to(self.device)
            counter.forward(batch)
            counter.reset()

            # Time the collection
            torch.cuda.synchronize() if self.device.type == "cuda" else None
            start = time.perf_counter()

            counter.collect(dataloader, num_batches=num_batches, show_progress=False)

            torch.cuda.synchronize() if self.device.type == "cuda" else None
            end = time.perf_counter()

            times.append((end - start) * 1000)  # Convert to ms

            del counter
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return TimingResult(times=times, unit="ms")

    def measure_inference_time(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        batch_size: int = 1,
        warmup_runs: Optional[int] = None,
        timing_runs: Optional[int] = None,
    ) -> TimingResult:
        """
        Measure model inference latency.

        Args:
            model: Model to benchmark.
            input_shape: Shape of input tensor (without batch dim).
            batch_size: Batch size for inference.
            warmup_runs: Number of warmup iterations.
            timing_runs: Number of timed iterations.

        Returns:
            TimingResult with latency measurements.
        """
        warmup_runs = warmup_runs or self.warmup_runs
        timing_runs = timing_runs or self.timing_runs

        model = model.to(self.device)
        model.eval()

        # Create dummy input
        dummy_input = torch.randn(batch_size, *input_shape, device=self.device)

        # Warmup
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(dummy_input)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()

        # Time
        times = []
        with torch.no_grad():
            for _ in range(timing_runs):
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()

                _ = model(dummy_input)

                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                end = time.perf_counter()

                times.append((end - start) * 1000)  # ms

        return TimingResult(times=times, unit="ms")

    def measure_pruning_time(
        self,
        model: nn.Module,
        statistics: Dict,
        method: str,
        sparsity: float,
        n_runs: int = 5,
    ) -> TimingResult:
        """
        Measure time to perform pruning.

        Args:
            model: Model to prune.
            statistics: Winner statistics dictionary.
            method: Pruning method name.
            sparsity: Target sparsity.
            n_runs: Number of timing runs.

        Returns:
            TimingResult with measurements.
        """
        times = []

        for _ in range(n_runs):
            model_copy = copy.deepcopy(model).to(self.device)

            start = time.perf_counter()

            if method == "tropical":
                has_conv = any(isinstance(m, nn.Conv2d) for m in model_copy.modules())
                if has_conv:
                    from tropical_pruning.conv_pruner import ConvTropicalPruner
                    pruner = ConvTropicalPruner(model_copy, statistics)
                else:
                    from tropical_pruning.pruner import TropicalPruner
                    pruner = TropicalPruner(model_copy, statistics)
                _ = pruner.prune(sparsity, inplace=True)

            elif method.startswith("magnitude"):
                from tropical_pruning.baselines import MagnitudeStructuredPruner
                norm = "l2" if "l2" in method else "l1"
                pruner = MagnitudeStructuredPruner(model_copy, norm=norm)
                _ = pruner.prune(sparsity, inplace=True)

            elif method == "random":
                from tropical_pruning.baselines import RandomStructuredPruner
                pruner = RandomStructuredPruner(model_copy)
                _ = pruner.prune(sparsity, inplace=True)

            end = time.perf_counter()
            times.append((end - start) * 1000)

            del model_copy, pruner
            gc.collect()

        return TimingResult(times=times, unit="ms")

    def get_model_stats(self, model: nn.Module) -> ModelStats:
        """
        Get statistics about a model.

        Args:
            model: Model to analyze.

        Returns:
            ModelStats with counts and estimates.
        """
        total_params = 0
        trainable_params = 0
        conv_params = 0
        linear_params = 0
        bn_params = 0
        num_conv = 0
        num_linear = 0
        num_bn = 0
        estimated_flops = 0

        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                num_conv += 1
                params = module.weight.numel()
                if module.bias is not None:
                    params += module.bias.numel()
                conv_params += params
                # Simplified FLOP estimate (ignoring spatial dimensions)
                estimated_flops += 2 * module.weight.numel()

            elif isinstance(module, nn.Linear):
                num_linear += 1
                params = module.weight.numel()
                if module.bias is not None:
                    params += module.bias.numel()
                linear_params += params
                estimated_flops += 2 * module.weight.numel()

            elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                num_bn += 1
                if module.affine:
                    bn_params += module.weight.numel() + module.bias.numel()

        for p in model.parameters():
            total_params += p.numel()
            if p.requires_grad:
                trainable_params += p.numel()

        # Estimate memory (parameters + gradients + some overhead)
        param_bytes = total_params * 4  # float32
        memory_mb = param_bytes / (1024 ** 2) * 2.5  # Rough estimate with gradients

        return ModelStats(
            total_params=total_params,
            trainable_params=trainable_params,
            conv_params=conv_params,
            linear_params=linear_params,
            bn_params=bn_params,
            num_conv_layers=num_conv,
            num_linear_layers=num_linear,
            num_bn_layers=num_bn,
            estimated_flops=estimated_flops,
            memory_mb=memory_mb,
        )

    def compare_methods(
        self,
        model_fn: Callable[[], nn.Module],
        calibration_loader: DataLoader,
        test_loader: DataLoader,
        methods: List[str] = ["tropical", "magnitude_l1", "magnitude_l2", "random"],
        sparsity: float = 0.5,
        input_shape: Tuple[int, ...] = (3, 224, 224),
        measure_inference: bool = True,
    ) -> "ComparisonResult":
        """
        Compare multiple pruning methods.

        Args:
            model_fn: Function that returns a fresh model.
            calibration_loader: DataLoader for statistics collection.
            test_loader: DataLoader for accuracy evaluation.
            methods: List of method names to compare.
            sparsity: Target sparsity level.
            input_shape: Input tensor shape.
            measure_inference: Whether to measure inference time.

        Returns:
            ComparisonResult with all benchmarks.
        """
        results = []

        # Get baseline stats
        baseline_model = model_fn().to(self.device)
        baseline_stats = self.get_model_stats(baseline_model)

        if measure_inference:
            baseline_inference = self.measure_inference_time(
                baseline_model, input_shape
            )
        else:
            baseline_inference = None

        # Evaluate baseline accuracy
        baseline_acc = self._evaluate_accuracy(baseline_model, test_loader)

        # Add baseline result
        results.append(BenchmarkResult(
            method_name="baseline",
            sparsity=0.0,
            accuracy=baseline_acc,
            inference_time=baseline_inference,
            model_stats=baseline_stats,
            compression_ratio=1.0,
            speedup=1.0,
        ))

        # Collect statistics once (used by tropical method)
        from tropical_pruning.counter import create_winner_counter
        counter = create_winner_counter(baseline_model, device=self.device)
        stats = counter.collect(calibration_loader, show_progress=False)
        counter.remove_hooks()

        # Benchmark each method
        for method in methods:
            print(f"\nBenchmarking: {method}")

            # Get fresh model
            model = model_fn().to(self.device)

            # Measure statistics collection time (only for tropical)
            stats_time = None
            if method == "tropical":
                stats_time = self.measure_statistics_collection_time(
                    model, calibration_loader, n_runs=3
                )

            # Measure pruning time
            pruning_time = self.measure_pruning_time(
                model, stats, method, sparsity
            )

            # Perform actual pruning
            pruned_model = self._prune_model(model, stats, method, sparsity)

            # Get pruned model stats
            pruned_stats = self.get_model_stats(pruned_model)

            # Measure inference time
            if measure_inference:
                inference_time = self.measure_inference_time(
                    pruned_model, input_shape
                )
            else:
                inference_time = None

            # Evaluate accuracy
            accuracy = self._evaluate_accuracy(pruned_model, test_loader)

            # Calculate compression and speedup
            compression_ratio = baseline_stats.total_params / max(pruned_stats.total_params, 1)
            speedup = (
                baseline_inference.mean / inference_time.mean
                if baseline_inference and inference_time else 1.0
            )

            results.append(BenchmarkResult(
                method_name=method,
                sparsity=sparsity,
                accuracy=accuracy,
                statistics_time=stats_time,
                pruning_time=pruning_time,
                inference_time=inference_time,
                model_stats=pruned_stats,
                compression_ratio=compression_ratio,
                speedup=speedup,
            ))

            del pruned_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return ComparisonResult(results=results, baseline_accuracy=baseline_acc)

    def _prune_model(
        self,
        model: nn.Module,
        statistics: Dict,
        method: str,
        sparsity: float,
    ) -> nn.Module:
        """Apply pruning to model."""
        model_copy = copy.deepcopy(model)

        if method == "tropical":
            has_conv = any(isinstance(m, nn.Conv2d) for m in model_copy.modules())
            if has_conv:
                from tropical_pruning.conv_pruner import ConvTropicalPruner
                pruner = ConvTropicalPruner(model_copy, statistics)
            else:
                from tropical_pruning.pruner import TropicalPruner
                pruner = TropicalPruner(model_copy, statistics)
            return pruner.prune(sparsity, inplace=True)

        elif method.startswith("magnitude"):
            from tropical_pruning.baselines import MagnitudeStructuredPruner
            norm = "l2" if "l2" in method else "l1"
            pruner = MagnitudeStructuredPruner(model_copy, norm=norm)
            return pruner.prune(sparsity, inplace=True)

        elif method == "random":
            from tropical_pruning.baselines import RandomStructuredPruner
            pruner = RandomStructuredPruner(model_copy, seed=42)
            return pruner.prune(sparsity, inplace=True)

        else:
            raise ValueError(f"Unknown method: {method}")

    def _evaluate_accuracy(
        self,
        model: nn.Module,
        test_loader: DataLoader,
    ) -> float:
        """Evaluate model accuracy."""
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in test_loader:
                if isinstance(batch, (list, tuple)):
                    inputs, targets = batch[0], batch[1]
                else:
                    continue

                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        return 100.0 * correct / total if total > 0 else 0.0


@dataclass
class ComparisonResult:
    """Results from comparing multiple methods."""

    results: List[BenchmarkResult]
    baseline_accuracy: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "baseline_accuracy": self.baseline_accuracy,
            "methods": [r.to_dict() for r in self.results],
        }

    def to_table(self) -> str:
        """Generate markdown table."""
        lines = [
            "| Method | Sparsity | Accuracy | Params | Compression | Inference (ms) | Speedup |",
            "|--------|----------|----------|--------|-------------|----------------|---------|",
        ]

        for r in self.results:
            params = r.model_stats.total_params if r.model_stats else "-"
            if isinstance(params, int):
                params = f"{params:,}"

            inference = f"{r.inference_time.mean:.2f}" if r.inference_time else "-"
            speedup = f"{r.speedup:.2f}x" if r.speedup != 1.0 else "-"

            line = (
                f"| {r.method_name} | {r.sparsity:.0%} | {r.accuracy:.2f}% | "
                f"{params} | {r.compression_ratio:.2f}x | {inference} | {speedup} |"
            )
            lines.append(line)

        return "\n".join(lines)

    def to_dataframe(self) -> "pd.DataFrame":
        """Convert to pandas DataFrame."""
        if not HAS_PANDAS:
            raise ImportError("pandas required for to_dataframe()")

        data = []
        for r in self.results:
            row = {
                "method": r.method_name,
                "sparsity": r.sparsity,
                "accuracy": r.accuracy,
                "compression_ratio": r.compression_ratio,
                "speedup": r.speedup,
            }
            if r.model_stats:
                row["params"] = r.model_stats.total_params
            if r.inference_time:
                row["inference_ms"] = r.inference_time.mean
            if r.statistics_time:
                row["stats_collection_ms"] = r.statistics_time.mean
            if r.pruning_time:
                row["pruning_ms"] = r.pruning_time.mean
            data.append(row)

        return pd.DataFrame(data)
