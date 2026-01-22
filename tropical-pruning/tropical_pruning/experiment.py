"""
Experiment Infrastructure: Multi-seed experiments with statistical analysis.

This module provides tools for running reproducible experiments with multiple
random seeds and computing statistical metrics (mean, std, confidence intervals).

Key Components:
    - ExperimentConfig: Configuration dataclass for experiments
    - ResultStats: Statistical aggregation of results
    - ExperimentRunner: Orchestrates multi-seed experiment runs
"""

import copy
import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import random

from tqdm import tqdm


@dataclass
class ResultStats:
    """Statistical aggregation of experiment results."""

    values: List[float]
    metric_name: str = "accuracy"

    @property
    def mean(self) -> float:
        """Mean of values."""
        return np.mean(self.values)

    @property
    def std(self) -> float:
        """Standard deviation."""
        return np.std(self.values, ddof=1) if len(self.values) > 1 else 0.0

    @property
    def min(self) -> float:
        """Minimum value."""
        return np.min(self.values)

    @property
    def max(self) -> float:
        """Maximum value."""
        return np.max(self.values)

    @property
    def ci_95(self) -> Tuple[float, float]:
        """95% confidence interval (assuming normal distribution)."""
        if len(self.values) < 2:
            return (self.mean, self.mean)
        se = self.std / np.sqrt(len(self.values))
        margin = 1.96 * se
        return (self.mean - margin, self.mean + margin)

    @property
    def n(self) -> int:
        """Number of samples."""
        return len(self.values)

    def __str__(self) -> str:
        """Format as 'mean ± std'."""
        return f"{self.mean:.2f} ± {self.std:.2f}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        ci_low, ci_high = self.ci_95
        return {
            "values": self.values,
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "max": self.max,
            "ci_95_low": ci_low,
            "ci_95_high": ci_high,
            "n": self.n,
        }


@dataclass
class ExperimentConfig:
    """Configuration for pruning experiments."""

    # Experiment identification
    name: str = "tropical_pruning_experiment"
    description: str = ""

    # Seeds for reproducibility
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456, 789, 1234])

    # Sparsity levels to test
    sparsity_levels: Tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

    # Pruning methods to compare
    methods: List[str] = field(
        default_factory=lambda: ["tropical", "magnitude_l1", "magnitude_l2", "random"]
    )

    # Data settings
    calibration_samples: int = 1000
    batch_size: int = 64

    # Model settings
    model_name: str = "vgg16"
    pretrained: bool = True

    # Output settings
    output_dir: str = "results"
    save_models: bool = False
    save_intermediate: bool = True

    # Evaluation
    eval_batch_size: int = 100
    track_margin: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentConfig":
        """Create config from dictionary."""
        return cls(**d)

    def save(self, path: str) -> None:
        """Save config to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ExperimentConfig":
        """Load config from JSON file."""
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))


@dataclass
class ExperimentResults:
    """Container for experiment results."""

    config: ExperimentConfig
    results: Dict[str, Dict[float, ResultStats]]  # method -> sparsity -> stats
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_table(self) -> str:
        """Generate a formatted results table."""
        lines = []

        # Header
        header = "| Method | " + " | ".join(
            f"{s*100:.0f}%" for s in sorted(self.config.sparsity_levels)
        ) + " |"
        lines.append(header)
        lines.append("|" + "---|" * (len(self.config.sparsity_levels) + 1))

        # Rows
        for method in self.config.methods:
            if method not in self.results:
                continue
            row = f"| {method} |"
            for sparsity in sorted(self.config.sparsity_levels):
                if sparsity in self.results[method]:
                    stats = self.results[method][sparsity]
                    row += f" {stats.mean:.2f}±{stats.std:.2f} |"
                else:
                    row += " - |"
            lines.append(row)

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        results_dict = {}
        for method, sparsity_results in self.results.items():
            results_dict[method] = {}
            for sparsity, stats in sparsity_results.items():
                results_dict[method][str(sparsity)] = stats.to_dict()

        return {
            "config": self.config.to_dict(),
            "results": results_dict,
            "metadata": self.metadata,
        }

    def save(self, path: str) -> None:
        """Save results to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ExperimentResults":
        """Load results from JSON file."""
        with open(path, "r") as f:
            d = json.load(f)

        config = ExperimentConfig.from_dict(d["config"])
        results = {}
        for method, sparsity_results in d["results"].items():
            results[method] = {}
            for sparsity_str, stats_dict in sparsity_results.items():
                sparsity = float(sparsity_str)
                results[method][sparsity] = ResultStats(
                    values=stats_dict["values"],
                    metric_name="accuracy",
                )

        return cls(
            config=config,
            results=results,
            metadata=d.get("metadata", {}),
        )


class ExperimentRunner:
    """
    Orchestrates multi-seed pruning experiments.

    Example:
        >>> config = ExperimentConfig(
        ...     seeds=[42, 123, 456],
        ...     sparsity_levels=(0.3, 0.5, 0.7),
        ...     methods=["tropical", "magnitude_l1"],
        ... )
        >>> runner = ExperimentRunner(config)
        >>> results = runner.run(
        ...     model_fn=lambda: torchvision.models.vgg16(pretrained=True),
        ...     train_loader=train_loader,
        ...     test_loader=test_loader,
        ...     calibration_loader=calibration_loader,
        ... )
        >>> print(results.get_table())
    """

    def __init__(
        self,
        config: ExperimentConfig,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize experiment runner.

        Args:
            config: Experiment configuration.
            device: Device for computation. If None, auto-detects.
        """
        self.config = config
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def set_seed(self, seed: int) -> None:
        """Set all random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def run(
        self,
        model_fn: Callable[[], nn.Module],
        train_loader: DataLoader,
        test_loader: DataLoader,
        calibration_loader: Optional[DataLoader] = None,
        pruner_factory: Optional[Callable] = None,
        eval_fn: Optional[Callable] = None,
        finetune_fn: Optional[Callable] = None,
        show_progress: bool = True,
    ) -> ExperimentResults:
        """
        Run the full experiment across all seeds, methods, and sparsity levels.

        Args:
            model_fn: Function that returns a fresh model instance.
            train_loader: DataLoader for training (fine-tuning).
            test_loader: DataLoader for evaluation.
            calibration_loader: DataLoader for calibration statistics.
                              If None, uses a subset of train_loader.
            pruner_factory: Optional custom pruner factory.
                           Signature: (model, stats, method) -> pruner
            eval_fn: Optional custom evaluation function.
                    Signature: (model, test_loader, device) -> accuracy
            finetune_fn: Optional fine-tuning function.
                        Signature: (model, train_loader, epochs) -> model
            show_progress: Whether to show progress bars.

        Returns:
            ExperimentResults with aggregated statistics.
        """
        if calibration_loader is None:
            # Create calibration loader from train data
            calibration_loader = self._create_calibration_loader(train_loader)

        if eval_fn is None:
            eval_fn = self._default_eval_fn

        # Results container: method -> sparsity -> list of accuracies
        all_results: Dict[str, Dict[float, List[float]]] = {
            method: {sparsity: [] for sparsity in self.config.sparsity_levels}
            for method in self.config.methods
        }

        # Main experiment loop
        total_runs = (
            len(self.config.seeds)
            * len(self.config.methods)
            * len(self.config.sparsity_levels)
        )

        pbar = tqdm(total=total_runs, desc="Running experiments", disable=not show_progress)

        for seed in self.config.seeds:
            self.set_seed(seed)

            for method in self.config.methods:
                # Create fresh model for each method
                model = model_fn().to(self.device)

                # Collect statistics once per seed-method combination
                if method != "random":
                    stats = self._collect_statistics(model, calibration_loader)
                else:
                    stats = None

                for sparsity in self.config.sparsity_levels:
                    # Create fresh copy of model for pruning
                    model_copy = copy.deepcopy(model)

                    # Prune
                    pruned_model = self._prune_model(
                        model_copy, stats, method, sparsity, seed, pruner_factory
                    )

                    # Optional fine-tuning
                    if finetune_fn is not None:
                        pruned_model = finetune_fn(pruned_model, train_loader)

                    # Evaluate
                    accuracy = eval_fn(pruned_model, test_loader, self.device)
                    all_results[method][sparsity].append(accuracy)

                    pbar.update(1)
                    pbar.set_postfix({
                        "seed": seed,
                        "method": method,
                        "sparsity": f"{sparsity:.0%}",
                        "acc": f"{accuracy:.2f}%",
                    })

                    # Save intermediate results
                    if self.config.save_intermediate:
                        self._save_intermediate_results(all_results)

        pbar.close()

        # Aggregate results
        results = ExperimentResults(
            config=self.config,
            results={
                method: {
                    sparsity: ResultStats(values=accs, metric_name="accuracy")
                    for sparsity, accs in sparsity_results.items()
                    if accs  # Only include non-empty
                }
                for method, sparsity_results in all_results.items()
            },
            metadata={
                "timestamp": datetime.now().isoformat(),
                "device": str(self.device),
                "torch_version": torch.__version__,
            },
        )

        # Save final results
        results.save(self.output_dir / f"{self.config.name}_results.json")

        return results

    def run_single(
        self,
        model: nn.Module,
        method: str,
        sparsity: float,
        calibration_loader: DataLoader,
        test_loader: DataLoader,
        seed: int = 42,
        eval_fn: Optional[Callable] = None,
    ) -> float:
        """
        Run a single pruning experiment.

        Args:
            model: Model to prune.
            method: Pruning method name.
            sparsity: Target sparsity.
            calibration_loader: DataLoader for statistics collection.
            test_loader: DataLoader for evaluation.
            seed: Random seed.
            eval_fn: Evaluation function.

        Returns:
            Accuracy after pruning.
        """
        self.set_seed(seed)
        model = copy.deepcopy(model).to(self.device)

        if eval_fn is None:
            eval_fn = self._default_eval_fn

        # Collect statistics
        if method != "random":
            stats = self._collect_statistics(model, calibration_loader)
        else:
            stats = None

        # Prune
        pruned_model = self._prune_model(model, stats, method, sparsity, seed, None)

        # Evaluate
        return eval_fn(pruned_model, test_loader, self.device)

    def _create_calibration_loader(self, train_loader: DataLoader) -> DataLoader:
        """Create calibration loader from training data."""
        dataset = train_loader.dataset
        num_samples = min(self.config.calibration_samples, len(dataset))
        indices = torch.randperm(len(dataset))[:num_samples].tolist()
        subset = Subset(dataset, indices)
        return DataLoader(
            subset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=getattr(train_loader, "num_workers", 0),
        )

    def _collect_statistics(
        self,
        model: nn.Module,
        calibration_loader: DataLoader,
    ) -> Dict:
        """Collect winner statistics from calibration data."""
        from tropical_pruning.counter import WinnerCounter

        counter = WinnerCounter(
            model,
            track_margin=self.config.track_margin,
            device=self.device,
        )
        return counter.collect(calibration_loader, show_progress=False)

    def _prune_model(
        self,
        model: nn.Module,
        stats: Optional[Dict],
        method: str,
        sparsity: float,
        seed: int,
        pruner_factory: Optional[Callable],
    ) -> nn.Module:
        """Apply pruning method to model."""
        if pruner_factory is not None:
            pruner = pruner_factory(model, stats, method)
            return pruner.prune(sparsity, inplace=True)

        if method == "tropical":
            from tropical_pruning.pruner import TropicalPruner
            pruner = TropicalPruner(model, stats)
            return pruner.prune(sparsity, inplace=True)

        elif method == "magnitude_l1":
            from tropical_pruning.baselines import MagnitudeStructuredPruner
            pruner = MagnitudeStructuredPruner(model, norm="l1")
            return pruner.prune(sparsity, inplace=True)

        elif method == "magnitude_l2":
            from tropical_pruning.baselines import MagnitudeStructuredPruner
            pruner = MagnitudeStructuredPruner(model, norm="l2")
            return pruner.prune(sparsity, inplace=True)

        elif method == "random":
            from tropical_pruning.baselines import RandomStructuredPruner
            pruner = RandomStructuredPruner(model, seed=seed)
            return pruner.prune(sparsity, inplace=True)

        elif method == "activation":
            from tropical_pruning.baselines import ActivationSparsityPruner
            pruner = ActivationSparsityPruner(model)
            pruner.collect_activations(self._calibration_loader)
            return pruner.prune(sparsity, inplace=True)

        else:
            raise ValueError(f"Unknown pruning method: {method}")

    def _default_eval_fn(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        device: torch.device,
    ) -> float:
        """Default evaluation function - classification accuracy."""
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in test_loader:
                if isinstance(batch, (list, tuple)):
                    inputs, targets = batch[0], batch[1]
                else:
                    inputs, targets = batch, None

                inputs = inputs.to(device)
                if targets is not None:
                    targets = targets.to(device)

                outputs = model(inputs)
                _, predicted = outputs.max(1)

                if targets is not None:
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

        return 100.0 * correct / total if total > 0 else 0.0

    def _save_intermediate_results(self, all_results: Dict) -> None:
        """Save intermediate results to file."""
        path = self.output_dir / f"{self.config.name}_intermediate.json"
        with open(path, "w") as f:
            json.dump(all_results, f, indent=2)


def run_calibration_ablation(
    model_fn: Callable[[], nn.Module],
    train_dataset: torch.utils.data.Dataset,
    test_loader: DataLoader,
    sample_sizes: List[int] = [100, 500, 1000, 2000, 5000],
    sparsity: float = 0.5,
    seeds: List[int] = [42, 123, 456],
    device: Optional[torch.device] = None,
    show_progress: bool = True,
) -> Dict[int, ResultStats]:
    """
    Run calibration sample ablation study.

    Tests how the number of calibration samples affects pruning quality.

    Args:
        model_fn: Function returning fresh model.
        train_dataset: Training dataset for calibration samples.
        test_loader: Test data loader.
        sample_sizes: List of calibration sample sizes to test.
        sparsity: Target sparsity level.
        seeds: Random seeds for trials.
        device: Computation device.
        show_progress: Show progress bar.

    Returns:
        Dictionary mapping sample size to ResultStats.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results: Dict[int, List[float]] = {n: [] for n in sample_sizes}

    total_runs = len(sample_sizes) * len(seeds)
    pbar = tqdm(total=total_runs, desc="Calibration ablation", disable=not show_progress)

    for seed in seeds:
        # Set seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        for n_samples in sample_sizes:
            # Create calibration subset
            indices = torch.randperm(len(train_dataset))[:n_samples].tolist()
            subset = Subset(train_dataset, indices)
            cal_loader = DataLoader(subset, batch_size=64, shuffle=False)

            # Get fresh model
            model = model_fn().to(device)

            # Collect statistics
            from tropical_pruning.counter import WinnerCounter
            counter = WinnerCounter(model, device=device)
            stats = counter.collect(cal_loader, show_progress=False)

            # Prune
            from tropical_pruning.pruner import TropicalPruner
            pruner = TropicalPruner(model, stats)

            pruned_model = pruner.prune(sparsity, inplace=True)

            # Evaluate
            accuracy = _evaluate_accuracy(pruned_model, test_loader, device)
            results[n_samples].append(accuracy)

            pbar.update(1)
            pbar.set_postfix({
                "samples": n_samples,
                "seed": seed,
                "acc": f"{accuracy:.2f}%",
            })

    pbar.close()

    return {
        n: ResultStats(values=accs, metric_name="accuracy")
        for n, accs in results.items()
    }


def _evaluate_accuracy(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> float:
    """Evaluate model accuracy."""
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
