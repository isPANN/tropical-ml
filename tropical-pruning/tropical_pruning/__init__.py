"""
Tropical Pruning: Winner-based Tropical Structured Pruning for Neural Networks.

This package implements tropical-geometry-aware pruning methods that leverage
the argmax structure of tropical GEMM operations to identify and prune neurons
that never "win" in the max-plus competition.

Core Components:
    - WinnerCounter: Tracks winner statistics for Linear layers
    - ConvWinnerCounter: Tracks winner statistics for Conv2d layers
    - TropicalPruner: Pruning interface for Linear layers
    - ConvTropicalPruner: Filter pruning for Conv2d layers
    - create_winner_counter: Factory to auto-select counter type
    - TropicalLinear: Drop-in replacement for nn.Linear with tropical ops

Experiment Infrastructure:
    - ExperimentRunner: Multi-seed experiment runner with statistical analysis
    - FinetuneConfig: Configuration for fine-tuning pruned models
    - PruningBenchmark: Timing and efficiency benchmarking

Baseline Methods:
    - MagnitudeStructuredPruner: L1/L2 weight magnitude pruning
    - NetworkSlimmingPruner: BatchNorm gamma-based pruning (Liu et al. ICCV 2017)
    - TaylorPruner: Gradient-based importance (Molchanov et al. CVPR 2019)

Example (Linear models):
    >>> from tropical_pruning import WinnerCounter, TropicalPruner
    >>>
    >>> counter = WinnerCounter(model)
    >>> stats = counter.collect(calibration_loader)
    >>> pruner = TropicalPruner(model, stats)
    >>> pruned_model = pruner.prune(sparsity=0.5)

Example (Conv models):
    >>> from tropical_pruning import ConvWinnerCounter, ConvTropicalPruner
    >>>
    >>> counter = ConvWinnerCounter(model)
    >>> stats = counter.collect(calibration_loader)
    >>> pruner = ConvTropicalPruner(model, stats)
    >>> pruned_model = pruner.prune(sparsity=0.5)

Example (Auto-detection):
    >>> from tropical_pruning import create_winner_counter
    >>>
    >>> counter = create_winner_counter(model)  # Auto-selects type
    >>> stats = counter.collect(calibration_loader)

Example (Multi-seed experiments):
    >>> from tropical_pruning import ExperimentRunner, ExperimentConfig
    >>>
    >>> config = ExperimentConfig(seeds=[42, 123, 456], sparsity_levels=[0.5, 0.7])
    >>> runner = ExperimentRunner(config)
    >>> results = runner.run_multi_seed(model_fn, train_loader, val_loader)
"""

__version__ = "0.2.0"

# Linear layer support
from tropical_pruning.counter import WinnerCounter, WinnerStatistics, create_winner_counter
from tropical_pruning.pruner import TropicalPruner
from tropical_pruning.layers import TropicalLinear

# Conv2d layer support
from tropical_pruning.conv_counter import ConvWinnerCounter, ConvWinnerStatistics
from tropical_pruning.conv_pruner import ConvTropicalPruner

# Pruning criteria
from tropical_pruning.criteria import (
    PruningCriterion,
    WinnerFrequencyCriterion,
    WinnerMarginCriterion,
    CombinedCriterion,
    get_criterion,
)

# Baseline methods
from tropical_pruning.baselines import (
    MagnitudeStructuredPruner,
    RandomStructuredPruner,
    ActivationSparsityPruner,
    NetworkSlimmingPruner,
    TaylorPruner,
)

# Experiment infrastructure
from tropical_pruning.experiment import (
    ExperimentConfig,
    ResultStats,
    ExperimentRunner,
    run_calibration_ablation,
)

# Fine-tuning
from tropical_pruning.finetuning import (
    FinetuneConfig,
    FinetuneResult,
    finetune_pruned_model,
    quick_finetune,
    knowledge_distillation_finetune,
)

# Benchmarking
from tropical_pruning.benchmarks import (
    PruningBenchmark,
    TimingResult,
    ModelStats,
    BenchmarkResult,
    ComparisonResult,
)

# Visualization
from tropical_pruning.visualization import (
    plot_winner_frequency_histogram,
    plot_sparsity_accuracy_curve,
    plot_finetuning_curve,
    plot_layer_importance,
    plot_calibration_ablation,
    plot_compression_comparison,
    plot_method_comparison_heatmap,
    setup_publication_style,
)

__all__ = [
    # Core classes - Linear
    "WinnerCounter",
    "WinnerStatistics",
    "TropicalPruner",
    "TropicalLinear",
    # Core classes - Conv2d
    "ConvWinnerCounter",
    "ConvWinnerStatistics",
    "ConvTropicalPruner",
    # Factory function
    "create_winner_counter",
    # Pruning criteria
    "PruningCriterion",
    "WinnerFrequencyCriterion",
    "WinnerMarginCriterion",
    "CombinedCriterion",
    "get_criterion",
    # Baselines
    "MagnitudeStructuredPruner",
    "RandomStructuredPruner",
    "ActivationSparsityPruner",
    "NetworkSlimmingPruner",
    "TaylorPruner",
    # Experiment infrastructure
    "ExperimentConfig",
    "ResultStats",
    "ExperimentRunner",
    "run_calibration_ablation",
    # Fine-tuning
    "FinetuneConfig",
    "FinetuneResult",
    "finetune_pruned_model",
    "quick_finetune",
    "knowledge_distillation_finetune",
    # Benchmarking
    "PruningBenchmark",
    "TimingResult",
    "ModelStats",
    "BenchmarkResult",
    "ComparisonResult",
    # Visualization
    "plot_winner_frequency_histogram",
    "plot_sparsity_accuracy_curve",
    "plot_finetuning_curve",
    "plot_layer_importance",
    "plot_calibration_ablation",
    "plot_compression_comparison",
    "plot_method_comparison_heatmap",
    "setup_publication_style",
]
