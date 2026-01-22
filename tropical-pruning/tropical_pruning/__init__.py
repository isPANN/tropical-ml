"""
Tropical Pruning: Winner-based Tropical Structured Pruning for Neural Networks.

This package implements tropical-geometry-aware pruning methods that leverage
the argmax structure of tropical GEMM operations to identify and prune neurons
that never "win" in the max-plus competition.

Core Components:
    - WinnerCounter: Tracks winner statistics during tropical forward passes
    - TropicalPruner: Main pruning interface with multiple criteria
    - TropicalLinear: Drop-in replacement for nn.Linear with tropical ops

Example:
    >>> from tropical_pruning import WinnerCounter, TropicalPruner
    >>>
    >>> # Collect winner statistics
    >>> counter = WinnerCounter(model)
    >>> for batch in calibration_loader:
    ...     counter.forward(batch)
    >>> stats = counter.get_statistics()
    >>>
    >>> # Prune based on winner frequency
    >>> pruner = TropicalPruner(model, stats)
    >>> pruned_model = pruner.prune(sparsity=0.5, criterion="winner_frequency")
"""

__version__ = "0.1.0"

from tropical_pruning.counter import WinnerCounter, WinnerStatistics
from tropical_pruning.pruner import TropicalPruner
from tropical_pruning.layers import TropicalLinear
from tropical_pruning.criteria import (
    PruningCriterion,
    WinnerFrequencyCriterion,
    WinnerMarginCriterion,
    CombinedCriterion,
)
from tropical_pruning.baselines import (
    MagnitudeStructuredPruner,
    RandomStructuredPruner,
    ActivationSparsityPruner,
)

__all__ = [
    # Core classes
    "WinnerCounter",
    "WinnerStatistics",
    "TropicalPruner",
    "TropicalLinear",
    # Pruning criteria
    "PruningCriterion",
    "WinnerFrequencyCriterion",
    "WinnerMarginCriterion",
    "CombinedCriterion",
    # Baselines
    "MagnitudeStructuredPruner",
    "RandomStructuredPruner",
    "ActivationSparsityPruner",
]
