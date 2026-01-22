"""
Tropical Pruning: Winner-based Tropical Structured Pruning for Neural Networks.

This package implements tropical-geometry-aware pruning methods that leverage
the argmax structure of tropical GEMM operations to identify and prune neurons
that never "win" in the max-plus competition.

Mathematical Foundation:
    In the tropical semiring (max-plus algebra):
    - Addition: a ⊕ b = max(a, b)
    - Multiplication: a ⊗ b = a + b

    For a linear layer Y = WX, the tropical version is:
        Y_i^trop = max_j(W_ij + X_j)

    The "winner" for output i is: w_i = argmax_j(W_ij + X_j)

    Theorem: If input neuron j is never a winner (∀i: j ≠ argmax_k(W_ik + X_k)),
    then removing j does not change the tropical output.

Core Components:
    - WinnerCounter: Tracks winner statistics for Linear layers
    - TropicalPruner: Pruning interface for Linear layers
    - TropicalLinear: Drop-in replacement for nn.Linear with tropical ops

Example:
    >>> from tropical_pruning import WinnerCounter, TropicalPruner
    >>>
    >>> counter = WinnerCounter(model)
    >>> stats = counter.collect(calibration_loader)
    >>> pruner = TropicalPruner(model, stats)
    >>> pruned_model = pruner.prune(sparsity=0.5)
"""

__version__ = "0.3.0"

# Core: Linear layer support
from tropical_pruning.counter import WinnerCounter, WinnerStatistics
from tropical_pruning.pruner import TropicalPruner
from tropical_pruning.layers import TropicalLinear

# Pruning criteria
from tropical_pruning.criteria import (
    PruningCriterion,
    WinnerFrequencyCriterion,
    WinnerMarginCriterion,
    CombinedCriterion,
    get_criterion,
)

# Baseline methods for comparison
from tropical_pruning.baselines import (
    MagnitudeStructuredPruner,
    RandomStructuredPruner,
    ActivationSparsityPruner,
)

# Fine-tuning
from tropical_pruning.finetuning import (
    FinetuneConfig,
    FinetuneResult,
    finetune_pruned_model,
    quick_finetune,
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
    "get_criterion",
    # Baselines
    "MagnitudeStructuredPruner",
    "RandomStructuredPruner",
    "ActivationSparsityPruner",
    # Fine-tuning
    "FinetuneConfig",
    "FinetuneResult",
    "finetune_pruned_model",
    "quick_finetune",
]
