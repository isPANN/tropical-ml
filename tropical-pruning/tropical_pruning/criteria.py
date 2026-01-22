"""
Pruning Criteria: Different methods to score neuron importance.

This module provides various criteria for determining which neurons to prune
based on winner statistics collected from tropical forward passes.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import torch

from tropical_pruning.counter import WinnerStatistics


class PruningCriterion(ABC):
    """Abstract base class for pruning criteria."""

    @abstractmethod
    def compute_importance(
        self,
        stats: WinnerStatistics,
    ) -> torch.Tensor:
        """
        Compute importance scores for each neuron.

        Args:
            stats: WinnerStatistics for the layer.

        Returns:
            Tensor of importance scores, shape (num_neurons,).
            Higher scores indicate more important neurons.
        """
        pass

    def get_pruning_mask(
        self,
        stats: WinnerStatistics,
        sparsity: float,
    ) -> torch.Tensor:
        """
        Get a mask indicating which neurons to keep.

        Args:
            stats: WinnerStatistics for the layer.
            sparsity: Target sparsity (fraction of neurons to prune, 0-1).

        Returns:
            Boolean mask, shape (num_neurons,). True = keep, False = prune.
        """
        importance = self.compute_importance(stats)
        num_neurons = importance.numel()
        num_to_prune = int(num_neurons * sparsity)

        if num_to_prune == 0:
            return torch.ones(num_neurons, dtype=torch.bool, device=importance.device)

        # Get threshold: neurons below this importance are pruned
        threshold = torch.kthvalue(importance, num_to_prune).values
        mask = importance > threshold

        return mask


class WinnerFrequencyCriterion(PruningCriterion):
    """
    Prune neurons based on their winner frequency.

    Winner frequency = (times neuron achieved argmax) / (total positions)

    Neurons that rarely or never "win" the tropical max competition
    are considered geometrically useless.

    This is the primary criterion recommended in the research directions.
    """

    def __init__(self, min_frequency: float = 0.0):
        """
        Args:
            min_frequency: Minimum frequency threshold. Neurons below this
                          are always pruned regardless of sparsity target.
        """
        self.min_frequency = min_frequency

    def compute_importance(self, stats: WinnerStatistics) -> torch.Tensor:
        """Importance = winner frequency."""
        return stats.winner_frequency


class WinnerMarginCriterion(PruningCriterion):
    """
    Prune neurons based on their average winning margin.

    Winning margin = gap between max and 2nd-max when this neuron wins.

    Neurons that win by a large margin are more confidently important,
    while neurons that barely win might be borderline.

    This criterion requires track_margin=True in WinnerCounter.
    """

    def __init__(self, frequency_weight: float = 0.5):
        """
        Args:
            frequency_weight: Weight for combining frequency with margin.
                             Higher values give more weight to frequency.
        """
        self.frequency_weight = frequency_weight

    def compute_importance(self, stats: WinnerStatistics) -> torch.Tensor:
        """Importance = weighted combination of frequency and margin."""
        frequency = stats.winner_frequency

        avg_margin = stats.average_margin
        if avg_margin is None:
            return frequency

        # Normalize margin to [0, 1] range for combination
        margin_normalized = avg_margin / (avg_margin.max() + 1e-8)

        # Combine: neurons that win often AND win decisively are most important
        importance = (
            self.frequency_weight * frequency +
            (1 - self.frequency_weight) * margin_normalized
        )

        return importance


class CombinedCriterion(PruningCriterion):
    """
    Combine multiple criteria with configurable weights.

    Example:
        >>> criterion = CombinedCriterion([
        ...     (WinnerFrequencyCriterion(), 0.7),
        ...     (WinnerMarginCriterion(), 0.3),
        ... ])
    """

    def __init__(self, criteria: List[Tuple[PruningCriterion, float]]):
        """
        Args:
            criteria: List of (criterion, weight) tuples.
        """
        self.criteria = criteria
        total_weight = sum(w for _, w in criteria)
        self.criteria = [(c, w / total_weight) for c, w in criteria]

    def compute_importance(self, stats: WinnerStatistics) -> torch.Tensor:
        """Weighted average of all criteria."""
        importance = None
        for criterion, weight in self.criteria:
            score = criterion.compute_importance(stats)
            if importance is None:
                importance = weight * score
            else:
                importance = importance + weight * score
        return importance


class MagnitudeCriterion(PruningCriterion):
    """
    Traditional L1/L2 magnitude-based pruning for comparison.

    This serves as a baseline to compare against tropical winner-based pruning.
    """

    def __init__(self, weight: torch.Tensor, norm: str = "l1"):
        """
        Args:
            weight: Layer weight tensor, shape (out_features, in_features).
            norm: "l1" or "l2" norm.
        """
        self.weight = weight
        self.norm = norm

    def compute_importance(self, stats: WinnerStatistics) -> torch.Tensor:
        """Importance = norm of weights along output dimension."""
        if self.norm == "l1":
            # Sum of absolute values along output dimension
            importance = self.weight.abs().sum(dim=0)
        elif self.norm == "l2":
            # L2 norm along output dimension
            importance = self.weight.pow(2).sum(dim=0).sqrt()
        else:
            raise ValueError(f"Unknown norm: {self.norm}")

        return importance


class ActivationSparsityCriterion(PruningCriterion):
    """
    Prune neurons based on activation sparsity.

    Neurons that produce zero or near-zero activations most of the time
    can be pruned. This is another baseline for comparison.
    """

    def __init__(self, activations: torch.Tensor, threshold: float = 1e-6):
        """
        Args:
            activations: Collected activations, shape (num_samples, num_neurons).
            threshold: Threshold below which activation is considered zero.
        """
        self.activations = activations
        self.threshold = threshold

    def compute_importance(self, stats: WinnerStatistics) -> torch.Tensor:
        """Importance = fraction of non-zero activations."""
        non_zero = (self.activations.abs() > self.threshold).float()
        importance = non_zero.mean(dim=0)
        return importance


def get_criterion(name: str, **kwargs) -> PruningCriterion:
    """
    Factory function to create pruning criteria by name.

    Args:
        name: Criterion name ("winner_frequency", "winner_margin", "combined",
              "magnitude_l1", "magnitude_l2", "activation_sparsity").
        **kwargs: Additional arguments for the criterion.

    Returns:
        PruningCriterion instance.
    """
    criteria_map = {
        "winner_frequency": WinnerFrequencyCriterion,
        "winner_margin": WinnerMarginCriterion,
        "combined": CombinedCriterion,
        "magnitude_l1": lambda **kw: MagnitudeCriterion(norm="l1", **kw),
        "magnitude_l2": lambda **kw: MagnitudeCriterion(norm="l2", **kw),
        "activation_sparsity": ActivationSparsityCriterion,
    }

    if name not in criteria_map:
        raise ValueError(f"Unknown criterion: {name}. Available: {list(criteria_map.keys())}")

    return criteria_map[name](**kwargs)
