"""
LLM FFN Tropical Pruning: Winner-based pruning for LLM Feed-Forward Networks.

This subpackage extends tropical pruning to LLM architectures (LLaMA, Mistral, etc.)
that use SwiGLU-based FFN layers.

The key insight: The `down_proj` layer in SwiGLU is a standard linear layer where
tropical winner counting applies directly:

    FFN(x) = down_proj(SiLU(gate_proj(x)) * up_proj(x))
                      └── intermediate activations ──┘

    down_proj: (intermediate, hidden) → tropical applies here
    output[h] = max_j(W_down[h,j] + intermediate[j])
    argmax_j → which intermediate neuron "wins"

Pruning is coupled across gate_proj, up_proj, and down_proj to maintain consistency:
- gate_proj columns (same intermediate neurons)
- up_proj columns (same intermediate neurons)
- down_proj rows (same intermediate neurons)

Example:
    >>> from tropical_pruning.llm import (
    ...     load_model_and_tokenizer,
    ...     FFNWinnerCounter,
    ...     FFNTropicalPruner,
    ...     CalibrationDataset,
    ...     LLMEvaluator,
    ... )
    >>>
    >>> # Load model
    >>> model, tokenizer = load_model_and_tokenizer("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    >>>
    >>> # Get calibration data
    >>> dataloader = CalibrationDataset.from_wikitext2(tokenizer, num_samples=128)
    >>>
    >>> # Collect FFN winner statistics
    >>> counter = FFNWinnerCounter(model)
    >>> stats = counter.collect(dataloader)
    >>>
    >>> # Prune FFN layers to 30% sparsity
    >>> pruner = FFNTropicalPruner(model, stats)
    >>> pruned_model = pruner.prune(sparsity=0.3)
    >>>
    >>> # Evaluate
    >>> evaluator = LLMEvaluator(pruned_model, tokenizer)
    >>> ppl = evaluator.compute_perplexity()
"""

from tropical_pruning.llm.loader import (
    load_model_and_tokenizer,
    get_ffn_layer_names,
    LLMArchitecture,
    detect_architecture,
)

from tropical_pruning.llm.ffn_counter import (
    FFNStatistics,
    FFNWinnerCounter,
)

from tropical_pruning.llm.ffn_pruner import (
    FFNTropicalPruner,
)

from tropical_pruning.llm.calibration import (
    CalibrationDataset,
)

from tropical_pruning.llm.evaluation import (
    LLMEvaluator,
)

from tropical_pruning.llm.baselines import (
    FFNBaselinePruner,
    MagnitudePruner,
    ActivationPruner,
    WandaStylePruner,
    FLAPStylePruner,
    get_baseline_pruner,
)

__all__ = [
    # Model loading
    "load_model_and_tokenizer",
    "get_ffn_layer_names",
    "LLMArchitecture",
    "detect_architecture",
    # Winner statistics
    "FFNStatistics",
    "FFNWinnerCounter",
    # Pruning
    "FFNTropicalPruner",
    # Calibration
    "CalibrationDataset",
    # Evaluation
    "LLMEvaluator",
    # Baselines
    "FFNBaselinePruner",
    "MagnitudePruner",
    "ActivationPruner",
    "WandaStylePruner",
    "FLAPStylePruner",
    "get_baseline_pruner",
]
