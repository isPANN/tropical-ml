# Tropical Pruning

**Winner-based Tropical Structured Pruning for Neural Networks**

This package implements tropical-geometry-aware pruning methods that leverage the argmax structure of tropical GEMM operations to identify and prune neurons that never "win" in the max-plus competition.

## Core Idea

In tropical GEMM: `C_ij = max_k(A_ik + B_kj)`

The `argmax` indices reveal which neurons actually contribute to the output. Neurons with zero or low "winner count" are geometrically useless and can be safely pruned.

## Installation

```bash
# First, install tropical-gemm from the sibling directory
cd tropical-gemm/crates/tropical-gemm-python
pip install -e ".[torch]"

# Then install tropical-pruning
cd ../../../tropical-pruning
pip install -e .

# With development dependencies
pip install -e ".[dev]"

# With experiment tracking (wandb, matplotlib, etc.)
pip install -e ".[experiment]"

# Everything
pip install -e ".[all]"
```

## Performance

Uses [tropical-gemm](https://pypi.org/project/tropical-gemm/) for high-performance tropical operations:
- **CPU**: Rust SIMD-accelerated (AVX2/AVX-512/NEON)

## Quick Start

```python
from tropical_pruning import WinnerCounter, TropicalPruner

# 1. Collect winner statistics on calibration data
counter = WinnerCounter(model)
for batch, _ in calibration_loader:
    counter.forward(batch)
stats = counter.get_statistics()

# 2. Analyze which neurons are important
pruner = TropicalPruner(model, stats)
analysis = pruner.analyze_winners()
print(analysis)  # Shows never_win, rarely_win, etc.

# 3. Prune to 50% sparsity
pruned_model = pruner.prune(sparsity=0.5, criterion="winner_frequency")

# 4. Check compression
print(pruner.get_compression_stats())
```

## Pruning Criteria

| Criterion | Description | Use Case |
|-----------|-------------|----------|
| `winner_frequency` | How often a neuron achieves argmax | Primary criterion (recommended) |
| `winner_margin` | Average gap to 2nd place when winning | Confidence of importance |
| `combined` | Weighted combination of multiple criteria | Balanced pruning |
| `magnitude_l1` / `magnitude_l2` | Traditional weight magnitude | Baseline comparison |

## Metrics Tracked

| Metric | Definition | Use |
|--------|------------|-----|
| Winner Count | Times neuron achieves argmax | Primary pruning criterion |
| Winner Frequency | Count / total samples | Normalized importance |
| Average Margin | Mean gap to 2nd place | Confidence of importance |

## Example: MNIST Pruning

```bash
# Run the pilot experiment
python examples/mnist_tropical_pruning.py \
    --epochs 10 \
    --hidden-sizes 256 128 \
    --device cuda

# Skip training if model already exists
python examples/mnist_tropical_pruning.py \
    --skip-training \
    --model-path mnist_mlp.pt
```

## Project Structure

```
tropical-pruning/
├── tropical_pruning/
│   ├── __init__.py      # Main exports
│   ├── counter.py       # WinnerCounter, WinnerStatistics
│   ├── pruner.py        # TropicalPruner
│   ├── criteria.py      # Pruning criteria (frequency, margin, etc.)
│   └── layers.py        # TropicalLinear layer
├── examples/
│   └── mnist_tropical_pruning.py
├── tests/
│   └── test_counter.py
└── pyproject.toml
```

## Research Background

This implementation is based on the observation that ReLU networks output piecewise-linear functions, and tropical networks naturally represent such functions. The winner-based pruning criterion is novel: instead of using weight magnitude (L1/L2), we prune neurons based on their geometric contribution to the tropical computation.

**Key Papers:**
- Zhang et al. (2018) - *Tropical Geometry of Deep Neural Networks* (ICML)
- TropNNC (2024) - *Structured Neural Network Compression Using Tropical Geometry*

## Comparison with Baselines

Winner-based tropical pruning provides a geometry-based pruning criterion that:
- **vs L1/L2 magnitude**: Uses actual contribution instead of weight size
- **vs Taylor expansion**: No gradient computation needed
- **vs Activation sparsity**: Considers max-plus structure, not just zeros
- **vs TropNNC (2024)**: Supports end-to-end fine-tuning (via our tropical backward)

## License

MIT
