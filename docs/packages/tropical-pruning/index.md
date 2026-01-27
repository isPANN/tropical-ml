# Tropical Pruning

**Neural Network Pruning via Tropical Geometry**

!!! warning "Experimental"
    This package is experimental and under active development.

## Overview

Tropical Pruning uses tropical geometry to analyze and prune neural networks. The key insight is that tropical rank (the rank of a matrix under max-plus algebra) can reveal structural redundancy in weight matrices.

## Installation

```bash
pip install tropical-pruning
```

## Concepts

### Tropical Rank

The tropical rank of a matrix $A$ is the smallest $r$ such that $A$ can be written as a tropical product of matrices with inner dimension $r$:

$$A = B \otimes C$$

where $B$ is $m \times r$ and $C$ is $r \times n$, and $\otimes$ denotes tropical matrix multiplication.

### Pruning via Tropical Decomposition

If a weight matrix has low tropical rank, it can be factorized into smaller matrices, reducing parameters while preserving the max-plus structure.

## Applications

1. **Structured pruning**: Remove entire neurons/channels based on tropical rank
2. **Low-rank factorization**: Decompose weight matrices tropically
3. **Architecture search**: Identify minimal-rank architectures

## API Reference

*Documentation coming soon.*

## Related Work

- [Tropical GEMM](../tropical-gemm/index.md) provides the matrix operations
- Research on tropical linear algebra and matrix factorization
