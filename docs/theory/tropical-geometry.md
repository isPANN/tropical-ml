# Tropical Geometry

This page provides background on the mathematical foundations of tropical geometry.

## The Tropical Semiring

Tropical geometry is built on the **tropical semiring** (also called max-plus algebra):

### Max-Plus Semiring

$$(\mathbb{R} \cup \{-\infty\}, \oplus, \otimes)$$

where:

- $a \oplus b = \max(a, b)$ (tropical addition)
- $a \otimes b = a + b$ (tropical multiplication)
- Additive identity: $-\infty$ (since $\max(a, -\infty) = a$)
- Multiplicative identity: $0$ (since $a + 0 = a$)

### Min-Plus Semiring

Alternatively, we can use the min-plus semiring:

$$(\mathbb{R} \cup \{+\infty\}, \oplus, \otimes)$$

where:

- $a \oplus b = \min(a, b)$
- $a \otimes b = a + b$

## Tropical Matrix Multiplication

For matrices $A \in \mathbb{R}^{m \times k}$ and $B \in \mathbb{R}^{k \times n}$:

### Max-Plus

$$(A \otimes B)[i,j] = \bigoplus_{l=1}^{k} A[i,l] \otimes B[l,j] = \max_{l=1}^{k}(A[i,l] + B[l,j])$$

### Connection to Graph Algorithms

- **Shortest paths**: Min-plus matmul computes shortest paths in weighted graphs
- **Longest paths**: Max-plus matmul computes longest paths
- **All-pairs shortest paths**: $(A^{\otimes n})$ gives shortest paths of length $n$

## Tropical Projective Space

In tropical geometry, vectors that differ by a constant are equivalent:

$$x \sim y \iff x = y + c \cdot \mathbf{1}$$

for some scalar $c$. This defines **tropical projective space** $\mathbb{TP}^{n-1}$.

### Tropical Simplex

A representative of each equivalence class is chosen by normalizing:

$$\tilde{x}_i = x_i - \max_j(x_j)$$

This places vectors on the **tropical simplex** where $\max_i(\tilde{x}_i) = 0$.

## Tropical Polynomials

A tropical polynomial is:

$$p(x) = \bigoplus_{i} c_i \otimes x^{\otimes a_i} = \max_i(c_i + a_i \cdot x)$$

This is a **piecewise-linear** function! The "roots" are where the maximum is achieved by multiple terms.

## Why Tropical for ML?

### 1. Piecewise-Linear Networks

ReLU networks compute piecewise-linear functions. Tropical geometry provides tools to analyze these functions algebraically.

### 2. Computational Efficiency

Max and addition are simpler operations than multiply-accumulate, potentially enabling:

- Faster inference
- Lower power consumption
- Simpler hardware implementations

### 3. Interpretability

Tropical operations have clear geometric interpretations:

- Max-plus matmul = finding longest paths
- Tropical rank = structural complexity
- Hilbert distance = angular separation

### 4. Sparse Gradients

Tropical operations naturally produce sparse gradients (only argmax contributes), which can:

- Reduce memory during training
- Provide implicit feature selection
- Improve robustness

## Further Reading

- Maclagan & Sturmfels, *Introduction to Tropical Geometry*
- Joswig, *Essentials of Tropical Combinatorics*
- Pachter & Sturmfels, *Algebraic Statistics for Computational Biology* (Chapter 8)
