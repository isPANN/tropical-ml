"""
Performance benchmarks for Tropical Attention vs Standard Attention.

Compares:
- Forward pass latency
- Backward pass latency
- Memory usage
- Scaling with sequence length
"""

import argparse
import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from tropical_attention import TropicalMultiheadAttention


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    config: str
    tropical_forward_ms: float
    standard_forward_ms: float
    tropical_backward_ms: float
    standard_backward_ms: float
    tropical_memory_mb: float
    standard_memory_mb: float

    @property
    def forward_ratio(self) -> float:
        return self.tropical_forward_ms / self.standard_forward_ms

    @property
    def backward_ratio(self) -> float:
        return self.tropical_backward_ms / self.standard_backward_ms

    def __str__(self) -> str:
        return (
            f"{self.config}\n"
            f"  Forward:  Tropical={self.tropical_forward_ms:7.2f}ms, "
            f"Standard={self.standard_forward_ms:7.2f}ms, "
            f"Ratio={self.forward_ratio:.2f}x\n"
            f"  Backward: Tropical={self.tropical_backward_ms:7.2f}ms, "
            f"Standard={self.standard_backward_ms:7.2f}ms, "
            f"Ratio={self.backward_ratio:.2f}x\n"
            f"  Memory:   Tropical={self.tropical_memory_mb:7.1f}MB, "
            f"Standard={self.standard_memory_mb:7.1f}MB"
        )


def get_memory_mb(device: str) -> float:
    """Get current GPU memory usage in MB."""
    if device == "cuda":
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


def benchmark_forward(
    model: nn.Module,
    x: torch.Tensor,
    num_warmup: int = 10,
    num_runs: int = 100,
) -> float:
    """Benchmark forward pass, return average time in ms."""
    device = x.device

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(x, x, x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(x, x, x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    return (elapsed / num_runs) * 1000  # Convert to ms


def benchmark_backward(
    model: nn.Module,
    x: torch.Tensor,
    num_warmup: int = 5,
    num_runs: int = 50,
) -> float:
    """Benchmark backward pass, return average time in ms."""
    device = x.device

    # Warmup
    for _ in range(num_warmup):
        x_clone = x.clone().requires_grad_(True)
        out, _ = model(x_clone, x_clone, x_clone)
        loss = out.sum()
        loss.backward()

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_runs):
        x_clone = x.clone().requires_grad_(True)
        out, _ = model(x_clone, x_clone, x_clone)
        loss = out.sum()
        loss.backward()

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    return (elapsed / num_runs) * 1000  # Convert to ms


def benchmark_memory(
    model: nn.Module,
    x: torch.Tensor,
) -> float:
    """Measure peak memory usage during forward+backward pass."""
    device = x.device

    if device.type != "cuda":
        return 0.0

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    x_clone = x.clone().requires_grad_(True)
    out, _ = model(x_clone, x_clone, x_clone)
    loss = out.sum()
    loss.backward()

    torch.cuda.synchronize()
    peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024

    return peak_memory


def run_benchmark(
    batch_size: int,
    seq_len: int,
    d_model: int,
    num_heads: int,
    device: str = "cpu",
    num_warmup: int = 10,
    num_runs: int = 100,
) -> BenchmarkResult:
    """Run complete benchmark comparing tropical vs standard attention."""

    config = f"B={batch_size}, S={seq_len}, D={d_model}, H={num_heads}, device={device}"

    # Create models
    tropical_attn = TropicalMultiheadAttention(
        d_model=d_model,
        num_heads=num_heads,
        batch_first=True,
    ).to(device)

    standard_attn = nn.MultiheadAttention(
        embed_dim=d_model,
        num_heads=num_heads,
        batch_first=True,
    ).to(device)

    # Create input
    x = torch.randn(batch_size, seq_len, d_model, device=device)

    # Benchmark forward
    tropical_forward = benchmark_forward(
        tropical_attn, x, num_warmup=num_warmup, num_runs=num_runs
    )
    standard_forward = benchmark_forward(
        standard_attn, x, num_warmup=num_warmup, num_runs=num_runs
    )

    # Benchmark backward
    tropical_backward = benchmark_backward(
        tropical_attn, x, num_warmup=num_warmup // 2, num_runs=num_runs // 2
    )
    standard_backward = benchmark_backward(
        standard_attn, x, num_warmup=num_warmup // 2, num_runs=num_runs // 2
    )

    # Benchmark memory
    tropical_memory = benchmark_memory(tropical_attn, x)
    standard_memory = benchmark_memory(standard_attn, x)

    return BenchmarkResult(
        config=config,
        tropical_forward_ms=tropical_forward,
        standard_forward_ms=standard_forward,
        tropical_backward_ms=tropical_backward,
        standard_backward_ms=standard_backward,
        tropical_memory_mb=tropical_memory,
        standard_memory_mb=standard_memory,
    )


def run_scaling_benchmark(
    d_model: int = 256,
    num_heads: int = 4,
    batch_size: int = 32,
    device: str = "cpu",
    seq_lengths: Optional[list] = None,
) -> list[BenchmarkResult]:
    """Benchmark scaling with sequence length."""

    if seq_lengths is None:
        seq_lengths = [64, 128, 256, 512, 1024]

    results = []
    for seq_len in seq_lengths:
        print(f"  Benchmarking seq_len={seq_len}...")
        try:
            result = run_benchmark(
                batch_size=batch_size,
                seq_len=seq_len,
                d_model=d_model,
                num_heads=num_heads,
                device=device,
                num_warmup=5,
                num_runs=50,
            )
            results.append(result)
        except RuntimeError as e:
            print(f"    Skipping seq_len={seq_len}: {e}")
            break

    return results


def print_scaling_table(results: list[BenchmarkResult]) -> None:
    """Print scaling results as a table."""
    print("\n" + "=" * 80)
    print("SCALING WITH SEQUENCE LENGTH")
    print("=" * 80)
    print(
        f"{'Seq Len':>8} | {'Tropical Fwd':>12} | {'Standard Fwd':>12} | "
        f"{'Ratio':>6} | {'Tropical Bwd':>12} | {'Standard Bwd':>12} | {'Ratio':>6}"
    )
    print("-" * 80)

    for r in results:
        # Extract seq_len from config
        seq_len = int(r.config.split("S=")[1].split(",")[0])
        print(
            f"{seq_len:>8} | {r.tropical_forward_ms:>10.2f}ms | "
            f"{r.standard_forward_ms:>10.2f}ms | {r.forward_ratio:>5.2f}x | "
            f"{r.tropical_backward_ms:>10.2f}ms | {r.standard_backward_ms:>10.2f}ms | "
            f"{r.backward_ratio:>5.2f}x"
        )


def main():
    parser = argparse.ArgumentParser(description="Benchmark Tropical vs Standard Attention")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-warmup", type=int, default=10)
    parser.add_argument("--num-runs", type=int, default=100)
    parser.add_argument("--scaling", action="store_true", help="Run scaling benchmark")
    args = parser.parse_args()

    # Check CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    print("=" * 80)
    print("TROPICAL ATTENTION PERFORMANCE BENCHMARK")
    print("=" * 80)
    print(f"Device: {args.device}")
    if args.device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    if args.scaling:
        # Run scaling benchmark
        print("Running scaling benchmark...")
        results = run_scaling_benchmark(
            d_model=args.d_model,
            num_heads=args.num_heads,
            batch_size=args.batch_size,
            device=args.device,
        )
        print_scaling_table(results)
    else:
        # Run single benchmark
        print("Running single configuration benchmark...")
        result = run_benchmark(
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            d_model=args.d_model,
            num_heads=args.num_heads,
            device=args.device,
            num_warmup=args.num_warmup,
            num_runs=args.num_runs,
        )
        print()
        print(result)

    # Run standard configurations
    print("\n" + "=" * 80)
    print("STANDARD CONFIGURATIONS")
    print("=" * 80)

    configs = [
        (32, 128, 256, 4, "Small"),
        (32, 256, 512, 8, "Medium"),
        (16, 512, 768, 12, "BERT-base"),
    ]

    for batch, seq, d_model, heads, name in configs:
        print(f"\n{name} (B={batch}, S={seq}, D={d_model}, H={heads}):")
        try:
            result = run_benchmark(
                batch_size=batch,
                seq_len=seq,
                d_model=d_model,
                num_heads=heads,
                device=args.device,
                num_warmup=5,
                num_runs=30,
            )
            print(f"  Forward:  Tropical={result.tropical_forward_ms:.2f}ms, "
                  f"Standard={result.standard_forward_ms:.2f}ms, "
                  f"Ratio={result.forward_ratio:.2f}x")
            print(f"  Backward: Tropical={result.tropical_backward_ms:.2f}ms, "
                  f"Standard={result.standard_backward_ms:.2f}ms, "
                  f"Ratio={result.backward_ratio:.2f}x")
            if result.tropical_memory_mb > 0:
                print(f"  Memory:   Tropical={result.tropical_memory_mb:.1f}MB, "
                      f"Standard={result.standard_memory_mb:.1f}MB")
        except RuntimeError as e:
            print(f"  Skipped: {e}")


if __name__ == "__main__":
    main()
