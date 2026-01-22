#!/usr/bin/env python3
"""
LLM FFN Pruning: Multi-Sparsity Sweep

Compare pruning methods across multiple sparsity levels (20%, 30%, 50%, 70%).
Generates a comprehensive comparison table and identifies the best method at each level.

Results are automatically saved to results/ directory with timestamp.

Usage:
    python llm_sparsity_sweep.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
"""

import os
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import argparse
import gc
import json
import csv
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import torch


def get_system_info() -> Dict[str, Any]:
    """Collect system and environment information."""
    info = {
        "timestamp": datetime.now().isoformat(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
        },
        "torch": {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
    }

    # GPU info
    if torch.cuda.is_available():
        info["gpu"] = {
            "device_count": torch.cuda.device_count(),
            "devices": [],
        }
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info["gpu"]["devices"].append({
                "name": props.name,
                "total_memory_gb": round(props.total_memory / 1024**3, 2),
                "compute_capability": f"{props.major}.{props.minor}",
            })

    # Try to get git commit
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        info["git_commit"] = git_hash
    except Exception:
        info["git_commit"] = None

    return info


def save_results(
    results: Dict,
    args: argparse.Namespace,
    system_info: Dict,
    baseline_ppl: float,
    original_params: int,
    never_win_neurons: int,
    total_neurons: int,
    output_dir: str = "results",
) -> str:
    """Save results to JSON and CSV files."""
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = args.model.split("/")[-1].replace("-", "_")
    base_name = f"ffn_pruning_{model_short}_{timestamp}"

    # Prepare full data
    full_data = {
        "experiment": {
            "name": "LLM FFN Pruning Comparison",
            "script": "llm_sparsity_sweep.py",
        },
        "config": {
            "model": args.model,
            "sparsities": [float(s) for s in args.sparsities.split(",")],
            "num_samples": args.num_samples,
            "seq_length": args.seq_length,
            "eval_samples": args.eval_samples,
        },
        "system": system_info,
        "baseline": {
            "perplexity": baseline_ppl,
            "parameters": original_params,
        },
        "analysis": {
            "never_win_neurons": never_win_neurons,
            "total_neurons": total_neurons,
            "never_win_ratio": never_win_neurons / total_neurons,
        },
        "results": results,
    }

    # Save JSON
    json_path = output_path / f"{base_name}.json"
    with open(json_path, "w") as f:
        json.dump(full_data, f, indent=2)

    # Save CSV (flattened for easy analysis)
    csv_path = output_path / f"{base_name}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)

        # Header
        writer.writerow([
            "model", "method", "sparsity", "params", "compression",
            "ppl", "delta_ppl", "delta_ppl_pct", "baseline_ppl",
            "num_samples", "seq_length", "timestamp"
        ])

        # Data rows
        for method, sparsity_results in results.items():
            for sparsity, data in sparsity_results.items():
                writer.writerow([
                    args.model,
                    method,
                    sparsity,
                    data["params"],
                    round(data["compression"], 4),
                    round(data["ppl"], 4),
                    round(data["delta_ppl"], 4),
                    round(data["delta_ppl_pct"], 4),
                    round(baseline_ppl, 4),
                    args.num_samples,
                    args.seq_length,
                    system_info["timestamp"],
                ])

    return str(json_path), str(csv_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-sparsity pruning comparison")
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--sparsities", type=str, default="0.2,0.3,0.5,0.7",
                       help="Comma-separated sparsity levels")
    parser.add_argument("--num-samples", type=int, default=64, help="Calibration samples")
    parser.add_argument("--seq-length", type=int, default=512)
    parser.add_argument("--eval-samples", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default="results",
                       help="Directory to save results (default: results/)")
    parser.add_argument("--no-save", action="store_true",
                       help="Don't save results to files")
    return parser.parse_args()


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


def compute_perplexity(model, tokenizer, num_samples=50, seq_length=512) -> float:
    from datasets import load_dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])
    encodings = tokenizer(text, return_tensors="pt", truncation=False, add_special_tokens=False)

    if hasattr(model, 'device'):
        device = model.device
    else:
        device = next(model.parameters()).device
    input_ids = encodings["input_ids"].to(device)

    model.eval()
    nlls = []
    stride = seq_length // 2
    seq_len = input_ids.size(1)

    with torch.no_grad():
        for i in range(0, min(num_samples * stride, seq_len - seq_length), stride):
            chunk = input_ids[:, i:i + seq_length]
            outputs = model(chunk)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = chunk[..., 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="mean"
            )
            nlls.append(loss)
            if len(nlls) >= num_samples:
                break

    return torch.exp(torch.stack(nlls).mean()).item()


def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    args = parse_args()
    sparsities = [float(s.strip()) for s in args.sparsities.split(",")]

    # Collect system info at start
    system_info = get_system_info()

    print("=" * 90)
    print("LLM FFN Pruning: Multi-Sparsity Comparison")
    print("=" * 90)
    print(f"Timestamp: {system_info['timestamp']}")
    print(f"Model: {args.model}")
    print(f"Sparsity levels: {[f'{s*100:.0f}%' for s in sparsities]}")
    print(f"Calibration samples: {args.num_samples}")
    if system_info["torch"]["cuda_available"]:
        gpu_info = system_info["gpu"]["devices"][0]
        print(f"GPU: {gpu_info['name']} ({gpu_info['total_memory_gb']}GB)")
    print()

    from tropical_pruning.llm import (
        load_model_and_tokenizer,
        FFNWinnerCounter,
        FFNTropicalPruner,
        CalibrationDataset,
    )
    from tropical_pruning.llm.baselines import (
        MagnitudePruner,
        WandaStylePruner,
        FLAPStylePruner,
    )

    # Load model
    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer(
        args.model,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    original_params = count_parameters(model)
    print(f"  Original parameters: {original_params:,}")
    print()

    # Baseline perplexity
    print("Computing baseline perplexity...")
    baseline_ppl = compute_perplexity(model, tokenizer, args.eval_samples, args.seq_length)
    print(f"  Baseline PPL: {baseline_ppl:.2f}")
    print()

    # Create calibration data
    print("Creating calibration dataset...")
    calibration_loader = CalibrationDataset.from_wikitext2(
        tokenizer, num_samples=args.num_samples, seq_length=args.seq_length
    )
    print()

    # Collect tropical statistics (once, reuse for all sparsities)
    print("Collecting tropical winner statistics...")
    counter = FFNWinnerCounter(model, track_margin=False)
    tropical_stats = counter.collect(calibration_loader, show_progress=True)
    counter.remove_hooks()

    # Analyze never-win neurons
    analysis = counter.analyze()
    total_never_win = sum(a["never_win"] for a in analysis.values())
    total_neurons = sum(a["intermediate_size"] for a in analysis.values())
    print(f"  Never-win neurons: {total_never_win:,} / {total_neurons:,} ({total_never_win/total_neurons*100:.1f}%)")
    print()

    # Create baseline pruners (once, reuse for all sparsities)
    print("Initializing baseline pruners...")
    magnitude_pruner = MagnitudePruner(model, norm="l1")
    wanda_pruner = WandaStylePruner(model, calibration_loader)
    flap_pruner = FLAPStylePruner(model, calibration_loader)
    print()

    # Results storage
    results: Dict[str, Dict[float, Dict]] = {
        "Tropical": {},
        "Magnitude": {},
        "Wanda": {},
        "FLAP": {},
    }

    methods = [
        ("Tropical", lambda s: FFNTropicalPruner(model, tropical_stats).prune(s, inplace=False)),
        ("Magnitude", lambda s: magnitude_pruner.prune(s, inplace=False)),
        ("Wanda", lambda s: wanda_pruner.prune(s, inplace=False)),
        ("FLAP", lambda s: flap_pruner.prune(s, inplace=False)),
    ]

    # Test each sparsity level
    for sparsity in sparsities:
        print("=" * 90)
        print(f"SPARSITY: {sparsity*100:.0f}%")
        print("=" * 90)

        for method_name, prune_fn in methods:
            print(f"  {method_name}...", end=" ", flush=True)

            pruned_model = prune_fn(sparsity)
            pruned_params = count_parameters(pruned_model)
            ppl = compute_perplexity(pruned_model, tokenizer, args.eval_samples, args.seq_length)

            results[method_name][sparsity] = {
                "params": pruned_params,
                "compression": original_params / pruned_params,
                "ppl": ppl,
                "delta_ppl": ppl - baseline_ppl,
                "delta_ppl_pct": (ppl - baseline_ppl) / baseline_ppl * 100,
            }

            print(f"PPL={ppl:.2f} (Δ={ppl-baseline_ppl:+.2f})")

            del pruned_model
            free_memory()

        print()

    # === Final Summary ===
    print("=" * 90)
    print("COMPREHENSIVE RESULTS")
    print("=" * 90)
    print()

    # Table header
    header = f"{'Sparsity':<10}"
    for method in results.keys():
        header += f" {method:>12}"
    print(header)
    print("-" * 90)

    # Baseline row
    row = f"{'0% (base)':<10}"
    for _ in results.keys():
        row += f" {baseline_ppl:>12.2f}"
    print(row)

    # Results rows
    for sparsity in sparsities:
        row = f"{f'{sparsity*100:.0f}%':<10}"
        best_ppl = min(results[m][sparsity]["ppl"] for m in results.keys())
        for method in results.keys():
            ppl = results[method][sparsity]["ppl"]
            marker = " *" if ppl == best_ppl else "  "
            row += f" {ppl:>10.2f}{marker}"
        print(row)

    print("-" * 90)
    print("(* = best at this sparsity level)")
    print()

    # Delta PPL table
    print("Perplexity Increase (Δ PPL):")
    print("-" * 90)
    header = f"{'Sparsity':<10}"
    for method in results.keys():
        header += f" {method:>12}"
    print(header)
    print("-" * 90)

    for sparsity in sparsities:
        row = f"{f'{sparsity*100:.0f}%':<10}"
        best_delta = min(results[m][sparsity]["delta_ppl"] for m in results.keys())
        for method in results.keys():
            delta = results[method][sparsity]["delta_ppl"]
            marker = " *" if delta == best_delta else "  "
            row += f" {delta:>+10.2f}{marker}"
        print(row)

    print("-" * 90)
    print()

    # Win count
    print("Method Rankings:")
    print("-" * 90)
    wins = {m: 0 for m in results.keys()}
    for sparsity in sparsities:
        best_method = min(results.keys(), key=lambda m: results[m][sparsity]["ppl"])
        wins[best_method] += 1

    for method, win_count in sorted(wins.items(), key=lambda x: -x[1]):
        bar = "█" * win_count + "░" * (len(sparsities) - win_count)
        print(f"  {method:<12} {bar} ({win_count}/{len(sparsities)} wins)")

    print()

    # Compression vs PPL trade-off summary
    print("Compression vs PPL Trade-off at 50% sparsity:")
    print("-" * 90)
    if 0.5 in sparsities:
        for method in results.keys():
            data = results[method][0.5]
            print(f"  {method:<12}: {data['compression']:.2f}x compression, PPL={data['ppl']:.2f} (Δ={data['delta_ppl']:+.2f})")

    print("=" * 90)

    # Auto-save results
    if not args.no_save:
        json_path, csv_path = save_results(
            results=results,
            args=args,
            system_info=system_info,
            baseline_ppl=baseline_ppl,
            original_params=original_params,
            never_win_neurons=total_never_win,
            total_neurons=total_neurons,
            output_dir=args.output_dir,
        )
        print(f"\nResults saved to:")
        print(f"  JSON: {json_path}")
        print(f"  CSV:  {csv_path}")


if __name__ == "__main__":
    main()
