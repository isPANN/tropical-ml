#!/usr/bin/env python3
"""
LLM FFN Pruning Comparison: Tropical vs SOTA Baselines

Compare tropical pruning with multiple baseline methods on LLM FFN layers:
- Magnitude pruning (L1 norm)
- Wanda-style pruning (weight * activation)
- FLAP-style pruning (weight * fluctuation)

Supports single or multiple sparsity levels. Results auto-saved to results/.

Usage:
    # Single sparsity
    python llm_ffn_comparison.py --sparsity 0.3

    # Multiple sparsities (sweep)
    python llm_ffn_comparison.py --sparsity 0.2,0.3,0.5,0.7

    # Custom model
    python llm_ffn_comparison.py --model meta-llama/Llama-2-7b-hf --sparsity 0.3,0.5

References:
- LLM-Pruner: https://github.com/horseee/LLM-Pruner (NeurIPS 2023)
- FLAP: https://github.com/CASIA-IVA-Lab/FLAP (AAAI 2024)
- Wanda: https://github.com/locuslab/wanda (ICLR 2024)
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
from typing import Dict, List, Any
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
            })

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
) -> tuple:
    """Save results to JSON and CSV."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = args.model.split("/")[-1].replace("-", "_")
    sparsities = [float(s.strip()) for s in args.sparsity.split(",")]
    sparsity_str = "_".join([f"{int(s*100)}" for s in sparsities])
    base_name = f"ffn_comparison_{model_short}_s{sparsity_str}_{timestamp}"

    full_data = {
        "experiment": {
            "name": "LLM FFN Pruning Comparison",
            "script": "llm_ffn_comparison.py",
        },
        "config": {
            "model": args.model,
            "sparsities": sparsities,
            "num_samples": args.num_samples,
            "seq_length": args.seq_length,
            "eval_samples": args.eval_samples,
            "methods": args.methods,
        },
        "system": system_info,
        "baseline": {
            "perplexity": baseline_ppl,
            "parameters": original_params,
        },
        "analysis": {
            "never_win_neurons": never_win_neurons,
            "total_neurons": total_neurons,
            "never_win_ratio": never_win_neurons / total_neurons if total_neurons > 0 else 0,
        },
        "results": results,
    }

    json_path = output_path / f"{base_name}.json"
    with open(json_path, "w") as f:
        json.dump(full_data, f, indent=2)

    csv_path = output_path / f"{base_name}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "method", "sparsity", "params", "compression",
            "ppl", "delta_ppl", "delta_ppl_pct", "baseline_ppl",
            "num_samples", "seq_length", "timestamp"
        ])
        for method, sparsity_results in results.items():
            if method == "Baseline":
                continue
            for sparsity, data in sparsity_results.items():
                writer.writerow([
                    args.model, method, sparsity,
                    data["params"], round(data["compression"], 4),
                    round(data["ppl"], 4), round(data["delta_ppl"], 4),
                    round(data["delta_ppl_pct"], 4), round(baseline_ppl, 4),
                    args.num_samples, args.seq_length, system_info["timestamp"],
                ])

    return str(json_path), str(csv_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare tropical vs baseline pruning methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --sparsity 0.3                    # Single sparsity
  %(prog)s --sparsity 0.2,0.3,0.5,0.7        # Multiple sparsities
  %(prog)s --model meta-llama/Llama-2-7b-hf  # Different model
        """
    )
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                       help="HuggingFace model name or path")
    parser.add_argument("--sparsity", type=str, default="0.2,0.3,0.5,0.7",
                       help="Sparsity level(s), comma-separated (e.g., '0.3' or '0.2,0.3,0.5,0.7')")
    parser.add_argument("--num-samples", type=int, default=64,
                       help="Number of calibration samples")
    parser.add_argument("--seq-length", type=int, default=512,
                       help="Sequence length for calibration")
    parser.add_argument("--eval-samples", type=int, default=50,
                       help="Number of samples for perplexity evaluation")
    parser.add_argument("--methods", type=str, default="all",
                       help="Methods to compare: all, or comma-separated list (tropical,magnitude,wanda,flap)")
    parser.add_argument("--output-dir", type=str, default="results",
                       help="Directory to save results")
    parser.add_argument("--no-save", action="store_true",
                       help="Don't save results to files")
    return parser.parse_args()


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


def compute_perplexity(model, tokenizer, num_samples=50, seq_length=512) -> float:
    """Compute perplexity on WikiText-2 test set."""
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
    sparsities = [float(s.strip()) for s in args.sparsity.split(",")]
    system_info = get_system_info()

    # Determine methods to run
    if args.methods == "all":
        methods = ["tropical", "magnitude", "wanda", "flap"]
    else:
        methods = [m.strip().lower() for m in args.methods.split(",")]

    print("=" * 90)
    print("LLM FFN Pruning Comparison: Tropical vs SOTA Baselines")
    print("=" * 90)
    print(f"Timestamp: {system_info['timestamp']}")
    print(f"Model: {args.model}")
    print(f"Sparsities: {[f'{s*100:.0f}%' for s in sparsities]}")
    print(f"Methods: {methods}")
    print(f"Calibration: {args.num_samples} samples x {args.seq_length} tokens")
    if system_info["torch"]["cuda_available"]:
        gpu = system_info["gpu"]["devices"][0]
        print(f"GPU: {gpu['name']} ({gpu['total_memory_gb']}GB)")
    print()

    # Import modules
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
    print(f"  Parameters: {original_params:,}")
    print()

    # Compute baseline perplexity
    print("Computing baseline perplexity...")
    baseline_ppl = compute_perplexity(model, tokenizer, args.eval_samples, args.seq_length)
    print(f"  Baseline PPL: {baseline_ppl:.2f}")
    print()

    # Create calibration data
    print("Creating calibration dataset...")
    calibration_loader = CalibrationDataset.from_wikitext2(
        tokenizer, num_samples=args.num_samples, seq_length=args.seq_length
    )
    print(f"  Created {len(calibration_loader)} batches")
    print()

    # Collect tropical statistics (once, reuse for all sparsities)
    total_never_win = 0
    total_neurons = 0
    tropical_stats = None

    if "tropical" in methods:
        print("Collecting tropical winner statistics...")
        counter = FFNWinnerCounter(model, track_margin=False)
        tropical_stats = counter.collect(calibration_loader, show_progress=True)
        counter.remove_hooks()

        analysis = counter.analyze()
        total_never_win = sum(a["never_win"] for a in analysis.values())
        total_neurons = sum(a["intermediate_size"] for a in analysis.values())
        print(f"  Never-win neurons: {total_never_win:,} / {total_neurons:,} ({total_never_win/total_neurons*100:.1f}%)")
        print()

    # Initialize baseline pruners (once, reuse for all sparsities)
    pruners = {}
    if "magnitude" in methods:
        print("Initializing magnitude pruner...")
        pruners["Magnitude"] = MagnitudePruner(model, norm="l1")

    if "wanda" in methods:
        print("Initializing Wanda pruner (collecting activations)...")
        pruners["Wanda"] = WandaStylePruner(model, calibration_loader)

    if "flap" in methods:
        print("Initializing FLAP pruner (collecting fluctuations)...")
        pruners["FLAP"] = FLAPStylePruner(model, calibration_loader)

    print()

    # Results storage: {method: {sparsity: {metrics}}}
    results: Dict[str, Dict[float, Dict]] = {name: {} for name in ["Tropical"] + list(pruners.keys())}

    # Run comparison for each sparsity
    for sparsity in sparsities:
        print("=" * 90)
        print(f"SPARSITY: {sparsity*100:.0f}%")
        print("=" * 90)

        # Tropical pruning
        if "tropical" in methods and tropical_stats is not None:
            print(f"  Tropical...", end=" ", flush=True)
            pruner = FFNTropicalPruner(model, tropical_stats)
            pruned = pruner.prune(sparsity=sparsity, inplace=False)
            params = count_parameters(pruned)
            ppl = compute_perplexity(pruned, tokenizer, args.eval_samples, args.seq_length)

            results["Tropical"][sparsity] = {
                "params": params,
                "compression": original_params / params,
                "ppl": ppl,
                "delta_ppl": ppl - baseline_ppl,
                "delta_ppl_pct": (ppl - baseline_ppl) / baseline_ppl * 100,
            }
            print(f"PPL={ppl:.2f} (Δ={ppl-baseline_ppl:+.2f})")

            del pruned
            free_memory()

        # Baseline methods
        for name, pruner in pruners.items():
            print(f"  {name}...", end=" ", flush=True)
            pruned = pruner.prune(sparsity=sparsity, inplace=False)
            params = count_parameters(pruned)
            ppl = compute_perplexity(pruned, tokenizer, args.eval_samples, args.seq_length)

            results[name][sparsity] = {
                "params": params,
                "compression": original_params / params,
                "ppl": ppl,
                "delta_ppl": ppl - baseline_ppl,
                "delta_ppl_pct": (ppl - baseline_ppl) / baseline_ppl * 100,
            }
            print(f"PPL={ppl:.2f} (Δ={ppl-baseline_ppl:+.2f})")

            del pruned
            free_memory()

        print()

    # === Final Summary ===
    print("=" * 90)
    print("COMPARISON SUMMARY")
    print("=" * 90)
    print()

    # Filter out empty results
    results = {k: v for k, v in results.items() if v}
    method_names = list(results.keys())

    # Header
    header = f"{'Sparsity':<12}"
    for name in method_names:
        header += f"{name:>12}"
    header += f"{'Best':>12}"
    print(header)
    print("-" * 90)

    # Baseline row
    row = f"{'0% (base)':<12}"
    for _ in method_names:
        row += f"{baseline_ppl:>12.2f}"
    row += f"{'-':>12}"
    print(row)

    # Results rows
    wins = {name: 0 for name in method_names}
    for sparsity in sparsities:
        row = f"{f'{sparsity*100:.0f}%':<12}"
        best_ppl = float("inf")
        best_method = ""

        for name in method_names:
            ppl = results[name][sparsity]["ppl"]
            row += f"{ppl:>12.2f}"
            if ppl < best_ppl:
                best_ppl = ppl
                best_method = name

        wins[best_method] += 1
        row += f"{best_method:>12}"
        print(row)

    print("-" * 90)

    # Wins row
    row = f"{'Wins':<12}"
    for name in method_names:
        row += f"{wins[name]:>12}"
    print(row)
    print()

    # Delta PPL table
    print("Perplexity Increase (Δ PPL):")
    print("-" * 90)
    header = f"{'Sparsity':<12}"
    for name in method_names:
        header += f"{name:>12}"
    print(header)
    print("-" * 90)

    for sparsity in sparsities:
        row = f"{f'{sparsity*100:.0f}%':<12}"
        for name in method_names:
            delta = results[name][sparsity]["delta_ppl"]
            row += f"{delta:>+12.2f}"
        print(row)

    print("-" * 90)
    print()

    # Winner summary
    best_method = max(wins.items(), key=lambda x: x[1])
    print(f"Best method: {best_method[0]} ({best_method[1]}/{len(sparsities)} wins)")

    if "Tropical" in results:
        print()
        print("Tropical vs others (average Δ PPL difference):")
        for name in method_names:
            if name != "Tropical":
                diffs = []
                for sparsity in sparsities:
                    t_ppl = results["Tropical"][sparsity]["ppl"]
                    o_ppl = results[name][sparsity]["ppl"]
                    diffs.append(o_ppl - t_ppl)
                avg_diff = sum(diffs) / len(diffs)
                if avg_diff > 0:
                    print(f"  Tropical beats {name} by avg {avg_diff:.2f} PPL")
                else:
                    print(f"  {name} beats Tropical by avg {-avg_diff:.2f} PPL")

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
