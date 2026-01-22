#!/usr/bin/env python3
"""
LLM FFN Pruning Comparison: Tropical vs SOTA Baselines

Compare tropical pruning with multiple baseline methods on LLM FFN layers:
- Magnitude pruning (L1 norm)
- Activation-based pruning
- Wanda-style pruning (weight * activation)
- FLAP-style pruning (weight * fluctuation)

Results are automatically saved to results/ directory with timestamp.

Usage:
    python llm_ffn_comparison.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --sparsity 0.3

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
from typing import Dict, Any
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


def save_results(results: Dict, args, system_info: Dict, output_dir: str = "results") -> tuple:
    """Save results to JSON and CSV."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = args.model.split("/")[-1].replace("-", "_")
    sparsity_str = f"{int(args.sparsity * 100)}pct"
    base_name = f"ffn_comparison_{model_short}_{sparsity_str}_{timestamp}"

    full_data = {
        "experiment": "LLM FFN Pruning Comparison",
        "config": {
            "model": args.model,
            "sparsity": args.sparsity,
            "num_samples": args.num_samples,
            "seq_length": args.seq_length,
            "eval_samples": args.eval_samples,
            "methods": args.methods,
        },
        "system": system_info,
        "results": results,
    }

    json_path = output_path / f"{base_name}.json"
    with open(json_path, "w") as f:
        json.dump(full_data, f, indent=2)

    csv_path = output_path / f"{base_name}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "sparsity", "method", "params", "compression",
            "ppl", "delta_ppl", "timestamp"
        ])
        for method, data in results.items():
            writer.writerow([
                args.model, args.sparsity, method,
                data["params"], round(data["compression"], 4),
                round(data["ppl"], 4), round(data["delta_ppl"], 4),
                system_info["timestamp"],
            ])

    return str(json_path), str(csv_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare tropical vs baseline pruning")
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--sparsity", type=float, default=0.3)
    parser.add_argument("--num-samples", type=int, default=64, help="Calibration samples")
    parser.add_argument("--seq-length", type=int, default=512)
    parser.add_argument("--eval-samples", type=int, default=50, help="Perplexity eval samples")
    parser.add_argument("--methods", type=str, default="all",
                       help="Methods to compare: all, tropical, magnitude, wanda, flap")
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
    """Free GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    args = parse_args()
    system_info = get_system_info()

    print("=" * 80)
    print("LLM FFN Pruning Comparison: Tropical vs SOTA Baselines")
    print("=" * 80)
    print(f"Timestamp: {system_info['timestamp']}")
    print(f"Model: {args.model}")
    print(f"Target sparsity: {args.sparsity * 100:.1f}%")
    print(f"Calibration samples: {args.num_samples}")
    print(f"Sequence length: {args.seq_length}")
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
        get_baseline_pruner,
    )

    # Determine which methods to run
    if args.methods == "all":
        methods = ["tropical", "magnitude", "wanda", "flap"]
    else:
        methods = [m.strip() for m in args.methods.split(",")]

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

    # Compute baseline perplexity
    print("Computing baseline perplexity...")
    baseline_ppl = compute_perplexity(model, tokenizer, args.eval_samples, args.seq_length)
    print(f"  Baseline PPL: {baseline_ppl:.2f}")
    print()

    # Create calibration data (shared across methods)
    print("Creating calibration dataset...")
    calibration_loader = CalibrationDataset.from_wikitext2(
        tokenizer,
        num_samples=args.num_samples,
        seq_length=args.seq_length,
    )
    print(f"  Created {len(calibration_loader)} batches")
    print()

    # Results storage
    results: Dict[str, Dict] = {
        "Baseline": {
            "params": original_params,
            "compression": 1.0,
            "ppl": baseline_ppl,
            "delta_ppl": 0.0,
        }
    }

    # === Method 1: Tropical Pruning ===
    if "tropical" in methods:
        print("-" * 80)
        print("Method: TROPICAL PRUNING (winner-based)")
        print("-" * 80)

        print("  Collecting winner statistics...")
        counter = FFNWinnerCounter(model, track_margin=False)
        stats = counter.collect(calibration_loader, show_progress=True)
        counter.remove_hooks()

        # Analyze never-win neurons
        analysis = counter.analyze()
        total_never_win = sum(a["never_win"] for a in analysis.values())
        total_neurons = sum(a["intermediate_size"] for a in analysis.values())
        print(f"  Never-win neurons: {total_never_win:,} / {total_neurons:,} ({total_never_win/total_neurons*100:.1f}%)")

        print("  Pruning...")
        pruner = FFNTropicalPruner(model, stats)
        tropical_model = pruner.prune(sparsity=args.sparsity, inplace=False)
        tropical_params = count_parameters(tropical_model)

        print("  Computing perplexity...")
        tropical_ppl = compute_perplexity(tropical_model, tokenizer, args.eval_samples, args.seq_length)

        results["Tropical"] = {
            "params": tropical_params,
            "compression": original_params / tropical_params,
            "ppl": tropical_ppl,
            "delta_ppl": tropical_ppl - baseline_ppl,
        }
        print(f"  Parameters: {tropical_params:,} ({results['Tropical']['compression']:.2f}x)")
        print(f"  Perplexity: {tropical_ppl:.2f} (Δ = +{tropical_ppl - baseline_ppl:.2f})")
        print()

        del tropical_model
        free_memory()

    # === Method 2: Magnitude Pruning ===
    if "magnitude" in methods:
        print("-" * 80)
        print("Method: MAGNITUDE PRUNING (L1 norm baseline)")
        print("-" * 80)

        print("  Pruning...")
        pruner = MagnitudePruner(model, norm="l1")
        magnitude_model = pruner.prune(sparsity=args.sparsity, inplace=False)
        magnitude_params = count_parameters(magnitude_model)

        print("  Computing perplexity...")
        magnitude_ppl = compute_perplexity(magnitude_model, tokenizer, args.eval_samples, args.seq_length)

        results["Magnitude"] = {
            "params": magnitude_params,
            "compression": original_params / magnitude_params,
            "ppl": magnitude_ppl,
            "delta_ppl": magnitude_ppl - baseline_ppl,
        }
        print(f"  Parameters: {magnitude_params:,} ({results['Magnitude']['compression']:.2f}x)")
        print(f"  Perplexity: {magnitude_ppl:.2f} (Δ = +{magnitude_ppl - baseline_ppl:.2f})")
        print()

        del magnitude_model
        free_memory()

    # === Method 3: Wanda-style Pruning ===
    if "wanda" in methods:
        print("-" * 80)
        print("Method: WANDA-STYLE PRUNING (weight * activation)")
        print("-" * 80)

        print("  Collecting activation statistics...")
        pruner = WandaStylePruner(model, calibration_loader)

        print("  Pruning...")
        wanda_model = pruner.prune(sparsity=args.sparsity, inplace=False)
        wanda_params = count_parameters(wanda_model)

        print("  Computing perplexity...")
        wanda_ppl = compute_perplexity(wanda_model, tokenizer, args.eval_samples, args.seq_length)

        results["Wanda"] = {
            "params": wanda_params,
            "compression": original_params / wanda_params,
            "ppl": wanda_ppl,
            "delta_ppl": wanda_ppl - baseline_ppl,
        }
        print(f"  Parameters: {wanda_params:,} ({results['Wanda']['compression']:.2f}x)")
        print(f"  Perplexity: {wanda_ppl:.2f} (Δ = +{wanda_ppl - baseline_ppl:.2f})")
        print()

        del wanda_model
        free_memory()

    # === Method 4: FLAP-style Pruning ===
    if "flap" in methods:
        print("-" * 80)
        print("Method: FLAP-STYLE PRUNING (weight * fluctuation)")
        print("-" * 80)

        print("  Collecting fluctuation statistics...")
        pruner = FLAPStylePruner(model, calibration_loader)

        print("  Pruning...")
        flap_model = pruner.prune(sparsity=args.sparsity, inplace=False)
        flap_params = count_parameters(flap_model)

        print("  Computing perplexity...")
        flap_ppl = compute_perplexity(flap_model, tokenizer, args.eval_samples, args.seq_length)

        results["FLAP"] = {
            "params": flap_params,
            "compression": original_params / flap_params,
            "ppl": flap_ppl,
            "delta_ppl": flap_ppl - baseline_ppl,
        }
        print(f"  Parameters: {flap_params:,} ({results['FLAP']['compression']:.2f}x)")
        print(f"  Perplexity: {flap_ppl:.2f} (Δ = +{flap_ppl - baseline_ppl:.2f})")
        print()

        del flap_model
        free_memory()

    # === Summary ===
    print("=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print(f"{'Method':<15} {'Params':>18} {'Compress':>10} {'PPL':>10} {'Δ PPL':>10}")
    print("-" * 80)

    for method, data in results.items():
        print(f"{method:<15} {data['params']:>18,} {data['compression']:>10.2f}x {data['ppl']:>10.2f} {data['delta_ppl']:>+10.2f}")

    print("-" * 80)

    # Find winner (lowest PPL among pruned methods)
    pruned_results = {k: v for k, v in results.items() if k != "Baseline"}
    if pruned_results:
        best_method = min(pruned_results.keys(), key=lambda k: pruned_results[k]["ppl"])
        best_ppl = pruned_results[best_method]["ppl"]

        print(f"\nBest method: {best_method} (PPL = {best_ppl:.2f})")

        # Compare tropical with others
        if "Tropical" in pruned_results:
            tropical_ppl = pruned_results["Tropical"]["ppl"]
            for method, data in pruned_results.items():
                if method != "Tropical":
                    diff = data["ppl"] - tropical_ppl
                    if diff > 0:
                        print(f"  Tropical beats {method} by {diff:.2f} PPL ({diff/data['ppl']*100:.1f}% better)")
                    elif diff < 0:
                        print(f"  {method} beats Tropical by {-diff:.2f} PPL ({-diff/tropical_ppl*100:.1f}% better)")
                    else:
                        print(f"  Tropical ties with {method}")

    print("=" * 80)

    # Auto-save results
    if not args.no_save:
        json_path, csv_path = save_results(results, args, system_info, args.output_dir)
        print(f"\nResults saved to:")
        print(f"  JSON: {json_path}")
        print(f"  CSV:  {csv_path}")


if __name__ == "__main__":
    main()
