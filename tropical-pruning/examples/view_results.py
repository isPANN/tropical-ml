#!/usr/bin/env python3
"""
View and compare saved experiment results.

Usage:
    # List all results
    python view_results.py --list

    # View specific result
    python view_results.py results/ffn_pruning_xxx.json

    # Compare multiple results
    python view_results.py results/*.json --compare
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict
import glob


def list_results(results_dir: str = "results"):
    """List all saved results."""
    path = Path(results_dir)
    if not path.exists():
        print(f"No results directory found: {results_dir}")
        return

    json_files = sorted(path.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)

    if not json_files:
        print(f"No results found in {results_dir}/")
        return

    print(f"Found {len(json_files)} result(s) in {results_dir}/")
    print("=" * 90)
    print(f"{'Filename':<50} {'Model':<25} {'Timestamp':<20}")
    print("-" * 90)

    for f in json_files:
        try:
            with open(f) as fp:
                data = json.load(fp)
            model = data.get("config", {}).get("model", "N/A")
            model_short = model.split("/")[-1][:24]
            timestamp = data.get("system", {}).get("timestamp", "N/A")[:19]
            print(f"{f.name:<50} {model_short:<25} {timestamp:<20}")
        except Exception as e:
            print(f"{f.name:<50} {'(error reading)':<25}")


def view_result(filepath: str):
    """View a single result file."""
    with open(filepath) as f:
        data = json.load(f)

    print("=" * 80)
    print(f"Experiment: {data.get('experiment', {}).get('name', data.get('experiment', 'N/A'))}")
    print("=" * 80)

    # Config
    config = data.get("config", {})
    print(f"\nConfiguration:")
    print(f"  Model: {config.get('model', 'N/A')}")
    if "sparsity" in config:
        print(f"  Sparsity: {config['sparsity'] * 100:.1f}%")
    if "sparsities" in config:
        print(f"  Sparsities: {[f'{s*100:.0f}%' for s in config['sparsities']]}")
    print(f"  Calibration samples: {config.get('num_samples', 'N/A')}")
    print(f"  Sequence length: {config.get('seq_length', 'N/A')}")

    # System info
    system = data.get("system", {})
    print(f"\nSystem:")
    print(f"  Timestamp: {system.get('timestamp', 'N/A')}")
    if "gpu" in system:
        for gpu in system["gpu"].get("devices", []):
            print(f"  GPU: {gpu.get('name', 'N/A')} ({gpu.get('total_memory_gb', 'N/A')}GB)")
    print(f"  PyTorch: {system.get('torch', {}).get('version', 'N/A')}")
    if system.get("git_commit"):
        print(f"  Git commit: {system['git_commit']}")

    # Baseline
    baseline = data.get("baseline", {})
    if baseline:
        print(f"\nBaseline:")
        print(f"  Parameters: {baseline.get('parameters', 'N/A'):,}")
        print(f"  Perplexity: {baseline.get('perplexity', 'N/A'):.2f}")

    # Analysis
    analysis = data.get("analysis", {})
    if analysis:
        print(f"\nAnalysis:")
        print(f"  Never-win neurons: {analysis.get('never_win_neurons', 'N/A'):,} / {analysis.get('total_neurons', 'N/A'):,}")
        print(f"  Never-win ratio: {analysis.get('never_win_ratio', 0) * 100:.1f}%")

    # Results
    results = data.get("results", {})
    if results:
        print(f"\nResults:")
        print("-" * 80)

        # Check if multi-sparsity or single
        first_val = list(results.values())[0]
        if isinstance(first_val, dict) and "ppl" not in first_val:
            # Multi-sparsity format
            sparsities = list(list(results.values())[0].keys())
            print(f"{'Method':<15}", end="")
            for s in sparsities:
                print(f" {float(s)*100:>6.0f}%", end="")
            print()
            print("-" * 80)

            for method, sparsity_data in results.items():
                print(f"{method:<15}", end="")
                for s in sparsities:
                    ppl = sparsity_data.get(s, sparsity_data.get(float(s), {})).get("ppl", 0)
                    print(f" {ppl:>7.2f}", end="")
                print()
        else:
            # Single sparsity format
            print(f"{'Method':<15} {'Params':>15} {'Compression':>12} {'PPL':>10} {'Δ PPL':>10}")
            print("-" * 80)
            for method, d in results.items():
                print(f"{method:<15} {d.get('params', 0):>15,} {d.get('compression', 1):>12.2f}x {d.get('ppl', 0):>10.2f} {d.get('delta_ppl', 0):>+10.2f}")

    print("=" * 80)


def compare_results(filepaths: List[str]):
    """Compare multiple result files with detailed analysis."""
    all_data = []
    for fp in filepaths:
        try:
            with open(fp) as f:
                data = json.load(f)
                data["_filepath"] = fp
                all_data.append(data)
        except Exception:
            print(f"Error reading {fp}")

    if not all_data:
        print("No valid results to compare")
        return

    print("=" * 100)
    print("Comparison of Experiments")
    print("=" * 100)

    # Summary table
    print(f"\n{'File':<45} {'Model':<20} {'Sparsity':<15} {'Best Method':<20}")
    print("-" * 100)

    for data in all_data:
        filepath = Path(data["_filepath"]).name[:44]
        model = data.get("config", {}).get("model", "N/A").split("/")[-1][:19]

        results = data.get("results", {})
        if not results:
            continue

        # Get sparsity info
        config = data.get("config", {})
        if "sparsity" in config:
            sparsity = f"{config['sparsity']*100:.0f}%"
        elif "sparsities" in config:
            sparsity = ",".join([f"{s*100:.0f}%" for s in config["sparsities"]])
        else:
            sparsity = "N/A"

        # Find best method
        first_val = list(results.values())[0]
        if isinstance(first_val, dict) and "ppl" in first_val:
            # Single sparsity
            pruned = {k: v for k, v in results.items() if k != "Baseline"}
            if pruned:
                best = min(pruned.items(), key=lambda x: x[1].get("ppl", float("inf")))
                best_method = f"{best[0]} ({best[1].get('ppl', 0):.2f})"
            else:
                best_method = "N/A"
        else:
            # Multi-sparsity - show wins
            wins = {method: 0 for method in results.keys()}
            sparsity_keys = list(list(results.values())[0].keys())
            for s in sparsity_keys:
                best_ppl = float("inf")
                best_m = None
                for method, sparsity_data in results.items():
                    ppl = sparsity_data.get(s, sparsity_data.get(str(s), {})).get("ppl", float("inf"))
                    if ppl < best_ppl:
                        best_ppl = ppl
                        best_m = method
                if best_m:
                    wins[best_m] += 1
            best = max(wins.items(), key=lambda x: x[1])
            best_method = f"{best[0]} ({best[1]}/{len(sparsity_keys)} wins)"

        print(f"{filepath:<45} {model:<20} {sparsity:<15} {best_method:<20}")

    print("-" * 100)

    # Detailed comparison for multi-sparsity results
    sweep_data = [d for d in all_data if "sparsities" in d.get("config", {})]
    if sweep_data:
        print("\n" + "=" * 100)
        print("Detailed Sweep Comparison")
        print("=" * 100)

        for data in sweep_data:
            filepath = Path(data["_filepath"]).name
            config = data.get("config", {})
            results = data.get("results", {})
            baseline = data.get("baseline", {})

            print(f"\n{filepath}")
            print(f"Model: {config.get('model', 'N/A')}")
            print(f"Baseline PPL: {baseline.get('perplexity', 'N/A')}")
            print()

            # Get all sparsities and methods
            methods = list(results.keys())
            sparsities = list(list(results.values())[0].keys())

            # Print header
            header = f"{'Sparsity':<12}"
            for m in methods:
                header += f"{m:>12}"
            header += f"{'Best':>12}"
            print(header)
            print("-" * (12 + 12 * len(methods) + 12))

            # Print each sparsity row
            for s in sparsities:
                row = f"{float(s)*100:>10.0f}% "
                best_ppl = float("inf")
                best_method = ""
                for m in methods:
                    ppl = results[m].get(s, results[m].get(str(s), {})).get("ppl", 0)
                    row += f"{ppl:>12.2f}"
                    if ppl < best_ppl:
                        best_ppl = ppl
                        best_method = m
                row += f"{best_method:>12}"
                print(row)

            # Print wins summary
            print("-" * (12 + 12 * len(methods) + 12))
            wins = {m: 0 for m in methods}
            for s in sparsities:
                best_ppl = float("inf")
                best_m = None
                for m in methods:
                    ppl = results[m].get(s, results[m].get(str(s), {})).get("ppl", float("inf"))
                    if ppl < best_ppl:
                        best_ppl = ppl
                        best_m = m
                if best_m:
                    wins[best_m] += 1

            row = f"{'Wins':<12}"
            for m in methods:
                row += f"{wins[m]:>12}"
            print(row)

    print("\n" + "=" * 100)


def main():
    parser = argparse.ArgumentParser(description="View experiment results")
    parser.add_argument("files", nargs="*", help="Result files to view")
    parser.add_argument("--list", "-l", action="store_true", help="List all results")
    parser.add_argument("--compare", "-c", action="store_true", help="Compare multiple results")
    parser.add_argument("--dir", "-d", type=str, default="results", help="Results directory")
    args = parser.parse_args()

    if args.list:
        list_results(args.dir)
    elif args.compare and args.files:
        # Expand globs
        expanded = []
        for f in args.files:
            expanded.extend(glob.glob(f))
        compare_results(expanded)
    elif args.files:
        for f in args.files:
            for expanded in glob.glob(f):
                view_result(expanded)
    else:
        list_results(args.dir)


if __name__ == "__main__":
    main()
