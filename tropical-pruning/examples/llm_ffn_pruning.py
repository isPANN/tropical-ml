#!/usr/bin/env python3
"""
LLM FFN Tropical Pruning Example

This script demonstrates how to use tropical pruning on LLM FFN layers.
It prunes the intermediate neurons in SwiGLU-based FFN blocks based on
winner statistics from the down_proj layer.

Usage:
    # Basic usage with TinyLlama (default)
    python llm_ffn_pruning.py

    # Custom model and sparsity
    python llm_ffn_pruning.py --model meta-llama/Llama-2-7b-hf --sparsity 0.3

    # Full evaluation run
    python llm_ffn_pruning.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --sparsity 0.3 --num-samples 256 --eval

Requirements:
    pip install tropical-pruning[llm]
"""

# Set HuggingFace mirror BEFORE any imports
import os
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import argparse
import sys
from typing import Optional

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tropical pruning for LLM FFN layers"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--sparsity",
        type=float,
        default=0.3,
        help="Target sparsity (fraction of intermediate neurons to prune)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=128,
        help="Number of calibration samples",
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=512,
        help="Sequence length for calibration (default: 512, reduce if OOM)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for calibration",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run perplexity evaluation after pruning",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Path to save pruned model",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cuda, cpu)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float16", "bfloat16", "float32"],
        default="float16",
        help="Model dtype",
    )
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Load model in 8-bit quantization (saves memory)",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load model in 4-bit quantization (saves more memory)",
    )
    parser.add_argument(
        "--low-cpu-mem-usage",
        action="store_true",
        default=True,
        help="Use low CPU memory usage mode (default: True)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("LLM FFN Tropical Pruning")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Target sparsity: {args.sparsity * 100:.1f}%")
    print(f"Calibration samples: {args.num_samples}")
    print(f"Sequence length: {args.seq_length}")
    print(f"Batch size: {args.batch_size}")
    if args.load_in_4bit:
        print("Quantization: 4-bit")
    elif args.load_in_8bit:
        print("Quantization: 8-bit")
    else:
        print(f"Quantization: None (dtype: {args.dtype})")
    print()

    # Import LLM pruning modules
    try:
        from tropical_pruning.llm import (
            load_model_and_tokenizer,
            get_ffn_layer_names,
            FFNWinnerCounter,
            FFNTropicalPruner,
            CalibrationDataset,
            LLMEvaluator,
        )
        from tropical_pruning.llm.loader import count_parameters
    except ImportError as e:
        print(f"Error: {e}")
        print("\nPlease install LLM dependencies:")
        print("  pip install tropical-pruning[llm]")
        sys.exit(1)

    # Determine dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    # Step 1: Load model
    print("Step 1: Loading model...")
    load_kwargs = {}
    if args.load_in_8bit:
        load_kwargs["load_in_8bit"] = True
        print("  Using 8-bit quantization")
    elif args.load_in_4bit:
        load_kwargs["load_in_4bit"] = True
        print("  Using 4-bit quantization")
    if args.low_cpu_mem_usage:
        load_kwargs["low_cpu_mem_usage"] = True
    
    model, tokenizer = load_model_and_tokenizer(
        args.model,
        device_map=args.device,
        torch_dtype=torch_dtype,
        **load_kwargs,
    )
    original_params = count_parameters(model)
    print(f"  Parameters: {original_params:,}")

    # Show FFN layer info
    ffn_layers = get_ffn_layer_names(model)
    print(f"  FFN layers: {len(ffn_layers)}")
    if ffn_layers:
        print(f"  Intermediate size: {ffn_layers[0].intermediate_size}")
        print(f"  Hidden size: {ffn_layers[0].hidden_size}")
    print()

    # Step 2: Create calibration dataset
    print("Step 2: Creating calibration dataset...")
    calibration_loader = CalibrationDataset.from_wikitext2(
        tokenizer,
        num_samples=args.num_samples,
        seq_length=args.seq_length,
        batch_size=args.batch_size,
    )
    print(f"  Created {len(calibration_loader)} batches")
    print()

    # Step 3: Collect FFN winner statistics
    print("Step 3: Collecting FFN winner statistics...")
    counter = FFNWinnerCounter(model, track_margin=True)
    stats = counter.collect(calibration_loader, show_progress=True)
    counter.remove_hooks()
    print(f"  Collected statistics for {len(stats)} FFN layers")

    # Analyze statistics
    analysis = counter.analyze()
    total_never_win = sum(a["never_win"] for a in analysis.values())
    total_neurons = sum(a["intermediate_size"] for a in analysis.values())
    print(f"  Neurons that never win: {total_never_win:,} / {total_neurons:,} "
          f"({total_never_win / total_neurons * 100:.1f}%)")
    print()

    # Step 4: Prune FFN layers
    print("Step 4: Pruning FFN layers...")
    pruner = FFNTropicalPruner(model, stats)
    pruned_model = pruner.prune(sparsity=args.sparsity, inplace=False)

    # Get compression stats
    compression = pruner.get_compression_stats()
    print(f"  Original parameters: {compression['original_parameters']:,}")
    print(f"  Pruned parameters: {compression['pruned_parameters']:,}")
    print(f"  Compression ratio: {compression['compression_ratio']:.2f}x")
    print(f"  FFN compression ratio: {compression['ffn_compression_ratio']:.2f}x")
    print(f"  Overall sparsity: {compression['sparsity_achieved'] * 100:.1f}%")
    print(f"  FFN sparsity: {compression['ffn_sparsity_achieved'] * 100:.1f}%")
    print()

    # Step 5: Evaluate (optional)
    if args.eval:
        print("Step 5: Evaluating perplexity...")
        evaluator = LLMEvaluator(pruned_model, tokenizer)
        baseline_evaluator = LLMEvaluator(model, tokenizer)

        print("  Computing baseline perplexity...")
        baseline_ppl = baseline_evaluator.compute_perplexity(
            dataset="wikitext2",
            split="test",
            max_length=args.seq_length,
        )
        print(f"  Baseline perplexity: {baseline_ppl:.2f}")

        print("  Computing pruned model perplexity...")
        pruned_ppl = evaluator.compute_perplexity(
            dataset="wikitext2",
            split="test",
            max_length=args.seq_length,
        )
        print(f"  Pruned perplexity: {pruned_ppl:.2f}")
        print(f"  Perplexity increase: {pruned_ppl - baseline_ppl:.2f} "
              f"({(pruned_ppl - baseline_ppl) / baseline_ppl * 100:.1f}%)")
        print()

    # Step 6: Save model (optional)
    if args.save_path:
        print(f"Step 6: Saving pruned model to {args.save_path}...")
        pruned_model.save_pretrained(args.save_path)
        tokenizer.save_pretrained(args.save_path)
        print("  Saved!")
        print()

    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Sparsity target: {args.sparsity * 100:.1f}%")
    print(f"Parameters reduced: {original_params:,} -> {compression['pruned_parameters']:,}")
    print(f"Compression: {compression['compression_ratio']:.2f}x")
    if args.eval:
        print(f"Perplexity: {baseline_ppl:.2f} -> {pruned_ppl:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
