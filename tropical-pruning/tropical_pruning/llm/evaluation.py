"""
LLM Evaluation: Perplexity computation for evaluating pruned models.

Implements sliding window perplexity computation on standard benchmarks like WikiText-2.
"""

from typing import Any, Dict, Optional
import torch
import torch.nn as nn
from tqdm import tqdm


class LLMEvaluator:
    """
    Evaluate LLM performance using perplexity.

    Implements sliding window perplexity computation which is the standard
    evaluation metric for language model pruning.

    Example:
        >>> model, tokenizer = load_model_and_tokenizer("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        >>> evaluator = LLMEvaluator(model, tokenizer)
        >>> ppl = evaluator.compute_perplexity()
        >>> print(f"Perplexity: {ppl:.2f}")
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the evaluator.

        Args:
            model: The LLM to evaluate.
            tokenizer: HuggingFace tokenizer.
            device: Device for computation. If None, uses model's device.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device

    def compute_perplexity(
        self,
        dataset: str = "wikitext2",
        split: str = "test",
        max_length: int = 2048,
        stride: Optional[int] = None,
        show_progress: bool = True,
    ) -> float:
        """
        Compute perplexity on a dataset.

        Uses sliding window evaluation where each position is predicted using
        a context of max_length tokens.

        Args:
            dataset: Dataset to evaluate on ("wikitext2", "c4", or a custom text).
            split: Dataset split ("test", "validation", "train").
            max_length: Maximum context length for each prediction.
            stride: Stride for sliding window. Defaults to max_length // 2.
            show_progress: Whether to show a progress bar.

        Returns:
            Perplexity score (lower is better).
        """
        if stride is None:
            stride = max_length // 2

        # Load dataset
        text = self._load_dataset_text(dataset, split)

        # Tokenize
        encodings = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=False,
            add_special_tokens=False,
        )
        input_ids = encodings["input_ids"].to(self.device)

        seq_len = input_ids.size(1)

        # Sliding window evaluation
        nlls = []
        n_tokens = 0

        self.model.eval()

        # Calculate number of windows
        num_windows = max(1, (seq_len - max_length) // stride + 1)
        iterator = range(0, seq_len - 1, stride)

        if show_progress:
            iterator = tqdm(iterator, desc="Computing perplexity", total=num_windows)

        with torch.no_grad():
            for begin_loc in iterator:
                end_loc = min(begin_loc + max_length, seq_len)
                trg_len = end_loc - begin_loc - 1

                if trg_len <= 0:
                    continue

                input_chunk = input_ids[:, begin_loc:end_loc]

                # Get model outputs
                outputs = self.model(input_chunk)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

                # Compute loss for this window
                # Shift so that tokens < n predict token n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_chunk[..., 1:].contiguous()

                # Calculate loss per token
                loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )

                # For overlapping windows, only count tokens in the second half
                # (to avoid double-counting)
                if begin_loc > 0:
                    # Only count the new tokens (those after the stride)
                    loss = loss[-(end_loc - begin_loc - stride - 1) :]

                nlls.append(loss.sum())
                n_tokens += loss.numel()

                if end_loc >= seq_len:
                    break

        # Compute perplexity
        total_nll = torch.stack(nlls).sum()
        ppl = torch.exp(total_nll / n_tokens)

        return ppl.item()

    def _load_dataset_text(self, dataset: str, split: str) -> str:
        """Load text from a dataset."""
        if dataset == "wikitext2":
            return self._load_wikitext2(split)
        elif dataset == "c4":
            return self._load_c4(split)
        else:
            # Assume it's a raw text string
            return dataset

    def _load_wikitext2(self, split: str) -> str:
        """Load WikiText-2 dataset."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "datasets is required for evaluation. "
                "Install with: pip install tropical-pruning[llm]"
            )

        # Map split names
        split_map = {
            "test": "test",
            "validation": "validation",
            "val": "validation",
            "train": "train",
        }
        split = split_map.get(split, split)

        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        return "\n\n".join(dataset["text"])

    def _load_c4(self, split: str, max_samples: int = 1000) -> str:
        """Load C4 dataset (subset for evaluation)."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "datasets is required for evaluation. "
                "Install with: pip install tropical-pruning[llm]"
            )

        # Use validation split for testing
        actual_split = "validation" if split in ["test", "val", "validation"] else "train"

        dataset = load_dataset(
            "allenai/c4",
            "en",
            split=actual_split,
            streaming=True,
            trust_remote_code=True,
        )

        texts = []
        for i, item in enumerate(dataset):
            if i >= max_samples:
                break
            texts.append(item["text"])

        return "\n\n".join(texts)

    def compare_perplexity(
        self,
        baseline_model: nn.Module,
        dataset: str = "wikitext2",
        split: str = "test",
        max_length: int = 2048,
        show_progress: bool = True,
    ) -> Dict[str, float]:
        """
        Compare perplexity between this model and a baseline.

        Args:
            baseline_model: The baseline (unpruned) model.
            dataset: Dataset to evaluate on.
            split: Dataset split.
            max_length: Maximum context length.
            show_progress: Whether to show progress bars.

        Returns:
            Dictionary with perplexity values and relative change.
        """
        # Evaluate baseline
        baseline_evaluator = LLMEvaluator(baseline_model, self.tokenizer, self.device)
        baseline_ppl = baseline_evaluator.compute_perplexity(
            dataset=dataset,
            split=split,
            max_length=max_length,
            show_progress=show_progress,
        )

        # Evaluate current model
        current_ppl = self.compute_perplexity(
            dataset=dataset,
            split=split,
            max_length=max_length,
            show_progress=show_progress,
        )

        return {
            "baseline_perplexity": baseline_ppl,
            "pruned_perplexity": current_ppl,
            "perplexity_increase": current_ppl - baseline_ppl,
            "relative_increase_pct": (current_ppl - baseline_ppl) / baseline_ppl * 100,
        }


def quick_perplexity(
    model: nn.Module,
    tokenizer: Any,
    text: Optional[str] = None,
    dataset: str = "wikitext2",
    max_length: int = 2048,
) -> float:
    """
    Quick perplexity evaluation.

    Args:
        model: The LLM to evaluate.
        tokenizer: HuggingFace tokenizer.
        text: Optional custom text. If provided, dataset is ignored.
        dataset: Dataset to evaluate on (if text not provided).
        max_length: Maximum context length.

    Returns:
        Perplexity score.
    """
    evaluator = LLMEvaluator(model, tokenizer)

    if text is not None:
        return evaluator.compute_perplexity(
            dataset=text,
            max_length=max_length,
            show_progress=False,
        )

    return evaluator.compute_perplexity(
        dataset=dataset,
        max_length=max_length,
        show_progress=False,
    )
