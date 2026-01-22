"""
Calibration Datasets: Utilities for creating calibration data loaders.

Provides standardized calibration data from common datasets like WikiText-2
for collecting winner statistics during pruning.
"""

from typing import Any, List, Optional, Union
import torch
from torch.utils.data import DataLoader, Dataset


class TokenizedDataset(Dataset):
    """A simple dataset wrapping tokenized sequences."""

    def __init__(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        Initialize the dataset.

        Args:
            input_ids: Tensor of token IDs, shape (num_samples, seq_length).
            attention_mask: Optional attention mask, shape (num_samples, seq_length).
        """
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def __len__(self) -> int:
        return self.input_ids.shape[0]

    def __getitem__(self, idx: int):
        item = {"input_ids": self.input_ids[idx]}
        if self.attention_mask is not None:
            item["attention_mask"] = self.attention_mask[idx]
        return item


class CalibrationDataset:
    """
    Factory for creating calibration data loaders from standard datasets.

    Example:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        >>> dataloader = CalibrationDataset.from_wikitext2(tokenizer, num_samples=128)
        >>>
        >>> for batch in dataloader:
        ...     input_ids = batch["input_ids"]
        ...     attention_mask = batch["attention_mask"]
    """

    @staticmethod
    def from_wikitext2(
        tokenizer: Any,
        num_samples: int = 128,
        seq_length: int = 2048,
        batch_size: int = 1,
        split: str = "train",
        seed: int = 42,
    ) -> DataLoader:
        """
        Create a calibration DataLoader from WikiText-2.

        Args:
            tokenizer: HuggingFace tokenizer.
            num_samples: Number of calibration samples to generate.
            seq_length: Sequence length for each sample.
            batch_size: Batch size for the DataLoader.
            split: Dataset split to use ("train", "validation", "test").
            seed: Random seed for reproducibility.

        Returns:
            DataLoader yielding batches with "input_ids" and "attention_mask".
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "datasets is required for calibration data. "
                "Install with: pip install tropical-pruning[llm]"
            )

        # Load WikiText-2
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)

        # Concatenate all text
        text = "\n\n".join(dataset["text"])

        # Tokenize
        tokenized = tokenizer(
            text,
            return_tensors="pt",
            truncation=False,
            add_special_tokens=False,
        )
        all_ids = tokenized["input_ids"].squeeze(0)

        # Create non-overlapping chunks
        return CalibrationDataset._create_chunks(
            all_ids,
            tokenizer,
            num_samples=num_samples,
            seq_length=seq_length,
            batch_size=batch_size,
            seed=seed,
        )

    @staticmethod
    def from_c4(
        tokenizer: Any,
        num_samples: int = 128,
        seq_length: int = 2048,
        batch_size: int = 1,
        seed: int = 42,
    ) -> DataLoader:
        """
        Create a calibration DataLoader from C4 dataset.

        Args:
            tokenizer: HuggingFace tokenizer.
            num_samples: Number of calibration samples to generate.
            seq_length: Sequence length for each sample.
            batch_size: Batch size for the DataLoader.
            seed: Random seed for reproducibility.

        Returns:
            DataLoader yielding batches with "input_ids" and "attention_mask".
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "datasets is required for calibration data. "
                "Install with: pip install tropical-pruning[llm]"
            )

        # Load C4 (streaming to avoid downloading entire dataset)
        dataset = load_dataset(
            "allenai/c4",
            "en",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )

        # Collect enough text
        texts = []
        total_tokens = 0
        target_tokens = num_samples * seq_length * 2  # Extra buffer

        for item in dataset:
            texts.append(item["text"])
            total_tokens += len(item["text"].split())  # Rough estimate
            if total_tokens >= target_tokens:
                break

        text = "\n\n".join(texts)

        # Tokenize
        tokenized = tokenizer(
            text,
            return_tensors="pt",
            truncation=False,
            add_special_tokens=False,
        )
        all_ids = tokenized["input_ids"].squeeze(0)

        return CalibrationDataset._create_chunks(
            all_ids,
            tokenizer,
            num_samples=num_samples,
            seq_length=seq_length,
            batch_size=batch_size,
            seed=seed,
        )

    @staticmethod
    def from_text(
        text: Union[str, List[str]],
        tokenizer: Any,
        num_samples: int = 128,
        seq_length: int = 2048,
        batch_size: int = 1,
        seed: int = 42,
    ) -> DataLoader:
        """
        Create a calibration DataLoader from custom text.

        Args:
            text: Text string or list of strings.
            tokenizer: HuggingFace tokenizer.
            num_samples: Number of calibration samples to generate.
            seq_length: Sequence length for each sample.
            batch_size: Batch size for the DataLoader.
            seed: Random seed for reproducibility.

        Returns:
            DataLoader yielding batches with "input_ids" and "attention_mask".
        """
        if isinstance(text, list):
            text = "\n\n".join(text)

        # Tokenize
        tokenized = tokenizer(
            text,
            return_tensors="pt",
            truncation=False,
            add_special_tokens=False,
        )
        all_ids = tokenized["input_ids"].squeeze(0)

        return CalibrationDataset._create_chunks(
            all_ids,
            tokenizer,
            num_samples=num_samples,
            seq_length=seq_length,
            batch_size=batch_size,
            seed=seed,
        )

    @staticmethod
    def _create_chunks(
        all_ids: torch.Tensor,
        tokenizer: Any,
        num_samples: int,
        seq_length: int,
        batch_size: int,
        seed: int,
    ) -> DataLoader:
        """Create non-overlapping chunks from tokenized text."""
        # Set random seed
        torch.manual_seed(seed)

        total_tokens = all_ids.numel()
        num_possible_chunks = total_tokens // seq_length

        if num_possible_chunks < num_samples:
            # Use overlapping chunks if not enough tokens
            stride = max(1, (total_tokens - seq_length) // num_samples)
            starts = torch.arange(0, total_tokens - seq_length, stride)[:num_samples]
        else:
            # Random non-overlapping selection
            chunk_starts = torch.arange(num_possible_chunks) * seq_length
            perm = torch.randperm(num_possible_chunks)[:num_samples]
            starts = chunk_starts[perm]

        # Create chunks
        input_ids_list = []
        for start in starts:
            chunk = all_ids[start : start + seq_length]
            if len(chunk) < seq_length:
                # Pad if necessary
                padding = torch.full(
                    (seq_length - len(chunk),),
                    tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
                chunk = torch.cat([chunk, padding])
            input_ids_list.append(chunk)

        input_ids = torch.stack(input_ids_list)

        # Create attention mask (1 for real tokens, 0 for padding)
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        attention_mask = (input_ids != pad_id).long()

        dataset = TokenizedDataset(input_ids, attention_mask)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)

    @staticmethod
    def from_random(
        tokenizer: Any,
        num_samples: int = 128,
        seq_length: int = 2048,
        batch_size: int = 1,
        seed: int = 42,
    ) -> DataLoader:
        """
        Create a calibration DataLoader with random tokens.

        This is primarily useful for testing and debugging.

        Args:
            tokenizer: HuggingFace tokenizer.
            num_samples: Number of calibration samples to generate.
            seq_length: Sequence length for each sample.
            batch_size: Batch size for the DataLoader.
            seed: Random seed for reproducibility.

        Returns:
            DataLoader yielding batches with "input_ids" and "attention_mask".
        """
        torch.manual_seed(seed)

        vocab_size = tokenizer.vocab_size
        input_ids = torch.randint(0, vocab_size, (num_samples, seq_length))
        attention_mask = torch.ones(num_samples, seq_length, dtype=torch.long)

        dataset = TokenizedDataset(input_ids, attention_mask)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)
