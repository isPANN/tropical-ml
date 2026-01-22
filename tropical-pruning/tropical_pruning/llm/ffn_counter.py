"""
FFN Winner Counter: Track winner statistics for LLM FFN layers.

The key insight is that down_proj is a standard linear layer where tropical
winner counting applies directly. The intermediate neuron that "wins" the
argmax in the tropical sense indicates which FFN neurons are geometrically
important.

    output[h] = max_j(W_down[h,j] + intermediate[j])
    argmax_j → which intermediate neuron "wins"
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from tqdm import tqdm

from tropical_pruning.llm.loader import (
    FFNLayerInfo,
    get_ffn_layer_names,
    detect_architecture,
)


@dataclass
class FFNStatistics:
    """Statistics collected from tropical forward passes for an FFN layer."""

    layer_idx: int
    # Shape: (intermediate_size,) - how many times each intermediate neuron won
    intermediate_winner_count: torch.Tensor
    # Total number of output positions processed (batch * seq_len * hidden_size)
    intermediate_total_positions: int
    # Shape: (intermediate_size,) - sum of margins when winning
    intermediate_margin_sum: Optional[torch.Tensor] = None
    # Shape: (intermediate_size,) - count for margin averaging
    intermediate_margin_count: Optional[torch.Tensor] = None

    @property
    def intermediate_size(self) -> int:
        """Number of intermediate neurons tracked."""
        return self.intermediate_winner_count.numel()

    @property
    def winner_frequency(self) -> torch.Tensor:
        """Normalized winner count (count / total_positions)."""
        return self.intermediate_winner_count.float() / max(self.intermediate_total_positions, 1)

    @property
    def average_margin(self) -> Optional[torch.Tensor]:
        """Average margin when winning. Higher = more confident importance."""
        if self.intermediate_margin_sum is None or self.intermediate_margin_count is None:
            return None
        return self.intermediate_margin_sum / self.intermediate_margin_count.clamp(min=1)

    def to(self, device: torch.device) -> "FFNStatistics":
        """Move statistics to specified device."""
        return FFNStatistics(
            layer_idx=self.layer_idx,
            intermediate_winner_count=self.intermediate_winner_count.to(device),
            intermediate_total_positions=self.intermediate_total_positions,
            intermediate_margin_sum=(
                self.intermediate_margin_sum.to(device)
                if self.intermediate_margin_sum is not None
                else None
            ),
            intermediate_margin_count=(
                self.intermediate_margin_count.to(device)
                if self.intermediate_margin_count is not None
                else None
            ),
        )


class _FFNLayerCounter:
    """Internal counter for a single FFN layer's down_proj."""

    def __init__(
        self,
        layer_idx: int,
        intermediate_size: int,
        track_margin: bool = True,
        device: Optional[torch.device] = None,
    ):
        self.layer_idx = layer_idx
        self.intermediate_size = intermediate_size
        self.track_margin = track_margin
        self.device = device or torch.device("cpu")

        self.winner_count = torch.zeros(
            intermediate_size, dtype=torch.long, device=self.device
        )
        self.total_positions = 0

        if track_margin:
            self.margin_sum = torch.zeros(
                intermediate_size, dtype=torch.float32, device=self.device
            )
            self.margin_count = torch.zeros(
                intermediate_size, dtype=torch.long, device=self.device
            )
        else:
            self.margin_sum = None
            self.margin_count = None

    def update(
        self,
        argmax_indices: torch.Tensor,
        margin: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Update counters with new argmax indices.

        Args:
            argmax_indices: Tensor of argmax indices, shape (..., hidden_size).
                           Values are in [0, intermediate_size).
            margin: Optional tensor of margins, shape (..., hidden_size).
        """
        # Flatten indices
        flat_indices = argmax_indices.flatten().to(self.device)

        # Clamp indices to valid range (safety check)
        flat_indices = flat_indices.clamp(0, self.intermediate_size - 1)

        self.total_positions += flat_indices.numel()

        # Count winners
        ones = torch.ones_like(flat_indices, dtype=torch.long)
        self.winner_count.scatter_add_(0, flat_indices, ones)

        # Track margins if enabled
        if self.track_margin and margin is not None:
            flat_margin = margin.flatten().to(self.device)
            # Only count valid margins (not inf)
            valid_mask = torch.isfinite(flat_margin)
            if valid_mask.any():
                valid_indices = flat_indices[valid_mask]
                valid_margins = flat_margin[valid_mask]

                self.margin_sum.scatter_add_(0, valid_indices, valid_margins)
                margin_ones = torch.ones_like(valid_indices, dtype=torch.long)
                self.margin_count.scatter_add_(0, valid_indices, margin_ones)

    def get_statistics(self) -> FFNStatistics:
        """Return FFNStatistics for this layer."""
        return FFNStatistics(
            layer_idx=self.layer_idx,
            intermediate_winner_count=self.winner_count.clone(),
            intermediate_total_positions=self.total_positions,
            intermediate_margin_sum=(
                self.margin_sum.clone() if self.margin_sum is not None else None
            ),
            intermediate_margin_count=(
                self.margin_count.clone() if self.margin_count is not None else None
            ),
        )

    def reset(self) -> None:
        """Reset counters to zero."""
        self.winner_count.zero_()
        self.total_positions = 0
        if self.margin_sum is not None:
            self.margin_sum.zero_()
        if self.margin_count is not None:
            self.margin_count.zero_()


class FFNWinnerCounter:
    """
    Track winner statistics for LLM FFN layers.

    This class hooks into down_proj layers of FFN blocks and tracks which
    intermediate neurons "win" the tropical argmax. These statistics directly
    indicate the geometric importance of intermediate neurons.

    Example:
        >>> model, tokenizer = load_model_and_tokenizer("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        >>> counter = FFNWinnerCounter(model)
        >>> for batch in calibration_loader:
        ...     counter.forward(batch)
        >>> stats = counter.get_statistics()
        >>> print(stats[0].winner_frequency)  # Layer 0 FFN statistics
    """

    def __init__(
        self,
        model: nn.Module,
        layer_indices: Optional[List[int]] = None,
        track_margin: bool = True,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the FFNWinnerCounter.

        Args:
            model: The LLM model to analyze.
            layer_indices: Specific layer indices to track. If None, tracks all layers.
            track_margin: Whether to track winner margins (gap to 2nd place).
            device: Device for statistics storage. If None, uses model's device.
        """
        self.model = model
        self.track_margin = track_margin
        self.device = device or next(model.parameters()).device

        # Get FFN layer information
        self.ffn_layers = get_ffn_layer_names(model)

        if layer_indices is not None:
            self.ffn_layers = [l for l in self.ffn_layers if l.layer_idx in layer_indices]

        # Initialize counters for each FFN layer
        self._counters: Dict[int, _FFNLayerCounter] = {}
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []

        self._setup_hooks()

    def _setup_hooks(self) -> None:
        """Register forward hooks on down_proj layers."""
        modules_dict = dict(self.model.named_modules())

        for layer_info in self.ffn_layers:
            down_proj_name = layer_info.down_proj_name

            if down_proj_name not in modules_dict:
                continue

            module = modules_dict[down_proj_name]
            if not isinstance(module, nn.Linear):
                continue

            # Initialize counter
            self._counters[layer_info.layer_idx] = _FFNLayerCounter(
                layer_idx=layer_info.layer_idx,
                intermediate_size=layer_info.intermediate_size,
                track_margin=self.track_margin,
                device=self.device,
            )

            # Register hook
            hook = module.register_forward_hook(
                self._create_hook(layer_info.layer_idx, module)
            )
            self._hooks.append(hook)

    def _create_hook(self, layer_idx: int, module: nn.Linear):
        """Create a forward hook for the down_proj layer."""

        def hook(
            module: nn.Linear,
            input: Tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ):
            # input[0] shape: (batch, seq_len, intermediate_size)
            # weight shape: (hidden_size, intermediate_size)
            # output shape: (batch, seq_len, hidden_size)
            x = input[0]
            weight = module.weight

            # Compute tropical argmax: which intermediate neuron wins for each output
            argmax_indices, margin = self._tropical_argmax(x, weight)

            # Update counter
            self._counters[layer_idx].update(argmax_indices, margin)

        return hook

    def _tropical_argmax(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute tropical argmax for down_proj operation.

        In tropical algebra: output[h] = max_j(W[h,j] + x[j])
        We compute argmax_j for each output position.

        Args:
            x: Input tensor, shape (..., intermediate_size)
            weight: Weight matrix, shape (hidden_size, intermediate_size)

        Returns:
            argmax_indices: Shape (..., hidden_size), values in [0, intermediate_size)
            margin: Optional shape (..., hidden_size), gap to 2nd place winner
        """
        # Tropical max-plus: x + weight.T (broadcast over batch/seq dims)
        # x: (..., intermediate_size), weight.T: (intermediate_size, hidden_size)
        # Result: (..., hidden_size) after max over intermediate dim

        # Expand for broadcasting
        # x: (..., intermediate_size) -> (..., intermediate_size, 1)
        # weight.T: (intermediate_size, hidden_size)
        x_expanded = x.unsqueeze(-1)  # (..., intermediate_size, 1)
        weight_t = weight.t()  # (intermediate_size, hidden_size)

        # Tropical sum: x + W^T
        tropical_sum = x_expanded + weight_t  # (..., intermediate_size, hidden_size)

        # Argmax over intermediate dimension (dim=-2)
        max_values, argmax_indices = tropical_sum.max(dim=-2)  # (..., hidden_size)

        # Compute margin if needed
        margin = None
        if self.track_margin:
            margin = self._compute_margin(tropical_sum, argmax_indices, max_values)

        return argmax_indices, margin

    def _compute_margin(
        self,
        tropical_sum: torch.Tensor,
        argmax_indices: torch.Tensor,
        max_values: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the margin (gap to 2nd place) for each output position.

        Args:
            tropical_sum: Shape (..., intermediate_size, hidden_size)
            argmax_indices: Shape (..., hidden_size)
            max_values: Shape (..., hidden_size)

        Returns:
            Margin tensor, shape (..., hidden_size)
        """
        # Mask out the max and find second max
        # argmax_indices: (..., hidden_size) -> (..., 1, hidden_size) for scatter
        mask = torch.zeros_like(tropical_sum, dtype=torch.bool)
        mask.scatter_(-2, argmax_indices.unsqueeze(-2), True)

        # Set max positions to -inf and find second max
        masked = tropical_sum.masked_fill(mask, float("-inf"))
        second_max = masked.max(dim=-2).values  # (..., hidden_size)

        margin = max_values - second_max
        return margin

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Run forward pass and collect winner statistics.

        Args:
            input_ids: Input token IDs, shape (batch, seq_len).
            attention_mask: Optional attention mask.
            **kwargs: Additional arguments passed to model.

        Returns:
            Model output.
        """
        # Get model device - handle both HuggingFace models and standard nn.Module
        if hasattr(self.model, "device"):
            model_device = self.model.device
        else:
            model_device = next(self.model.parameters()).device

        input_ids = input_ids.to(model_device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(model_device)

        with torch.no_grad():
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs,
            )

    def collect(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_batches: Optional[int] = None,
        show_progress: bool = True,
    ) -> Dict[int, FFNStatistics]:
        """
        Collect statistics from a dataloader.

        Args:
            dataloader: DataLoader providing calibration data with input_ids.
            num_batches: Maximum number of batches to process. If None, processes all.
            show_progress: Whether to show a progress bar.

        Returns:
            Dictionary mapping layer indices to their FFNStatistics.
        """
        self.model.eval()

        iterator = dataloader
        if show_progress:
            total = num_batches if num_batches else len(dataloader)
            iterator = tqdm(dataloader, total=total, desc="Collecting FFN statistics")

        for i, batch in enumerate(iterator):
            if num_batches is not None and i >= num_batches:
                break

            # Handle different batch formats
            if isinstance(batch, dict):
                input_ids = batch.get("input_ids")
                attention_mask = batch.get("attention_mask", None)
            elif isinstance(batch, (list, tuple)):
                input_ids = batch[0]
                attention_mask = batch[1] if len(batch) > 1 else None
            else:
                input_ids = batch
                attention_mask = None

            self.forward(input_ids, attention_mask)

        return self.get_statistics()

    def get_statistics(self) -> Dict[int, FFNStatistics]:
        """
        Get collected FFN statistics for all layers.

        Returns:
            Dictionary mapping layer indices to their FFNStatistics.
        """
        return {idx: counter.get_statistics() for idx, counter in self._counters.items()}

    def reset(self) -> None:
        """Reset all counters to zero."""
        for counter in self._counters.values():
            counter.reset()

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def __del__(self):
        """Clean up hooks on deletion."""
        self.remove_hooks()

    def analyze(self) -> Dict[int, Dict]:
        """
        Analyze winner statistics for each FFN layer.

        Returns:
            Dictionary with analysis per layer.
        """
        analysis = {}
        for layer_idx, stats in self.get_statistics().items():
            freq = stats.winner_frequency

            analysis[layer_idx] = {
                "intermediate_size": stats.intermediate_size,
                "total_positions": stats.intermediate_total_positions,
                "never_win": (freq == 0).sum().item(),
                "never_win_pct": ((freq == 0).sum().item() / stats.intermediate_size * 100),
                "rarely_win": (freq < 0.001).sum().item(),
                "rarely_win_pct": ((freq < 0.001).sum().item() / stats.intermediate_size * 100),
                "frequently_win": (freq > 0.01).sum().item(),
                "frequently_win_pct": ((freq > 0.01).sum().item() / stats.intermediate_size * 100),
                "min_frequency": freq.min().item(),
                "max_frequency": freq.max().item(),
                "mean_frequency": freq.mean().item(),
                "std_frequency": freq.std().item(),
            }

            if stats.average_margin is not None:
                margin = stats.average_margin
                valid_margins = margin[margin > 0]
                analysis[layer_idx]["mean_margin"] = (
                    valid_margins.mean().item() if len(valid_margins) > 0 else 0
                )
                analysis[layer_idx]["max_margin"] = margin.max().item()

        return analysis
