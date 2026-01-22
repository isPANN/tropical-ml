"""
Model Loading: Utilities for loading LLM models and detecting architectures.

Supports LLaMA, Mistral, and other HuggingFace models with SwiGLU-based FFN layers.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn


class LLMArchitecture(Enum):
    """Supported LLM architectures."""
    LLAMA = "llama"
    MISTRAL = "mistral"
    QWEN = "qwen"
    QWEN2 = "qwen2"
    PHI = "phi"
    GEMMA = "gemma"
    UNKNOWN = "unknown"


@dataclass
class FFNLayerInfo:
    """Information about FFN layers in a transformer block."""
    layer_idx: int
    gate_proj_name: str
    up_proj_name: str
    down_proj_name: str
    intermediate_size: int
    hidden_size: int


def detect_architecture(model: nn.Module) -> LLMArchitecture:
    """
    Detect the LLM architecture from the model.

    Args:
        model: The loaded model.

    Returns:
        Detected architecture type.
    """
    model_type = getattr(model.config, "model_type", "").lower()

    architecture_map = {
        "llama": LLMArchitecture.LLAMA,
        "mistral": LLMArchitecture.MISTRAL,
        "qwen": LLMArchitecture.QWEN,
        "qwen2": LLMArchitecture.QWEN2,
        "phi": LLMArchitecture.PHI,
        "gemma": LLMArchitecture.GEMMA,
    }

    return architecture_map.get(model_type, LLMArchitecture.UNKNOWN)


def get_ffn_layer_pattern(architecture: LLMArchitecture) -> Dict[str, str]:
    """
    Get the naming pattern for FFN layers based on architecture.

    Returns:
        Dictionary with keys 'gate', 'up', 'down' mapping to layer name patterns.
        Use {layer_idx} as placeholder for layer number.
    """
    patterns = {
        LLMArchitecture.LLAMA: {
            "gate": "model.layers.{layer_idx}.mlp.gate_proj",
            "up": "model.layers.{layer_idx}.mlp.up_proj",
            "down": "model.layers.{layer_idx}.mlp.down_proj",
        },
        LLMArchitecture.MISTRAL: {
            "gate": "model.layers.{layer_idx}.mlp.gate_proj",
            "up": "model.layers.{layer_idx}.mlp.up_proj",
            "down": "model.layers.{layer_idx}.mlp.down_proj",
        },
        LLMArchitecture.QWEN: {
            "gate": "transformer.h.{layer_idx}.mlp.gate_proj",
            "up": "transformer.h.{layer_idx}.mlp.up_proj",
            "down": "transformer.h.{layer_idx}.mlp.down_proj",
        },
        LLMArchitecture.QWEN2: {
            "gate": "model.layers.{layer_idx}.mlp.gate_proj",
            "up": "model.layers.{layer_idx}.mlp.up_proj",
            "down": "model.layers.{layer_idx}.mlp.down_proj",
        },
        LLMArchitecture.PHI: {
            "gate": "model.layers.{layer_idx}.mlp.gate_up_proj",  # Phi combines gate+up
            "up": "model.layers.{layer_idx}.mlp.gate_up_proj",
            "down": "model.layers.{layer_idx}.mlp.down_proj",
        },
        LLMArchitecture.GEMMA: {
            "gate": "model.layers.{layer_idx}.mlp.gate_proj",
            "up": "model.layers.{layer_idx}.mlp.up_proj",
            "down": "model.layers.{layer_idx}.mlp.down_proj",
        },
    }

    return patterns.get(architecture, patterns[LLMArchitecture.LLAMA])


def get_ffn_layer_names(
    model: nn.Module,
    architecture: Optional[LLMArchitecture] = None,
) -> List[FFNLayerInfo]:
    """
    Get information about all FFN layers in the model.

    Args:
        model: The loaded model.
        architecture: Optional architecture type. If None, auto-detect.

    Returns:
        List of FFNLayerInfo for each transformer block.
    """
    if architecture is None:
        architecture = detect_architecture(model)

    if architecture == LLMArchitecture.UNKNOWN:
        # Try to auto-detect by searching for common patterns
        return _detect_ffn_layers_heuristic(model)

    pattern = get_ffn_layer_pattern(architecture)
    num_layers = getattr(model.config, "num_hidden_layers", None)

    if num_layers is None:
        # Try alternative config attribute names
        num_layers = getattr(model.config, "n_layer", None)
        if num_layers is None:
            num_layers = getattr(model.config, "num_layers", None)

    if num_layers is None:
        return _detect_ffn_layers_heuristic(model)

    layers = []
    modules_dict = dict(model.named_modules())

    for layer_idx in range(num_layers):
        gate_name = pattern["gate"].format(layer_idx=layer_idx)
        up_name = pattern["up"].format(layer_idx=layer_idx)
        down_name = pattern["down"].format(layer_idx=layer_idx)

        # Verify layers exist
        if down_name not in modules_dict:
            continue

        down_module = modules_dict[down_name]
        if not isinstance(down_module, nn.Linear):
            continue

        # Get dimensions from down_proj: (hidden, intermediate)
        intermediate_size = down_module.in_features
        hidden_size = down_module.out_features

        layers.append(FFNLayerInfo(
            layer_idx=layer_idx,
            gate_proj_name=gate_name,
            up_proj_name=up_name,
            down_proj_name=down_name,
            intermediate_size=intermediate_size,
            hidden_size=hidden_size,
        ))

    return layers


def _detect_ffn_layers_heuristic(model: nn.Module) -> List[FFNLayerInfo]:
    """
    Heuristically detect FFN layers by looking for common naming patterns.
    """
    layers = []
    modules_dict = dict(model.named_modules())

    # Common FFN layer name patterns
    down_patterns = ["down_proj", "fc2", "w2", "dense_4h_to_h"]
    gate_patterns = ["gate_proj", "gate", "w1"]
    up_patterns = ["up_proj", "fc1", "w3", "dense_h_to_4h"]

    # Find all down_proj layers and infer layer structure
    layer_groups: Dict[str, Dict[str, str]] = {}

    for name, module in modules_dict.items():
        if not isinstance(module, nn.Linear):
            continue

        for pattern in down_patterns:
            if pattern in name:
                # Extract layer prefix (e.g., "model.layers.0.mlp")
                parts = name.rsplit(pattern, 1)
                if len(parts) == 2:
                    prefix = parts[0].rstrip(".")
                    if prefix not in layer_groups:
                        layer_groups[prefix] = {}
                    layer_groups[prefix]["down"] = name
                break

    # Now find matching gate and up projections
    for prefix, group in layer_groups.items():
        for name, module in modules_dict.items():
            if not name.startswith(prefix):
                continue
            if not isinstance(module, nn.Linear):
                continue

            for pattern in gate_patterns:
                if pattern in name and "down" not in name and "up" not in name:
                    group["gate"] = name
                    break

            for pattern in up_patterns:
                if pattern in name and "down" not in name and "gate" not in name:
                    group["up"] = name
                    break

    # Build FFNLayerInfo list
    for idx, (prefix, group) in enumerate(sorted(layer_groups.items())):
        if "down" not in group:
            continue

        down_module = modules_dict[group["down"]]

        layers.append(FFNLayerInfo(
            layer_idx=idx,
            gate_proj_name=group.get("gate", ""),
            up_proj_name=group.get("up", ""),
            down_proj_name=group["down"],
            intermediate_size=down_module.in_features,
            hidden_size=down_module.out_features,
        ))

    return layers


def load_model_and_tokenizer(
    model_name_or_path: str,
    device_map: str = "auto",
    torch_dtype: Optional[torch.dtype] = None,
    trust_remote_code: bool = True,
    **kwargs: Any,
) -> Tuple[nn.Module, Any]:
    """
    Load a HuggingFace model and tokenizer.

    Args:
        model_name_or_path: Model identifier or path.
        device_map: Device mapping strategy ("auto", "cpu", "cuda", etc.).
        torch_dtype: Data type for model weights. If None, uses model default.
        trust_remote_code: Whether to trust remote code from HuggingFace Hub.
        **kwargs: Additional arguments passed to from_pretrained.

    Returns:
        Tuple of (model, tokenizer).
    """
    # Support HuggingFace mirror (hf-mirror.com)
    # Set HF_ENDPOINT BEFORE importing transformers to ensure it's used
    import os
    if "HF_ENDPOINT" not in os.environ:
        # Default to hf-mirror.com mirror for better connectivity in China
        # This environment variable is used by huggingface_hub library
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    
    # Also try to configure huggingface_hub directly if available
    try:
        import huggingface_hub
        # Force update endpoint if not already set
        if hasattr(huggingface_hub, 'constants'):
            # Update the endpoint in huggingface_hub
            huggingface_hub.constants.ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
    except (ImportError, AttributeError):
        pass
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        raise ImportError(
            "transformers is required for LLM support. "
            "Install with: pip install tropical-pruning[llm]"
        )

    # Determine dtype
    if torch_dtype is None:
        if torch.cuda.is_available():
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
    )

    # Ensure padding token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Handle quantization if requested
    load_kwargs = kwargs.copy()
    load_in_8bit = load_kwargs.pop("load_in_8bit", False)
    load_in_4bit = load_kwargs.pop("load_in_4bit", False)
    low_cpu_mem_usage = load_kwargs.pop("low_cpu_mem_usage", True)
    
    # Apply quantization if requested
    if load_in_8bit or load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
            if load_in_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch_dtype,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            else:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            load_kwargs["quantization_config"] = quantization_config
        except ImportError:
            raise ImportError(
                "bitsandbytes is required for quantization. "
                "Install with: pip install bitsandbytes"
            )
    
    # Load model with memory optimizations
    # If device_map is "auto" and CUDA is available, use balanced memory allocation
    if device_map == "auto" and torch.cuda.is_available():
        # Try to use max_memory to prevent OOM
        try:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            # Reserve some memory for other operations
            max_memory = {0: f"{int(gpu_memory * 0.9)}GiB"}
            load_kwargs["max_memory"] = max_memory
        except Exception:
            pass  # If max_memory fails, continue without it
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=low_cpu_mem_usage,
        **load_kwargs,
    )

    model.eval()

    return model, tokenizer


def get_model_device(model: nn.Module) -> torch.device:
    """Get the device of the model's parameters."""
    return next(model.parameters()).device


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Count model parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())
