"""
Open-weight model registry for VRAM estimation.

Used by scale_and_cost_projection to determine how much GPU memory is needed to run a
self-hosted model and whether it fits on a given GPU instance.

Param counts and typical precision sourced from published model cards.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelSizeOverride:
    param_count_b: float
    precision: str = "fp16"  # "fp16" | "int8" | "int4"
    num_layers: int | None = None  # inferred from param_count_b when None


@dataclass
class VramEstimate:
    model_id: str
    param_count_b: float
    precision: str  # "fp16" | "int8" | "int4"
    weights_gb: float
    kv_cache_gb_p95: float
    total_vram_gb: float
    source: str  # "registry" | "user_override"


# Registry: model_id prefix → (param_count_b, default_precision, num_layers)
# Keyed by the part after "vllm/" or the full model_string.
# Num layers from public model cards (approximate for quant variants).
# Source for each entry: https://huggingface.co/api/models
_MODEL_REGISTRY: dict[str, tuple[float, str, int]] = {
    # Llama 3 family
    "meta-llama/Llama-3-8B": (8.0, "fp16", 32),
    "meta-llama/Llama-3-8B-Instruct": (8.0, "fp16", 32),
    "meta-llama/Llama-3-70B": (70.0, "fp16", 80),
    "meta-llama/Llama-3-70B-Instruct": (70.0, "fp16", 80),
    "meta-llama/Llama-3.1-8B": (8.0, "fp16", 32),
    "meta-llama/Llama-3.1-8B-Instruct": (8.0, "fp16", 32),
    "meta-llama/Llama-3.1-70B": (70.0, "fp16", 80),
    "meta-llama/Llama-3.1-70B-Instruct": (70.0, "fp16", 80),
    "meta-llama/Llama-3.1-405B": (405.0, "fp16", 126),
    "meta-llama/Llama-3.2-3B": (3.0, "fp16", 28),
    "meta-llama/Llama-3.2-3B-Instruct": (3.0, "fp16", 28),
    "meta-llama/Llama-3.2-11B": (11.0, "fp16", 32),
    "meta-llama/Llama-3.2-11B-Vision-Instruct": (11.0, "fp16", 32),
    "meta-llama/Llama-3.3-70B-Instruct": (70.0, "fp16", 80),
    # Qwen family
    "Qwen/Qwen2.5-0.5B-Instruct": (0.5, "fp16", 24),
    "Qwen/Qwen2.5-1.5B-Instruct": (1.5, "fp16", 28),
    "Qwen/Qwen2.5-7B-Instruct": (7.0, "fp16", 28),
    "Qwen/Qwen2.5-14B-Instruct": (14.0, "fp16", 48),
    "Qwen/Qwen2.5-32B-Instruct": (32.0, "fp16", 64),
    "Qwen/Qwen2.5-72B-Instruct": (72.0, "fp16", 80),
    "Qwen/Qwen2.5-Coder-7B-Instruct": (7.0, "fp16", 28),
    "Qwen/Qwen2.5-Coder-32B-Instruct": (32.0, "fp16", 64),
    # Mistral / Mixtral family
    "mistralai/Mistral-7B-Instruct-v0.3": (7.0, "fp16", 32),
    "mistralai/Mistral-Nemo-Instruct-2407": (12.0, "fp16", 40),
    "mistralai/Mixtral-8x7B-Instruct-v0.1": (46.7, "fp16", 32),
    "mistralai/Mixtral-8x22B-Instruct-v0.1": (141.0, "fp16", 56),
    # Gemma family
    "google/gemma-2-2b-it": (2.0, "fp16", 26),
    "google/gemma-2-9b-it": (9.0, "fp16", 42),
    "google/gemma-2-27b-it": (27.0, "fp16", 46),
    # Phi family
    "microsoft/phi-3-mini-4k-instruct": (3.8, "fp16", 32),
    "microsoft/phi-3-small-8k-instruct": (7.0, "fp16", 32),
    "microsoft/phi-3-medium-4k-instruct": (14.0, "fp16", 40),
    "microsoft/phi-4": (14.0, "fp16", 40),
    # Ollama (local example models)
    "llava": (7.0, "fp16", 32),  # LLaVA-1.5-7B (default ollama/llava tag)
    "minicpm-v": (8.0, "fp16", 28),  # MiniCPM-V 2.6: Qwen2-7B + SigLIP-400M
    # DeepSeek
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": (7.0, "fp16", 28),
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": (14.0, "fp16", 48),
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B": (70.0, "fp16", 80),
}
