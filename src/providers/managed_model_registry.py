"""
LLM registry — static list of managed inference providers with TPM ceilings and pricing.

Pricing as of 2026-06. TPM ceilings are approximate tier-1 defaults; actual limits
vary by account tier and region.

Pricing sources:
  Gemini:    https://ai.google.dev/gemini-api/docs/pricing
  OpenAI:    https://platform.openai.com/docs/pricing
  Anthropic: https://www.anthropic.com/pricing#anthropic-api
  DeepSeek:  https://api-docs.deepseek.com/quick_start/pricing
  xAI:       https://docs.x.ai/docs/models
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelPricing:
    input_per_1m: float
    output_per_1m: float


@dataclass
class ModelProvider:
    id: str
    name: str
    tpm_ceiling: int
    pricing: ModelPricing


_PROVIDERS: list[ModelProvider] = [
    # Gemini — https://ai.google.dev/gemini-api/docs/pricing
    ModelProvider(
        id="gemini/gemini-2.5-flash",
        name="Gemini 2.5 Flash",
        tpm_ceiling=4_000_000,
        pricing=ModelPricing(input_per_1m=0.30, output_per_1m=2.50),
    ),
    ModelProvider(
        id="gemini/gemini-2.5-flash-lite",
        name="Gemini 2.5 Flash Lite",
        tpm_ceiling=4_000_000,
        pricing=ModelPricing(input_per_1m=0.10, output_per_1m=0.40),
    ),
    ModelProvider(
        id="gemini/gemini-2.5-pro",
        name="Gemini 2.5 Pro",
        tpm_ceiling=2_000_000,
        pricing=ModelPricing(input_per_1m=1.25, output_per_1m=10.00),
    ),
    ModelProvider(
        id="gemini/gemini-3.1-flash-lite",
        name="Gemini 3.1 Flash Lite",
        tpm_ceiling=4_000_000,
        pricing=ModelPricing(input_per_1m=0.25, output_per_1m=1.50),
    ),
    ModelProvider(
        id="gemini/gemini-3.1-pro-preview",
        name="Gemini 3.1 Pro Preview",
        tpm_ceiling=2_000_000,
        pricing=ModelPricing(input_per_1m=2.00, output_per_1m=12.00),
    ),
    # OpenAI — https://platform.openai.com/docs/pricing
    ModelProvider(
        id="openai/gpt-5.4",
        name="GPT-5.4",
        tpm_ceiling=2_000_000,
        pricing=ModelPricing(input_per_1m=2.50, output_per_1m=15.00),
    ),
    ModelProvider(
        id="openai/gpt-5.4-mini",
        name="GPT-5.4 Mini",
        tpm_ceiling=2_000_000,
        pricing=ModelPricing(input_per_1m=0.75, output_per_1m=4.50),
    ),
    ModelProvider(
        id="openai/gpt-5.4-nano",
        name="GPT-5.4 Nano",
        tpm_ceiling=2_000_000,
        pricing=ModelPricing(input_per_1m=0.20, output_per_1m=1.25),
    ),
    # Anthropic — https://www.anthropic.com/pricing#anthropic-api
    ModelProvider(
        id="anthropic/claude-sonnet-4-6",
        name="Claude Sonnet 4.6",
        tpm_ceiling=1_600_000,
        pricing=ModelPricing(input_per_1m=3.00, output_per_1m=15.00),
    ),
    ModelProvider(
        id="anthropic/claude-haiku-4-5",
        name="Claude Haiku 4.5",
        tpm_ceiling=1_600_000,
        pricing=ModelPricing(input_per_1m=1.00, output_per_1m=5.00),
    ),
    ModelProvider(
        id="anthropic/claude-opus-4-8",
        name="Claude Opus 4.8",
        tpm_ceiling=1_600_000,
        pricing=ModelPricing(input_per_1m=5.00, output_per_1m=25.00),
    ),
    # DeepSeek — https://api-docs.deepseek.com/quick_start/pricing
    # deepseek-chat / deepseek-reasoner deprecated 2026-07-24; use v4 variants
    ModelProvider(
        id="deepseek/deepseek-v4-flash",
        name="DeepSeek V4 Flash",
        tpm_ceiling=1_000_000,
        pricing=ModelPricing(input_per_1m=0.14, output_per_1m=0.28),
    ),
    ModelProvider(
        id="deepseek/deepseek-v4-pro",
        name="DeepSeek V4 Pro",
        tpm_ceiling=1_000_000,
        pricing=ModelPricing(input_per_1m=0.435, output_per_1m=0.87),
    ),
    # Qwen (Alibaba Cloud DashScope) — https://www.alibabacloud.com/help/en/model-studio/getting-started/models
    ModelProvider(
        id="qwen/qwen3.7-plus",
        name="Qwen3.7-Plus",
        tpm_ceiling=1_000_000,
        pricing=ModelPricing(input_per_1m=0.32, output_per_1m=0.032),
    ),
    ModelProvider(
        id="qwen/qwen3.6-flash",
        name="Qwen3.6-Flash",
        tpm_ceiling=1_000_000,
        pricing=ModelPricing(input_per_1m=0.25, output_per_1m=1.5),
    ),
    ModelProvider(
        id="qwen/qwen3.7-max",
        name="Qwen3.7-Max",
        tpm_ceiling=1_000_000,
        pricing=ModelPricing(input_per_1m=1.25, output_per_1m=3.75),
    ),
    ModelProvider(
        id="qwen/qwen3.6-35b-a3b",
        name="Qwen3.6-Open-Source",
        tpm_ceiling=1_000_000,
        pricing=ModelPricing(input_per_1m=0.375, output_per_1m=2.25),
    ),
    # xAI — https://docs.x.ai/docs/models
    ModelProvider(
        id="xai/grok-3",
        name="Grok 3",
        tpm_ceiling=2_000_000,
        pricing=ModelPricing(input_per_1m=3.00, output_per_1m=15.00),
    ),
    ModelProvider(
        id="xai/grok-3-mini",
        name="Grok 3 Mini",
        tpm_ceiling=2_000_000,
        pricing=ModelPricing(input_per_1m=0.30, output_per_1m=0.50),
    ),
]


def get_model_providers() -> list[ModelProvider]:
    return list(_PROVIDERS)


def get_managed_model_providers() -> set[str]:
    return {p.id for p in _PROVIDERS}


def estimate_cost(model_id: str, input_tokens: int, output_tokens: int) -> float:
    providers = {p.id: p for p in _PROVIDERS}
    if model_id not in providers:
        raise ValueError(f"Unknown model: {model_id!r}")
    pricing = providers[model_id].pricing
    return (input_tokens * pricing.input_per_1m + output_tokens * pricing.output_per_1m) / 1_000_000
