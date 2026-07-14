"""
Static constants for LLM providers.

Provider names match the prefix used in model strings (e.g. "openai/gpt-4o").
"""

from __future__ import annotations


class Provider:
    GEMINI = "gemini"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    TOGETHER = "together"
    FIREWORKS = "fireworks"
    DEEPSEEK = "deepseek"
    QWEN = "qwen"
    OLLAMA = "ollama"
    VLLM = "vllm"


# Maps provider prefix → env var name for API-key-based providers.
PROVIDERS_WITH_API_KEY: dict[str, str] = {
    Provider.GEMINI: "GEMINI_API_KEY",
    Provider.OPENAI: "OPENAI_API_KEY",
    Provider.ANTHROPIC: "ANTHROPIC_API_KEY",
    Provider.TOGETHER: "TOGETHER_API_KEY",
    Provider.FIREWORKS: "FIREWORKS_API_KEY",
    Provider.DEEPSEEK: "DEEPSEEK_API_KEY",
    Provider.QWEN: "DASHSCOPE_API_KEY",
}

# Maps provider prefix → env var name for host-based providers.
PROVIDERS_WITH_HOST: dict[str, str] = {
    Provider.VLLM: "VLLM_HOST",
    Provider.OLLAMA: "OLLAMA_HOST",
}

# Maps provider prefix → base URL for OpenAI-compatible endpoints.
PROVIDER_BASE_URLS: dict[str, str] = {
    Provider.ANTHROPIC: "https://api.anthropic.com/v1/",
    Provider.GEMINI: "https://generativelanguage.googleapis.com/v1beta/openai/",
    Provider.TOGETHER: "https://api.together.xyz/v1/",
    Provider.FIREWORKS: "https://api.fireworks.ai/inference/v1/",
    Provider.DEEPSEEK: "https://api.deepseek.com/v1/",
    Provider.QWEN: "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    Provider.OLLAMA: "http://localhost:11434/v1/",
}
