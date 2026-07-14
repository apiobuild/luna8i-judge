"""
LLM provider adapters — shared types only.

Provider implementations live in their own modules:
  openai_compatible_adapter.py  — all OpenAI-compatible providers
  anthropic_adapter.py          — Anthropic native SDK (stub)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, TypedDict, runtime_checkable


@dataclass
class GenerationConfig:
    temperature: float = 0.0
    top_p: float = 1.0
    response_format: dict[str, Any] | None = None
    extra_params: dict[str, Any] | None = None


@dataclass
class ModelResponse:
    content: str
    input_tokens: int
    output_tokens: int


class ContentPart(TypedDict, total=False):
    # OpenAI-compatible content part shape.
    # type values: "text" | "image_url" | "image" | "audio" | "audio_url" | "video" | "video_url"
    type: str
    text: str
    image_url: dict[str, str]  # {"url": "..."}  — OpenAI image_url format
    audio_url: dict[str, str]  # {"url": "..."}
    video_url: dict[str, str]  # {"url": "..."}


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant"
    content: str | list[ContentPart]


@runtime_checkable
class ModelClient(Protocol):
    def complete(self, messages: list[Message], config: GenerationConfig | None = None) -> ModelResponse: ...
