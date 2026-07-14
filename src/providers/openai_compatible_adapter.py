from __future__ import annotations

import base64
import logging
import mimetypes
import re

import httpx

from src.providers.adapters import GenerationConfig, Message, ModelResponse
from src.providers.managed_model_provider_constants import PROVIDERS_WITH_API_KEY, Provider

logger = logging.getLogger(__name__)

_RETRY_DELAY_RE = re.compile(r"retry[^\d]*(\d+(?:\.\d+)?)\s*s", re.IGNORECASE)

RATE_LIMIT_MAX_ATTEMPTS: int = 4
RATE_LIMIT_BACKOFF_MULTIPLIER: float = 5.0  # seconds; doubles each attempt
RATE_LIMIT_BACKOFF_MIN: float = 5.0
RATE_LIMIT_BACKOFF_MAX: float = 60.0


class OpenAICompatibleAdapter:
    def __init__(
        self, model: str, api_key: str | None, base_url: str | None = None, provider: str | None = None
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._provider = provider

    def _url_to_data_uri(self, url: str) -> str:
        resp = httpx.get(url, timeout=30.0, follow_redirects=True)
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "image/jpeg").split(";")[0].strip()
        if not content_type:
            ext = url.rsplit(".", 1)[-1].lower()
            content_type = mimetypes.types_map.get(f".{ext}", "image/jpeg")
        encoded = base64.b64encode(resp.content).decode()
        return f"data:{content_type};base64,{encoded}"

    def _adapt_content_for_ollama(self, content: str | list) -> str | list:
        # Ollama's OpenAI-compat endpoint rejects image_url with HTTP URLs — convert to base64 data URIs.
        if not isinstance(content, list):
            return content
        adapted = []
        for part in content:
            if part.get("type") == "image_url":
                url = (part.get("image_url") or {}).get("url", "")
                if url.startswith("http://") or url.startswith("https://"):
                    url = self._url_to_data_uri(url)
                adapted.append({"type": "image_url", "image_url": {"url": url}})
            else:
                adapted.append(part)
        return adapted

    def _adapt_response_format(self, response_format: dict) -> dict:
        # If a raw JSON Schema is passed (has "properties" / "$schema" / type "object"),
        # wrap it as an OpenAI structured-output response_format before provider-specific adaptation.
        is_raw_schema = (
            "properties" in response_format or "$schema" in response_format or response_format.get("type") == "object"
        )
        if is_raw_schema:
            response_format = {
                "type": "json_schema",
                "json_schema": {"name": "output", "schema": response_format, "strict": False},
            }

        # Gemini and Qwen (DashScope) only accept {"type": "json_object"} or {"type": "text"}.
        # Strip full JSON Schema and downgrade to json_object mode.
        if self._provider in (Provider.GEMINI, Provider.QWEN):
            return {"type": "json_object"}
        return response_format

    def complete(self, messages: list[Message], config: GenerationConfig | None = None) -> ModelResponse:
        import openai
        from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

        def _wait(retry_state) -> float:
            exc = retry_state.outcome.exception()
            m = _RETRY_DELAY_RE.search(str(exc)) if exc else None
            if m:
                return float(m.group(1))
            return wait_exponential(
                multiplier=RATE_LIMIT_BACKOFF_MULTIPLIER,
                min=RATE_LIMIT_BACKOFF_MIN,
                max=RATE_LIMIT_BACKOFF_MAX,
            )(retry_state)

        @retry(
            retry=retry_if_exception_type(openai.RateLimitError),
            stop=stop_after_attempt(RATE_LIMIT_MAX_ATTEMPTS),
            wait=_wait,
            reraise=True,
            before_sleep=lambda rs: logger.warning(
                "Rate limited by %s (attempt %d/%d), retrying in %.1fs",
                self._provider or self._model,
                rs.attempt_number,
                RATE_LIMIT_MAX_ATTEMPTS,
                rs.next_action.sleep,  # type: ignore[union-attr]
            ),
        )
        def _call() -> ModelResponse:
            cfg = config or GenerationConfig()

            if self._provider == Provider.OLLAMA:
                raw_messages = [
                    {"role": m.role, "content": self._adapt_content_for_ollama(m.content)} for m in messages
                ]
            else:
                raw_messages = [{"role": m.role, "content": m.content} for m in messages]

            kwargs: dict = {
                "model": self._model,
                "messages": raw_messages,
                "temperature": cfg.temperature,
                "top_p": cfg.top_p,
            }
            if cfg.response_format is not None:
                kwargs["response_format"] = self._adapt_response_format(cfg.response_format)
            if cfg.extra_params:
                kwargs.update(cfg.extra_params)

            api_key = self._api_key
            if self._provider in PROVIDERS_WITH_API_KEY and not api_key:
                raise ValueError(
                    f"API key for {self._provider or self._model} is not set — add it in LLM Providers settings"
                )
            # The openai client requires a non-None api_key even for providers that don't need one (e.g. Ollama, vLLM).
            client = openai.OpenAI(api_key=api_key or "no-key", base_url=self._base_url, timeout=30.0)
            response = client.chat.completions.create(**kwargs)

            choice = response.choices[0]
            content = choice.message.content or ""
            usage = response.usage
            return ModelResponse(
                content=content,
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
            )

        return _call()
