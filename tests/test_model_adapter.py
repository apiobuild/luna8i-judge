"""
Tests for OpenAICompatibleAdapter via get_client — covers vLLM and Ollama providers.
All openai SDK calls are mocked; no real network calls.
"""

from unittest.mock import MagicMock, patch

import openai
import pytest

from src.providers.adapters import GenerationConfig, Message
from src.providers.client import get_client, get_host_from_env
from src.providers.managed_model_provider_constants import Provider
from src.providers.openai_compatible_adapter import RATE_LIMIT_MAX_ATTEMPTS, OpenAICompatibleAdapter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_openai_response(content: str, prompt_tokens: int, completion_tokens: int) -> MagicMock:
    mock_choice = MagicMock()
    mock_choice.message.content = content

    mock_usage = MagicMock()
    mock_usage.prompt_tokens = prompt_tokens
    mock_usage.completion_tokens = completion_tokens

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = mock_usage
    return mock_response


# ---------------------------------------------------------------------------
# vLLM
# ---------------------------------------------------------------------------


def test_vllm_adapter_complete_returns_model_response(monkeypatch):
    monkeypatch.setenv("VLLM_HOST", "http://vllm:8000")
    import importlib

    import src.env as env_mod

    importlib.reload(env_mod)

    mock_openai_client = MagicMock()
    mock_openai_client.chat.completions.create.return_value = _mock_openai_response("Paris", 15, 3)

    with patch("openai.OpenAI", return_value=mock_openai_client):
        client = get_client("vllm/meta-llama/Llama-3.1-8B-Instruct", get_llm_provider_host_func=get_host_from_env)
        assert isinstance(client, OpenAICompatibleAdapter)
        assert client._base_url == "http://vllm:8000/v1"

        result = client.complete(
            [Message(role="user", content="What is the capital of France?")],
            GenerationConfig(temperature=0.0),
        )

    assert result.content == "Paris"
    assert result.input_tokens == 15
    assert result.output_tokens == 3


def test_vllm_adapter_propagates_server_error(monkeypatch):
    monkeypatch.setenv("VLLM_HOST", "http://vllm:8000")
    import importlib

    import src.env as env_mod

    importlib.reload(env_mod)

    mock_openai_client = MagicMock()
    mock_openai_client.chat.completions.create.side_effect = Exception("500 Internal Server Error")

    with patch("openai.OpenAI", return_value=mock_openai_client):
        client = get_client("vllm/my-model", get_llm_provider_host_func=get_host_from_env)
        with pytest.raises(Exception, match="500"):
            client.complete([Message(role="user", content="Hello")])


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------


def test_get_client_ollama_missing_host_raises(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    import importlib

    import src.env as env_mod

    importlib.reload(env_mod)
    # pydantic-settings may have picked up OLLAMA_HOST from the .env file;
    # override it directly so the test reflects a truly absent host.
    monkeypatch.setattr(env_mod.settings, "OLLAMA_HOST", None)

    with pytest.raises(RuntimeError, match="OLLAMA_HOST"):
        get_client("ollama/llama3.2", get_llm_provider_host_func=get_host_from_env)


def test_get_client_ollama_custom_host(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://192.168.1.5:11434")
    import importlib

    import src.env as env_mod

    importlib.reload(env_mod)

    client = get_client("ollama/llama3.2", get_llm_provider_host_func=get_host_from_env)
    assert isinstance(client, OpenAICompatibleAdapter)
    assert client._base_url == "http://192.168.1.5:11434/v1"


# ---------------------------------------------------------------------------
# Retry behaviour
# ---------------------------------------------------------------------------


def test_rate_limit_retries_then_succeeds():
    """RateLimitError is retried; succeeds on the final attempt before the limit."""
    mock_openai_client = MagicMock()
    rate_limit_exc = openai.RateLimitError(
        message="rate limited, retry in 0.01s",
        response=MagicMock(status_code=429, headers={}),
        body={},
    )
    mock_openai_client.chat.completions.create.side_effect = [
        rate_limit_exc,
        rate_limit_exc,
        _mock_openai_response("ok", 5, 2),
    ]

    with patch("openai.OpenAI", return_value=mock_openai_client):
        adapter = OpenAICompatibleAdapter(model="gemini-flash", api_key="key", provider=Provider.GEMINI)
        result = adapter.complete([Message(role="user", content="hi")])

    assert result.content == "ok"
    assert mock_openai_client.chat.completions.create.call_count == 3


def test_rate_limit_exhausted_raises():
    """RateLimitError raised on every attempt — re-raises after RATE_LIMIT_MAX_ATTEMPTS."""
    mock_openai_client = MagicMock()
    rate_limit_exc = openai.RateLimitError(
        message="rate limited, retry in 0.01s",
        response=MagicMock(status_code=429, headers={}),
        body={},
    )
    mock_openai_client.chat.completions.create.side_effect = rate_limit_exc

    with patch("openai.OpenAI", return_value=mock_openai_client):
        adapter = OpenAICompatibleAdapter(model="gemini-flash", api_key="key", provider=Provider.GEMINI)
        with pytest.raises(openai.RateLimitError):
            adapter.complete([Message(role="user", content="hi")])

    assert mock_openai_client.chat.completions.create.call_count == RATE_LIMIT_MAX_ATTEMPTS


def test_non_rate_limit_error_not_retried():
    """Non-rate-limit errors propagate immediately without retrying."""
    mock_openai_client = MagicMock()
    mock_openai_client.chat.completions.create.side_effect = ValueError("bad request")

    with patch("openai.OpenAI", return_value=mock_openai_client):
        adapter = OpenAICompatibleAdapter(model="gemini-flash", api_key="key", provider=Provider.GEMINI)
        with pytest.raises(ValueError, match="bad request"):
            adapter.complete([Message(role="user", content="hi")])

    assert mock_openai_client.chat.completions.create.call_count == 1


def test_ollama_adapter_complete_returns_model_response(monkeypatch):
    mock_openai_client = MagicMock()
    mock_openai_client.chat.completions.create.return_value = _mock_openai_response("a photo of a dog", 15, 8)

    with patch("openai.OpenAI", return_value=mock_openai_client):
        adapter = OpenAICompatibleAdapter(model="llama3.2", api_key=None, base_url="http://localhost:11434/v1")
        result = adapter.complete(
            [Message(role="user", content="Describe this image.")],
            GenerationConfig(temperature=0.0),
        )

    assert result.content == "a photo of a dog"
    assert result.input_tokens == 15
    assert result.output_tokens == 8
