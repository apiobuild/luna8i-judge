"""
T8d tests — six required cases for get_client() and OpenAICompatibleAdapter.complete().
All openai SDK calls are mocked; no real API keys required.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.providers.adapters import GenerationConfig, Message
from src.providers.client import get_client, get_host_from_env, get_key_from_env
from src.providers.openai_compatible_adapter import OpenAICompatibleAdapter

# ---------------------------------------------------------------------------
# Test 1: gemini/gemini-2.0-flash + GEMINI_API_KEY set → returns OpenAICompatibleAdapter
# ---------------------------------------------------------------------------


def test_get_client_gemini_returns_openai_compatible_adapter(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    import importlib

    import src.env as env_mod

    importlib.reload(env_mod)

    client = get_client("gemini/gemini-2.0-flash", get_managed_model_provider_api_key_func=get_key_from_env)
    assert isinstance(client, OpenAICompatibleAdapter)


# ---------------------------------------------------------------------------
# Test 2: mock openai SDK → ModelResponse with correct content + token counts
# ---------------------------------------------------------------------------


def test_openai_compatible_adapter_complete_returns_model_response():
    mock_choice = MagicMock()
    mock_choice.message.content = "extracted: INV-001"

    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 42
    mock_usage.completion_tokens = 10

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = mock_usage

    mock_openai_client = MagicMock()
    mock_openai_client.chat.completions.create.return_value = mock_response

    with patch("openai.OpenAI", return_value=mock_openai_client):
        adapter = OpenAICompatibleAdapter(model="gpt-4o", api_key="test-key")
        result = adapter.complete(
            [Message(role="user", content="Extract the invoice number.")],
            GenerationConfig(temperature=0.0),
        )

    assert result.content == "extracted: INV-001"
    assert result.input_tokens == 42
    assert result.output_tokens == 10


# ---------------------------------------------------------------------------
# Test 2b: response_format is forwarded when set in GenerationConfig
# ---------------------------------------------------------------------------


def test_openai_compatible_adapter_passes_response_format():
    mock_choice = MagicMock()
    mock_choice.message.content = '{"invoice": "INV-001"}'

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)

    mock_openai_client = MagicMock()
    mock_openai_client.chat.completions.create.return_value = mock_response

    fmt = {"type": "json_object"}
    with patch("openai.OpenAI", return_value=mock_openai_client):
        adapter = OpenAICompatibleAdapter(model="gpt-4o", api_key="test-key")
        adapter.complete(
            [Message(role="user", content="Extract.")],
            GenerationConfig(response_format=fmt),
        )

    call_kwargs = mock_openai_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["response_format"] == fmt


# ---------------------------------------------------------------------------
# Test 2c: usage=None → token counts default to 0
# ---------------------------------------------------------------------------


def test_openai_compatible_adapter_zero_tokens_when_usage_none():
    mock_choice = MagicMock()
    mock_choice.message.content = "hello"

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = None

    mock_openai_client = MagicMock()
    mock_openai_client.chat.completions.create.return_value = mock_response

    with patch("openai.OpenAI", return_value=mock_openai_client):
        adapter = OpenAICompatibleAdapter(model="gpt-4o", api_key="test-key")
        result = adapter.complete([Message(role="user", content="hi")])

    assert result.input_tokens == 0
    assert result.output_tokens == 0


# ---------------------------------------------------------------------------
# Test 2d: config=None → GenerationConfig defaults applied (temperature=0, top_p=1)
# ---------------------------------------------------------------------------


def test_openai_compatible_adapter_default_config_when_none():
    mock_choice = MagicMock()
    mock_choice.message.content = "ok"

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = MagicMock(prompt_tokens=1, completion_tokens=1)

    mock_openai_client = MagicMock()
    mock_openai_client.chat.completions.create.return_value = mock_response

    with patch("openai.OpenAI", return_value=mock_openai_client):
        adapter = OpenAICompatibleAdapter(model="gpt-4o", api_key="test-key")
        adapter.complete([Message(role="user", content="hi")], config=None)

    call_kwargs = mock_openai_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["temperature"] == 0.0
    assert call_kwargs["top_p"] == 1.0
    assert "response_format" not in call_kwargs


# ---------------------------------------------------------------------------
# Test 3: SDK raises rate-limit error → exception propagates unchanged
# ---------------------------------------------------------------------------


def test_openai_compatible_adapter_propagates_rate_limit_error():
    mock_openai_client = MagicMock()
    mock_openai_client.chat.completions.create.side_effect = Exception("429 rate limit exceeded")

    with patch("openai.OpenAI", return_value=mock_openai_client):
        adapter = OpenAICompatibleAdapter(model="gpt-4o", api_key="test-key")
        with pytest.raises(Exception, match="rate limit"):
            adapter.complete([Message(role="user", content="Hello")])


# ---------------------------------------------------------------------------
# Test 4: get_client("gemini/...") with GEMINI_API_KEY unset → RuntimeError
# ---------------------------------------------------------------------------


def test_get_client_gemini_missing_key_raises(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    import importlib

    import src.env as env_mod

    importlib.reload(env_mod)

    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        get_client("gemini/gemini-2.0-flash", get_managed_model_provider_api_key_func=get_key_from_env)


# ---------------------------------------------------------------------------
# Test 5: get_client("unknown/model") → ValueError
# ---------------------------------------------------------------------------


def test_get_client_unknown_provider_raises():
    with pytest.raises(ValueError, match="unknown"):
        get_client("unknown/some-model")


# ---------------------------------------------------------------------------
# Test 6: get_client("vllm/my-model") → OpenAICompatibleAdapter with correct base_url
# ---------------------------------------------------------------------------


def test_get_client_vllm_returns_openai_compatible_adapter_with_base_url(monkeypatch):
    monkeypatch.setenv("VLLM_HOST", "http://vllm:8000")
    import importlib

    import src.env as env_mod

    importlib.reload(env_mod)

    client = get_client(
        "vllm/my-model",
        get_managed_model_provider_api_key_func=get_key_from_env,
        get_llm_provider_host_func=get_host_from_env,
    )
    assert isinstance(client, OpenAICompatibleAdapter)
    assert client._base_url == "http://vllm:8000/v1"
    assert client._model == "my-model"


# ---------------------------------------------------------------------------
# T8j: Qwen (Alibaba Cloud) provider tests
# ---------------------------------------------------------------------------


def test_get_client_qwen_with_api_key(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-dashscope-key")
    import importlib

    import src.env as env_mod

    importlib.reload(env_mod)
    from src.providers.client import get_key_from_env as _get_key

    client = get_client("qwen/qwen3.7-plus", get_managed_model_provider_api_key_func=_get_key)
    assert isinstance(client, OpenAICompatibleAdapter)
    assert client._base_url == "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"


def test_get_client_qwen_missing_key_raises(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    import importlib

    import src.env as env_mod

    importlib.reload(env_mod)

    with pytest.raises(RuntimeError, match="DASHSCOPE_API_KEY"):
        get_client("qwen/qwen3.7-plus", get_managed_model_provider_api_key_func=get_key_from_env)


def test_get_client_qwen_complete_returns_model_response():
    mock_choice = MagicMock()
    mock_choice.message.content = "Positive sentiment"

    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 50
    mock_usage.completion_tokens = 8

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = mock_usage

    mock_openai_client = MagicMock()
    mock_openai_client.chat.completions.create.return_value = mock_response

    with patch("openai.OpenAI", return_value=mock_openai_client):
        adapter = OpenAICompatibleAdapter(
            model="qwen3.7-plus",
            api_key="test-dashscope-key",
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        )
        result = adapter.complete(
            [Message(role="user", content="Classify the sentiment.")],
            GenerationConfig(temperature=0.0),
        )

    assert result.content == "Positive sentiment"
    assert result.input_tokens == 50
    assert result.output_tokens == 8


def test_estimate_cost_qwen_plus():
    from src.providers.managed_model_registry import estimate_cost

    cost = estimate_cost("qwen/qwen3.7-plus", input_tokens=1_000_000, output_tokens=1_000_000)
    assert cost == pytest.approx(0.352)  # 0.32 + 0.032
