"""
Factory for instantiating the right ModelClient from a model string.

Model string format: `provider/model-name`
  gemini/gemini-2.0-flash
  openai/gpt-4o
  anthropic/claude-sonnet-4-6
  together/meta-llama/Llama-3-70b
  fireworks/accounts/fireworks/models/llama-v3-70b
  deepseek/deepseek-chat
  vllm/my-local-model

Key resolution is injected via a callable: (provider, env_var) -> str | None.
Use the pre-built resolvers:
  get_key_from_db  — reads DB (provider_keys table); falls back to env if absent (API path)
  get_key_from_env — reads env vars / .env file only; never touches the DB (CLI path)
"""

from __future__ import annotations

from collections.abc import Callable

from src.providers.adapters import GenerationConfig, Message, ModelClient
from src.providers.managed_model_provider_constants import PROVIDER_BASE_URLS, PROVIDERS_WITH_API_KEY, Provider
from src.providers.openai_compatible_adapter import OpenAICompatibleAdapter

GetManagedModelProviderAPIKeyFunc = Callable[[str, str], str | None]  # (provider, env_var) -> api_key | None
GetLLMProviderHostFunc = Callable[[str, str], str | None]  # (provider, env_var) -> host | None


def _from_db(model_cls: type, attr: str, provider: str, env_var: str) -> str | None:
    from src.db import get_session
    from src.env import settings

    with get_session() as session:
        row = session.get(model_cls, provider)
        if row:
            return getattr(row, attr)
    return getattr(settings, env_var, None)


def _from_env(_provider: str, env_var: str) -> str | None:
    from src.env import settings

    return getattr(settings, env_var, None)


def get_key_from_db(provider: str, env_var: str) -> str | None:
    from src.schemas.db import ProviderKey

    return _from_db(ProviderKey, "api_key", provider, env_var)


def get_key_from_env(provider: str, env_var: str) -> str | None:
    return _from_env(provider, env_var)


def get_host_from_db(provider: str, env_var: str) -> str | None:
    from src.schemas.db import ProviderHost

    return _from_db(ProviderHost, "host", provider, env_var)


def get_host_from_env(provider: str, env_var: str) -> str | None:
    return _from_env(provider, env_var)


def get_client(
    model_string: str,
    get_managed_model_provider_api_key_func: GetManagedModelProviderAPIKeyFunc = get_key_from_db,
    get_llm_provider_host_func: GetLLMProviderHostFunc = get_host_from_db,
) -> ModelClient:
    """
    Parse `provider/model-name` and return the appropriate adapter.

    get_managed_model_provider_api_key_func: callable(provider, env_var) -> api_key | None
      get_key_from_db  — DB-managed keys with env fallback (web API path)
      get_key_from_env — env vars only (CLI / local Docker path)

    Raises RuntimeError if a required API key is missing.
    Raises ValueError for unknown provider prefixes.
    """

    from src.providers.managed_model_provider_constants import PROVIDERS_WITH_HOST

    if "/" not in model_string:
        raise ValueError(f"Invalid model string '{model_string}': expected 'provider/model-name' format")

    provider, _, model = model_string.partition("/")

    if provider == Provider.VLLM:
        host = get_llm_provider_host_func(provider, PROVIDERS_WITH_HOST[Provider.VLLM])
        if not host:
            raise RuntimeError("VLLM_HOST is not set")
        return OpenAICompatibleAdapter(
            model=model,
            api_key=None,
            base_url=f"{host.rstrip('/')}/v1",
        )

    if provider == Provider.OLLAMA:
        host = get_llm_provider_host_func(provider, PROVIDERS_WITH_HOST[Provider.OLLAMA])
        if not host:
            raise RuntimeError("OLLAMA_HOST is not set")
        return OpenAICompatibleAdapter(
            model=model,
            api_key=None,
            base_url=host.rstrip("/") + "/v1",
            provider=Provider.OLLAMA,
        )

    if provider in PROVIDERS_WITH_API_KEY:
        env_var = PROVIDERS_WITH_API_KEY[provider]
        api_key = get_managed_model_provider_api_key_func(provider, env_var)
        if not api_key:
            raise RuntimeError(f"{env_var} is not set")
        return OpenAICompatibleAdapter(
            model=model,
            api_key=api_key,
            base_url=PROVIDER_BASE_URLS.get(provider),
            provider=provider,
        )

    raise ValueError(f"Unknown provider '{provider}' in model string '{model_string}'")


def complete_chat(
    model_string: str,
    messages: list[Message],
    config: GenerationConfig | None = None,
    system: str | None = None,
    get_managed_model_provider_api_key_func: GetManagedModelProviderAPIKeyFunc = get_key_from_db,
    get_llm_provider_host_func: GetLLMProviderHostFunc = get_host_from_db,
) -> str:
    """
    Send a messages list to the given model and return the text response.

    system: optional system prompt prepended before messages (skipped if already present).
    config: generation settings (temperature, top_p, response_format, …).
    get_managed_model_provider_api_key_func: key resolver — get_key_from_db (API) or get_key_from_env (CLI).
    get_llm_provider_host_func: host resolver — get_host_from_db (API) or get_host_from_env (CLI).
    """
    client = get_client(
        model_string,
        get_managed_model_provider_api_key_func=get_managed_model_provider_api_key_func,
        get_llm_provider_host_func=get_llm_provider_host_func,
    )
    system_count = sum(1 for m in messages if m.role == "system") + (1 if system else 0)
    if system_count > 1:
        raise ValueError("Only one system message is allowed per request")
    if system:
        messages = [Message(role="system", content=system)] + messages
    return client.complete(messages, config or GenerationConfig()).content
