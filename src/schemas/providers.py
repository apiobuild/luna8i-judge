from pydantic import BaseModel


class ProviderStatus(BaseModel):
    configured: bool
    masked: str | None


class LocalProviderStatus(BaseModel):
    configured: bool
    host: str | None


class KeysUpdate(BaseModel):
    gemini: str | None = None
    openai: str | None = None
    anthropic: str | None = None
    together: str | None = None
    fireworks: str | None = None
    deepseek: str | None = None
    vllm: str | None = None
    ollama: str | None = None
