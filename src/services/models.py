"""
Local inference provider services: vLLM and Ollama.

vLLM — thin proxy to vLLM's OpenAI-compatible model management API.
  Raises VLLMNotConfiguredError when VLLM_HOST is unset.

Ollama — thin proxy to Ollama's REST API.
  Raises OllamaNotConfiguredError when OLLAMA_HOST is unset and the
  default host (localhost:11434) is unreachable.
"""

from __future__ import annotations

import json

import httpx

from src.env import settings

# ---------- vLLM ----------

_VLLM_TIMEOUT = 10.0


class VLLMNotConfiguredError(Exception):
    pass


class VLLMError(Exception):
    def __init__(self, status_code: int, body: object) -> None:
        self.status_code = status_code
        self.body = body
        super().__init__(f"vLLM returned {status_code}")


class VLLMUnreachableError(Exception):
    pass


def _vllm_host() -> str:
    if not settings.VLLM_HOST:
        raise VLLMNotConfiguredError
    return settings.VLLM_HOST.rstrip("/")


def _vllm_proxy(method: str, path: str, json: object = None) -> object:
    url = f"{_vllm_host()}/v1{path}"
    try:
        resp = httpx.request(method, url, json=json, timeout=_VLLM_TIMEOUT)
    except httpx.ConnectError as exc:
        raise VLLMUnreachableError(str(exc)) from exc
    except httpx.RequestError as exc:
        raise VLLMUnreachableError(str(exc)) from exc
    if not resp.is_success:
        raise VLLMError(resp.status_code, resp.json() if resp.content else {})
    return resp.json()


def vllm_list_models() -> object:
    return _vllm_proxy("GET", "/models")


def vllm_load_model(model: str) -> dict:
    _vllm_proxy("POST", "/load_model", json={"model": model})
    return {"model": model, "status": "loaded"}


def vllm_unload_model(model: str) -> dict:
    _vllm_proxy("DELETE", "/unload_model", json={"model": model})
    return {"model": model, "status": "unloaded"}


# ---------- Ollama ----------

_OLLAMA_DEFAULT_HOST = "http://localhost:11434"
_OLLAMA_TIMEOUT = 10.0
_OLLAMA_PULL_TIMEOUT = 30.0


class OllamaNotConfiguredError(Exception):
    pass


class OllamaError(Exception):
    def __init__(self, status_code: int, body: object) -> None:
        self.status_code = status_code
        self.body = body
        super().__init__(f"Ollama returned {status_code}")


class OllamaUnreachableError(Exception):
    pass


def _ollama_host() -> str:
    return (settings.OLLAMA_HOST or _OLLAMA_DEFAULT_HOST).rstrip("/")


def _ollama_proxy(method: str, path: str, json: object = None, timeout: float = _OLLAMA_TIMEOUT) -> object:
    url = f"{_ollama_host()}{path}"
    try:
        resp = httpx.request(method, url, json=json, timeout=timeout)
    except httpx.ConnectError as exc:
        if not settings.OLLAMA_HOST:
            raise OllamaNotConfiguredError(str(exc)) from exc
        raise OllamaUnreachableError(str(exc)) from exc
    except httpx.RequestError as exc:
        raise OllamaUnreachableError(str(exc)) from exc
    if not resp.is_success:
        raise OllamaError(resp.status_code, resp.json() if resp.content else {})
    if not resp.content:
        return {}
    return resp.json()


def ollama_list_models() -> object:
    return _ollama_proxy("GET", "/api/tags")


def ollama_list_running() -> object:
    return _ollama_proxy("GET", "/api/ps")


def ollama_pull_model(model: str) -> dict:
    _ollama_proxy("POST", "/api/pull", json={"model": model}, timeout=_OLLAMA_PULL_TIMEOUT)
    return {"model": model, "status": "pulled"}


def ollama_pull_model_stream(model: str):
    """Yield parsed progress dicts from Ollama's NDJSON pull stream."""
    url = f"{_ollama_host()}/api/pull"
    try:
        with httpx.stream("POST", url, json={"model": model}, timeout=_OLLAMA_PULL_TIMEOUT) as resp:
            if not resp.is_success:
                resp.read()
                raise OllamaError(resp.status_code, resp.json() if resp.content else {})
            for line in resp.iter_lines():
                if line.strip():
                    yield json.loads(line)
    except httpx.ConnectError as exc:
        if not settings.OLLAMA_HOST:
            raise OllamaNotConfiguredError(str(exc)) from exc
        raise OllamaUnreachableError(str(exc)) from exc
    except httpx.RequestError as exc:
        raise OllamaUnreachableError(str(exc)) from exc


def ollama_unload_model(model: str) -> dict:
    _ollama_proxy("DELETE", "/api/delete", json={"model": model})
    return {"model": model, "status": "removed"}


def ollama_evict_then_delete(model: str) -> dict:
    """Evict from VRAM then delete from disk."""
    try:
        ollama_evict_model(model)
    except Exception:
        pass
    return ollama_unload_model(model)


def ollama_pull_with_output_fn(model: str, output_fn) -> None:
    """Pull a model and stream progress to output_fn(msg)."""
    last_status = ""
    for chunk in ollama_pull_model_stream(model):
        status = chunk.get("status", "")
        if status and status != last_status:
            output_fn(status)
            last_status = status


def ollama_evict_with_output_fn(model: str, output_fn) -> None:
    """Evict a model from VRAM, calling output_fn(msg) on warning."""
    try:
        ollama_evict_model(model)
    except Exception as exc:
        output_fn(f"Warning: failed to unload Ollama model '{model}': {exc}")


def ollama_evict_model(model: str) -> dict:
    """Evict a model from VRAM without deleting it (keep_alive=0)."""
    _ollama_proxy("POST", "/api/generate", json={"model": model, "keep_alive": 0})
    return {"model": model, "status": "evicted"}
