"""
Tests for local inference provider routers (vLLM + Ollama).

vLLM:
  - GET /api/vllm/models → model list
  - POST /api/vllm/models success → 200
  - POST /api/vllm/models vLLM 400 → forwarded
  - DELETE /api/vllm/models/{model} success → 200
  - DELETE /api/vllm/models/{model} vLLM 404 → 404
  - vLLM unreachable → 503
  - VLLM_HOST unset → 501 on all endpoints

Ollama:
  - GET /api/ollama/models → installed model list
  - GET /api/ollama/models/running → running model list
  - POST /api/ollama/models success → 200
  - POST /api/ollama/models Ollama 400 → forwarded
  - DELETE /api/ollama/models/{model} success → 200
  - DELETE /api/ollama/models/{model} not found → 404
  - Ollama unreachable → 503
  - OLLAMA_HOST unset and default host unreachable → 501
"""

from __future__ import annotations

import json
from unittest.mock import patch

import httpx
import pytest
from fastapi.testclient import TestClient

from src.main import app

PATCH_REQUEST = "src.services.models.httpx.request"
PATCH_SETTINGS = "src.services.models.settings"


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def _mock_resp(status_code: int, json_body: object) -> httpx.Response:
    return httpx.Response(status_code, json=json_body)


def _settings(**kwargs):
    return type("S", (), kwargs)()


# ===========================================================================
# vLLM
# ===========================================================================

VLLM_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
VLLM_MODELS_LIST = {"object": "list", "data": [{"id": VLLM_MODEL, "object": "model"}]}
_VLLM_CONFIGURED = _settings(VLLM_HOST="http://vllm:8000", OLLAMA_HOST=None)
_VLLM_UNCONFIGURED = _settings(VLLM_HOST=None, OLLAMA_HOST=None)


def test_vllm_list_models(client, monkeypatch):
    monkeypatch.setattr(PATCH_SETTINGS, _VLLM_CONFIGURED)
    with patch(PATCH_REQUEST, return_value=_mock_resp(200, VLLM_MODELS_LIST)):
        resp = client.get("/api/providers/models/hosted/vllm/models")
    assert resp.status_code == 200
    assert resp.json() == VLLM_MODELS_LIST


def test_vllm_load_model_success(client, monkeypatch):
    monkeypatch.setattr(PATCH_SETTINGS, _VLLM_CONFIGURED)
    with patch(PATCH_REQUEST, return_value=_mock_resp(200, {"status": "ok"})):
        resp = client.post("/api/providers/models/hosted/vllm/models", json={"model": VLLM_MODEL})
    assert resp.status_code == 200
    assert resp.json() == {"model": VLLM_MODEL, "status": "loaded"}


def test_vllm_load_model_400(client, monkeypatch):
    monkeypatch.setattr(PATCH_SETTINGS, _VLLM_CONFIGURED)
    error_body = {"error": "model not found on HuggingFace"}
    with patch(PATCH_REQUEST, return_value=_mock_resp(400, error_body)):
        resp = client.post("/api/providers/models/hosted/vllm/models", json={"model": "bad/model"})
    assert resp.status_code == 400
    assert resp.json() == error_body


def test_vllm_unload_model_success(client, monkeypatch):
    monkeypatch.setattr(PATCH_SETTINGS, _VLLM_CONFIGURED)
    with patch(PATCH_REQUEST, return_value=_mock_resp(200, {"status": "ok"})):
        resp = client.delete(f"/api/providers/models/hosted/vllm/models/{VLLM_MODEL}")
    assert resp.status_code == 200
    assert resp.json() == {"model": VLLM_MODEL, "status": "unloaded"}


def test_vllm_unload_model_not_found(client, monkeypatch):
    monkeypatch.setattr(PATCH_SETTINGS, _VLLM_CONFIGURED)
    error_body = {"error": "model not loaded"}
    with patch(PATCH_REQUEST, return_value=_mock_resp(404, error_body)):
        resp = client.delete(f"/api/providers/models/hosted/vllm/models/{VLLM_MODEL}")
    assert resp.status_code == 404
    assert resp.json() == error_body


def test_vllm_unreachable(client, monkeypatch):
    monkeypatch.setattr(PATCH_SETTINGS, _VLLM_CONFIGURED)
    with patch(PATCH_REQUEST, side_effect=httpx.ConnectError("connection refused")):
        resp = client.get("/api/providers/models/hosted/vllm/models")
    assert resp.status_code == 503


def test_vllm_not_configured(client, monkeypatch):
    monkeypatch.setattr(PATCH_SETTINGS, _VLLM_UNCONFIGURED)
    assert client.get("/api/providers/models/hosted/vllm/models").status_code == 501
    assert client.post("/api/providers/models/hosted/vllm/models", json={"model": VLLM_MODEL}).status_code == 501
    assert client.delete(f"/api/providers/models/hosted/vllm/models/{VLLM_MODEL}").status_code == 501


# ===========================================================================
# Ollama
# ===========================================================================

OLLAMA_MODEL = "qwen2.5:latest"
TAGS_RESPONSE = {"models": [{"name": OLLAMA_MODEL, "size": 4661224676}]}
PS_RESPONSE = {"models": [{"name": OLLAMA_MODEL, "size_vram": 4661224676}]}
_OLLAMA_CONFIGURED = _settings(OLLAMA_HOST="http://localhost:11434", VLLM_HOST=None)
_OLLAMA_UNCONFIGURED = _settings(OLLAMA_HOST=None, VLLM_HOST=None)


def test_ollama_list_models(client, monkeypatch):
    monkeypatch.setattr(PATCH_SETTINGS, _OLLAMA_CONFIGURED)
    with patch(PATCH_REQUEST, return_value=_mock_resp(200, TAGS_RESPONSE)):
        resp = client.get("/api/providers/models/hosted/ollama/models")
    assert resp.status_code == 200
    assert resp.json() == TAGS_RESPONSE


def test_ollama_list_running(client, monkeypatch):
    monkeypatch.setattr(PATCH_SETTINGS, _OLLAMA_CONFIGURED)
    with patch(PATCH_REQUEST, return_value=_mock_resp(200, PS_RESPONSE)):
        resp = client.get("/api/providers/models/hosted/ollama/models/running")
    assert resp.status_code == 200
    assert resp.json() == PS_RESPONSE


def test_ollama_pull_model_success(client, monkeypatch):
    monkeypatch.setattr(PATCH_SETTINGS, _OLLAMA_CONFIGURED)

    def _fake_stream(model):
        yield {"status": "pulling manifest"}
        yield {"status": "success"}

    with patch("src.routers.models.ollama_pull_model_stream", side_effect=_fake_stream):
        resp = client.post("/api/providers/models/hosted/ollama/pull", json={"model": "qwen2.5"})
    assert resp.status_code == 200
    lines = [json.loads(line) for line in resp.text.strip().splitlines()]
    assert lines[-1] == {"status": "success"}


def test_ollama_pull_model_400(client, monkeypatch):
    from src.services.models import OllamaError

    monkeypatch.setattr(PATCH_SETTINGS, _OLLAMA_CONFIGURED)
    error_body = {"error": "model 'bad-model' not found"}

    def _fake_stream(model):
        raise OllamaError(400, error_body)
        yield  # make it a generator

    with patch("src.routers.models.ollama_pull_model_stream", side_effect=_fake_stream):
        resp = client.post("/api/providers/models/hosted/ollama/pull", json={"model": "bad-model"})
    assert resp.status_code == 200  # streaming always returns 200; error is in body
    lines = [json.loads(line) for line in resp.text.strip().splitlines()]
    assert lines[-1] == error_body


def test_ollama_remove_model_success(client, monkeypatch):
    monkeypatch.setattr(PATCH_SETTINGS, _OLLAMA_CONFIGURED)
    with patch(PATCH_REQUEST, return_value=_mock_resp(200, {})):
        resp = client.delete(f"/api/providers/models/hosted/ollama/unload/{OLLAMA_MODEL}")
    assert resp.status_code == 200
    assert resp.json() == {"model": OLLAMA_MODEL, "status": "removed"}


def test_ollama_remove_model_not_found(client, monkeypatch):
    monkeypatch.setattr(PATCH_SETTINGS, _OLLAMA_CONFIGURED)
    error_body = {"error": f"model '{OLLAMA_MODEL}' not found"}
    with patch(PATCH_REQUEST, return_value=_mock_resp(404, error_body)):
        resp = client.delete(f"/api/providers/models/hosted/ollama/unload/{OLLAMA_MODEL}")
    assert resp.status_code == 404
    assert resp.json() == error_body


def test_ollama_unreachable(client, monkeypatch):
    monkeypatch.setattr(PATCH_SETTINGS, _OLLAMA_CONFIGURED)
    with patch(PATCH_REQUEST, side_effect=httpx.ConnectError("connection refused")):
        resp = client.get("/api/providers/models/hosted/ollama/models")
    assert resp.status_code == 503


def test_ollama_not_configured(client, monkeypatch):
    from src.services.models import OllamaNotConfiguredError

    monkeypatch.setattr(PATCH_SETTINGS, _OLLAMA_UNCONFIGURED)
    with patch(PATCH_REQUEST, side_effect=httpx.ConnectError("connection refused")):
        assert client.get("/api/providers/models/hosted/ollama/models").status_code == 501
        assert client.get("/api/providers/models/hosted/ollama/models/running").status_code == 501
        assert client.delete(f"/api/providers/models/hosted/ollama/unload/{OLLAMA_MODEL}").status_code == 501

    # POST /api/ollama/pull streams — error is embedded in NDJSON body (status 200)
    def _not_configured_stream(model):
        raise OllamaNotConfiguredError("not configured")
        yield  # make it a generator

    with patch("src.routers.models.ollama_pull_model_stream", side_effect=_not_configured_stream):
        resp = client.post("/api/providers/models/hosted/ollama/pull", json={"model": "qwen2.5"})
    assert resp.status_code == 200
    lines = [json.loads(line) for line in resp.text.strip().splitlines()]
    assert lines[-1] == {"error": "Ollama not configured"}
