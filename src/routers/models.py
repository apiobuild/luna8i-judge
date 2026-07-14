"""
Model management routers (mounted under /api/providers/models by providers.py).

  GET  /api/providers/models                    — managed + open model catalog
  GET/PUT /api/providers/models/managed/        — API key management
  GET  /api/providers/models/hosted/vllm/models          — list loaded vLLM models
  POST /api/providers/models/hosted/vllm/models          — load a model
  DELETE /api/providers/models/hosted/vllm/models/{id}   — unload a model
  GET  /api/providers/models/hosted/ollama/models         — list installed Ollama models
  GET  /api/providers/models/hosted/ollama/models/running — list running Ollama models
  POST /api/providers/models/hosted/ollama/pull           — pull a model
  DELETE /api/providers/models/hosted/ollama/unload/{id}  — evict + delete

Hosted endpoints return 501 when the provider is not configured.
"""

from __future__ import annotations

import json

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from src.providers.managed_model_provider_constants import Provider
from src.providers.managed_model_registry import ModelProvider, get_model_providers
from src.routers.managed_model_providers import providers_router as _managed_router
from src.schemas.models import ModelPricingResponse, ModelProviderResponse
from src.services.models import (
    OllamaError,
    OllamaNotConfiguredError,
    OllamaUnreachableError,
    VLLMError,
    VLLMNotConfiguredError,
    VLLMUnreachableError,
    ollama_evict_then_delete,
    ollama_list_models,
    ollama_list_running,
    ollama_pull_model_stream,
    vllm_list_models,
    vllm_load_model,
    vllm_unload_model,
)

# ---------- hosted/vllm ----------

_vllm_router = APIRouter(prefix=f"/{Provider.VLLM}", tags=[Provider.VLLM])


def _handle_vllm(fn, *args, **kwargs) -> JSONResponse:
    try:
        return JSONResponse(status_code=200, content=fn(*args, **kwargs))
    except VLLMNotConfiguredError:
        return JSONResponse(status_code=501, content={"error": "vLLM not configured"})
    except VLLMUnreachableError:
        return JSONResponse(status_code=503, content={"error": "vLLM service is unreachable"})
    except VLLMError as exc:
        return JSONResponse(status_code=exc.status_code, content=exc.body)


class LoadModelRequest(BaseModel):
    model: str


@_vllm_router.get("/models")
async def vllm_get_models() -> JSONResponse:
    return _handle_vllm(vllm_list_models)


@_vllm_router.post("/models")
async def vllm_post_load_model(body: LoadModelRequest) -> JSONResponse:
    return _handle_vllm(vllm_load_model, body.model)


@_vllm_router.delete("/models/{model_id:path}")
async def vllm_delete_unload_model(model_id: str) -> JSONResponse:
    return _handle_vllm(vllm_unload_model, model_id)


# ---------- hosted/ollama ----------

_ollama_router = APIRouter(prefix=f"/{Provider.OLLAMA}", tags=[Provider.OLLAMA])


def _ollama_error_response(exc: Exception) -> dict:
    if isinstance(exc, OllamaNotConfiguredError):
        return {"status_code": 501, "content": {"error": "Ollama not configured"}}
    if isinstance(exc, OllamaUnreachableError):
        return {"status_code": 503, "content": {"error": "Ollama service is unreachable"}}
    if isinstance(exc, OllamaError):
        return {"status_code": exc.status_code, "content": exc.body}
    raise exc


def _handle_ollama(fn, *args, **kwargs) -> JSONResponse:
    try:
        return JSONResponse(status_code=200, content=fn(*args, **kwargs))
    except (OllamaNotConfiguredError, OllamaUnreachableError, OllamaError) as exc:
        r = _ollama_error_response(exc)
        return JSONResponse(status_code=r["status_code"], content=r["content"])


class PullModelRequest(BaseModel):
    model: str


@_ollama_router.get("/models")
async def ollama_get_models() -> JSONResponse:
    return _handle_ollama(ollama_list_models)


@_ollama_router.get("/models/running")
async def ollama_get_running() -> JSONResponse:
    return _handle_ollama(ollama_list_running)


@_ollama_router.post("/pull")
async def ollama_post_pull_model(body: PullModelRequest) -> StreamingResponse:
    def _stream():
        try:
            for event in ollama_pull_model_stream(body.model):
                yield json.dumps(event) + "\n"
        except (OllamaNotConfiguredError, OllamaUnreachableError, OllamaError) as exc:
            r = _ollama_error_response(exc)
            yield json.dumps(r["content"]) + "\n"

    return StreamingResponse(_stream(), media_type="application/x-ndjson")


@_ollama_router.delete("/unload/{model_id:path}")
async def ollama_unload_model(model_id: str) -> JSONResponse:
    return _handle_ollama(ollama_evict_then_delete, model_id)


# ---------- hosted ----------

_hosted_router = APIRouter(prefix="/hosted", tags=["hosted"])
_hosted_router.include_router(_vllm_router)
_hosted_router.include_router(_ollama_router)

# ---------- /models ----------

models_router = APIRouter(prefix="/models")


def _model_to_response(m: ModelProvider) -> ModelProviderResponse:
    return ModelProviderResponse(
        id=m.id,
        name=m.name,
        provider=m.id.split("/")[0],
        tpm_ceiling=m.tpm_ceiling,
        pricing=ModelPricingResponse(
            input_per_1m=m.pricing.input_per_1m,
            output_per_1m=m.pricing.output_per_1m,
        ),
    )


@models_router.get("", tags=["models"])
async def list_models() -> list[ModelProviderResponse]:
    return [_model_to_response(m) for m in get_model_providers()]


models_router.include_router(_managed_router, prefix="/managed")
models_router.include_router(_hosted_router)
