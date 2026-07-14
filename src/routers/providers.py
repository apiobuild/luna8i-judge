"""
Provider router — mounts all provider sub-routers under /api/providers.

  GET     /api/providers/models                        — managed + open model catalog
  GET/PUT /api/providers/models/managed/               — API key management
  ...     /api/providers/models/hosted/vllm            — vLLM management
  ...     /api/providers/models/hosted/ollama          — Ollama management
  GET     /api/providers/infra/providers               — list cached GPU infra providers
  POST    /api/providers/infra/refresh                 — GPU pricing refresh
"""

from fastapi import APIRouter

from src.routers.infra import infra_router
from src.routers.models import models_router

providers_router = APIRouter(prefix="/api/providers")
providers_router.include_router(models_router)
providers_router.include_router(infra_router)
