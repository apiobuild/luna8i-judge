from dataclasses import asdict

from fastapi import APIRouter

from src.providers.infra_registry import InfraProvider, get_infra_providers, refresh_infra_providers

infra_router = APIRouter(prefix="/infra", tags=["infra"])


@infra_router.get("/")
async def list_infra_providers() -> list[dict]:
    providers: list[InfraProvider] = await get_infra_providers()
    return [asdict(p) for p in providers]


@infra_router.post("/refresh")
async def refresh_infra() -> list[dict]:
    providers: list[InfraProvider] = await refresh_infra_providers()
    return [asdict(p) for p in providers]
