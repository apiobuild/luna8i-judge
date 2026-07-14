"""
Provider routers:

  GET /api/providers  — list which provider API keys and hosts are configured (masked)
  PUT /api/providers  — upsert one or more provider API keys or hosts into the DB
"""

from __future__ import annotations

from fastapi import APIRouter
from sqlmodel import select

from src.db import get_session
from src.providers.managed_model_provider_constants import PROVIDERS_WITH_API_KEY, PROVIDERS_WITH_HOST
from src.schemas.db import ProviderHost, ProviderKey
from src.schemas.providers import KeysUpdate, LocalProviderStatus, ProviderStatus

providers_router = APIRouter(tags=["providers"])

_KEY_PROVIDERS: list[str] = list(PROVIDERS_WITH_API_KEY)
_HOST_PROVIDERS: list[str] = list(PROVIDERS_WITH_HOST)


def _masked(key: str | None) -> str | None:
    if not key:
        return None
    visible = key[:6]
    return visible + "*" * min(6, max(0, len(key) - 6))


def _load_keys(session) -> dict[str, str]:
    rows = session.exec(select(ProviderKey)).all()
    return {row.provider: row.api_key for row in rows}


def _load_hosts(session) -> dict[str, str]:
    rows = session.exec(select(ProviderHost)).all()
    return {row.provider: row.host for row in rows}


def _build_response(keys: dict[str, str], hosts: dict[str, str]) -> dict:
    return {
        **{
            provider: ProviderStatus(
                configured=provider in keys,
                masked=_masked(keys.get(provider)),
            )
            for provider in _KEY_PROVIDERS
        },
        **{
            provider: LocalProviderStatus(
                configured=provider in hosts,
                host=hosts.get(provider),
            )
            for provider in _HOST_PROVIDERS
        },
    }


@providers_router.get("/")
async def get_providers() -> dict:
    with get_session() as session:
        keys = _load_keys(session)
        hosts = _load_hosts(session)
    return _build_response(keys, hosts)


@providers_router.put("/")
async def update_providers(body: KeysUpdate) -> dict:
    """
    Upsert API keys or host URLs into the database.
    Send null or omit a field to leave it unchanged.
    Send an empty string to remove a key/host.
    """

    def _upsert(provider: str, row_cls, field: str, value: str) -> None:
        row = session.get(row_cls, provider)
        if value:
            if row:
                setattr(row, field, value)
            else:
                session.add(row_cls(provider=provider, **{field: value}))
        elif row:
            session.delete(row)

    updated: list[str] = []
    with get_session() as session:
        for provider, row_cls, field in [(p, ProviderKey, "api_key") for p in _KEY_PROVIDERS] + [
            (p, ProviderHost, "host") for p in _HOST_PROVIDERS
        ]:
            value = getattr(body, provider, None)
            if value is None:
                continue
            _upsert(provider, row_cls, field, value)
            updated.append(provider)

        session.commit()

        keys = _load_keys(session)
        hosts = _load_hosts(session)

    return {
        "updated": updated,
        "providers": _build_response(keys, hosts),
    }
