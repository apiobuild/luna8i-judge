"""Tests for T8c — Infra registry."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.providers.infra_registry import (
    InfraProvider,
    InfraProviderPricing,
    _fetch_runpod,
    _gpu_id,
    fetch_infra_providers,
    get_infra_providers,
    refresh_infra_providers,
)

# ---------------------------------------------------------------------------
# _gpu_id
# ---------------------------------------------------------------------------


def test_gpu_id_slugifies_display_name():
    assert _gpu_id("A100 SXM4 80GB", 80) == "runpod_a100_sxm4_80gb_80gb"
    assert _gpu_id("RTX 4090", 24) == "runpod_rtx_4090_24gb"


# ---------------------------------------------------------------------------
# _fetch_runpod
# ---------------------------------------------------------------------------

_RUNPOD_OK_RESPONSE = {
    "data": {
        "gpuTypes": [
            {
                "displayName": "A100 SXM4 80GB",
                "memoryInGb": 80,
                "securePrice": 2.49,
                "communityPrice": 1.89,
            },
            {
                "displayName": "RTX 4090",
                "memoryInGb": 24,
                "securePrice": 0.74,
                "communityPrice": 0.69,
            },
            {
                "displayName": "Free GPU",
                "memoryInGb": 8,
                "securePrice": 0,
                "communityPrice": 0,
            },
        ]
    }
}


def test_fetch_runpod_parses_live_response():
    mock_resp = MagicMock()
    mock_resp.json.return_value = _RUNPOD_OK_RESPONSE
    mock_resp.raise_for_status = MagicMock()

    with patch("src.providers.infra_registry.httpx.Client") as mock_client_cls:
        mock_client_cls.return_value.__enter__.return_value.post.return_value = mock_resp
        providers = _fetch_runpod()

    assert len(providers) == 2  # zero-price GPU filtered out
    a100 = next(p for p in providers if "a100" in p.id)
    assert a100.cloud_provider == "runpod"
    assert a100.is_live is True
    assert a100.gpu_memory_gb == 80
    assert a100.pricing.spot == 1.89
    assert a100.pricing.on_demand == 2.49


def test_fetch_runpod_filters_zero_price_gpus():
    mock_resp = MagicMock()
    mock_resp.json.return_value = _RUNPOD_OK_RESPONSE
    mock_resp.raise_for_status = MagicMock()

    with patch("src.providers.infra_registry.httpx.Client") as mock_client_cls:
        mock_client_cls.return_value.__enter__.return_value.post.return_value = mock_resp
        providers = _fetch_runpod()

    ids = {p.id for p in providers}
    assert not any("free" in pid for pid in ids)


def test_fetch_runpod_community_only_sets_spot_none_when_secure_present():
    resp = {
        "data": {
            "gpuTypes": [
                {"displayName": "RTX 3090", "memoryInGb": 24, "securePrice": 0.44, "communityPrice": 0},
            ]
        }
    }
    mock_resp = MagicMock()
    mock_resp.json.return_value = resp
    mock_resp.raise_for_status = MagicMock()

    with patch("src.providers.infra_registry.httpx.Client") as mock_client_cls:
        mock_client_cls.return_value.__enter__.return_value.post.return_value = mock_resp
        providers = _fetch_runpod()

    assert len(providers) == 1
    assert providers[0].pricing.spot is None
    assert providers[0].pricing.on_demand == 0.44


def test_fetch_runpod_500_raises():
    with patch("src.providers.infra_registry.httpx.Client") as mock_client_cls:
        mock_client_cls.return_value.__enter__.return_value.post.side_effect = Exception("500")
        with pytest.raises(Exception, match="500"):
            _fetch_runpod()


# ---------------------------------------------------------------------------
# fetch_infra_providers — combines RunPod + AWS + GCP
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_infra_providers_happy_path():
    mock_runpod = [
        InfraProvider(
            id="runpod_a100_sxm4_80gb_80gb",
            cloud_provider="runpod",
            name="RunPod A100 SXM4 80GB",
            instance_type="A100 SXM4 80GB 80GB",
            gpu_count=1,
            gpu_memory_gb=80,
            pricing=InfraProviderPricing(spot=1.89, on_demand=2.49),
            is_live=True,
        )
    ]

    with patch("src.providers.infra_registry._fetch_runpod", return_value=mock_runpod):
        providers = await fetch_infra_providers()

    ids = {p.id for p in providers}
    # RunPod + hardcoded AWS + GCP
    assert "runpod_a100_sxm4_80gb_80gb" in ids
    assert "aws_p4d_24xlarge" in ids
    assert "gcp_a2_highgpu_1g" in ids
    assert "gcp_a2_highgpu_4g" in ids


@pytest.mark.asyncio
async def test_fetch_infra_providers_runpod_error_returns_static_only():
    with patch("src.providers.infra_registry._fetch_runpod", side_effect=Exception("network error")):
        providers = await fetch_infra_providers()

    ids = {p.id for p in providers}
    assert "aws_p4d_24xlarge" in ids
    assert "gcp_a2_highgpu_1g" in ids
    assert not any(p.cloud_provider == "runpod" for p in providers)


# ---------------------------------------------------------------------------
# get_infra_providers — SQLite cache
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_infra_providers_warm_cache_skips_fetch():
    from src.schemas.db import InfraProviderRecord

    mock_records = [
        InfraProviderRecord(
            id="runpod_a100_sxm4_80gb_80gb",
            cloud_provider="runpod",
            name="RunPod A100 SXM4 80GB",
            instance_type="A100 SXM4 80GB 80GB",
            gpu_count=1,
            gpu_memory_gb=80,
            spot_price=1.89,
            on_demand_price=2.49,
            is_live=True,
            gpu_type="a100_80gb",
            fetched_at="2026-01-01T00:00:00+00:00",
        ),
        InfraProviderRecord(
            id="aws_p4d_24xlarge",
            cloud_provider="aws",
            name="AWS p4d.24xlarge",
            instance_type="p4d.24xlarge",
            gpu_count=8,
            gpu_memory_gb=40,
            spot_price=9.83,
            on_demand_price=32.77,
            is_live=False,
            gpu_type="a100_40gb",
            fetched_at="2026-01-01T00:00:00+00:00",
        ),
    ]

    mock_exec_result = MagicMock()
    mock_exec_result.all.return_value = mock_records
    mock_session = MagicMock()
    mock_session.exec.return_value = mock_exec_result
    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__ = MagicMock(return_value=False)

    with (
        patch("src.providers.infra_registry.get_session", return_value=mock_session),
        patch("src.providers.infra_registry.fetch_infra_providers", new_callable=AsyncMock) as mock_fetch,
    ):
        providers = await get_infra_providers()

    mock_fetch.assert_not_called()
    assert len(providers) == 2


@pytest.mark.asyncio
async def test_get_infra_providers_cold_cache_calls_fetch_and_saves():
    mock_providers = [
        InfraProvider(
            id="aws_p4d_24xlarge",
            cloud_provider="aws",
            name="AWS p4d.24xlarge",
            instance_type="p4d.24xlarge",
            gpu_count=8,
            gpu_memory_gb=40,
            pricing=InfraProviderPricing(spot=9.83, on_demand=32.77),
            is_live=False,
        )
    ]

    with (
        patch(
            "src.providers.infra_registry._load_providers_from_db",
            return_value=None,
        ),
        patch(
            "src.providers.infra_registry.fetch_infra_providers",
            new_callable=AsyncMock,
            return_value=mock_providers,
        ) as mock_fetch,
        patch("src.providers.infra_registry._persist_providers") as mock_save,
    ):
        providers = await get_infra_providers()

    mock_fetch.assert_called_once()
    mock_save.assert_called_once()
    assert len(providers) == 1


# ---------------------------------------------------------------------------
# refresh_infra_providers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_refresh_infra_providers_calls_fetch_and_updates_cache():
    mock_providers = [
        InfraProvider(
            id="runpod_a100_sxm4_80gb_80gb",
            cloud_provider="runpod",
            name="RunPod A100 SXM4 80GB",
            instance_type="A100 SXM4 80GB 80GB",
            gpu_count=1,
            gpu_memory_gb=80,
            pricing=InfraProviderPricing(spot=1.75, on_demand=2.30),
            is_live=True,
        ),
        InfraProvider(
            id="aws_p4d_24xlarge",
            cloud_provider="aws",
            name="AWS p4d.24xlarge",
            instance_type="p4d.24xlarge",
            gpu_count=8,
            gpu_memory_gb=40,
            pricing=InfraProviderPricing(spot=9.83, on_demand=32.77),
            is_live=False,
        ),
    ]

    with (
        patch(
            "src.providers.infra_registry.fetch_infra_providers",
            new_callable=AsyncMock,
            return_value=mock_providers,
        ),
        patch("src.providers.infra_registry._persist_providers") as mock_save,
    ):
        result = await refresh_infra_providers()

    mock_save.assert_called_once()
    assert len(result) == 2
    runpod = next(p for p in result if p.id == "runpod_a100_sxm4_80gb_80gb")
    assert runpod.pricing.spot == 1.75
    assert runpod.is_live is True
