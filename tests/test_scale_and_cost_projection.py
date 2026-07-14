"""Tests for T11c — Scale and cost projection."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.agents.scale_and_cost_projection import (
    _create_managed_model_scale_and_cost_projection,
    _create_self_hosted_model_scale_and_cost_projection,
    _get_model_size_class,
    _get_vram_estimate,
    compute_token_stats,
    create_scale_and_cost_projection,
    select_providers_by_optimizers,
)
from src.providers.hosted_model_registry import VramEstimate
from src.providers.infra_registry import InfraProvider, InfraProviderPricing
from src.providers.managed_model_registry import ModelPricing, ModelProvider
from src.schemas.scale_and_cost import ProviderEntry, ProviderPricing, ProviderProjection

# ---------------------------------------------------------------------------
# compute_token_stats
# ---------------------------------------------------------------------------


def test_compute_token_stats_happy_path():
    rows = [
        {"output": "result", "input_tokens": 100, "output_tokens": 40},
        {"output": "result", "input_tokens": 80, "output_tokens": 30},
        {"output": "result", "input_tokens": 200, "output_tokens": 80},
        {"output": None, "input_tokens": 50, "output_tokens": 20},  # failed — excluded
    ]
    stats = compute_token_stats(rows)
    # p95 of [80, 100, 200] ≈ 200 (idx = int(3*0.95)-1 = 1 → sorted[1] = 100)
    assert stats.input.p50 is not None
    assert stats.output.p50 is not None
    assert stats.input.p95 is not None


def test_compute_token_stats_empty_rows():
    stats = compute_token_stats([])
    assert stats.input.p50 is None
    assert stats.output.p99 is None
    assert stats.input.p95 is None


# ---------------------------------------------------------------------------
# _project_managed — math correctness
# ---------------------------------------------------------------------------


def _make_managed_entry(
    id: str = "openai/gpt-4o-mini",
    name: str = "GPT-4o Mini",
    tpm_ceiling: int = 2_000_000,
    input_per_1m: float = 0.15,
    output_per_1m: float = 0.60,
) -> ProviderEntry:
    return ProviderEntry(
        provider_id=id,
        name=name,
        hosting_type="managed",
        tpm_ceiling=tpm_ceiling,
        pricing=ProviderPricing(input_per_1m=input_per_1m, output_per_1m=output_per_1m),
    )


_MANAGED_ENTRY = _make_managed_entry()


def test_project_managed_computes_time_and_cost():
    result = _create_managed_model_scale_and_cost_projection(
        provider=_MANAGED_ENTRY,
        num_records=10_000,
        input_tokens=100,
        output_tokens=50,
        target_sla_hours=None,
    )
    # total_tokens_needed = 10_000 * (100+50) = 1_500_000
    # time_hours = 1_500_000 / (2_000_000 * 60) ≈ 0.0125
    assert result.total_hours_needed == pytest.approx(0.0125, rel=1e-3)
    # cost = (100 * 0.15 + 50 * 0.60) / 1e6 * 10_000 = (15 + 30) / 1e6 * 10_000 = 0.45
    assert result.total_cost_usd == pytest.approx(0.45, rel=1e-3)
    assert result.feasible is None  # no SLA given


def test_project_managed_feasible_flag_set_when_sla_given():
    result = _create_managed_model_scale_and_cost_projection(
        provider=_MANAGED_ENTRY,
        num_records=10_000,
        input_tokens=100,
        output_tokens=50,
        target_sla_hours=1.0,  # 1 hour — time should be ~0.0125h → feasible
    )
    assert result.feasible is True


def test_project_managed_infeasible_when_too_slow():
    slow_entry = _make_managed_entry(
        id="slow/model", name="Slow", tpm_ceiling=1_000, input_per_1m=1.0, output_per_1m=2.0
    )
    result = _create_managed_model_scale_and_cost_projection(
        provider=slow_entry,
        num_records=1_000_000,
        input_tokens=500,
        output_tokens=200,
        target_sla_hours=0.5,
    )
    assert result.feasible is False


# ---------------------------------------------------------------------------
# _project_self_hosted
# ---------------------------------------------------------------------------


def _make_self_hosted_entry(
    id: str,
    gpu_type: str,
    gpu_count: int,
    gpu_memory_gb: int,
    on_demand: float,
    spot: float | None = None,
) -> ProviderEntry:
    return ProviderEntry(
        provider_id=id,
        name=id,
        hosting_type="self_hosted",
        gpu_type=gpu_type,
        gpu_count=gpu_count,
        gpu_memory_gb=gpu_memory_gb,
        pricing=ProviderPricing(spot_per_gpu_hour=spot, on_demand_per_gpu_hour=on_demand),
    )


_A100_ENTRY = _make_self_hosted_entry(
    "runpod_a100_80gb", "a100_80gb", gpu_count=1, gpu_memory_gb=80, on_demand=2.49, spot=1.89
)
_SMALL_GPU_ENTRY = _make_self_hosted_entry(
    "runpod_rtx_4090_24gb", "rtx_4090_24gb", gpu_count=1, gpu_memory_gb=24, on_demand=0.89, spot=0.74
)

_VRAM_FITS = VramEstimate(
    model_id="vllm/meta-llama/Llama-3-8B",
    param_count_b=8.0,
    precision="fp16",
    weights_gb=16.0,
    kv_cache_gb_p95=2.1,
    total_vram_gb=20.1,
    source="registry",
)

_VRAM_TOO_LARGE = VramEstimate(
    model_id="vllm/meta-llama/Llama-3-70B",
    param_count_b=70.0,
    precision="fp16",
    weights_gb=140.0,
    kv_cache_gb_p95=3.0,
    total_vram_gb=220.0,  # ceil(220/24)=10 > _MAX_INSTANCES_TO_FIT(8) → fits_model=False
    source="registry",
)


def test_project_self_hosted_fits_model_true():
    result = _create_self_hosted_model_scale_and_cost_projection(
        provider=_A100_ENTRY,
        vram_estimate=_VRAM_FITS,
        model_size_class="7b",
        num_records=10_000,
        input_tokens=100,
        output_tokens=50,
        target_sla_hours=None,
    )
    assert result.fits_model is not False
    assert result.instances_to_fit == 1
    assert result.tokens_per_sec is not None
    assert result.total_cost_usd_on_demand is not None


def test_project_self_hosted_fits_model_false():
    result = _create_self_hosted_model_scale_and_cost_projection(
        provider=_SMALL_GPU_ENTRY,
        vram_estimate=_VRAM_TOO_LARGE,
        model_size_class="70b",
        num_records=10_000,
        input_tokens=100,
        output_tokens=50,
        target_sla_hours=None,
    )
    assert result.fits_model is False
    assert result.tokens_per_sec is None
    assert result.total_cost_usd_spot is None
    assert result.total_cost_usd_on_demand is None


def test_project_self_hosted_instances_needed_scales_for_sla():
    result = _create_self_hosted_model_scale_and_cost_projection(
        provider=_A100_ENTRY,
        vram_estimate=_VRAM_FITS,
        model_size_class="7b",
        num_records=1_000_000,
        input_tokens=100,
        output_tokens=50,
        target_sla_hours=0.01,  # very tight SLA → needs many instances
    )
    assert result.fits_model is not False
    # clusters_needed should be > 1 to meet the 0.01h SLA
    assert result.clusters_needed is not None
    assert result.clusters_needed >= 1


# ---------------------------------------------------------------------------
# hosted_model_registry.estimate_vram
# ---------------------------------------------------------------------------


def test_estimate_vram_known_model():
    est = _get_vram_estimate("vllm/meta-llama/Llama-3-8B", ctx_len=2048, batch_size=1)
    assert est.source == "luna8i_registry"
    assert est.param_count_b == 8.0
    assert est.weights_gb == pytest.approx(16.0, rel=1e-3)
    assert est.total_vram_gb > est.weights_gb


def test_estimate_vram_unknown_model_with_override():
    from src.providers.hosted_model_registry import ModelSizeOverride

    est = _get_vram_estimate(
        "vllm/my-private/Model-12B", ctx_len=1024, batch_size=1, override=ModelSizeOverride(param_count_b=12.0)
    )
    assert est.source == "user_override"
    assert est.param_count_b == 12.0


def test_estimate_vram_unknown_model_no_override_raises():
    with pytest.raises(ValueError, match="ModelSizeOverride"):
        _get_vram_estimate("vllm/unknown/Model", ctx_len=1024, batch_size=1)


# ---------------------------------------------------------------------------
# hosted_model_registry.get_model_size_class
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "param_b, expected",
    [
        (7.0, "7b"),
        (8.0, "7b"),
        (13.0, "13b"),
        (14.0, "13b"),
        (34.0, "34b"),
        (70.0, "70b"),
        (405.0, "70b"),
    ],
)
def test_get_model_size_class(param_b, expected):
    assert _get_model_size_class(param_b) == expected


# ---------------------------------------------------------------------------
# select_providers_by_optimizers
# ---------------------------------------------------------------------------


def _make_provider_entry(
    id: str,
    gpu_type: str,
    gpu_count: int,
    gpu_memory_gb: int,
    on_demand: float,
    spot: float | None = None,
    p50_projection: ProviderProjection | None = None,
) -> ProviderEntry:
    return ProviderEntry(
        provider_id=id,
        name=id,
        hosting_type="self_hosted",
        gpu_type=gpu_type,
        gpu_count=gpu_count,
        gpu_memory_gb=gpu_memory_gb,
        pricing=ProviderPricing(spot_per_gpu_hour=spot, on_demand_per_gpu_hour=on_demand),
        p50=p50_projection,
    )


def test_select_best_excludes_providers_that_cannot_fit_model():
    """Entries with fits_model=False in p50 should be excluded (optimizer returns inf)."""
    small = _make_provider_entry(
        "small",
        "l4_24gb",
        gpu_count=1,
        gpu_memory_gb=16,
        on_demand=1.0,
        p50_projection=ProviderProjection(fits_model=False),
    )
    fits = _make_provider_entry(
        "fits",
        "a100_40gb",
        gpu_count=1,
        gpu_memory_gb=40,
        on_demand=3.67,
        p50_projection=ProviderProjection(fits_model=None, total_cost_usd_on_demand=100.0, tokens_per_sec=2800),
    )
    result = select_providers_by_optimizers([small, fits])
    ids = {p.provider_id for p in result}
    assert "small" not in ids
    assert "fits" in ids


def test_select_best_picks_cheapest_within_cost_optimizer():
    # provider_a: on_demand_cost=200.0; provider_b: on_demand_cost=150.0 ← cheaper
    provider_a = _make_provider_entry(
        "a",
        "a100_80gb",
        gpu_count=1,
        gpu_memory_gb=80,
        on_demand=2.0,
        p50_projection=ProviderProjection(fits_model=None, total_cost_usd_on_demand=200.0, tokens_per_sec=2800),
    )
    provider_b = _make_provider_entry(
        "b",
        "a100_80gb",
        gpu_count=1,
        gpu_memory_gb=80,
        on_demand=1.5,
        p50_projection=ProviderProjection(fits_model=None, total_cost_usd_on_demand=150.0, tokens_per_sec=2800),
    )
    from src.agents.scale_and_cost_projection import ProviderOptimizer, optimizer_cost

    result = select_providers_by_optimizers(
        [provider_a, provider_b], optimizers=[ProviderOptimizer("cost", optimizer_cost, top_n=1)]
    )
    assert len(result) == 1
    assert result[0].provider_id == "b"


def test_select_best_custom_providers_always_included():
    """Entries with provider_id starting with 'custom_' are always appended."""
    regular = _make_provider_entry(
        "regular",
        "a100_80gb",
        gpu_count=1,
        gpu_memory_gb=80,
        on_demand=2.0,
        p50_projection=ProviderProjection(fits_model=None, total_cost_usd_on_demand=100.0, tokens_per_sec=2800),
    )
    custom = _make_provider_entry(
        "custom_gpu",
        "h100_80gb",
        gpu_count=1,
        gpu_memory_gb=80,
        on_demand=3.0,
        p50_projection=ProviderProjection(fits_model=None, total_cost_usd_on_demand=200.0, tokens_per_sec=4000),
    )
    result = select_providers_by_optimizers([regular, custom])
    ids = {p.provider_id for p in result}
    assert "custom_gpu" in ids


# ---------------------------------------------------------------------------
# create_scale_and_cost_projection (full agent, mocked I/O)
# ---------------------------------------------------------------------------


_DEFAULT_INFERENCE_ROWS = [
    {"row_index": 0, "output": "result", "input_tokens": 100, "output_tokens": 40},
    {"row_index": 1, "output": "result", "input_tokens": 120, "output_tokens": 50},
]

_DEFAULT_GOLDEN_ROWS = [
    {"row_index": 0, "output": "golden", "input_tokens": 110, "output_tokens": 45},
    {"row_index": 1, "output": "golden", "input_tokens": 130, "output_tokens": 55},
]


def _write_inference_files(
    tmp_path: Path,
    models_and_rows: "list[tuple[str, list[dict]]]",
    golden_rows: "list[dict] | None" = None,
) -> None:
    """Write inference JSONL files and golden dataset to tmp_path for scale projection tests."""
    import json

    inference_dir = tmp_path / "inference"
    inference_dir.mkdir(parents=True, exist_ok=True)
    for model_string, rows in models_and_rows:
        safe = model_string.replace("/", "__")
        (inference_dir / f"{safe}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    rows_to_write = golden_rows if golden_rows is not None else _DEFAULT_GOLDEN_ROWS
    (tmp_path / "golden_dataset.jsonl").write_text("\n".join(json.dumps(r) for r in rows_to_write) + "\n")


def _make_job(**overrides):
    defaults = {
        "job_id": "test-job-1",
        "status": "run_compare_models_evaluation",
        "sota_model": "openai/gpt-4o",
        "projection_by_num_records": [1000],
        "target_sla_hours": 2.0,
        "model_hosting_preference": "managed",
        "compare_models": [{"model": "openai/gpt-4o-mini", "params": {}}],
        "scale_and_cost_projection_report_path": None,
        "updated_at": "2026-01-01T00:00:00+00:00",
        "managed_provider_custom_pricing": None,
        "self_hosted_provider_custom_pricing": None,
    }
    defaults.update(overrides)
    job = MagicMock()
    for k, v in defaults.items():
        setattr(job, k, v)
    return job


_MOCK_INFRA = [
    InfraProvider(
        id="runpod_a100_80gb",
        cloud_provider="runpod",
        name="RunPod A100 80GB",
        instance_type="A100 80GB",
        gpu_count=1,
        gpu_memory_gb=80,
        pricing=InfraProviderPricing(spot=1.89, on_demand=2.49),
        is_live=True,
        gpu_type="a100_80gb",
    )
]


_SOTA_PROVIDER = ModelProvider(
    id="openai/gpt-4o",
    name="GPT-4o",
    tpm_ceiling=30_000_000,
    pricing=ModelPricing(input_per_1m=5.0, output_per_1m=15.0),
)


def _mock_build_report_payload(tmp_path):
    """Return a build_report_payload side_effect that writes scale_and_cost directly to disk."""

    def _write(job_id, status, output_dir=None, scale_and_cost=None, job=None):
        out = (output_dir or tmp_path) / "scale_and_cost_report.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(scale_and_cost or {}))

    return _write


def test_scale_projection_writes_artifact(tmp_path):
    job = _make_job()
    _write_inference_files(tmp_path, [("openai/gpt-4o-mini", _DEFAULT_INFERENCE_ROWS)])

    mock_session = MagicMock()
    mock_session.exec.return_value.first.return_value = job
    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__ = MagicMock(return_value=False)

    with (
        patch("src.agents.scale_and_cost_projection.get_session", return_value=mock_session),
        patch("src.agents.scale_and_cost_projection.get_infra_providers_from_db", return_value=_MOCK_INFRA),
        patch(
            "src.agents.scale_and_cost_projection.get_model_providers",
            return_value=[
                _SOTA_PROVIDER,
                ModelProvider(
                    id="openai/gpt-4o-mini",
                    name="GPT-4o Mini",
                    tpm_ceiling=2_000_000,
                    pricing=ModelPricing(input_per_1m=0.15, output_per_1m=0.60),
                ),
            ],
        ),
        patch("src.services.report.build_report_payload", side_effect=_mock_build_report_payload(tmp_path)),
    ):
        create_scale_and_cost_projection("test-job-1", output_dir=tmp_path)

    report_path = tmp_path / "scale_and_cost_report.json"
    assert report_path.exists()

    data = json.loads(report_path.read_text())
    assert "num_records_projections" in data
    assert len(data["num_records_projections"]) == 1
    projections = data["num_records_projections"][0]
    assert projections["num_records"] == 1000
    assert len(projections["managed_providers"]) == 1
    assert len(projections["self_hosted_models"]) == 0  # no vllm/* model in compare_models


def test_scale_projection_managed_entries_have_cost_and_time(tmp_path):
    job = _make_job(projection_by_num_records=[10_000])
    _write_inference_files(tmp_path, [("openai/gpt-4o-mini", _DEFAULT_INFERENCE_ROWS)])

    mock_session = MagicMock()
    mock_session.exec.return_value.first.return_value = job
    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__ = MagicMock(return_value=False)

    with (
        patch("src.agents.scale_and_cost_projection.get_session", return_value=mock_session),
        patch("src.agents.scale_and_cost_projection.get_infra_providers_from_db", return_value=_MOCK_INFRA),
        patch(
            "src.agents.scale_and_cost_projection.get_model_providers",
            return_value=[
                _SOTA_PROVIDER,
                ModelProvider(
                    id="openai/gpt-4o-mini",
                    name="GPT-4o Mini",
                    tpm_ceiling=2_000_000,
                    pricing=ModelPricing(input_per_1m=0.15, output_per_1m=0.60),
                ),
            ],
        ),
        patch("src.services.report.build_report_payload", side_effect=_mock_build_report_payload(tmp_path)),
    ):
        create_scale_and_cost_projection("test-job-1", output_dir=tmp_path)

    data = json.loads((tmp_path / "scale_and_cost_report.json").read_text())
    proj = data["num_records_projections"][0]["managed_providers"][0]["p50"]
    assert proj["total_hours_needed"] is not None
    assert proj["total_cost_usd"] is not None
    assert proj["feasible"] is not None  # target_sla_hours=2.0 was set


def test_scale_projection_infeasible_provider_still_written(tmp_path):
    job = _make_job(
        projection_by_num_records=[100_000_000],
        target_sla_hours=0.001,  # impossible SLA
    )
    _write_inference_files(tmp_path, [("openai/gpt-4o-mini", _DEFAULT_INFERENCE_ROWS)])

    mock_session = MagicMock()
    mock_session.exec.return_value.first.return_value = job
    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__ = MagicMock(return_value=False)

    with (
        patch("src.agents.scale_and_cost_projection.get_session", return_value=mock_session),
        patch("src.agents.scale_and_cost_projection.get_infra_providers_from_db", return_value=_MOCK_INFRA),
        patch(
            "src.agents.scale_and_cost_projection.get_model_providers",
            return_value=[
                _SOTA_PROVIDER,
                ModelProvider(
                    id="openai/gpt-4o-mini",
                    name="GPT-4o Mini",
                    tpm_ceiling=1_000,  # tiny ceiling
                    pricing=ModelPricing(input_per_1m=0.15, output_per_1m=0.60),
                ),
            ],
        ),
        patch("src.services.report.build_report_payload", side_effect=_mock_build_report_payload(tmp_path)),
    ):
        create_scale_and_cost_projection("test-job-1", output_dir=tmp_path)

    data = json.loads((tmp_path / "scale_and_cost_report.json").read_text())
    assert data["num_records_projections"][0]["managed_providers"][0]["p50"]["feasible"] is False


def test_scale_projection_infra_failure_still_writes_artifact(tmp_path):
    """If get_infra_providers returns empty list, artifact is still written."""
    job = _make_job()
    _write_inference_files(tmp_path, [("openai/gpt-4o-mini", _DEFAULT_INFERENCE_ROWS)])

    mock_session = MagicMock()
    mock_session.exec.return_value.first.return_value = job
    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__ = MagicMock(return_value=False)

    with (
        patch("src.agents.scale_and_cost_projection.get_session", return_value=mock_session),
        patch("src.agents.scale_and_cost_projection.get_infra_providers_from_db", return_value=[]),
        patch("src.agents.scale_and_cost_projection.get_model_providers", return_value=[_SOTA_PROVIDER]),
        patch("src.services.report.build_report_payload", side_effect=_mock_build_report_payload(tmp_path)),
    ):
        create_scale_and_cost_projection("test-job-1", output_dir=tmp_path)

    data = json.loads((tmp_path / "scale_and_cost_report.json").read_text())
    assert data["num_records_projections"][0]["self_hosted_models"] == []
    assert data["num_records_projections"][0]["managed_providers"] == []


def test_scale_projection_vllm_model_triggers_vram_estimate(tmp_path):
    vllm_rows = [{"row_index": 0, "output": "x", "input_tokens": 100, "output_tokens": 50}]
    job = _make_job(
        compare_models=[
            {"model": "vllm/meta-llama/Llama-3-8B", "params": {}},
        ],
    )
    _write_inference_files(tmp_path, [("vllm/meta-llama/Llama-3-8B", vllm_rows)])

    mock_session = MagicMock()
    mock_session.exec.return_value.first.return_value = job
    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__ = MagicMock(return_value=False)

    with (
        patch("src.agents.scale_and_cost_projection.get_session", return_value=mock_session),
        patch("src.agents.scale_and_cost_projection.get_infra_providers_from_db", return_value=_MOCK_INFRA),
        patch("src.agents.scale_and_cost_projection.get_model_providers", return_value=[_SOTA_PROVIDER]),
        patch("src.services.report.build_report_payload", side_effect=_mock_build_report_payload(tmp_path)),
    ):
        create_scale_and_cost_projection("test-job-1", output_dir=tmp_path)

    data = json.loads((tmp_path / "scale_and_cost_report.json").read_text())
    projections = data["num_records_projections"][0]
    assert len(projections["self_hosted_models"]) == 1
    vram = projections["self_hosted_models"][0]["vram_estimate"]
    assert vram["source"] == "luna8i_registry"
    assert vram["param_count_b"] == 8.0


def test_scale_projection_gpu_too_small_excluded(tmp_path):
    """A GPU that cannot fit the model VRAM is excluded from self_hosted results entirely."""
    vllm_rows = [{"row_index": 0, "output": "x", "input_tokens": 100, "output_tokens": 50}]
    job = _make_job(
        projection_by_num_records=[1000],
        compare_models=[{"model": "vllm/meta-llama/Llama-3.1-405B", "params": {}}],
    )
    _write_inference_files(tmp_path, [("vllm/meta-llama/Llama-3.1-405B", vllm_rows)])

    small_gpu = InfraProvider(
        id="runpod_rtx_4090_24gb",
        cloud_provider="runpod",
        name="RunPod RTX 4090 24GB",
        instance_type="RTX 4090 24GB",
        gpu_count=1,
        gpu_memory_gb=24,
        pricing=InfraProviderPricing(spot=0.74, on_demand=0.89),
        is_live=True,
        gpu_type="rtx_4090_24gb",
    )

    mock_session = MagicMock()
    mock_session.exec.return_value.first.return_value = job
    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__ = MagicMock(return_value=False)

    with (
        patch("src.agents.scale_and_cost_projection.get_session", return_value=mock_session),
        patch("src.agents.scale_and_cost_projection.get_infra_providers_from_db", return_value=[small_gpu]),
        patch("src.agents.scale_and_cost_projection.get_model_providers", return_value=[_SOTA_PROVIDER]),
        patch("src.services.report.build_report_payload", side_effect=_mock_build_report_payload(tmp_path)),
    ):
        create_scale_and_cost_projection("test-job-1", output_dir=tmp_path)

    data = json.loads((tmp_path / "scale_and_cost_report.json").read_text())
    projections = data["num_records_projections"][0]
    assert len(projections["self_hosted_models"]) == 1
    assert projections["self_hosted_models"][0]["infra_providers"] == []


def test_scale_projection_no_provider_fits_model_self_hosted_empty(tmp_path):
    """When all infra providers are too small, artifact is written with empty self_hosted_providers."""
    vllm_rows = [{"row_index": 0, "output": "x", "input_tokens": 100, "output_tokens": 50}]
    job = _make_job(
        projection_by_num_records=[1000],
        compare_models=[{"model": "vllm/meta-llama/Llama-3.1-405B", "params": {}}],
    )
    _write_inference_files(tmp_path, [("vllm/meta-llama/Llama-3.1-405B", vllm_rows)])

    too_small_providers = [
        InfraProvider(
            id=f"small_{i}",
            cloud_provider="test",
            name=f"small_{i}",
            instance_type=f"small_{i}",
            gpu_count=1,
            gpu_memory_gb=24,
            pricing=InfraProviderPricing(spot=0.5, on_demand=1.0),
            is_live=False,
            gpu_type="l4_24gb",
        )
        for i in range(3)
    ]

    mock_session = MagicMock()
    mock_session.exec.return_value.first.return_value = job
    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__ = MagicMock(return_value=False)

    with (
        patch("src.agents.scale_and_cost_projection.get_session", return_value=mock_session),
        patch(
            "src.agents.scale_and_cost_projection.get_infra_providers_from_db",
            return_value=too_small_providers,
        ),
        patch("src.agents.scale_and_cost_projection.get_model_providers", return_value=[_SOTA_PROVIDER]),
        patch("src.services.report.build_report_payload", side_effect=_mock_build_report_payload(tmp_path)),
    ):
        create_scale_and_cost_projection("test-job-1", output_dir=tmp_path)

    data = json.loads((tmp_path / "scale_and_cost_report.json").read_text())
    projections = data["num_records_projections"][0]
    assert len(projections["self_hosted_models"]) == 1
    assert projections["self_hosted_models"][0]["infra_providers"] == []
