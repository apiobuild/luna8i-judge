"""
T11c — Scale and cost projection.

Reads inference_output (token stats) and job fields (projection_by_num_records,
target_sla_hours, model_hosting_preference) to pre-compute
cost and feasibility projections for all managed + self-hosted providers.

Output artifact: scale_and_cost_report.json written under output/{job_id}/.
The job's scale_and_cost_projection_report_path column is updated to point at that file.
"""

from __future__ import annotations

import logging
import math
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from sqlmodel import select

from src.db import get_session
from src.providers.hosted_model_registry import _MODEL_REGISTRY, ModelSizeOverride, VramEstimate
from src.providers.infra_registry import (
    InfraProvider,
    get_infra_providers_from_db,
    get_prefill_multiplier,
    get_tokens_per_sec,
)
from src.providers.managed_model_registry import ModelProvider, get_model_providers
from src.schemas.db import JobConfig
from src.schemas.jobs import CustomManagedProviderPricing, CustomSelfHostedProviderPricing
from src.schemas.scale_and_cost import (
    CostByTokenUsage,
    ModelCostSeries,
    ModelVramEstimate,
    NumRecordsProjection,
    ProviderEntry,
    ProviderPricing,
    ProviderProjection,
    ScaleAndCostReport,
    SelfHostedModelEntry,
    TokenPercentiles,
    TokenStats,
)
from src.services.job_utils import (
    job_golden_path,
    job_inference_dir,
    job_scale_and_cost_report_path,
)
from src.utils.jsonl import load_inference, read_rows

_PRECISION_FACTOR: dict[str, float] = {
    "fp16": 2.0,
    "int8": 1.0,
    "int4": 0.5,
}

_KV_BYTES_PER_TOKEN_PER_LAYER = 128
_OVERHEAD_GB = 2.0


_SELF_HOSTED_PREFIXES = ("vllm/", "ollama/")


_MAX_INSTANCES_TO_FIT = 8


class ProviderOptimizer:
    """A named optimizer to select top_n provider. Lower score = better."""

    def __init__(self, name: str, optimizer: Callable[[ProviderEntry], float], top_n: int = 1) -> None:
        self.name = name
        self.optimizer = optimizer
        self.top_n = top_n


# ---------------------------------------------------------------------------
# Built-in optimizer functions (lower = better; return inf to exclude)
# ---------------------------------------------------------------------------


def optimizer_cost(entry: ProviderEntry) -> float:
    """Rank by on-demand cost (canonical baseline; spot shown for reference only)."""
    proj = entry.p50
    if proj is None or proj.fits_model is False:
        return float("inf")
    if proj.total_cost_usd_on_demand is not None:
        return proj.total_cost_usd_on_demand
    return float("inf")


def optimizer_tps(entry: ProviderEntry) -> float:
    """Rank by throughput — highest TPS wins (inverted so lower = better)."""
    proj = entry.p50
    if proj is None or proj.fits_model is False or not proj.tokens_per_sec:
        return float("inf")
    return 1.0 / proj.tokens_per_sec


def optimizer_balanced(entry: ProviderEntry) -> float:
    """Rank by cost-efficiency: on-demand cost per token/sec (knee of the curve)."""
    proj = entry.p50
    if proj is None or proj.fits_model is False or not proj.tokens_per_sec:
        return float("inf")
    cost = optimizer_cost(entry)
    return float("inf") if cost == float("inf") else cost / proj.tokens_per_sec


DEFAULT_PROVIDER_OPTIMIZERS: list[ProviderOptimizer] = [
    ProviderOptimizer("cost", optimizer_cost, top_n=1),
    ProviderOptimizer("tps", optimizer_tps, top_n=1),
    ProviderOptimizer("balanced", optimizer_balanced, top_n=1),
]


def select_providers_by_optimizers(
    entries: list[ProviderEntry],
    optimizers: list[ProviderOptimizer] | None = None,
) -> list[ProviderEntry]:
    """
    Return a deduplicated shortlist by applying each criterion's optimizer and
    picking its top-N entries. Custom entries (provider_id starting with "custom_")
    are always appended unless they already appear in the shortlist.
    Entries where the optimizer returns inf are excluded from ranking.
    """
    if optimizers is None:
        optimizers = DEFAULT_PROVIDER_OPTIMIZERS

    custom_provider = [e for e in entries if e.provider_id.startswith("custom_")]
    providers = [e for e in entries if not e.provider_id.startswith("custom_")]

    seen: set[str] = set()
    selected: list[ProviderEntry] = []

    for criterion in optimizers:
        scored = sorted(
            [(criterion.optimizer(e), e) for e in providers if criterion.optimizer(e) < float("inf")],
            key=lambda x: x[0],
        )
        for score, entry in scored[: criterion.top_n]:
            if entry.provider_id not in seen:
                selected.append(entry)
                seen.add(entry.provider_id)

    for entry in custom_provider:
        if entry.provider_id not in seen:
            selected.append(entry)
            seen.add(entry.provider_id)

    return selected


def _strip_hosted_model_prefix(model_id: str) -> str:
    for prefix in _SELF_HOSTED_PREFIXES:
        if model_id.startswith(prefix):
            return model_id[len(prefix) :]
    return model_id


def _is_self_hosted_model(model_string: str) -> bool:
    return any(model_string.startswith(p) for p in _SELF_HOSTED_PREFIXES)


def _get_vram_estimate(
    model_id: str,
    ctx_len: int,
    batch_size: int,
    override: ModelSizeOverride | None = None,
) -> VramEstimate:
    """
    Estimate VRAM required to run `model_id` at inference time.

    Returns a VramEstimate with source="registry" when the model is known,
    or source="user_override" when an override is provided for an unknown model.

    Raises ValueError if the model is unknown and no override is given.
    """
    key = _strip_hosted_model_prefix(model_id)
    entry = _MODEL_REGISTRY.get(key)

    if entry is not None:
        param_count_b, default_precision, num_layers = entry
        precision = (override.precision if override else None) or default_precision
        source = "luna8i_registry"
    elif override is not None:
        param_count_b = override.param_count_b
        precision = override.precision
        # Estimate num_layers from param count using the standard dense-transformer
        # identity: params = 12 × L × d_model², with the empirical ratio d_model ≈ 128 × L
        # (holds for Llama 3.x, Qwen 2.5, Mistral, Gemma, Phi; ~2–18% error).
        # Derived from Kaplan et al. (2020) "Scaling Laws for Neural Language Models", §D.
        # NOT reliable for MoE models (e.g. Mixtral) — supply num_layers explicitly for those.
        num_layers = override.num_layers or max(16, round((param_count_b * 1e9 / (12 * 128**2)) ** (1 / 3)))
        source = "user_override"
    else:
        raise ValueError(
            f"Model '{model_id}' not found in hosted_model_registry. "
            "Provide a ModelSizeOverride to estimate VRAM for unknown models."
        )

    if precision not in _PRECISION_FACTOR:
        raise ValueError(f"Unknown precision '{precision}'. Must be one of: {list(_PRECISION_FACTOR)}")
    factor = _PRECISION_FACTOR[precision]
    weights_gb = param_count_b * factor

    # KV cache: ctx_len × batch × 128 b/token/layer × num_layers.
    # 128 b/token/layer follows the MHA memory layout in Kwon et al., "Efficient Memory
    # Management for LLM Serving with PagedAttention" (SOSP 2023), §3.2. GQA models
    # (Llama 3.x, Qwen 2.5) use fewer KV heads so actual usage is lower — treat this
    # as a conservative upper bound. See Ainslie et al., "GQA" (EMNLP 2023) for GQA math.
    kv_bytes = ctx_len * batch_size * _KV_BYTES_PER_TOKEN_PER_LAYER * num_layers
    kv_cache_gb_p95 = kv_bytes / 1e9

    # (default) +2 GB flat overhead for CUDA context, activations, and framework buffers.
    # See HuggingFace "Model Memory Anatomy" (hf.co/docs/transformers/model_memory_anatomy).
    total_vram_gb = weights_gb + kv_cache_gb_p95 + _OVERHEAD_GB

    return VramEstimate(
        model_id=model_id,
        param_count_b=param_count_b,
        precision=precision,
        weights_gb=round(weights_gb, 2),
        kv_cache_gb_p95=round(kv_cache_gb_p95, 2),
        total_vram_gb=round(total_vram_gb, 2),
        source=source,
    )


def _get_model_size_class(param_count_b: float) -> str:
    """Map param count to the size class used in throughput lookup tables."""
    if param_count_b <= 10:
        return "7b"
    if param_count_b <= 20:
        return "13b"
    if param_count_b <= 50:
        return "34b"
    return "70b"


logger = logging.getLogger(__name__)


@dataclass
class SelfHostedModelStats:
    vram_estimate: VramEstimate
    model_size_class: str
    token_stats: TokenStats


# ---------------------------------------------------------------------------
# Token stat helpers
# ---------------------------------------------------------------------------


def _percentile(data: list[int], p: int) -> int | None:
    if not data:
        return None
    if len(data) == 1:
        return data[0]
    # statistics.quantiles uses inclusive method matching p50/p95/p99 semantics
    return int(statistics.quantiles(data, n=100, method="inclusive")[p - 1])


def _mean(data: list[int]) -> float | None:
    return round(statistics.mean(data), 1) if data else None


def compute_token_stats(inference_rows: list[dict]) -> TokenStats:
    """
    Aggregate token counts from raw inference JSONL rows across all models.

    Uses all successful rows from all compare_model entries.
    """
    input_tokens: list[int] = []
    output_tokens: list[int] = []

    for row in inference_rows:
        if row.get("output") is None:
            continue
        it = row.get("input_tokens")
        ot = row.get("output_tokens")
        if it is not None:
            input_tokens.append(int(it))
        if ot is not None:
            output_tokens.append(int(ot))

    return TokenStats(
        input=TokenPercentiles(
            p50=_percentile(input_tokens, 50),
            p95=_percentile(input_tokens, 95),
            p99=_percentile(input_tokens, 99),
        ),
        output=TokenPercentiles(
            p50=_percentile(output_tokens, 50),
            p95=_percentile(output_tokens, 95),
            p99=_percentile(output_tokens, 99),
        ),
        total=TokenPercentiles(
            p50=_percentile(input_tokens + output_tokens, 50),
            p95=_percentile(input_tokens + output_tokens, 95),
            p99=_percentile(input_tokens + output_tokens, 99),
        ),
    )


# ---------------------------------------------------------------------------
# Managed inference projection
# ---------------------------------------------------------------------------


def _create_managed_model_scale_and_cost_projection(
    provider: ProviderEntry,
    num_records: int,
    input_tokens: int,
    output_tokens: int,
    target_sla_hours: float | None,
    custom_pricing: dict | None = None,
) -> ProviderProjection:
    total_tokens_needed = num_records * (input_tokens + output_tokens)

    tpm_ceiling = provider.tpm_ceiling
    input_per_1m = provider.pricing.input_per_1m
    output_per_1m = provider.pricing.output_per_1m

    total_hours_needed = total_tokens_needed / (tpm_ceiling * 60) if tpm_ceiling else None

    feasible: bool | None = None
    if total_hours_needed is not None and target_sla_hours is not None:
        feasible = total_hours_needed <= target_sla_hours

    if input_per_1m is None or output_per_1m is None:
        raise ValueError("provider input_per_1m and output_per_1m must be provided to create projection")

    cost_usd = (input_tokens * input_per_1m + output_tokens * output_per_1m) / 1_000_000 * num_records

    cost_usd_custom: float | None = None
    if (
        custom_pricing
        and custom_pricing.get("input_per_1m") is not None
        and custom_pricing.get("output_per_1m") is not None
    ):
        cost_usd_custom = (
            (input_tokens * custom_pricing["input_per_1m"] + output_tokens * custom_pricing["output_per_1m"])
            / 1_000_000
            * num_records
        )

    return ProviderProjection(
        total_hours_needed=round(total_hours_needed, 4) if total_hours_needed is not None else None,
        total_cost_usd=round(cost_usd, 4),
        feasible=feasible,
        total_cost_usd_custom=round(cost_usd_custom, 4) if cost_usd_custom is not None else None,
    )


# ---------------------------------------------------------------------------
# Self-hosted GPU projection
# ---------------------------------------------------------------------------


def _create_self_hosted_model_scale_and_cost_projection(
    provider: ProviderEntry,
    vram_estimate: VramEstimate,
    model_size_class: str | None,
    num_records: int,
    input_tokens: int,
    output_tokens: int,
    target_sla_hours: float | None,
    tps_override: int | None = None,
) -> ProviderProjection:
    if vram_estimate is None:
        raise ValueError("vram_estimate is required for self-hosted projection but is not provided")

    instance_total_vram = (provider.gpu_memory_gb or 0) * (provider.gpu_count or 0)

    instances_to_fit = math.ceil(vram_estimate.total_vram_gb / instance_total_vram)
    if instances_to_fit > _MAX_INSTANCES_TO_FIT:
        return ProviderProjection(fits_model=False, instances_to_fit=instances_to_fit)

    if not model_size_class:
        raise ValueError("model_size_class is required for self-hosted projection but was not resolved.")
    if not provider.gpu_type:
        raise ValueError(f"Provider '{provider.name}' is missing gpu_type; cannot look up throughput.")

    tps = tps_override
    if tps is None:
        raw_tps = get_tokens_per_sec(provider.gpu_type, model_size_class)
        if raw_tps is None:
            raise ValueError(
                f"No throughput data for gpu_type='{provider.gpu_type}', model_size_class='{model_size_class}'. "
                "Add an entry to the infra registry or supply tokens_per_sec explicitly."
            )

        # raw_tps is benchmarked for one GPU. One instance has gpu_count GPUs, so throughput
        # scales linearly across GPUs within the instance (tensor parallelism / pipeline parallelism).
        if not provider.gpu_count:
            raise ValueError(f"Provider '{provider.name}' is missing gpu_count; cannot compute per-instance TPS.")
        tps_per_instance: int = raw_tps * provider.gpu_count  # type: ignore[operator]

        # One cluster = the minimum number of instances needed to hold one model replica in VRAM.
        # More instances per cluster means the model is sharded across more hardware, but it is
        # still ONE replica — throughput stays at tps_per_instance, not tps_per_instance × N.
        # Adding more *clusters* (full replicas) is what scales throughput linearly.
        tps = tps_per_instance
    if tps is None:
        raise ValueError(f"unable to calculate tokens_per_second estimate per gpu_type: '{provider.gpu_type}'")

    # Decode tokens are the bottleneck; input tokens are processed during prefill which runs
    # ~5× faster than decode, so they contribute proportionally less to wall-clock time.
    # Prefill (input tokens) runs ~5× faster than decode (output tokens); weight input tokens
    # by 1/prefill_multiplier before summing.
    total_prefill_and_decode_tokens_weighted = (output_tokens + input_tokens / get_prefill_multiplier()) * num_records
    num_hours_needed_per_cluster: float = total_prefill_and_decode_tokens_weighted / (tps * 3600)

    clusters_needed: int = 1
    num_hours_needed: float = num_hours_needed_per_cluster

    if target_sla_hours is not None:
        clusters_needed = max(1, math.ceil(num_hours_needed_per_cluster / target_sla_hours))
        num_hours_needed = target_sla_hours

    total_instances: int = clusters_needed * instances_to_fit

    total_cost_usd_spot: float | None = None
    if provider.pricing.spot_per_gpu_hour is not None:
        total_cost_usd_spot = round(total_instances * num_hours_needed * provider.pricing.spot_per_gpu_hour, 4)

    total_cost_usd_on_demand = round(
        total_instances * num_hours_needed * (provider.pricing.on_demand_per_gpu_hour or 0), 4
    )

    return ProviderProjection(
        fits_model=True,
        instances_to_fit=instances_to_fit,
        tokens_per_sec=tps,
        clusters_needed=clusters_needed,
        total_instances=total_instances,
        num_hours_needed_per_cluster=round(num_hours_needed_per_cluster, 4),
        total_cost_usd_spot=total_cost_usd_spot,
        total_cost_usd_on_demand=total_cost_usd_on_demand,
    )


# ---------------------------------------------------------------------------
# Provider entry builders
# ---------------------------------------------------------------------------


def _percentile_projections(
    projection_percentiles: list[str],
    num_records: int,
    project_fn: Callable[[str], ProviderProjection],
) -> dict[str, ProviderProjection | None]:
    return {p: (project_fn(p) if num_records > 0 else None) for p in projection_percentiles}


def _build_self_hosted_provider_entry(
    provider: InfraProvider | CustomSelfHostedProviderPricing,
    provider_id: str,
    vram_estimate: VramEstimate,
    model_size_class: str | None,
    num_records: int,
    projection_percentiles: list[str],
    token_stats: TokenStats,
    target_sla_hours: float | None,
    tps_override: int | None = None,
) -> ProviderEntry:
    provider_entity = ProviderEntry(
        provider_id=provider_id,
        name=provider.name,
        hosting_type="self_hosted",
        vram_gb=vram_estimate.total_vram_gb,
        gpu_memory_gb=provider.gpu_memory_gb,
        gpu_count=provider.gpu_count,
        gpu_type=provider.gpu_type,
        pricing=ProviderPricing(
            spot_per_gpu_hour=provider.pricing.spot,
            on_demand_per_gpu_hour=provider.pricing.on_demand,
        ),
    )
    projections = _percentile_projections(
        projection_percentiles,
        num_records,
        lambda p: _create_self_hosted_model_scale_and_cost_projection(
            provider_entity,
            vram_estimate=vram_estimate,
            model_size_class=model_size_class,
            num_records=num_records,
            input_tokens=int(getattr(token_stats.input, p) or 0),
            output_tokens=int(getattr(token_stats.output, p) or 0),
            target_sla_hours=target_sla_hours,
            tps_override=tps_override,
        ),
    )
    return provider_entity.model_copy(
        update={
            "p50": projections.get("p50"),
            "p95": projections.get("p95"),
            "p99": projections.get("p99"),
            "mean": projections.get("mean"),
        }
    )


def _build_managed_provider_entry(
    provider: ModelProvider | CustomManagedProviderPricing,
    provider_id: str,
    num_records: int,
    projection_percentiles: list[str],
    token_stats: TokenStats,
    target_sla_hours: float | None,
) -> ProviderEntry:
    if isinstance(provider, CustomManagedProviderPricing):
        tpm_ceiling = provider.tpm_ceiling or 0
        input_per_1m = provider.input_per_1m
        output_per_1m = provider.output_per_1m
        name = provider.name
    else:
        tpm_ceiling = provider.tpm_ceiling
        input_per_1m = provider.pricing.input_per_1m
        output_per_1m = provider.pricing.output_per_1m
        name = provider.name

    provider_entity = ProviderEntry(
        provider_id=provider_id,
        name=name,
        hosting_type="managed",
        tpm_ceiling=tpm_ceiling,
        pricing=ProviderPricing(
            input_per_1m=input_per_1m,
            output_per_1m=output_per_1m,
        ),
    )
    projections = _percentile_projections(
        projection_percentiles,
        num_records,
        lambda p: _create_managed_model_scale_and_cost_projection(
            provider_entity,
            num_records,
            int(getattr(token_stats.input, p) or 0),
            int(getattr(token_stats.output, p) or 0),
            target_sla_hours,
        ),
    )
    return provider_entity.model_copy(
        update={
            "p50": projections.get("p50"),
            "p95": projections.get("p95"),
            "p99": projections.get("p99"),
            "mean": projections.get("mean"),
        }
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


_VALID_PERCENTILES = {"p50", "p95", "p99"}
_DEFAULT_PROJECTION_BY_NUM_RECORDS: list[int] = [10_000, 100_000, 500_000, 1_000_000]


def create_scale_and_cost_projection(
    job_id: str,
    output_dir: Path | None = None,
    projection_percentiles: list[str] = ["p50", "p95", "p99"],  # noqa: B006
    batch_size: int = 1,
    html_output: Path | None = None,
) -> None:
    """
    LangGraph node — compute scale and cost projections and write the artifact.

    Reads job fields and inference_output token stats; writes
    scale_and_cost_report.json under output/{job_id}/.
    Projections are computed for each value in `projection_percentiles` (e.g. ["p50", "p95", "p99"]).
    Valid values: "p50", "p95", "p99".
    VRAM sizing uses the highest numeric percentile.
    `batch_size` is passed to VRAM estimation for KV-cache sizing.
    """
    invalid = set(projection_percentiles) - _VALID_PERCENTILES
    if invalid:
        raise ValueError(f"Invalid percentiles: {invalid}. Must be one of: {_VALID_PERCENTILES}")
    if not projection_percentiles:
        raise ValueError("projection_percentiles must not be empty")

    with get_session() as session:
        job = session.exec(select(JobConfig).where(JobConfig.job_id == job_id)).first()
        if job is None:
            raise ValueError(f"Job '{job_id}' not found.")
        projection_by_num_records: list[int] | None = job.projection_by_num_records
        target_sla_hours: float | None = job.target_sla_hours
        compare_models: list[dict] = job.compare_models or []
        managed_provider_custom_pricing: CustomManagedProviderPricing | None = job.managed_provider_custom_pricing
        self_hosted_provider_custom_pricing: CustomSelfHostedProviderPricing | None = (
            job.self_hosted_provider_custom_pricing
        )
        sota_model: str | None = job.sota_model
        job_detail_data: dict = {
            "job_id": job.job_id,
            "alias": job.alias,
            "workload_type": job.workload_type,
            "input_modality": job.input_modality,
            "sota_model": job.sota_model,
            "compare_models": job.compare_models,
            "projection_by_num_records": job.projection_by_num_records,
            "target_sla_hours": job.target_sla_hours,
            "model_hosting_preference": job.model_hosting_preference,
            "created_at": job.created_at,
        }

    projection_by_num_records = (
        projection_by_num_records if projection_by_num_records else _DEFAULT_PROJECTION_BY_NUM_RECORDS
    )

    # ------------------------------------------------------------------
    # Token stats — read from per-model JSONL artifacts on disk
    # ------------------------------------------------------------------
    inference_dir = job_inference_dir(job_id, output_dir)
    inference_data: dict[str, dict[int, dict]] = {
        m["model"]: load_inference(inference_dir, m["model"]) for m in compare_models
    }

    model_token_stats: dict[str, TokenStats] = {}
    for model_id, data in inference_data.items():
        model_token_stats[model_id] = compute_token_stats(list(data.values()))

    # ------------------------------------------------------------------
    # Sota model token stats — from golden dataset (same JSONL schema)
    # ------------------------------------------------------------------
    sota_token_stats: TokenStats | None = None
    sota_registry_provider: ModelProvider | None = None
    if sota_model is None:
        raise ValueError("Sota model must be provided to run scale and cost projection.")

    golden_dataset_path = job_golden_path(job_id, output_dir)
    if not golden_dataset_path.exists():
        raise FileNotFoundError(
            f"Golden dataset not found at '{golden_dataset_path}'. Run golden generation before scale projection."
        )
    golden_dataset_rows = read_rows(golden_dataset_path)
    sota_token_stats = compute_token_stats(golden_dataset_rows)
    all_providers_by_id = {p.id: p for p in get_model_providers()}
    sota_registry_provider = all_providers_by_id.get(sota_model)
    if sota_registry_provider is None:
        raise ValueError(f"sota_model '{sota_model}' not found in managed registry; cannot compute sota projection.")

    # ------------------------------------------------------------------
    # Split compare_models into managed vs self-hosted
    # ------------------------------------------------------------------
    compare_managed_model: set[str] = {
        m.get("model", "") for m in compare_models if not _is_self_hosted_model(m.get("model", ""))
    }
    compare_self_hosted_models: list[dict] = [m for m in compare_models if _is_self_hosted_model(m.get("model", ""))]

    managed_providers: list[ModelProvider] = [p for p in get_model_providers() if p.id in compare_managed_model]
    available_infra_providers: list[InfraProvider] = get_infra_providers_from_db()

    # ------------------------------------------------------------------
    # Per-model stat helpers
    # ------------------------------------------------------------------
    vram_pct = max(projection_percentiles, key=lambda p: int(p[1:]))

    # ------------------------------------------------------------------
    # Pre-compute VRAM estimates (independent of volume)
    # ------------------------------------------------------------------
    self_hosted_models: dict[str, SelfHostedModelStats] = {}
    for model in compare_self_hosted_models:
        model_name = model.get("model", "")
        stats = model_token_stats[model_name]
        proj_ctx_len = int(getattr(stats.input, vram_pct) or 0) + int(getattr(stats.output, vram_pct) or 0)
        vram_estimate = _get_vram_estimate(model_name, proj_ctx_len, batch_size)
        model_size_class: str = _get_model_size_class(vram_estimate.param_count_b)
        if not available_infra_providers:
            logger.warning(
                "[job_id=%s] No infra providers available; self_hosted will be empty for %s.",
                job_id,
                model_name,
            )
        self_hosted_models[model_name] = SelfHostedModelStats(
            vram_estimate=vram_estimate,
            model_size_class=model_size_class,
            token_stats=stats,
        )

    # ------------------------------------------------------------------
    # Build one NumRecordsProjection per num_records projection point
    # ------------------------------------------------------------------
    num_records_projections: list[NumRecordsProjection] = []
    for num_records in projection_by_num_records:
        num_records_managed: list[ProviderEntry] = []
        for provider in managed_providers:
            num_records_managed.append(
                _build_managed_provider_entry(
                    provider=provider,
                    provider_id=provider.id,
                    num_records=num_records,
                    projection_percentiles=projection_percentiles,
                    token_stats=model_token_stats[provider.id],
                    target_sla_hours=target_sla_hours,
                )
            )
        if managed_provider_custom_pricing:
            num_records_managed.append(
                _build_managed_provider_entry(
                    provider=managed_provider_custom_pricing,
                    provider_id="custom_managed_provider",
                    num_records=num_records,
                    projection_percentiles=projection_percentiles,
                    token_stats=model_token_stats[managed_providers[0].id],
                    target_sla_hours=target_sla_hours,
                )
            )

        num_records_self_hosted: list[SelfHostedModelEntry] = []
        for model_name, model_stats in self_hosted_models.items():
            infra_entries: list[ProviderEntry] = []
            for infra in available_infra_providers:
                infra_entries.append(
                    _build_self_hosted_provider_entry(
                        provider=infra,
                        provider_id=infra.id,
                        vram_estimate=model_stats.vram_estimate,
                        model_size_class=model_stats.model_size_class,
                        num_records=num_records,
                        projection_percentiles=projection_percentiles,
                        token_stats=model_stats.token_stats,
                        target_sla_hours=target_sla_hours,
                    )
                )
            if self_hosted_provider_custom_pricing:
                custom_tps: int | None = self_hosted_provider_custom_pricing.tokens_per_sec
                if custom_tps is None and self_hosted_provider_custom_pricing.gpu_type:
                    raw_tps = get_tokens_per_sec(
                        self_hosted_provider_custom_pricing.gpu_type, model_stats.model_size_class
                    )
                    if raw_tps is not None:
                        custom_tps = raw_tps * self_hosted_provider_custom_pricing.gpu_count
                if custom_tps is None:
                    raise ValueError(
                        "Cannot compute custom self-hosted projection: tokens_per_sec could not be resolved. "
                        "Set tokens_per_sec explicitly, or provide a gpu_type that exists in the infra registry."
                    )
                infra_entries.append(
                    _build_self_hosted_provider_entry(
                        provider=self_hosted_provider_custom_pricing,
                        provider_id="custom_self_hosted_provider",
                        vram_estimate=model_stats.vram_estimate,
                        model_size_class=model_stats.model_size_class,
                        num_records=num_records,
                        projection_percentiles=projection_percentiles,
                        token_stats=model_stats.token_stats,
                        target_sla_hours=target_sla_hours,
                        tps_override=custom_tps,
                    )
                )
            num_records_self_hosted.append(
                SelfHostedModelEntry(
                    model_id=model_name,
                    vram_estimate=ModelVramEstimate(
                        model_id=model_stats.vram_estimate.model_id,
                        param_count_b=model_stats.vram_estimate.param_count_b,
                        precision=model_stats.vram_estimate.precision,
                        weights_gb=model_stats.vram_estimate.weights_gb,
                        kv_cache_gb_p95=model_stats.vram_estimate.kv_cache_gb_p95,
                        total_vram_gb=model_stats.vram_estimate.total_vram_gb,
                        source=model_stats.vram_estimate.source,
                    ),
                    infra_providers=select_providers_by_optimizers(infra_entries),
                )
            )

        sota_provider_entry: ProviderEntry | None = None
        if sota_registry_provider and sota_token_stats:
            sota_provider_entry = _build_managed_provider_entry(
                provider=sota_registry_provider,
                provider_id=sota_registry_provider.id,
                num_records=num_records,
                projection_percentiles=projection_percentiles,
                token_stats=sota_token_stats,
                target_sla_hours=target_sla_hours,
            )

        num_records_projections.append(
            NumRecordsProjection(
                num_records=num_records,
                managed_providers=num_records_managed,
                self_hosted_models=num_records_self_hosted,
                sota_provider=sota_provider_entry,
            )
        )

    # ------------------------------------------------------------------
    # Build per-model cost series for line chart (x=volume, y=cost, lines=percentiles)
    # ------------------------------------------------------------------
    model_cost_series: list[ModelCostSeries] = []

    # Managed models: provider_id == model id for standard providers
    for provider in managed_providers:
        cost_by_token_usage: list[CostByTokenUsage] = []
        for num_records_projection in num_records_projections:
            provider_entry = next(
                (e for e in num_records_projection.managed_providers if e.provider_id == provider.id), None
            )
            if provider_entry is None:
                continue
            cost_by_token_usage.append(
                CostByTokenUsage(
                    num_records=num_records_projection.num_records,
                    p50_usd=provider_entry.p50.total_cost_usd if provider_entry.p50 else None,
                    p95_usd=provider_entry.p95.total_cost_usd if provider_entry.p95 else None,
                    p99_usd=provider_entry.p99.total_cost_usd if provider_entry.p99 else None,
                )
            )
        if cost_by_token_usage:
            model_cost_series.append(
                ModelCostSeries(
                    model_id=provider.id,
                    hosting_type="managed",
                    cost_by_token_usage=cost_by_token_usage,
                )
            )

    # Self-hosted models: one series per model using on-demand cost from best provider
    for model_name in self_hosted_models:
        cost_by_token_usage = []
        for num_records_projection in num_records_projections:
            model_entry = next((e for e in num_records_projection.self_hosted_models if e.model_id == model_name), None)
            if model_entry is None or not model_entry.infra_providers:
                continue
            # Use the first (best-ranked) provider's on-demand cost
            best = model_entry.infra_providers[0]
            cost_by_token_usage.append(
                CostByTokenUsage(
                    num_records=num_records_projection.num_records,
                    p50_usd=best.p50.total_cost_usd_on_demand if best.p50 else None,
                    p95_usd=best.p95.total_cost_usd_on_demand if best.p95 else None,
                    p99_usd=best.p99.total_cost_usd_on_demand if best.p99 else None,
                )
            )
        if cost_by_token_usage:
            model_cost_series.append(
                ModelCostSeries(
                    model_id=model_name,
                    hosting_type="self_hosted",
                    cost_by_token_usage=cost_by_token_usage,
                )
            )

    # ------------------------------------------------------------------
    # Sota cost series for benchmark reference line
    # ------------------------------------------------------------------
    sota_model_cost_series: ModelCostSeries | None = None
    if sota_registry_provider and sota_token_stats:
        sota_cost_by_token_usage: list[CostByTokenUsage] = []
        for nrp in num_records_projections:
            if nrp.sota_provider is None:
                continue
            sota_cost_by_token_usage.append(
                CostByTokenUsage(
                    num_records=nrp.num_records,
                    p50_usd=nrp.sota_provider.p50.total_cost_usd if nrp.sota_provider.p50 else None,
                    p95_usd=nrp.sota_provider.p95.total_cost_usd if nrp.sota_provider.p95 else None,
                    p99_usd=nrp.sota_provider.p99.total_cost_usd if nrp.sota_provider.p99 else None,
                )
            )
        sota_model_cost_series = ModelCostSeries(
            model_id=sota_registry_provider.id,
            hosting_type="managed",
            cost_by_token_usage=sota_cost_by_token_usage,
        )

    # ------------------------------------------------------------------
    # Assemble artifact
    # ------------------------------------------------------------------
    artifact = ScaleAndCostReport(
        model_token_stats=model_token_stats,
        projection_percentiles=projection_percentiles,
        num_records_projections=num_records_projections,
        model_cost_series=model_cost_series,
        sota_model_cost_series=sota_model_cost_series,
    )

    # ------------------------------------------------------------------
    # Write to disk and update DB
    # ------------------------------------------------------------------
    from src.services.report import build_report_payload, render_html_report

    report_path = job_scale_and_cost_report_path(job_id, output_dir)
    build_report_payload(
        job_id,
        status=None,
        output_dir=output_dir,
        scale_and_cost=artifact.model_dump(),
        job=job_detail_data,
    )

    logger.info("[%s] create_scale_and_cost_projection: wrote %s", job_id, report_path)

    if html_output is not None:
        render_html_report(report_path, html_output)
