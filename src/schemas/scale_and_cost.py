"""
Output schemas for the scale-and-cost projection artifact (T11c).

These mirror the dicts produced by:
  - _create_managed_model_scale_and_cost_projection
  - _create_self_hosted_model_scale_and_cost_projection
  - create_scale_and_cost_projection (top-level artifact)
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Token stats
# ---------------------------------------------------------------------------


class TokenPercentiles(BaseModel):
    p50: int | None
    p95: int | None
    p99: int | None


class TokenStats(BaseModel):
    input: TokenPercentiles
    output: TokenPercentiles
    total: TokenPercentiles


# ---------------------------------------------------------------------------
# VRAM estimate
# ---------------------------------------------------------------------------


class ModelVramEstimate(BaseModel):
    model_id: str
    param_count_b: float
    precision: str
    weights_gb: float
    kv_cache_gb_p95: float
    total_vram_gb: float
    source: str  # "registry" | "user_override"


# ---------------------------------------------------------------------------
# Unified provider projection
# ---------------------------------------------------------------------------


class ProviderProjection(BaseModel):
    # Managed fields
    total_hours_needed: float | None = None
    total_cost_usd: float | None = None
    feasible: bool | None = None

    # Self-hosted-only fields
    fits_model: bool | None = None
    instances_to_fit: int | None = None
    tokens_per_sec: int | None = None
    clusters_needed: int | None = None
    total_instances: int | None = None
    num_hours_needed_per_cluster: float | None = None
    total_cost_usd_spot: float | None = None
    total_cost_usd_on_demand: float | None = None

    # Populated only when the job has custom pricing for this provider.
    total_cost_usd_custom: float | None = None


class ProviderPricing(BaseModel):
    # Managed fields — rate per 1M tokens
    input_per_1m: float | None = None
    output_per_1m: float | None = None

    # Self-hosted fields — rate per GPU-hour
    spot_per_gpu_hour: float | None = None
    on_demand_per_gpu_hour: float | None = None


class ProviderEntry(BaseModel):
    provider_id: str
    name: str
    hosting_type: Literal["managed", "self_hosted"]
    pricing: ProviderPricing

    # Managed-only
    tpm_ceiling: int | None = None

    # Self-hosted-only
    is_live: bool | None = None
    vram_gb: float | None = None
    gpu_memory_gb: int | None = None
    gpu_count: int | None = None
    gpu_type: str | None = None

    # Keyed by percentile string (e.g. "p50", "p95", "mean"). None when num_records == 0.
    p50: ProviderProjection | None = None
    p95: ProviderProjection | None = None
    p99: ProviderProjection | None = None


# ---------------------------------------------------------------------------
# Self-hosted model grouping
# ---------------------------------------------------------------------------


class SelfHostedModelEntry(BaseModel):
    model_id: str
    vram_estimate: ModelVramEstimate
    infra_providers: list[ProviderEntry]  # shortlisted/optimized


# ---------------------------------------------------------------------------
# Num records projection — one entry per num_records value
# ---------------------------------------------------------------------------


class NumRecordsProjection(BaseModel):
    num_records: int
    managed_providers: list[ProviderEntry]
    self_hosted_models: list[SelfHostedModelEntry]
    # Cost projection for the sota_model used as the quality benchmark.
    sota_provider: ProviderEntry | None = None


# ---------------------------------------------------------------------------
# Per-model cost series (for line chart: x=num_records, y=cost, lines=percentiles)
# ---------------------------------------------------------------------------


class CostByTokenUsage(BaseModel):
    num_records: int
    p50_usd: float | None = None
    p95_usd: float | None = None
    p99_usd: float | None = None


class ModelCostSeries(BaseModel):
    model_id: str
    hosting_type: Literal["managed", "self_hosted"]
    # For self-hosted, cost is on-demand; for managed, cost is standard list price.
    cost_by_token_usage: list[CostByTokenUsage]


# ---------------------------------------------------------------------------
# Top-level artifact
# ---------------------------------------------------------------------------


class ScaleAndCostReport(BaseModel):
    model_token_stats: dict[str, TokenStats] = {}
    projection_percentiles: list[str]
    # The projection record counts used. Sourced from projection_by_num_records when set;
    # falls back to _DEFAULT_PROJECTION_BY_NUM_RECORDS otherwise.
    num_records_projections: list[NumRecordsProjection]
    # Pre-computed cost series per model for line chart rendering.
    model_cost_series: list[ModelCostSeries] = []
    # Cost series for the sota_model benchmark (golden-generation model), for comparison.
    sota_model_cost_series: ModelCostSeries | None = None
