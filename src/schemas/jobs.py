from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, field_validator


class RunEntry(BaseModel):
    model: str
    params: dict = {}


class CustomManagedProviderPricing(BaseModel):
    input_per_1m: float
    output_per_1m: float
    tpm_ceiling: int | None = None

    @property
    def name(self) -> str:
        return "Custom Managed"


class InfraProviderPricing(BaseModel):
    spot: float | None = None
    on_demand: float


class CustomSelfHostedProviderPricing(BaseModel):
    instance_type: str
    gpu_count: int
    gpu_memory_gb: int
    # Provide either tokens_per_sec (direct estimate) or gpu_type (registry lookup). Both optional;
    # if neither is given, throughput and time estimates are omitted but cost per GPU-hour still shows.
    tokens_per_sec: int | None = None
    gpu_type: str | None = None
    pricing: InfraProviderPricing

    @property
    def name(self) -> str:
        if self.gpu_type:
            import re

            base = re.sub(r"_\d+gb$", "", self.gpu_type)
            label = base.upper().replace("_", " ")
        else:
            label = "GPU"
        return f"Custom {self.gpu_count}× {label} {self.gpu_memory_gb}GB"


class JudgeCriterion(BaseModel):
    name: str
    description: str


class GoldenGenerationConfig(BaseModel):
    temperature: float = 0.0
    top_p: float = 1.0
    max_retries: int = 2


class SubmissionRequest(BaseModel):
    prompt_template: str
    sota_model: str | None = None
    output_json_schema: dict | None = None
    eval_fields: list[str] | None = None
    compare_models: list[RunEntry] = []
    judge_criteria: list[JudgeCriterion] | None = None
    workload_type: Literal["classification", "extraction", "summarization", "captioning"] | None = None
    modality: Literal["text", "image", "audio", "video"] | None = None
    alias: str | None = None
    projection_by_num_records: list[int] = [10_000, 100_000, 500_000, 1_000_000]
    target_sla_hours: float | None = None
    hosting_preference: Literal["managed", "self_hosted"] | None = None

    @field_validator("projection_by_num_records")
    @classmethod
    def _validate_projection_by_num_records(cls, v: list[int]) -> list[int]:
        if len(v) == 0 or any(n <= 0 for n in v):
            raise ValueError("projection_by_num_records must be a non-empty list of positive integers")
        return v

    managed_provider_custom_pricing: CustomManagedProviderPricing | None = None
    self_hosted_provider_custom_pricing: CustomSelfHostedProviderPricing | None = None
    golden_generation_config: GoldenGenerationConfig | None = None


class JobStatus(BaseModel):
    job_id: str
    status: str
    error_message: str | None = None
    detected_workload_details: dict | None = None
    inference_progress: dict | None = None
    evaluation_progress: dict | None = None
    pipeline_log: list[dict] | None = None
    step_durations: dict[str, float] | None = None
    step_status: dict[str, str] | None = None
    sota_model: str | None = None
    workload_type: str | None = None


class JobDetail(BaseModel):
    job_id: str
    alias: Optional[str] = None
    prompt_template: Optional[str] = None
    output_json_schema: Optional[dict] = None
    workload_type: Optional[str] = None
    input_modality: Optional[str] = None
    sota_model: Optional[str] = None
    golden_dataset_generation_config: Optional[dict] = None
    evaluation_fields: Optional[list[str]] = None
    judge_criteria: Optional[list[dict]] = None
    projection_by_num_records: Optional[list[int]] = None
    target_sla_hours: Optional[float] = None
    model_hosting_preference: Optional[str] = None
    managed_provider_custom_pricing: Optional[CustomManagedProviderPricing] = None
    self_hosted_provider_custom_pricing: Optional[CustomSelfHostedProviderPricing] = None
    compare_models: Optional[list[dict]] = None
    created_at: str
    updated_at: str


class UploadResponse(BaseModel):
    upload_id: str


class JobSubmitRequest(SubmissionRequest):
    upload_id: str


class JobSummary(BaseModel):
    job_id: str
    alias: Optional[str] = None
    status: str
    created_at: str


class BenchmarkJobReport(BaseModel):
    job_id: str
    workload_type: str | None = None
    sota_model: str | None = None
    exec_summary: dict | None = None
    technical_detail: dict | None = None
    scale_projection: dict | None = None
