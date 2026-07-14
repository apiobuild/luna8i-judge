from typing import Any, Optional

from sqlmodel import JSON, Column, Field, SQLModel

from src.schemas.jobs import CustomManagedProviderPricing, CustomSelfHostedProviderPricing


class UploadRecord(SQLModel, table=True):
    __tablename__ = "uploads"  # type: ignore[assignment]

    upload_id: str = Field(primary_key=True)
    raw_jsonl: bytes
    created_at: str


class JobConfig(SQLModel, table=True):
    __tablename__ = "jobs"  # type: ignore[assignment]

    job_id: str = Field(primary_key=True)
    alias: Optional[str] = None
    status: str
    error_message: Optional[str] = None
    prompt_template: Optional[str] = None
    output_json_schema: Optional[dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    workload_type: Optional[str] = None
    input_modality: Optional[str] = None
    detected_workload_details: Optional[dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    sota_model: Optional[str] = None
    golden_dataset_generation_config: Optional[dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    evaluation_fields: Optional[list[str]] = Field(default=None, sa_column=Column(JSON))
    judge_criteria: Optional[list[dict[str, Any]]] = Field(default=None, sa_column=Column(JSON))
    projection_by_num_records: Optional[list[int]] = Field(default=None, sa_column=Column(JSON))
    target_sla_hours: Optional[float] = None
    model_hosting_preference: Optional[str] = None
    managed_provider_custom_pricing: Optional[CustomManagedProviderPricing] = Field(
        default=None, sa_column=Column(JSON)
    )
    self_hosted_provider_custom_pricing: Optional[CustomSelfHostedProviderPricing] = Field(
        default=None, sa_column=Column(JSON)
    )
    input_file_jsonl_path: Optional[str] = None
    compare_models: Optional[list[dict[str, Any]]] = Field(default=None, sa_column=Column(JSON))
    generated_golden_dataset_path: Optional[str] = None
    compare_models_inference_output_path: Optional[str] = None
    inference_progress: Optional[dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    evaluation_progress: Optional[dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    evaluating_inference_output_path: Optional[str] = None
    scale_and_cost_projection_report_path: Optional[str] = None
    pipeline_log: Optional[list[dict[str, Any]]] = Field(default=None, sa_column=Column(JSON))
    step_durations: Optional[dict[str, float]] = Field(default=None, sa_column=Column(JSON))
    step_status: Optional[dict[str, str]] = Field(default=None, sa_column=Column(JSON))
    output_dir: Optional[str] = None
    html_output_filename: Optional[str] = None
    created_at: str
    updated_at: str


class ProviderKey(SQLModel, table=True):
    __tablename__ = "provider_keys"  # type: ignore[assignment]

    provider: str = Field(primary_key=True)
    api_key: str


class ProviderHost(SQLModel, table=True):
    __tablename__ = "provider_hosts"  # type: ignore[assignment]

    provider: str = Field(primary_key=True)
    host: str


class InfraProviderRecord(SQLModel, table=True):
    __tablename__ = "infra_providers"  # type: ignore[assignment]

    id: str = Field(primary_key=True)
    cloud_provider: str
    name: str
    instance_type: str
    gpu_count: int
    gpu_memory_gb: int
    spot_price: Optional[float] = None
    on_demand_price: float
    is_live: bool
    gpu_type: str
    fetched_at: str
