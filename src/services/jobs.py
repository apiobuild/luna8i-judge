import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from sqlmodel import select

from src.db import get_session
from src.env import settings
from src.providers.client import GetManagedModelProviderAPIKeyFunc, get_key_from_db
from src.schemas.db import JobConfig, UploadRecord
from src.schemas.jobs import CustomManagedProviderPricing, CustomSelfHostedProviderPricing, JobDetail, JobStatus
from src.services.job_constants import INFERENCE_BLOB_FIELDS, RUNNING_STATUSES, TERMINAL_STATUSES, JobStatusStr
from src.services.job_utils import job_base_dir, job_golden_path, job_inference_dir
from src.utils.jsonl import model_filename, write_rows


class UploadError(ValueError):
    pass


def save_upload(raw: bytes) -> str:
    if not raw:
        raise UploadError("Uploaded file is empty.")
    upload_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    with get_session() as session:
        session.add(UploadRecord(upload_id=upload_id, raw_jsonl=raw, created_at=created_at))
        session.commit()
    return upload_id


def get_upload_raw(upload_id: str) -> bytes | None:
    with get_session() as session:
        record = session.get(UploadRecord, upload_id)
    return record.raw_jsonl if record else None


def create_job(
    rows: list[dict],
    prompt_template: str = "",
    sota_model: str | None = None,
    alias: str | None = None,
    workload_type: str | None = None,
    input_modality: str | None = None,
    detected_workload_details: dict | None = None,
    output_json_schema: dict | None = None,
    eval_fields: list[str] | None = None,
    compare_models: list[dict] | None = None,
    judge_criteria: list[dict] | None = None,
    projection_by_num_records: list[int] | None = None,
    target_sla_hours: float | None = None,
    hosting_preference: str | None = None,
    managed_provider_custom_pricing: CustomManagedProviderPricing | None = None,
    self_hosted_provider_custom_pricing: CustomSelfHostedProviderPricing | None = None,
    golden_generation_config: dict | None = None,
    output_dir: Path | None = None,
    html_output_filename: str | None = None,
) -> JobStatus:
    with get_session() as session:
        now = datetime.now(timezone.utc).isoformat()
        job_id = str(uuid.uuid4())

        if output_dir is None:
            output_dir = Path(settings.DATA_DIR) / "output" / job_id
        resolved_dir = job_base_dir(job_id, output_dir)
        input_path = resolved_dir / "input.jsonl"
        write_rows(input_path, rows)

        job = JobConfig(
            job_id=job_id,
            alias=alias,
            status=JobStatusStr.QUEUED,
            prompt_template=prompt_template,
            sota_model=sota_model,
            workload_type=workload_type,
            input_modality=input_modality,
            detected_workload_details=detected_workload_details,
            output_json_schema=output_json_schema,
            evaluation_fields=eval_fields,
            judge_criteria=judge_criteria,
            compare_models=compare_models,
            projection_by_num_records=projection_by_num_records,
            target_sla_hours=target_sla_hours,
            model_hosting_preference=hosting_preference,
            managed_provider_custom_pricing=managed_provider_custom_pricing,
            self_hosted_provider_custom_pricing=self_hosted_provider_custom_pricing,
            golden_dataset_generation_config=golden_generation_config,
            input_file_jsonl_path=str(input_path),
            output_dir=str(resolved_dir),
            html_output_filename=html_output_filename,
            created_at=now,
            updated_at=now,
        )
        session.add(job)
        session.commit()
        session.refresh(job)

    return JobStatus(
        job_id=job.job_id,
        status=JobStatusStr.QUEUED,
        detected_workload_details=detected_workload_details,
    )


class JobNotFoundError(ValueError):
    pass


def _resolve_output_dir(job: "JobConfig", output_dir: Path | None) -> Path | None:
    return output_dir or (Path(job.output_dir) if job.output_dir else None)


def _resolve_html_output(job: "JobConfig", html_output_filename: str | None) -> Path | None:
    filename = html_output_filename or job.html_output_filename
    if filename and job.output_dir:
        return Path(job.output_dir) / filename
    return None


class JobRunningError(ValueError):
    def __init__(self, job_id: str, reason: str = "job is currently running") -> None:
        super().__init__(f"job '{job_id}': {reason}")
        self.job_id = job_id


class EmptyCompareModelsError(ValueError):
    pass


# Statuses where a pipeline is actively executing — patching compare_models is unsafe.
_ACTIVE_STATUSES: frozenset[str] = RUNNING_STATUSES - {JobStatusStr.QUEUED}


def patch_compare_models(job_id: str, op: str, entries: list[dict], output_dir: Path | None = None) -> JobConfig:
    """Add or remove compare_models entries; clears inference blobs and resets status to queued."""
    with get_session() as session:
        job = session.get(JobConfig, job_id)
        if job is None:
            raise JobNotFoundError(job_id)
        if job.status in _ACTIVE_STATUSES:
            raise JobRunningError(job_id, f"cannot modify compare_models while job is {job.status!r}")

        current: list[dict] = list(job.compare_models or [])

        if op == "add":
            existing_keys = {(e["model"], str(e.get("params", {}))) for e in current}
            for entry in entries:
                key = (entry["model"], str(entry.get("params", {})))
                if key not in existing_keys:
                    current.append(entry)
                    existing_keys.add(key)
        elif op == "remove":
            remove_keys = {(e["model"], str(e.get("params", {}))) for e in entries}
            inference_dir = job_inference_dir(job_id, _resolve_output_dir(job, output_dir))
            for entry in entries:
                if inference_dir.exists() and (inference_dir / model_filename(entry["model"])).exists():
                    raise ValueError(
                        f"inference output already exists for {entry['model']!r}; use force=true to override"
                    )
            current = [e for e in current if (e["model"], str(e.get("params", {}))) not in remove_keys]
            if not current:
                raise EmptyCompareModelsError(job_id)
        else:
            raise ValueError(f"Unknown op '{op}'")

        now = datetime.now(timezone.utc).isoformat()
        job.compare_models = current
        job.status = JobStatusStr.QUEUED
        job.error_message = None
        if op == "remove":
            for blob in INFERENCE_BLOB_FIELDS:
                setattr(job, blob, None)
        job.updated_at = now
        session.add(job)
        session.commit()
        session.refresh(job)
        return job


def cancel_job(job_id: str) -> JobConfig:
    """Set job status to 'cancelled'. 409 if already in a terminal state."""
    with get_session() as session:
        job = session.get(JobConfig, job_id)
        if job is None:
            raise JobNotFoundError(job_id)
        if job.status in TERMINAL_STATUSES:
            raise JobRunningError(job_id)
        now = datetime.now(timezone.utc).isoformat()
        job.status = JobStatusStr.CANCELLED
        job.updated_at = now
        session.add(job)
        session.commit()
        session.refresh(job)
        return job


def run_job(
    job_id: str,
    step: str | None = None,
    force: bool = False,
    compare_models: list[dict] | None = None,
    golden_dataset_rows: list[dict] | None = None,
    retry: bool = False,
    output_dir: Path | None = None,
    on_node_start: Callable[[str], None] | None = None,
    on_node_complete: Callable[[str], None] | None = None,
    on_node_failed: Callable[[str, str], None] | None = None,
    on_generate_golden_dataset_progress: Callable[[int, int, str], None] | None = None,
    on_run_inference_progress: Callable[[int, int, str], None] | None = None,
    on_run_inference_model_start: Callable[[str, int, int], None] | None = None,
    on_run_inference_model_complete: Callable[[str, int, int, int, int], None] | None = None,
    on_run_inference_model_load: Callable[[str], None] | None = None,
    on_run_inference_model_unload: Callable[[str], None] | None = None,
    on_run_evaluation_model_start: Callable[[str, int, int], None] | None = None,
    on_run_evaluation_model_complete: Callable[[str, int, int, int, int], None] | None = None,
    on_run_evaluation_progress: Callable[[int, int, str], None] | None = None,
    get_managed_model_provider_api_key_func: GetManagedModelProviderAPIKeyFunc = get_key_from_db,
    run_models: list[str] | None = None,
    auto_load_and_unload_ollama_models: bool = False,
    html_output_filename: str | None = None,
) -> None:
    """Validate, prepare, and run (or resume) a pipeline job.

    Raises JobNotFoundError if the job does not exist.
    Raises JobRunningError if compare_models is provided but the job is currently running.
    Raises ValueError for invalid step or missing upstream blobs (re-raised from run_pipeline).
    """
    from src.agents.job_runner import _update_job, run_pipeline

    with get_session() as session:
        job = session.get(JobConfig, job_id)
    if job is None:
        raise JobNotFoundError(job_id)

    output_dir = _resolve_output_dir(job, output_dir)
    html_output = _resolve_html_output(job, html_output_filename)

    if compare_models:
        patch_compare_models(job_id, "add", compare_models, output_dir=output_dir)

    if golden_dataset_rows is not None:
        golden_path = job_golden_path(job_id, output_dir)
        write_rows(golden_path, golden_dataset_rows)
        _update_job(job_id, generated_golden_dataset_path=str(golden_path))

    run_pipeline(
        job_id,
        step=step,
        force=force,
        output_dir=output_dir,
        on_node_start=on_node_start,
        on_node_complete=on_node_complete,
        on_node_failed=on_node_failed,
        on_generate_golden_dataset_progress=on_generate_golden_dataset_progress,
        on_run_inference_progress=on_run_inference_progress,
        on_run_inference_model_start=on_run_inference_model_start,
        on_run_inference_model_complete=on_run_inference_model_complete,
        on_run_inference_model_load=on_run_inference_model_load,
        on_run_inference_model_unload=on_run_inference_model_unload,
        on_run_evaluation_model_start=on_run_evaluation_model_start,
        on_run_evaluation_model_complete=on_run_evaluation_model_complete,
        on_run_evaluation_progress=on_run_evaluation_progress,
        retry=retry,
        get_managed_model_provider_api_key_func=get_managed_model_provider_api_key_func,
        run_models=run_models,
        auto_load_and_unload_ollama_models=auto_load_and_unload_ollama_models,
        html_output=html_output,
    )


def list_jobs(
    status: str | None = None,
    alias: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list:
    from src.schemas.jobs import JobSummary

    with get_session() as session:
        query = select(JobConfig)
        if status is not None:
            query = query.where(JobConfig.status == status)
        if alias is not None:
            query = query.where(JobConfig.alias == alias)
        query = query.order_by(JobConfig.created_at.desc()).offset(offset).limit(limit)  # type: ignore[attr-defined]
        jobs = session.exec(query).all()

    return [
        JobSummary(
            job_id=j.job_id,
            alias=j.alias,
            status=j.status,
            created_at=j.created_at,
        )
        for j in jobs
    ]


def get_job(job_id: str) -> JobDetail | None:
    with get_session() as session:
        job = session.get(JobConfig, job_id)
    if job is None:
        return None
    return JobDetail(
        job_id=job.job_id,
        alias=job.alias,
        prompt_template=job.prompt_template,
        output_json_schema=job.output_json_schema,
        workload_type=job.workload_type,
        input_modality=job.input_modality,
        sota_model=job.sota_model,
        golden_dataset_generation_config=job.golden_dataset_generation_config,
        evaluation_fields=job.evaluation_fields,
        judge_criteria=job.judge_criteria,
        projection_by_num_records=job.projection_by_num_records,
        target_sla_hours=job.target_sla_hours,
        model_hosting_preference=job.model_hosting_preference,
        managed_provider_custom_pricing=job.managed_provider_custom_pricing,
        self_hosted_provider_custom_pricing=job.self_hosted_provider_custom_pricing,
        compare_models=job.compare_models,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


def get_job_status(job_id: str) -> JobStatus | None:
    with get_session() as session:
        job = session.get(JobConfig, job_id)
    if job is None:
        return None
    return JobStatus(
        job_id=job.job_id,
        status=job.status,
        error_message=job.error_message,
        detected_workload_details=job.detected_workload_details,
        inference_progress=job.inference_progress,
        evaluation_progress=job.evaluation_progress,
        pipeline_log=job.pipeline_log,
        step_durations=job.step_durations,
        step_status=job.step_status,
        sota_model=job.sota_model,
        workload_type=job.workload_type,
    )
