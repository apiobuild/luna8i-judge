"""
Job submission and status routers.

T_UP — POST /api/uploads
T6   — POST /api/jobs
T7   — GET  /api/jobs/{job_id}/status
T7a  — PATCH /api/jobs/{job_id}/compare_models
T17  — GET  /api/jobs/{job_id}/report
T19  — POST /api/jobs/{job_id}/cancel

Endpoint distinction:
  GET /api/jobs/{job_id}         — static config (submission params, never changes after creation)
  GET /api/jobs/{job_id}/status  — runtime state (status, step_status, pipeline_log, progress, errors)
  GET /api/jobs/{job_id}/report  — final eval results from disk (only available when status=completed)
"""

from __future__ import annotations

import json

from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.agents.job_runner import run_pipeline
from src.schemas.jobs import (
    JobDetail,
    JobStatus,
    JobSubmitRequest,
    JobSummary,
    RunEntry,
    SubmissionRequest,
    UploadResponse,
)
from src.services.job_submission import SubmissionError, parse_submission_from_upload_id
from src.services.job_utils import job_scale_and_cost_report_path
from src.services.jobs import (
    EmptyCompareModelsError,
    JobNotFoundError,
    JobRunningError,
    UploadError,
    cancel_job,
    get_job_status,
    list_jobs,
    patch_compare_models,
    run_job,
    save_upload,
)
from src.services.jobs import create_job as create_job_record
from src.services.jobs import get_job as _get_job

jobs_router = APIRouter(prefix="/api/jobs")


def _bad(msg: str) -> JSONResponse:
    return JSONResponse(status_code=400, content={"error": msg})


# ---------------------------------------------------------------------------
# T_UP — POST /api/uploads
# ---------------------------------------------------------------------------


@jobs_router.post("/upload", response_model=UploadResponse)
async def upload_file(input_file_jsonl: UploadFile) -> UploadResponse | JSONResponse:
    if not input_file_jsonl.filename or not input_file_jsonl.filename.endswith(".jsonl"):
        return _bad("File must be a .jsonl file.")

    raw = await input_file_jsonl.read()
    try:
        upload_id = save_upload(raw)
    except UploadError as exc:
        return _bad(str(exc))
    return UploadResponse(upload_id=upload_id)


# ---------------------------------------------------------------------------
# GET /api/jobs
# ---------------------------------------------------------------------------


@jobs_router.get("", response_model=list[JobSummary])
async def list_jobs_endpoint(
    status: str | None = None,
    alias: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[JobSummary]:
    return list_jobs(status=status, alias=alias, limit=limit, offset=offset)


# ---------------------------------------------------------------------------
# T6 — POST /api/jobs
# ---------------------------------------------------------------------------


@jobs_router.post("", response_model=None)
async def create_job(body: JobSubmitRequest, background_tasks: BackgroundTasks) -> JobStatus | JSONResponse:
    req = SubmissionRequest(**body.model_dump(exclude={"upload_id"}))

    try:
        sub = parse_submission_from_upload_id(body.upload_id, req)
    except SubmissionError as exc:
        return _bad(str(exc))

    job_status = create_job_record(
        rows=sub.rows,
        prompt_template=sub.prompt_template,
        sota_model=sub.sota_model,
        alias=sub.alias,
        workload_type=sub.workload_type,
        input_modality=sub.input_modality,
        detected_workload_details=sub.detected_workload_details,
        output_json_schema=sub.output_json_schema,
        eval_fields=sub.eval_fields,
        compare_models=sub.compare_models,
        judge_criteria=sub.judge_criteria,
        projection_by_num_records=sub.projection_by_num_records,
        target_sla_hours=sub.target_sla_hours,
        hosting_preference=sub.hosting_preference,
        managed_provider_custom_pricing=sub.managed_provider_custom_pricing,
        self_hosted_provider_custom_pricing=sub.self_hosted_provider_custom_pricing,
        golden_generation_config=sub.golden_generation_config,
    )
    background_tasks.add_task(run_pipeline, job_status.job_id)
    return job_status


# ---------------------------------------------------------------------------
# GET /api/jobs/{job_id}
# ---------------------------------------------------------------------------


@jobs_router.get("/{job_id}", response_model=JobDetail)
async def get_job(job_id: str) -> JobDetail:
    detail = _get_job(job_id)
    if detail is None:
        raise HTTPException(status_code=404, detail={"error": f"Job '{job_id}' not found."})
    return detail


# ---------------------------------------------------------------------------
# T7 — GET /api/jobs/{job_id}/status
# ---------------------------------------------------------------------------


@jobs_router.get("/{job_id}/status")
async def job_status(job_id: str) -> JobStatus:
    status = get_job_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail={"error": f"Job '{job_id}' not found."})
    return status


# ---------------------------------------------------------------------------
# T7a — PATCH /api/jobs/{job_id}/compare_models
# ---------------------------------------------------------------------------


class CompareModelsPatch(BaseModel):
    op: str  # "add" | "remove"
    compare_models: list[RunEntry]


@jobs_router.patch("/{job_id}/compare_models", response_model=None)
async def patch_job_compare_models(
    job_id: str, body: CompareModelsPatch, background_tasks: BackgroundTasks
) -> JobStatus | JSONResponse:
    try:
        job = patch_compare_models(job_id, body.op, [e.model_dump() for e in body.compare_models])
    except JobNotFoundError:
        raise HTTPException(status_code=404, detail={"error": f"Job '{job_id}' not found."})
    except JobRunningError:
        raise HTTPException(
            status_code=409,
            detail={"error": "Job is currently running. Wait for it to finish before modifying run entries."},
        )
    except EmptyCompareModelsError:
        return _bad("Cannot remove all run entries. At least one must remain.")
    except ValueError as exc:
        return _bad(str(exc))

    background_tasks.add_task(run_pipeline, job_id)
    return JobStatus(job_id=job.job_id, status=job.status)


# ---------------------------------------------------------------------------
# POST /api/jobs/{job_id}/run
# ---------------------------------------------------------------------------


class RunJobRequest(BaseModel):
    step: str | None = None
    force: bool = False
    compare_models: list[RunEntry] | None = None
    retry: bool = False
    run_models: list[str] | None = None


@jobs_router.post("/{job_id}/run", response_model=None)
async def run_job_endpoint(
    job_id: str, body: RunJobRequest, background_tasks: BackgroundTasks
) -> JobStatus | JSONResponse:
    status = get_job_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail={"error": f"Job '{job_id}' not found."})
    background_tasks.add_task(
        run_job,
        job_id,
        step=body.step,
        force=body.force,
        compare_models=[e.model_dump() for e in body.compare_models] if body.compare_models else None,
        retry=body.retry,
        run_models=body.run_models,
    )
    return status


# ---------------------------------------------------------------------------
# T17 — GET /api/jobs/{job_id}/report
# ---------------------------------------------------------------------------


@jobs_router.get("/{job_id}/report")
async def get_job_report(job_id: str) -> JSONResponse:
    status = get_job_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail={"error": f"Job '{job_id}' not found."})
    if status.status != "completed":
        raise HTTPException(status_code=409, detail={"error": f"Report not ready. Job status: {status.status}"})
    report_path = job_scale_and_cost_report_path(job_id)
    if not report_path.exists():
        raise HTTPException(status_code=404, detail={"error": "Report file not found."})
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    return JSONResponse(content=payload)


# ---------------------------------------------------------------------------
# T19 — POST /api/jobs/{job_id}/cancel
# ---------------------------------------------------------------------------


@jobs_router.post("/{job_id}/cancel", response_model=None)
async def cancel_job_endpoint(job_id: str) -> JobStatus | JSONResponse:
    try:
        job = cancel_job(job_id)
    except JobNotFoundError:
        raise HTTPException(status_code=404, detail={"error": f"Job '{job_id}' not found."})
    except JobRunningError:
        raise HTTPException(
            status_code=409,
            detail={"error": "Job is already in a terminal state and cannot be cancelled."},
        )
    return JobStatus(job_id=job.job_id, status=job.status)
