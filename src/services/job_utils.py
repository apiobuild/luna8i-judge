from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from sqlmodel import select

from src.db import get_session
from src.env import settings
from src.schemas.db import JobConfig


def job_base_dir(job_id: str, output_dir: Path | None = None) -> Path:
    return output_dir if output_dir is not None else Path(settings.DATA_DIR) / "output" / job_id


def job_inference_dir(job_id: str, output_dir: Path | None = None) -> Path:
    return job_base_dir(job_id, output_dir) / "inference"


def job_evaluation_dir(job_id: str, output_dir: Path | None = None) -> Path:
    return job_base_dir(job_id, output_dir) / "evaluation"


def job_evaluation_rows_path(job_id: str, model_string: str, output_dir: Path | None = None) -> Path:
    safe = model_string.replace("/", "__")
    return job_evaluation_dir(job_id, output_dir) / f"{safe}.jsonl"


def job_evaluation_result_path(job_id: str, model_string: str, output_dir: Path | None = None) -> Path:
    safe = model_string.replace("/", "__")
    return job_evaluation_dir(job_id, output_dir) / f"{safe}_evaluation_result.json"


def job_golden_path(job_id: str, output_dir: Path | None = None) -> Path:
    return job_base_dir(job_id, output_dir) / "golden_dataset.jsonl"


def job_scale_and_cost_report_path(job_id: str, output_dir: Path | None = None) -> Path:
    return job_base_dir(job_id, output_dir) / "scale_and_cost_report.json"


def job_report_path(job_id: str, output_dir: Path | None = None) -> Path:
    return job_base_dir(job_id, output_dir) / "report.json"


def write_job_progress(job_id: str, field: str, progress: dict, job_status: str | None = None) -> None:
    """Persist a progress snapshot to a JSON field on the job record."""
    now = datetime.now(timezone.utc).isoformat()
    with get_session() as s:
        job = s.exec(select(JobConfig).where(JobConfig.job_id == job_id)).first()
        if job:
            setattr(job, field, progress)
            if job_status:
                job.status = job_status
            job.updated_at = now
            s.add(job)
            s.commit()
