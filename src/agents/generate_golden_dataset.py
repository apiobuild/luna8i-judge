"""
T10 — Golden dataset generation.

Calls sota_model on all input rows; stores output blob on job record and writes
golden_dataset.jsonl to $DATA_DIR/output/{job_id}/.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlmodel import select

from src.agents.inference_runner import run
from src.agents.pipeline_log import _log as _pipeline_log
from src.db import get_session
from src.env import settings
from src.providers.adapters import GenerationConfig, Message, ModelResponse
from src.providers.client import GetManagedModelProviderAPIKeyFunc, get_client, get_key_from_db
from src.schemas.db import JobConfig
from src.services.job_constants import JobStatusStr
from src.services.job_utils import job_golden_path
from src.utils.jsonl import read_rows

logger = logging.getLogger(__name__)


def generate_golden_dataset(
    job_id: str,
    output_dir: Path | None = None,
    on_progress: Any = None,
    retry: bool = False,
    get_managed_model_provider_api_key_func: GetManagedModelProviderAPIKeyFunc = get_key_from_db,
) -> None:
    with get_session() as session:
        job = session.exec(select(JobConfig).where(JobConfig.job_id == job_id)).first()
        if job is None:
            raise ValueError(f"Job '{job_id}' not found.")

        sota_model = job.sota_model or settings.DEFAULT_SOTA_MODEL
        input_path = job.input_file_jsonl_path
        output_json_schema = job.output_json_schema
        raw_gen_config: dict = job.golden_dataset_generation_config or {}

    temperature: float = raw_gen_config.get("temperature", 0.0)
    top_p: float = raw_gen_config.get("top_p", 1.0)

    config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        response_format=output_json_schema,
    )

    def _log(message: str, level: str = "info", log_only: bool = False) -> None:
        _pipeline_log(job_id, "generate_golden_dataset", message, level=level, logger=logger, log_only=log_only)

    if not input_path:
        raise ValueError(f"Job '{job_id}' has no input_file_jsonl_path.")
    rows = read_rows(Path(input_path))

    client = get_client(sota_model, get_managed_model_provider_api_key_func=get_managed_model_provider_api_key_func)
    started_at = time.monotonic()
    _log(f"Starting golden generation: {len(rows)} rows via {sota_model}")

    if output_dir is None:
        output_dir = Path(settings.DATA_DIR) / "output" / job_id
    output_path = job_golden_path(job_id, output_dir)

    def build_messages(row: dict) -> list[Message]:
        return [Message(role=m["role"], content=m["content"]) for m in row.get("messages", [])]

    def parse_response(resp: ModelResponse, row: dict) -> dict:
        return {"input": row}

    def _on_progress(completed: int, total: int) -> None:
        if on_progress:
            on_progress(completed, total, sota_model)

    results = run(
        rows=rows,
        output_path=output_path,
        build_messages=build_messages,
        client=client,
        config=config,
        parse_response=parse_response,
        on_progress=_on_progress,
        retry=retry,
        log=_log,
    )

    _log(f"Wrote {len(results)} golden entries to {output_path}")

    now = datetime.now(timezone.utc).isoformat()
    with get_session() as session:
        job = session.exec(select(JobConfig).where(JobConfig.job_id == job_id)).first()
        if job is None:
            raise ValueError(f"Job '{job_id}' not found.")
        job.generated_golden_dataset_path = str(output_path)
        job.golden_dataset_generation_config = {"temperature": temperature, "top_p": top_p}
        job.status = JobStatusStr.GENERATE_GOLDEN_DATASET
        job.updated_at = now
        session.add(job)
        session.commit()
        elapsed = time.monotonic() - started_at
        _log(f"Done: {len(results)} rows in {elapsed:.1f}s")
