"""
T11a — compare_models inference.

Runs each compare_model entry against all input rows and stores the
inference_output blob on the job record.

Output layout: one JSONL file per model under output/{job_id}/inference/.
Filename is derived from the model string with '/' replaced by '__'
(e.g. openai/gpt-4o-mini → openai__gpt-4o-mini.jsonl).
This lets you add a new candidate model by running inference again —
only the new model's file is written; existing files are untouched.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlmodel import select

from src.agents.inference_runner import run
from src.agents.pipeline_log import _log as _pipeline_log
from src.db import get_session
from src.providers.adapters import GenerationConfig, Message, ModelResponse
from src.providers.client import (
    GetLLMProviderHostFunc,
    GetManagedModelProviderAPIKeyFunc,
    get_client,
    get_host_from_db,
    get_key_from_db,
)
from src.providers.managed_model_provider_constants import Provider
from src.schemas.db import JobConfig
from src.services.job_constants import JobStatusStr
from src.services.job_utils import job_inference_dir, write_job_progress
from src.services.models import ollama_evict_with_output_fn, ollama_pull_with_output_fn
from src.utils.jsonl import model_filename, read_rows

logger = logging.getLogger(__name__)

_PROGRESS_WRITE_INTERVAL = 10


def _ollama_pull(model: str, log: Any) -> None:
    log(f"Pulling Ollama model '{model}' …")
    try:
        ollama_pull_with_output_fn(model, lambda msg: log(f"ollama pull [{model}]: {msg}", log_only=True))
    except Exception as exc:
        raise RuntimeError(f"Failed to pull Ollama model '{model}': {exc}") from exc
    log(f"Ollama model '{model}' ready")


def _ollama_unload(model: str, log: Any) -> None:
    log(f"Unloading Ollama model '{model}' …", log_only=True)
    ollama_evict_with_output_fn(model, lambda msg: log(msg, level="warning", log_only=True))


def run_compare_models_inference(
    job_id: str,
    output_dir: Path | None = None,
    on_progress: Any = None,
    on_model_start: Any = None,
    on_model_complete: Any = None,
    on_model_load: Any = None,
    on_model_unload: Any = None,
    retry: bool = False,
    get_managed_model_provider_api_key_func: GetManagedModelProviderAPIKeyFunc = get_key_from_db,
    get_llm_provider_host_func: GetLLMProviderHostFunc = get_host_from_db,
    run_models: list[str] | None = None,
    auto_load_and_unload_ollama_models: bool = False,
) -> None:
    with get_session() as session:
        job = session.exec(select(JobConfig).where(JobConfig.job_id == job_id)).first()
        if job is None:
            raise ValueError(f"Job '{job_id}' not found.")

        input_file_jsonl_path = job.input_file_jsonl_path
        all_compare_models: list[dict] = job.compare_models or []
        if run_models is not None:
            registered = {e["model"] for e in all_compare_models}
            unknown = [m for m in run_models if m not in registered]
            if unknown:
                raise ValueError(
                    f"run-models contains model(s) not in compare_models: {unknown}. Registered: {sorted(registered)}"
                )
            compare_models = [e for e in all_compare_models if e["model"] in run_models]
        else:
            compare_models = all_compare_models
        output_json_schema = job.output_json_schema

    if not input_file_jsonl_path:
        raise ValueError(f"Job '{job_id}' has no input file path.")
    rows: list[dict] = read_rows(Path(input_file_jsonl_path))

    def _log(message: str, level: str = "info", log_only: bool = False) -> None:
        _pipeline_log(
            job_id, "running_compare_models_inference", message, level=level, logger=logger, log_only=log_only
        )

    _log(f"Starting inference: {len(compare_models)} model(s), {len(rows)} rows")

    total_rows = len(rows)
    total_models = len(compare_models)

    model_progress: list[dict] = [
        {"model": e["model"], "status": "pending", "completed": 0, "total": total_rows, "failed": 0}
        for e in compare_models
    ]

    def _write_progress(model: str, model_idx: int, completed: int, job_status: str | None = None) -> None:
        model_progress[model_idx]["completed"] = completed
        write_job_progress(
            job_id,
            "inference_progress",
            {
                "current_model": model,
                "model_index": model_idx,
                "total_models": total_models,
                "completed": completed,
                "total": total_rows,
                "models": list(model_progress),
            },
            job_status=job_status,
        )

    model_summaries: list[dict] = []

    for model_index, entry in enumerate(compare_models):
        model_string = entry["model"]
        params: dict = entry.get("params", {})

        model_progress[model_index]["status"] = "running"
        _write_progress(model_string, model_index, 0, job_status=JobStatusStr.RUNNING)

        if on_model_start:
            on_model_start(model_string, model_index, total_models)

        model_path = job_inference_dir(job_id, output_dir) / model_filename(model_string)

        # Skip if file already complete with no errors
        if model_path.exists():
            existing = read_rows(model_path)
            failed = sum(1 for r in existing if r.get("output") is None)
            if failed == 0 and len(existing) >= len(rows):
                failure_rate = 0.0
                model_summaries.append({"model": model_string, "params": params, "failure_rate": failure_rate})
                _log(f"Model {model_string}: output already exists at {model_path}, skipping", log_only=True)
                model_progress[model_index].update({"status": "skipped", "completed": total_rows, "failed": 0})
                _write_progress(model_string, model_index, total_rows)
                if on_model_complete:
                    on_model_complete(model_string, model_index, total_models, 0, len(rows))
                continue

        temperature = float(params.get("temperature", 0.0))
        top_p = float(params.get("top_p", 1.0))
        extra_params = {k: v for k, v in params.items() if k not in ("temperature", "top_p")}

        config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            response_format=output_json_schema,
            extra_params=extra_params or None,
        )

        provider, _, model_name = model_string.partition("/")
        if auto_load_and_unload_ollama_models and provider == Provider.OLLAMA:
            if on_model_load:
                on_model_load(model_string)
            _ollama_pull(model_name, _log)

        client = get_client(
            model_string,
            get_managed_model_provider_api_key_func=get_managed_model_provider_api_key_func,
            get_llm_provider_host_func=get_llm_provider_host_func,
        )
        completed_rows_for_model = 0

        def build_messages(row: dict) -> list[Message]:
            return [Message(role=m["role"], content=m["content"]) for m in row.get("messages", [])]

        def parse_response(
            resp: ModelResponse,  # noqa: ARG001
            _row: dict,
            _ms: str = model_string,
            _p: dict = params,
            _ep: dict = extra_params,
        ) -> dict:
            result: dict = {"model": _ms, "params": _p}
            if _ep:
                result["extra_params"] = _ep
            return result

        def _on_progress(completed: int, total: int, _model_idx: int = model_index, _ms: str = model_string) -> None:
            nonlocal completed_rows_for_model
            completed_rows_for_model = completed
            if completed % _PROGRESS_WRITE_INTERVAL == 0 or completed == total:
                _write_progress(_ms, _model_idx, completed)
            if on_progress:
                on_progress(completed, total, _ms)

        try:
            written = run(
                rows=rows,
                output_path=model_path,
                build_messages=build_messages,
                client=client,
                config=config,
                parse_response=parse_response,
                on_progress=_on_progress,
                retry=retry,
                log=_log,
            )
        finally:
            if auto_load_and_unload_ollama_models and provider == Provider.OLLAMA:
                if on_model_unload:
                    on_model_unload(model_string)
                _ollama_unload(model_name, _log)

        failed = sum(1 for r in written if r.get("output") is None)
        failure_rate = failed / len(rows) if rows else 0.0
        model_summaries.append({"model": model_string, "params": params, "failure_rate": failure_rate})
        _log(f"Model {model_string}: {len(rows) - failed}/{len(rows)} rows succeeded (failure_rate={failure_rate:.2%})")
        _log(f"Wrote inference output to {model_path}")

        if failure_rate == 1.0:
            sample_error = next((r.get("error") for r in written if r.get("error")), "unknown error")
            model_progress[model_index].update({"status": "failed", "completed": total_rows, "failed": failed})
            _write_progress(model_string, model_index, total_rows)
            raise RuntimeError(f"Model {model_string}: all {len(rows)} rows failed. First error: {sample_error}")

        model_progress[model_index].update({"status": "done", "completed": total_rows, "failed": failed})
        _write_progress(model_string, model_index, total_rows)
        if on_model_complete:
            on_model_complete(model_string, model_index, total_models, failed, len(rows))

    now = datetime.now(timezone.utc).isoformat()
    with get_session() as session:
        job = session.exec(select(JobConfig).where(JobConfig.job_id == job_id)).first()
        if job is None:
            raise ValueError(f"Job '{job_id}' not found.")
        job.compare_models_inference_output_path = str(job_inference_dir(job_id, output_dir))
        job.status = JobStatusStr.RUNNING
        job.updated_at = now
        session.add(job)
        session.commit()

    _log("Done")
