"""
T11 — LangGraph orchestrator.

Wires pipeline nodes into a sequential graph:
  validate → detect_workload → generate_golden → run_compare_models_inference → evaluate →
  create_scale_and_cost_projection

Each node reads the job record, does work, writes its output blob + new status to SQLite.
On node exception: status="failed" + error stored; graph halts.

T19 — Resumable pipeline:
  - run_pipeline(step=...) runs only the named node (all others skipped).
  - run_pipeline(force=True) clears blobs + step_status from the target step onward (or all if no step).
  - Each node checks for status="cancelled" before starting and halts cleanly.
  - Without step/force, resumes from the first node whose output blob is missing.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlmodel import select

from src.agents.detect_workload import DetectionResult, detect_workload, validate_prompt_consistency
from src.agents.generate_golden_dataset import generate_golden_dataset
from src.agents.pipeline_log import _log as _pipeline_log
from src.agents.run_compare_models_evaluation import run_compare_models_evaluation
from src.agents.run_compare_models_inference import run_compare_models_inference
from src.agents.scale_and_cost_projection import create_scale_and_cost_projection
from src.db import get_session
from src.providers.client import GetManagedModelProviderAPIKeyFunc, get_key_from_db
from src.schemas.db import JobConfig
from src.services.job_constants import TERMINAL_STATUSES, JobOutputFields, JobStatusStr
from src.utils.jsonl import read_rows

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Blob metadata: output blob field, required upstream blob, downstream blobs to
# clear when force=True.
# ---------------------------------------------------------------------------

# Map from node name → (output_blob_field, required_upstream_blob, downstream_blobs_to_clear)
_NODE_OUTPUT_DEPENDENCY_MAP: dict[str, tuple[str | None, str | None, list[str]]] = {
    JobStatusStr.VALIDATE_INPUT: (None, None, []),
    JobStatusStr.DETECTING_WORKLOAD_TYPE: (None, None, []),
    JobStatusStr.GENERATE_GOLDEN_DATASET: (
        JobOutputFields.GENERATED_GOLDEN_DATASET,
        None,
        [
            JobOutputFields.EVALUATING_INFERENCE_OUTPUT,
            JobOutputFields.SCALE_AND_COST_PROJECTION_REPORT,
        ],
    ),
    JobStatusStr.RUN_COMPARE_MODELS_INFERENCE: (
        JobOutputFields.COMPARE_MODELS_INFERENCE_OUTPUT,
        JobOutputFields.GENERATED_GOLDEN_DATASET,
        [
            JobOutputFields.EVALUATING_INFERENCE_OUTPUT,
            JobOutputFields.SCALE_AND_COST_PROJECTION_REPORT,
        ],
    ),
    JobStatusStr.EVALUATING_INFERENCE_OUTPUT: (
        JobOutputFields.EVALUATING_INFERENCE_OUTPUT,
        JobOutputFields.COMPARE_MODELS_INFERENCE_OUTPUT,
        [JobOutputFields.SCALE_AND_COST_PROJECTION_REPORT],
    ),
    JobStatusStr.CREATE_SCALE_AND_COST_PROJECTION_REPORT: (
        JobOutputFields.SCALE_AND_COST_PROJECTION_REPORT,
        JobOutputFields.EVALUATING_INFERENCE_OUTPUT,
        [],
    ),
    JobStatusStr.CREATE_SCALE_AND_COST_PROJECTION_REPORT_HTML: (
        None,
        JobOutputFields.SCALE_AND_COST_PROJECTION_REPORT,
        [],
    ),
}

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _update_job(job_id: str, **fields: Any) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with get_session() as session:
        job = session.exec(select(JobConfig).where(JobConfig.job_id == job_id)).first()
        if job is None:
            raise ValueError(f"Job '{job_id}' not found.")
        for key, value in fields.items():
            setattr(job, key, value)
        job.updated_at = now
        session.add(job)
        session.commit()


def _get_job(job_id: str) -> JobConfig:
    with get_session() as session:
        job = session.exec(select(JobConfig).where(JobConfig.job_id == job_id)).one_or_none()
        if job is None:
            raise ValueError(f"Job '{job_id}' not found.")
        session.expunge(job)
        return job


def _fail(job_id: str, error: str) -> None:
    _pipeline_log(job_id, "pipeline", f"failed: {error}", level="error", logger=logger)
    _update_job(job_id, status=JobStatusStr.FAILED, error_message=error)


def _clear_blobs(job_id: str, blob_fields: list[str]) -> None:
    """Set the given blob fields to None on the job record."""
    if not blob_fields:
        return
    _update_job(job_id, **{f: None for f in blob_fields})


# ---------------------------------------------------------------------------
# Pipeline nodes
# ---------------------------------------------------------------------------


def _node_validate(job_id: str) -> None:
    """Structural validation — prompt consistency check across all rows."""
    job = _get_job(job_id)
    _update_job(job_id, status=JobStatusStr.VALIDATE_INPUT)
    if not job.prompt_template:
        raise ValueError("prompt_template is required.")
    rows = read_rows(Path(job.input_file_jsonl_path)) if job.input_file_jsonl_path else []
    validate_prompt_consistency(job.prompt_template, rows)


def _node_detect_workload(
    job_id: str, get_managed_model_provider_api_key_func: GetManagedModelProviderAPIKeyFunc = get_key_from_db
) -> None:
    """LLM workload classification. Low confidence with no override → fail."""
    job = _get_job(job_id)

    # Detection already ran at submission time (confidence is "high" or "override").
    # Re-using those results avoids a redundant LLM call and preserves the correct confidence label.
    _update_job(job_id, status=JobStatusStr.DETECTING_WORKLOAD_TYPE)

    prior = job.detected_workload_details or {}
    if prior.get("confidence") in ("high", "override"):
        return

    if not job.prompt_template:
        raise ValueError("prompt_template is required for workload detection.")
    rows = read_rows(Path(job.input_file_jsonl_path)) if job.input_file_jsonl_path else []
    result: DetectionResult = detect_workload(
        job.prompt_template,
        rows,
        model=job.sota_model,
        get_managed_model_provider_api_key_func=get_managed_model_provider_api_key_func,
    )

    if result.confidence == "low":
        raise ValueError(
            f"Workload type could not be determined (confidence=low): {result.confidence_note} "
            "Set workload_type explicitly in the submission to proceed."
        )

    _update_job(
        job_id,
        status=JobStatusStr.DETECTING_WORKLOAD_TYPE,
        workload_type=result.workload_type,
        input_modality=result.modality,
        detected_workload_details={
            "workload_type": result.workload_type,
            "modality": result.modality,
            "confidence": result.confidence,
            "confidence_note": result.confidence_note,
        },
    )


def _node_generate_golden_dataset(
    job_id: str,
    output_dir: Path | None = None,
    on_progress: Any = None,
    retry: bool = False,
    get_managed_model_provider_api_key_func: GetManagedModelProviderAPIKeyFunc = get_key_from_db,
) -> None:
    """Golden dataset generation via sota_model."""
    generate_golden_dataset(
        job_id,
        output_dir=output_dir,
        on_progress=on_progress,
        retry=retry,
        get_managed_model_provider_api_key_func=get_managed_model_provider_api_key_func,
    )


def _node_run_compare_models_inference(
    job_id: str,
    output_dir: Path | None = None,
    on_progress: Any = None,
    on_model_start: Any = None,
    on_model_complete: Any = None,
    on_model_load: Any = None,
    on_model_unload: Any = None,
    retry: bool = False,
    get_managed_model_provider_api_key_func: GetManagedModelProviderAPIKeyFunc = get_key_from_db,
    run_models: list[str] | None = None,
    auto_load_and_unload_ollama_models: bool = False,
) -> None:
    job = _get_job(job_id)
    if not job.compare_models:
        raise ValueError(
            "compare_models is empty — no candidate models to run inference on.\n"
            f"Add models with:\n"
            f"  luna8i-judge job run {job_id} --step run_compare_models_inference "
            f'--compare-models \'[{{"model": "openai/gpt-4o-mini", "params": {{}}}}]\''
        )
    run_compare_models_inference(
        job_id,
        output_dir=output_dir,
        on_progress=on_progress,
        on_model_start=on_model_start,
        on_model_complete=on_model_complete,
        on_model_load=on_model_load,
        on_model_unload=on_model_unload,
        retry=retry,
        get_managed_model_provider_api_key_func=get_managed_model_provider_api_key_func,
        run_models=run_models,
        auto_load_and_unload_ollama_models=auto_load_and_unload_ollama_models,
    )


def _node_run_compare_models_evaluation(
    job_id: str,
    output_dir: Path | None = None,
    on_model_start: Any = None,
    on_model_complete: Any = None,
    on_progress: Any = None,
    get_managed_model_provider_api_key_func: GetManagedModelProviderAPIKeyFunc = get_key_from_db,
) -> None:
    run_compare_models_evaluation(
        job_id,
        output_dir=output_dir,
        on_model_start=on_model_start,
        on_model_complete=on_model_complete,
        on_progress=on_progress,
        get_managed_model_provider_api_key_func=get_managed_model_provider_api_key_func,
    )


def _node_create_scale_and_cost_projection(
    job_id: str, output_dir: Path | None = None, html_output: Path | None = None
) -> None:
    if html_output is None:
        raise ValueError("html_output is required for the create_scale_and_cost_projection step")
    create_scale_and_cost_projection(job_id, output_dir=output_dir, html_output=html_output)
    _update_job(job_id, status=JobStatusStr.COMPLETED)
    _pipeline_log(
        job_id,
        JobStatusStr.CREATE_SCALE_AND_COST_PROJECTION_REPORT,
        "Created scale and cost projection report",
        logger=logger,
    )
    if html_output is not None:
        _pipeline_log(
            job_id,
            JobStatusStr.CREATE_SCALE_AND_COST_PROJECTION_REPORT,
            f"HTML report written to {html_output.resolve()}",
            logger=logger,
        )


def _node_create_scale_and_cost_projection_report_html(
    job_id: str, output_dir: Path | None = None, html_output: Path | None = None
) -> None:
    if html_output is None:
        raise ValueError("html_output is required for create_scale_and_cost_projection_report_html step")

    from src.services.job_utils import job_scale_and_cost_report_path
    from src.services.report import render_html_report

    report_json = job_scale_and_cost_report_path(job_id, output_dir)
    render_html_report(report_json, html_output)
    _pipeline_log(
        job_id,
        JobStatusStr.CREATE_SCALE_AND_COST_PROJECTION_REPORT_HTML,
        f"HTML report written to {html_output.resolve()}",
        logger=logger,
    )


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

_NODES = [
    (JobStatusStr.VALIDATE_INPUT, _node_validate),
    (JobStatusStr.DETECTING_WORKLOAD_TYPE, _node_detect_workload),
    (JobStatusStr.GENERATE_GOLDEN_DATASET, _node_generate_golden_dataset),
    (JobStatusStr.RUN_COMPARE_MODELS_INFERENCE, _node_run_compare_models_inference),
    (JobStatusStr.EVALUATING_INFERENCE_OUTPUT, _node_run_compare_models_evaluation),
    (JobStatusStr.CREATE_SCALE_AND_COST_PROJECTION_REPORT, _node_create_scale_and_cost_projection),
    (JobStatusStr.CREATE_SCALE_AND_COST_PROJECTION_REPORT_HTML, _node_create_scale_and_cost_projection_report_html),
]

_NODE_NAMES = [name for name, _ in _NODES]


def run_pipeline(
    job_id: str,
    step: str | None = None,
    force: bool = False,
    output_dir: Path | None = None,
    html_output: Path | None = None,
    on_node_start: Any = None,
    on_node_complete: Any = None,
    on_node_failed: Any = None,
    on_generate_golden_dataset_progress: Any = None,
    on_run_inference_progress: Any = None,
    on_run_inference_model_start: Any = None,
    on_run_inference_model_complete: Any = None,
    on_run_inference_model_load: Any = None,
    on_run_inference_model_unload: Any = None,
    on_run_evaluation_model_start: Any = None,
    on_run_evaluation_model_complete: Any = None,
    on_run_evaluation_progress: Any = None,
    retry: bool = False,
    get_managed_model_provider_api_key_func: GetManagedModelProviderAPIKeyFunc = get_key_from_db,
    run_models: list[str] | None = None,
    auto_load_and_unload_ollama_models: bool = False,
) -> None:
    """
    Entry point called by the submission router via BackgroundTasks.

    step: node name constant (e.g. JobStatusStr.RUNNING_INFERENCE) — run only this node.
          When None, resumes from the first incomplete node (blob missing).
    force: clear the target node's blob + downstream before running.
    """

    def _log(node: str, message: str, level: str = "info") -> None:
        _pipeline_log(job_id, node, message, level=level, logger=logger)

    def _record_node_result(node: str, elapsed_s: float, status: str) -> None:
        with get_session() as session:
            job = session.exec(select(JobConfig).where(JobConfig.job_id == job_id)).first()
            if job is None:
                return
            job.step_durations = {**(job.step_durations or {}), node: elapsed_s}
            job.step_status = {**(job.step_status or {}), node: status}
            job.updated_at = datetime.now(timezone.utc).isoformat()
            session.add(job)
            session.commit()

    import time

    pipeline_start = time.monotonic()
    _log("pipeline", "started")

    # ------------------------------------------------------------------
    # Pre-flight: validate step, check prerequisites, apply force-clear
    # ------------------------------------------------------------------
    if step is not None:
        if step not in _NODE_OUTPUT_DEPENDENCY_MAP:
            _fail(job_id, f"Unknown step '{step}'. Valid steps: {list(_NODE_OUTPUT_DEPENDENCY_MAP)}")
            return

        output_blob, required_upstream, downstream = _NODE_OUTPUT_DEPENDENCY_MAP[step]
        if required_upstream is not None:
            job = _get_job(job_id)
            if getattr(job, required_upstream, None) is None:
                msg = f"{required_upstream} missing — run the preceding step first"
                _fail(job_id, msg)
                raise ValueError(msg)

    if force:
        # Determine scope: step-targeted rerun clears that step + downstream;
        # full-pipeline rerun clears everything.
        if step is not None:
            output_blob, _, downstream = _NODE_OUTPUT_DEPENDENCY_MAP[step]
            blobs_to_clear = ([output_blob] if output_blob else []) + downstream
            step_idx = _NODE_NAMES.index(step)
            steps_to_reset = _NODE_NAMES[step_idx:]
        else:
            blobs_to_clear = [
                JobOutputFields.GENERATED_GOLDEN_DATASET,
                JobOutputFields.EVALUATING_INFERENCE_OUTPUT,
                JobOutputFields.SCALE_AND_COST_PROJECTION_REPORT,
            ]
            steps_to_reset = _NODE_NAMES

        _clear_blobs(job_id, blobs_to_clear)
        _log("pipeline", f"force-cleared blobs: {blobs_to_clear}")

        stuck_job = _get_job(job_id)
        new_step_status = {k: v for k, v in (stuck_job.step_status or {}).items() if k not in steps_to_reset}
        update_kwargs: dict[str, Any] = {"step_status": new_step_status or None, "step_durations": None}
        if stuck_job.status not in TERMINAL_STATUSES and stuck_job.status != JobStatusStr.QUEUED:
            update_kwargs["status"] = JobStatusStr.QUEUED
            update_kwargs["error_message"] = None
        _update_job(job_id, **update_kwargs)

    for name, node_fn in _NODES:
        # ------------------------------------------------------------------
        # Cancellation check — before each node
        # ------------------------------------------------------------------
        current_job = _get_job(job_id)
        if current_job.status == JobStatusStr.CANCELLED:
            _log("pipeline", "cancelled. closing...")
            return

        # ------------------------------------------------------------------
        # Skip logic
        # ------------------------------------------------------------------
        if step is not None:
            # Single-step mode: skip every node except the target
            if name != step:
                _log(name, "skipping: not the target step")
                continue
        else:
            # Resume mode: skip nodes that previously completed successfully.
            job_snapshot = _get_job(job_id)
            if (job_snapshot.step_status or {}).get(name) == "completed" and not force:
                _log(name, f"skipping {name}: already completed")
                continue

        # ------------------------------------------------------------------
        # Run the node
        # ------------------------------------------------------------------
        if on_node_start:
            on_node_start(name)
        _log(name, "starting")
        node_start = time.monotonic()
        _record_node_result(name, 0.0, "running")
        try:

            def _omit_none(**kw: Any) -> dict[str, Any]:
                return {k: v for k, v in kw.items() if v is not None}

            _node_kwargs: dict[str, dict[str, Any]] = {
                JobStatusStr.DETECTING_WORKLOAD_TYPE: _omit_none(
                    get_managed_model_provider_api_key_func=get_managed_model_provider_api_key_func,
                ),
                JobStatusStr.GENERATE_GOLDEN_DATASET: _omit_none(
                    output_dir=output_dir,
                    retry=retry,
                    get_managed_model_provider_api_key_func=get_managed_model_provider_api_key_func,
                    on_progress=on_generate_golden_dataset_progress,
                ),
                JobStatusStr.RUN_COMPARE_MODELS_INFERENCE: _omit_none(
                    output_dir=output_dir,
                    retry=retry,
                    get_managed_model_provider_api_key_func=get_managed_model_provider_api_key_func,
                    on_progress=on_run_inference_progress,
                    on_model_start=on_run_inference_model_start,
                    on_model_complete=on_run_inference_model_complete,
                    on_model_load=on_run_inference_model_load,
                    on_model_unload=on_run_inference_model_unload,
                    run_models=run_models,
                    auto_load_and_unload_ollama_models=auto_load_and_unload_ollama_models or None,
                ),
                JobStatusStr.EVALUATING_INFERENCE_OUTPUT: _omit_none(
                    output_dir=output_dir,
                    get_managed_model_provider_api_key_func=get_managed_model_provider_api_key_func,
                    on_model_start=on_run_evaluation_model_start,
                    on_model_complete=on_run_evaluation_model_complete,
                    on_progress=on_run_evaluation_progress,
                ),
                JobStatusStr.CREATE_SCALE_AND_COST_PROJECTION_REPORT: _omit_none(
                    output_dir=output_dir,
                    html_output=html_output,
                ),
                JobStatusStr.CREATE_SCALE_AND_COST_PROJECTION_REPORT_HTML: _omit_none(
                    output_dir=output_dir,
                    html_output=html_output,
                ),
            }
            node_fn(job_id, **_node_kwargs.get(name, {}))
        except Exception as exc:
            elapsed = round(time.monotonic() - node_start, 3)
            _record_node_result(name, elapsed, "failed")
            _fail(job_id, str(exc))
            _log(name, f"halted: {exc}", level="error")
            if on_node_failed:
                on_node_failed(name, str(exc))
            return
        elapsed = round(time.monotonic() - node_start, 3)
        _record_node_result(name, elapsed, "completed")
        _log(name, "done")
        if on_node_complete:
            on_node_complete(name)

    total = round(time.monotonic() - pipeline_start, 3)
    _record_node_result("total", total, "completed")
    _log("pipeline", "complete")
