"""
T11 + T19 tests — LangGraph orchestrator (job_runner.py)

Tests cover:
  - Happy path (all nodes mocked) → final status="done", all blobs written
  - detect_workload low confidence, no override → status="failed"
  - Node exception mid-graph → status="failed", subsequent nodes did not run
  - Status written after each node (verified in SQLite between calls)
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime, timezone

import pytest
from sqlmodel import Session, SQLModel, create_engine, select

import src.schemas.db  # noqa: F401 — register table models
from src.agents.job_runner import run_pipeline
from src.schemas.db import JobConfig
from src.services.job_constants import JobStatusStr

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_ROWS = [
    {"messages": [{"role": "user", "content": "Classify this: urgent complaint"}]},
    {"messages": [{"role": "user", "content": "Classify this: billing inquiry"}]},
]


@pytest.fixture
def db_session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


@pytest.fixture
def patched_session(db_session, monkeypatch):
    """Redirect get_session in job_runner to use the test DB."""

    @contextmanager
    def _get_session():
        yield db_session

    monkeypatch.setattr("src.agents.job_runner.get_session", _get_session)
    monkeypatch.setattr("src.agents.pipeline_log.get_session", _get_session)
    return db_session


def _write_jsonl(rows: list[dict]) -> str:
    import tempfile

    fd, path = tempfile.mkstemp(suffix=".jsonl")
    import os

    with os.fdopen(fd, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return path


def _make_job(
    session: Session,
    rows: list[dict] | None = None,
    workload_type: str | None = "classification",
    prompt_template: str | None = "Classify this: .*",
) -> str:
    now = datetime.now(timezone.utc).isoformat()
    # Pre-populate detected_workload_details so the runner skips LLM detection
    # (mirrors what job_submission.py writes before the pipeline starts).
    details = (
        {
            "workload_type": workload_type,
            "modality": "text",
            "confidence": "high",
            "confidence_note": "pre-detected at submission",
        }
        if workload_type is not None
        else None
    )
    job = JobConfig(
        job_id="test-job-t11",
        status=JobStatusStr.QUEUED,
        sota_model="gemini/gemini-2.0-flash",
        workload_type=workload_type,
        prompt_template=prompt_template,
        input_file_jsonl_path=_write_jsonl(rows or SAMPLE_ROWS),
        compare_models=[{"model": "openai/gpt-4o-mini", "params": {}}],
        detected_workload_details=details,
        created_at=now,
        updated_at=now,
    )
    session.add(job)
    session.commit()
    return job.job_id


def _get_job(session: Session, job_id: str) -> JobConfig:
    session.expire_all()
    job = session.exec(select(JobConfig).where(JobConfig.job_id == job_id)).first()
    assert job is not None
    return job


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _patch_run_inference(monkeypatch, patched_session):
    """Patch run_compare_models_inference to write the running status via _update_job (keeps recording intact)."""

    def _fake_inference(job_id: str, **_) -> None:
        import src.agents.job_runner as _runner

        _runner._update_job(job_id, status=JobStatusStr.RUN_COMPARE_MODELS_INFERENCE)

    monkeypatch.setattr("src.agents.job_runner.run_compare_models_inference", _fake_inference)


def _patch_evaluate(monkeypatch):
    """Patch evaluate to write the evaluating status without doing real scoring."""

    def _fake_evaluate(job_id: str, **_) -> None:
        import src.agents.job_runner as _runner

        _runner._update_job(
            job_id,
            status=JobStatusStr.EVALUATING_INFERENCE_OUTPUT,
            evaluating_inference_output_path="",
        )

    monkeypatch.setattr("src.agents.job_runner.run_compare_models_evaluation", _fake_evaluate)


def _patch_scale_projection(monkeypatch):
    """Patch scale projection and HTML render steps to no-ops."""

    def _fake_create_scale_and_cost_projection(job_id: str, **_) -> None:
        import src.agents.job_runner as _runner

        _runner._update_job(
            job_id,
            status=JobStatusStr.CREATE_SCALE_AND_COST_PROJECTION_REPORT,
            scale_and_cost_projection_report_path="",
        )

    monkeypatch.setattr(
        "src.agents.job_runner.create_scale_and_cost_projection", _fake_create_scale_and_cost_projection
    )
    monkeypatch.setattr("src.services.report.render_html_report", lambda *_a, **_kw: None)


def _patch_golden(monkeypatch, patched_session):
    """Patch generate_golden_dataset to write the generate_golden_dataset status directly."""

    def _fake_golden(job_id: str, **_) -> None:
        now = datetime.now(timezone.utc).isoformat()
        golden_path = _write_jsonl([{"row_index": 0, "output": "cat1"}])
        job = patched_session.exec(select(JobConfig).where(JobConfig.job_id == job_id)).first()
        job.generated_golden_dataset_path = golden_path
        job.status = JobStatusStr.GENERATE_GOLDEN_DATASET
        job.updated_at = now
        patched_session.add(job)
        patched_session.commit()

    monkeypatch.setattr("src.agents.job_runner.generate_golden_dataset", _fake_golden)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_happy_path_final_status_done(patched_session, monkeypatch, tmp_path):
    """All nodes complete → status='done', all output blobs present."""
    job_id = _make_job(patched_session)
    _patch_golden(monkeypatch, patched_session)
    _patch_run_inference(monkeypatch, patched_session)
    _patch_evaluate(monkeypatch)
    _patch_scale_projection(monkeypatch)

    run_pipeline(job_id, html_output=tmp_path / "report.html")

    job = _get_job(patched_session, job_id)
    assert job.status == JobStatusStr.COMPLETED
    assert job.evaluating_inference_output_path == ""
    assert job.scale_and_cost_projection_report_path == ""
    assert job.error_message is None


def test_detect_workload_low_confidence_no_override_fails(patched_session, monkeypatch):
    """
    When detect_workload returns low confidence and no workload_type override is set,
    the graph should halt with status='failed'.
    """
    # prompt_template matches the sample rows so validate_prompt_consistency passes,
    # but our mocked detect_workload will return low confidence.
    job_id = _make_job(patched_session, workload_type=None, prompt_template="Classify this: .*")

    from src.agents.detect_workload import DetectionResult

    low_conf_result = DetectionResult(
        workload_type=None,
        modality="text",
        confidence="low",
        confidence_note="Prompt is ambiguous.",
    )
    monkeypatch.setattr("src.agents.job_runner.detect_workload", lambda *a, **kw: low_conf_result)
    _patch_scale_projection(monkeypatch)

    run_pipeline(job_id)

    job = _get_job(patched_session, job_id)
    assert job.status == JobStatusStr.FAILED
    assert job.error_message is not None
    err = job.error_message
    assert "confidence=low" in err
    assert "workload_type" in err


def test_node_exception_mid_graph_halts_and_subsequent_nodes_skipped(patched_session, monkeypatch):
    """
    Exception in generate_golden node → status='failed'; run_compare_models_inference and later nodes
    must NOT have run (inference_output stays None).
    """
    job_id = _make_job(patched_session)

    run_inference_called = []

    def _boom(job_id: str, **_) -> None:
        raise RuntimeError("golden gen exploded")

    def _spy_inference(job_id: str) -> None:
        run_inference_called.append(job_id)

    monkeypatch.setattr("src.agents.job_runner.generate_golden_dataset", _boom)
    monkeypatch.setattr("src.agents.job_runner._node_run_compare_models_inference", _spy_inference)

    run_pipeline(job_id)

    job = _get_job(patched_session, job_id)
    assert job.status == JobStatusStr.FAILED
    assert job.error_message is not None and "golden gen exploded" in job.error_message
    assert run_inference_called == [], "run_inference must not have run"


def test_status_written_after_each_node(patched_session, monkeypatch, tmp_path):
    """
    Verify that each node transition is persisted before the next node starts.
    We track the sequence of statuses by reading SQLite after each mock node call.
    """
    job_id = _make_job(patched_session)
    observed_statuses: list[str] = []

    original_update = __import__("src.agents.job_runner", fromlist=["_update_job"])._update_job

    def _recording_update(jid: str, **fields):
        original_update(jid, **fields)
        if "status" in fields:
            observed_statuses.append(fields["status"])

    monkeypatch.setattr("src.agents.job_runner._update_job", _recording_update)
    _patch_golden(monkeypatch, patched_session)
    _patch_run_inference(monkeypatch, patched_session)
    _patch_evaluate(monkeypatch)
    _patch_scale_projection(monkeypatch)

    run_pipeline(job_id, html_output=tmp_path / "report.html")

    assert JobStatusStr.VALIDATE_INPUT in observed_statuses
    assert JobStatusStr.DETECTING_WORKLOAD_TYPE in observed_statuses
    # golden_gen is set inside generate_golden_dataset (patched separately)
    assert JobStatusStr.RUN_COMPARE_MODELS_INFERENCE in observed_statuses
    assert JobStatusStr.EVALUATING_INFERENCE_OUTPUT in observed_statuses
    assert JobStatusStr.CREATE_SCALE_AND_COST_PROJECTION_REPORT in observed_statuses
    assert JobStatusStr.COMPLETED in observed_statuses

    # Order must be preserved
    expected_order = [
        JobStatusStr.VALIDATE_INPUT,
        JobStatusStr.DETECTING_WORKLOAD_TYPE,
        JobStatusStr.RUN_COMPARE_MODELS_INFERENCE,
        JobStatusStr.EVALUATING_INFERENCE_OUTPUT,
        JobStatusStr.CREATE_SCALE_AND_COST_PROJECTION_REPORT,
        JobStatusStr.COMPLETED,
    ]
    filtered = [s for s in observed_statuses if s in expected_order]
    assert filtered == expected_order


def test_validate_node_fails_without_prompt_template(patched_session):
    """validate node must fail immediately when prompt_template is absent."""
    job_id = _make_job(patched_session, prompt_template="")

    run_pipeline(job_id)

    job = _get_job(patched_session, job_id)
    assert job.status == "failed"
    assert job.error_message is not None and "prompt_template" in job.error_message


def test_detect_workload_node_fails_without_prompt_template(patched_session, monkeypatch):
    """detect_workload node must fail when prompt_template is absent and no override is set."""
    # Patch validate to pass so we reach detect_workload with a None prompt_template.
    monkeypatch.setattr(
        "src.agents.job_runner.validate_prompt_consistency",
        lambda template, rows: None,
    )
    job_id = _make_job(patched_session, workload_type=None, prompt_template="")

    run_pipeline(job_id)

    job = _get_job(patched_session, job_id)
    assert job.status == "failed"
    assert job.error_message is not None and "prompt_template" in job.error_message


# ---------------------------------------------------------------------------
# detect_workload node — confidence label regression tests
# ---------------------------------------------------------------------------


def _make_job_with_detection(
    session: Session,
    workload_type: str | None,
    detected_confidence: str | None,
) -> str:
    """Create a job with pre-populated detected_workload_details (simulates post-submission state)."""
    now = datetime.now(timezone.utc).isoformat()
    details = (
        {
            "workload_type": workload_type,
            "modality": "text",
            "confidence": detected_confidence,
            "confidence_note": "pre-detected at submission",
        }
        if detected_confidence is not None
        else None
    )
    job = JobConfig(
        job_id="test-job-detect",
        status="queued",
        sota_model="gemini/gemini-2.0-flash",
        workload_type=workload_type,
        prompt_template="Classify this: .*",
        input_file_jsonl_path=_write_jsonl(SAMPLE_ROWS),
        compare_models=[{"model": "openai/gpt-4o-mini", "params": {}}],
        detected_workload_details=details,
        created_at=now,
        updated_at=now,
    )
    session.add(job)
    session.commit()
    return job.job_id


def test_detect_workload_node_preserves_high_confidence_label(patched_session, monkeypatch, tmp_path):
    """
    When submission already ran LLM detection (confidence='high'), the runner must
    NOT overwrite it with 'override'. detect_workload should not be called at all.
    """
    job_id = _make_job_with_detection(patched_session, workload_type="captioning", detected_confidence="high")
    _patch_golden(monkeypatch, patched_session)
    _patch_run_inference(monkeypatch, patched_session)
    _patch_evaluate(monkeypatch)
    _patch_scale_projection(monkeypatch)

    llm_called = []
    monkeypatch.setattr("src.agents.job_runner.detect_workload", lambda *a, **kw: llm_called.append(1))

    run_pipeline(job_id, html_output=tmp_path / "report.html")

    job = _get_job(patched_session, job_id)
    assert job.status == JobStatusStr.COMPLETED
    assert llm_called == [], "detect_workload LLM must not be called when submission already detected high confidence"
    assert job.detected_workload_details is not None
    assert job.detected_workload_details["confidence"] == "high", (
        "confidence must remain 'high', not be overwritten to 'override'"
    )


def test_detect_workload_node_preserves_override_label(patched_session, monkeypatch, tmp_path):
    """
    When the user explicitly supplied workload_type (confidence='override'),
    the runner must keep confidence='override' and skip LLM detection.
    """
    job_id = _make_job_with_detection(patched_session, workload_type="classification", detected_confidence="override")
    _patch_golden(monkeypatch, patched_session)
    _patch_run_inference(monkeypatch, patched_session)
    _patch_evaluate(monkeypatch)
    _patch_scale_projection(monkeypatch)

    llm_called = []
    monkeypatch.setattr("src.agents.job_runner.detect_workload", lambda *a, **kw: llm_called.append(1))

    run_pipeline(job_id, html_output=tmp_path / "report.html")

    job = _get_job(patched_session, job_id)
    assert job.status == JobStatusStr.COMPLETED
    assert llm_called == []
    assert job.detected_workload_details is not None
    assert job.detected_workload_details["confidence"] == "override"


def test_detect_workload_node_runs_llm_when_no_prior_detection(patched_session, monkeypatch, tmp_path):
    """
    When detected_workload_details is absent (e.g. job created without submission step),
    the runner must call detect_workload and store the result.
    """
    job_id = _make_job_with_detection(patched_session, workload_type=None, detected_confidence=None)
    _patch_golden(monkeypatch, patched_session)
    _patch_run_inference(monkeypatch, patched_session)

    from src.agents.detect_workload import DetectionResult

    high_conf_result = DetectionResult(
        workload_type="classification",
        modality="text",
        confidence="high",
        confidence_note="Keywords matched.",
    )
    monkeypatch.setattr("src.agents.job_runner.detect_workload", lambda *a, **kw: high_conf_result)
    _patch_evaluate(monkeypatch)
    _patch_scale_projection(monkeypatch)

    run_pipeline(job_id, html_output=tmp_path / "report.html")

    job = _get_job(patched_session, job_id)
    assert job.status == JobStatusStr.COMPLETED
    assert job.workload_type == "classification"
    assert job.detected_workload_details is not None
    assert job.detected_workload_details["confidence"] == "high"


# ---------------------------------------------------------------------------
# T19 — Resumable pipeline tests
# ---------------------------------------------------------------------------


def _make_resumable_job(
    session: Session,
    golden_dataset: list | None = None,
    step_status: dict | None = None,
) -> str:
    """Create a job pre-seeded with optional blob data and step_status for run-job tests."""
    now = datetime.now(timezone.utc).isoformat()
    details = {
        "workload_type": "classification",
        "modality": "text",
        "confidence": "high",
        "confidence_note": "pre-detected",
    }
    golden_path = _write_jsonl(golden_dataset) if golden_dataset is not None else None
    job = JobConfig(
        job_id="test-run-job",
        status=JobStatusStr.QUEUED,
        sota_model="gemini/gemini-2.0-flash",
        workload_type="classification",
        prompt_template="Classify this: .*",
        input_file_jsonl_path=_write_jsonl(SAMPLE_ROWS),
        compare_models=[{"model": "openai/gpt-4o-mini", "params": {}}],
        detected_workload_details=details,
        generated_golden_dataset_path=golden_path,
        step_status=step_status,
        created_at=now,
        updated_at=now,
    )
    session.add(job)
    session.commit()
    return job.job_id


def test_run_pipeline_skips_completed_nodes_in_resume_mode(patched_session, monkeypatch, tmp_path):
    """Resume mode: nodes with step_status='completed' are skipped; others run."""
    # Seed validate, detect_workload, and generate_golden as already completed
    completed_nodes = {
        JobStatusStr.VALIDATE_INPUT: "completed",
        JobStatusStr.DETECTING_WORKLOAD_TYPE: "completed",
        JobStatusStr.GENERATE_GOLDEN_DATASET: "completed",
    }
    golden = [{"row_index": 0, "output": "cat1"}]
    job_id = _make_resumable_job(patched_session, golden_dataset=golden, step_status=completed_nodes)

    golden_called = []
    monkeypatch.setattr("src.agents.job_runner.generate_golden_dataset", lambda *a, **kw: golden_called.append(1))
    _patch_run_inference(monkeypatch, patched_session)
    _patch_evaluate(monkeypatch)
    _patch_scale_projection(monkeypatch)

    run_pipeline(job_id, html_output=tmp_path / "report.html")

    job = _get_job(patched_session, job_id)
    assert job.status == JobStatusStr.COMPLETED
    assert golden_called == [], "generate_golden_dataset must be skipped when step_status='completed'"


def test_run_pipeline_single_step_runs_only_target_node(patched_session, monkeypatch):
    """step='running_inference' with golden present → only inference runs; validate/detect/golden not called."""
    golden = [{"row_index": 0, "output": "cat1"}]
    job_id = _make_resumable_job(patched_session, golden_dataset=golden)

    validate_called = []
    detect_called = []
    golden_called = []

    monkeypatch.setattr("src.agents.job_runner.validate_prompt_consistency", lambda *a, **kw: validate_called.append(1))
    monkeypatch.setattr("src.agents.job_runner.detect_workload", lambda *a, **kw: detect_called.append(1))
    monkeypatch.setattr("src.agents.job_runner.generate_golden_dataset", lambda *a, **kw: golden_called.append(1))
    _patch_run_inference(monkeypatch, patched_session)

    run_pipeline(job_id, step=JobStatusStr.RUN_COMPARE_MODELS_INFERENCE)

    job = _get_job(patched_session, job_id)
    assert validate_called == [], "validate must not run in single-step mode"
    assert detect_called == [], "detect_workload must not run in single-step mode"
    assert golden_called == [], "generate_golden must not run in single-step mode"
    assert job.status == JobStatusStr.RUN_COMPARE_MODELS_INFERENCE


def test_run_pipeline_single_step_fails_when_upstream_blob_missing(patched_session, monkeypatch):
    """step='running_inference' with golden blob missing → ValueError before any node runs."""
    job_id = _make_resumable_job(patched_session, golden_dataset=None)

    inference_called = []
    monkeypatch.setattr(
        "src.agents.job_runner.run_compare_models_inference", lambda *a, **kw: inference_called.append(1)
    )

    with pytest.raises(ValueError, match="generated_golden_dataset_path missing"):
        run_pipeline(job_id, step=JobStatusStr.RUN_COMPARE_MODELS_INFERENCE)

    assert inference_called == []
    job = _get_job(patched_session, job_id)
    assert job.status == JobStatusStr.FAILED


def test_run_pipeline_force_clears_target_and_downstream_blobs(patched_session, monkeypatch):
    """step='running_inference' + force=True → downstream blobs cleared, then inference runs."""
    golden = [{"row_index": 0, "output": "cat1"}]
    job_id = _make_resumable_job(patched_session, golden_dataset=golden)
    # Pre-seed downstream blobs to verify they get cleared
    from src.agents.job_runner import _update_job

    _update_job(
        job_id,
        evaluating_inference_output_path="/old/eval",
        scale_and_cost_projection_report_path="/old/scale",
    )

    _patch_run_inference(monkeypatch, patched_session)

    run_pipeline(job_id, step=JobStatusStr.RUN_COMPARE_MODELS_INFERENCE, force=True)

    job = _get_job(patched_session, job_id)
    assert job.status == JobStatusStr.RUN_COMPARE_MODELS_INFERENCE
    assert job.evaluating_inference_output_path is None, "downstream blob must be cleared by force"
    assert job.scale_and_cost_projection_report_path is None, "downstream blob must be cleared by force"
    assert job.generated_golden_dataset_path is not None, "upstream blob must not be touched"


def test_run_pipeline_halts_when_job_is_cancelled(patched_session, monkeypatch):
    """Job status='cancelled' → pipeline detects at next node boundary and halts; completed blobs intact."""
    golden = [{"row_index": 0, "output": "cat1"}]
    job_id = _make_resumable_job(patched_session, golden_dataset=golden)

    inference_called = []

    def _cancelling_inference(jid: str, **kw) -> None:
        inference_called.append(jid)

    from src.agents.job_runner import _update_job

    _update_job(job_id, status=JobStatusStr.CANCELLED)

    monkeypatch.setattr("src.agents.job_runner.run_compare_models_inference", _cancelling_inference)

    run_pipeline(job_id)

    job = _get_job(patched_session, job_id)
    assert job.status == JobStatusStr.CANCELLED
    assert inference_called == [], "inference must not run when job is cancelled"
    assert job.generated_golden_dataset_path is not None, "completed blobs preserved on cancellation"


def test_run_pipeline_records_step_status_and_durations(patched_session, monkeypatch, tmp_path):
    """Each node records step_status='completed' and a positive duration in step_durations."""
    job_id = _make_job(patched_session)
    _patch_golden(monkeypatch, patched_session)
    _patch_run_inference(monkeypatch, patched_session)
    _patch_evaluate(monkeypatch)
    _patch_scale_projection(monkeypatch)

    run_pipeline(job_id, html_output=tmp_path / "report.html")

    job = _get_job(patched_session, job_id)
    assert job.step_status is not None
    for node_name in [
        JobStatusStr.VALIDATE_INPUT,
        JobStatusStr.DETECTING_WORKLOAD_TYPE,
        JobStatusStr.GENERATE_GOLDEN_DATASET,
        JobStatusStr.RUN_COMPARE_MODELS_INFERENCE,
        JobStatusStr.EVALUATING_INFERENCE_OUTPUT,
        JobStatusStr.CREATE_SCALE_AND_COST_PROJECTION_REPORT,
    ]:
        assert job.step_status.get(node_name) == "completed", f"{node_name} step_status must be 'completed'"
    assert job.step_durations is not None
    for node_name in job.step_status:
        assert node_name in job.step_durations, f"step_durations missing entry for {node_name}"
        assert job.step_durations[node_name] >= 0


def test_run_pipeline_on_node_start_fires_for_each_node_in_order(patched_session, monkeypatch, tmp_path):
    """on_node_start callback is invoked once per node that actually runs."""
    job_id = _make_job(patched_session)
    _patch_golden(monkeypatch, patched_session)
    _patch_run_inference(monkeypatch, patched_session)
    _patch_evaluate(monkeypatch)
    _patch_scale_projection(monkeypatch)

    started_nodes: list[str] = []
    run_pipeline(job_id, html_output=tmp_path / "report.html", on_node_start=started_nodes.append)

    from src.services.job_constants import JobStatusStr as S

    expected = [
        S.VALIDATE_INPUT,
        S.DETECTING_WORKLOAD_TYPE,
        S.GENERATE_GOLDEN_DATASET,
        S.RUN_COMPARE_MODELS_INFERENCE,
        S.EVALUATING_INFERENCE_OUTPUT,
        S.CREATE_SCALE_AND_COST_PROJECTION_REPORT,
        S.CREATE_SCALE_AND_COST_PROJECTION_REPORT_HTML,
    ]
    assert started_nodes == expected


def test_run_pipeline_unknown_step_name_fails_job(patched_session):
    """step='nonexistent' → status='failed', no nodes run."""
    job_id = _make_job(patched_session)

    run_pipeline(job_id, step="nonexistent_step")

    job = _get_job(patched_session, job_id)
    assert job.status == JobStatusStr.FAILED
    assert job.error_message is not None and "nonexistent_step" in job.error_message


def test_run_pipeline_node_exception_records_step_status_failed(patched_session, monkeypatch):
    """When a node raises, step_status for that node is 'failed'."""
    job_id = _make_job(patched_session)

    def _boom(*_args, **_kwargs) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr("src.agents.job_runner.generate_golden_dataset", _boom)

    run_pipeline(job_id)

    job = _get_job(patched_session, job_id)
    assert job.status == JobStatusStr.FAILED
    assert (job.step_status or {}).get(JobStatusStr.GENERATE_GOLDEN_DATASET) == "failed"
