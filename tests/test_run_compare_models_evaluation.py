"""
T11b tests — evaluate

Tests cover:
  - classification workload → accuracy and F1 computed correctly
  - extraction workload with eval_fields → field-level accuracy per field
  - summarization workload → LLM-as-judge called once per (candidate, golden) pair; aggregate score returned
  - row with null candidate output → excluded from metrics; failure_rate reflects it
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from sqlmodel import Session, SQLModel, create_engine

import src.schemas.db  # noqa: F401 — register table models
from src.agents.run_compare_models_evaluation import run_compare_models_evaluation as evaluate
from src.providers.adapters import ModelResponse
from src.schemas.db import JobConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db_engine(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)
    return engine


@pytest.fixture
def db_session(db_engine):
    with Session(db_engine) as session:
        yield session


@pytest.fixture
def patched_session(db_engine, db_session, monkeypatch):
    @contextmanager
    def _get_session():
        with Session(db_engine) as session:
            yield session

    monkeypatch.setattr("src.agents.run_compare_models_evaluation.get_session", _get_session)
    monkeypatch.setattr("src.agents.pipeline_log.get_session", _get_session)
    monkeypatch.setattr("src.services.job_utils.get_session", _get_session)
    return db_session


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_eval_file(eval_dir: str, model_string: str) -> list[dict]:
    filename = model_string.replace("/", "__") + "_evaluation_result.json"
    path = Path(eval_dir) / filename
    return [json.loads(path.read_text())]


def _write_golden(tmp_path: Path, job_id: str, rows: list[dict]) -> Path:
    path = tmp_path / "output" / job_id / "golden_dataset.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def _write_inference(tmp_path: Path, job_id: str, model_string: str, rows: list[dict]) -> None:
    filename = model_string.replace("/", "__") + ".jsonl"
    path = tmp_path / "output" / job_id / "inference" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def _make_job(
    session: Session,
    golden_path: Path,
    workload_type: str,
    compare_models: list[dict],
    eval_fields: list[str] | None = None,
    judge_criteria: list[dict] | None = None,
    job_id: str = "test-eval-job",
) -> str:
    now = datetime.now(timezone.utc).isoformat()
    job = JobConfig(
        job_id=job_id,
        status="run_compare_models_inference",
        sota_model="gemini/gemini-2.0-flash",
        workload_type=workload_type,
        generated_golden_dataset_path=str(golden_path),
        compare_models=compare_models,
        evaluation_fields=eval_fields,
        judge_criteria=judge_criteria,
        inference_output=[
            {"model": e["model"], "params": e.get("params", {}), "failure_rate": 0.0} for e in compare_models
        ],
        created_at=now,
        updated_at=now,
    )
    session.add(job)
    session.commit()
    return job.job_id


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_classification_accuracy_and_f1(patched_session, tmp_path, monkeypatch):
    """Classification workload: accuracy and F1 are computed correctly from golden vs. candidate."""
    job_id = "cls-job"
    # 4 rows: 3 correct, 1 wrong (positive predicted as negative)
    golden_rows = [
        {"row_index": 0, "input": {}, "output": "positive"},
        {"row_index": 1, "input": {}, "output": "negative"},
        {"row_index": 2, "input": {}, "output": "positive"},
        {"row_index": 3, "input": {}, "output": "negative"},
    ]
    inference_rows = [
        {"row_index": 0, "model": "gemini/gemini-2.0-flash", "params": {}, "output": "positive"},
        {"row_index": 1, "model": "gemini/gemini-2.0-flash", "params": {}, "output": "negative"},
        {"row_index": 2, "model": "gemini/gemini-2.0-flash", "params": {}, "output": "negative"},  # wrong
        {"row_index": 3, "model": "gemini/gemini-2.0-flash", "params": {}, "output": "negative"},
    ]

    golden_path = _write_golden(tmp_path, job_id, golden_rows)
    _write_inference(tmp_path, job_id, "gemini/gemini-2.0-flash", inference_rows)

    _make_job(
        patched_session,
        golden_path,
        "classification",
        [{"model": "gemini/gemini-2.0-flash", "params": {}}],
        job_id=job_id,
    )

    monkeypatch.setattr(
        "src.agents.run_compare_models_evaluation.settings",
        MagicMock(DATA_DIR=str(tmp_path), DEFAULT_SOTA_MODEL="gemini/gemini-2.0-flash"),
    )

    evaluate(job_id, output_dir=tmp_path / "output" / job_id)

    job = patched_session.get(JobConfig, job_id)
    assert job.status == "run_compare_models_evaluation"
    assert job.evaluating_inference_output_path is not None

    eval_rows = _read_eval_file(job.evaluating_inference_output_path, "gemini/gemini-2.0-flash")
    assert len(eval_rows) == 1
    result = eval_rows[0]
    assert result["workload_type"] == "classification"
    metrics = result["metrics"]
    # 3/4 correct
    assert metrics["accuracy"] == pytest.approx(0.75, abs=1e-3)
    assert "per_class" in metrics
    assert "positive" in metrics["per_class"]
    assert "negative" in metrics["per_class"]
    # macro F1: positive has 1 TP 1 FN → recall=0.5, precision=1.0, F1=0.667
    #           negative has 2 TP 0 FN 1 FP → recall=1.0, precision=0.667, F1=0.8
    # macro = (0.667 + 0.8) / 2 ≈ 0.733
    assert metrics["macro_f1"] == pytest.approx(0.733, abs=0.01)
    assert metrics["confusion_matrix"]["positive"]["negative"] == 1  # FN for positive


def test_extraction_field_level_accuracy(patched_session, tmp_path, monkeypatch):
    """Extraction workload: field-level accuracy computed per eval_field."""
    job_id = "ext-job"
    golden_rows = [
        {"row_index": 0, "input": {}, "output": json.dumps({"name": "Alice", "date": "2024-01-15"})},
        {"row_index": 1, "input": {}, "output": json.dumps({"name": "Bob", "date": "2024-03-10"})},
        {"row_index": 2, "input": {}, "output": json.dumps({"name": "Carol", "date": "2024-06-01"})},
    ]
    # name: 2/3 correct (Bob wrong), date: 3/3 correct (semantic: "Jan 15, 2024" ≈ "2024-01-15")
    inference_rows = [
        {
            "row_index": 0,
            "model": "openai/gpt-4o-mini",
            "params": {},
            "output": json.dumps({"name": "Alice", "date": "Jan 15, 2024"}),
        },
        {
            "row_index": 1,
            "model": "openai/gpt-4o-mini",
            "params": {},
            "output": json.dumps({"name": "Robert", "date": "2024-03-10"}),
        },
        {
            "row_index": 2,
            "model": "openai/gpt-4o-mini",
            "params": {},
            "output": json.dumps({"name": "Carol", "date": "2024-06-01"}),
        },
    ]

    golden_path = _write_golden(tmp_path, job_id, golden_rows)
    _write_inference(tmp_path, job_id, "openai/gpt-4o-mini", inference_rows)

    _make_job(
        patched_session,
        golden_path,
        "extraction",
        [{"model": "openai/gpt-4o-mini", "params": {}}],
        eval_fields=["name", "date"],
        job_id=job_id,
    )

    monkeypatch.setattr(
        "src.agents.run_compare_models_evaluation.settings",
        MagicMock(DATA_DIR=str(tmp_path), DEFAULT_SOTA_MODEL="gemini/gemini-2.0-flash"),
    )

    evaluate(job_id, output_dir=tmp_path / "output" / job_id)

    job = patched_session.get(JobConfig, job_id)
    eval_rows = _read_eval_file(job.evaluating_inference_output_path, "openai/gpt-4o-mini")
    result = eval_rows[0]
    metrics = result["metrics"]

    assert metrics["fields"]["name"] == pytest.approx(2 / 3, abs=0.01)
    # date: semantic match means "Jan 15, 2024" should match "2024-01-15" via substring/containment
    # Bob's date is exact match; Carol's is exact; Alice uses semantic
    assert metrics["fields"]["date"] is not None
    assert metrics["missing_field_rate"]["name"] == pytest.approx(0.0, abs=0.01)
    assert result["failure_rate"] == 0.0


def test_summarization_llm_judge_called_per_pair(patched_session, tmp_path, monkeypatch):
    """Summarization workload: LLM-as-judge called once per (candidate, golden) pair; aggregate returned."""
    job_id = "sum-job"
    golden_rows = [
        {"row_index": 0, "input": {}, "output": "Golden summary A."},
        {"row_index": 1, "input": {}, "output": "Golden summary B."},
    ]
    inference_rows = [
        {"row_index": 0, "model": "gemini/gemini-2.0-flash", "params": {}, "output": "Candidate summary A."},
        {"row_index": 1, "model": "gemini/gemini-2.0-flash", "params": {}, "output": "Candidate summary B."},
    ]

    golden_path = _write_golden(tmp_path, job_id, golden_rows)
    _write_inference(tmp_path, job_id, "gemini/gemini-2.0-flash", inference_rows)

    _make_job(
        patched_session,
        golden_path,
        "summarization",
        [{"model": "gemini/gemini-2.0-flash", "params": {}}],
        job_id=job_id,
    )

    monkeypatch.setattr(
        "src.agents.run_compare_models_evaluation.settings",
        MagicMock(DATA_DIR=str(tmp_path), DEFAULT_SOTA_MODEL="gemini/gemini-2.0-flash"),
    )

    judge_scores = {"faithfulness": 4, "completeness": 3, "conciseness": 5, "instruction following": 4}
    judge_response = json.dumps(judge_scores)
    mock_client = MagicMock()
    mock_client.complete.return_value = ModelResponse(content=judge_response, input_tokens=100, output_tokens=20)

    with patch("src.agents.run_compare_models_evaluation.get_client", return_value=mock_client):
        evaluate(job_id, output_dir=tmp_path / "output" / job_id)

    # complete() called once per (candidate, golden) pair = 2 rows × 1 model
    assert mock_client.complete.call_count == 2

    job = patched_session.get(JobConfig, job_id)
    assert job.status == "run_compare_models_evaluation"
    eval_rows = _read_eval_file(job.evaluating_inference_output_path, "gemini/gemini-2.0-flash")
    result = eval_rows[0]
    metrics = result["metrics"]

    assert metrics["evaluated_rows"] == 2
    assert metrics["overall_mean"] == pytest.approx(4.0, abs=0.01)  # (4+3+5+4)/4 = 4.0 per row, same for both rows
    for criterion in ("faithfulness", "completeness", "conciseness", "instruction following"):
        assert criterion in metrics["mean_scores"]
    assert metrics["judge_input_tokens"] == 200  # 100 × 2 rows
    assert metrics["judge_output_tokens"] == 40  # 20 × 2 rows


def test_null_candidate_output_excluded_from_metrics(patched_session, tmp_path, monkeypatch):
    """Rows with null candidate output are excluded from scoring; failure_rate reflects them."""
    job_id = "null-job"
    golden_rows = [
        {"row_index": 0, "input": {}, "output": "positive"},
        {"row_index": 1, "input": {}, "output": "negative"},
        {"row_index": 2, "input": {}, "output": "positive"},
    ]
    # Row 1 failed inference → output=None
    inference_rows = [
        {"row_index": 0, "model": "openai/gpt-4o-mini", "params": {}, "output": "positive"},
        {"row_index": 1, "model": "openai/gpt-4o-mini", "params": {}, "output": None, "error": "rate limit"},
        {"row_index": 2, "model": "openai/gpt-4o-mini", "params": {}, "output": "positive"},
    ]

    golden_path = _write_golden(tmp_path, job_id, golden_rows)
    _write_inference(tmp_path, job_id, "openai/gpt-4o-mini", inference_rows)

    _make_job(
        patched_session,
        golden_path,
        "classification",
        [{"model": "openai/gpt-4o-mini", "params": {}}],
        job_id=job_id,
    )

    monkeypatch.setattr(
        "src.agents.run_compare_models_evaluation.settings",
        MagicMock(DATA_DIR=str(tmp_path), DEFAULT_SOTA_MODEL="gemini/gemini-2.0-flash"),
    )

    evaluate(job_id, output_dir=tmp_path / "output" / job_id)

    job = patched_session.get(JobConfig, job_id)
    eval_rows = _read_eval_file(job.evaluating_inference_output_path, "openai/gpt-4o-mini")
    result = eval_rows[0]

    # failure_rate = 1/3 (one null out of 3 inference rows)
    assert result["failure_rate"] == pytest.approx(1 / 3, abs=0.01)
    metrics = result["metrics"]
    # Only 2 rows evaluated (row 1 skipped due to null output)
    assert metrics["evaluated_rows"] == 2
    # Both evaluated rows are correct (positive→positive, positive→positive)
    assert metrics["accuracy"] == pytest.approx(1.0, abs=1e-3)
