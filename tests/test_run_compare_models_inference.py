"""
T20 tests — inference_runner.run()

Tests cover:
  - happy path: all rows succeed → JSONL has one record per row in index order
  - partial existing file, retry_failed=False: missing rows filled in; successful rows untouched
  - error rows, retry_failed=True: error records re-run; successful retry replaces error; permanent failures remain
  - rate-limit retry: first two attempts raise RateLimitError, third succeeds → row written correctly
  - non-rate-limit error: breaks immediately, writes error record, no retry
  - generate_golden_dataset refactored: golden file written; output_path used correctly
  - run_compare_models_inference refactored: per-model JSONL written; per-model skip logic fires for complete files
  - evaluate judge path refactored: judge JSONL written under evaluation/judge/; aggregate scores correct
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import openai
import pytest
from sqlmodel import Session, SQLModel, create_engine

import src.schemas.db  # noqa: F401 — register table models
from src.agents.generate_golden_dataset import generate_golden_dataset
from src.agents.inference_runner import run
from src.agents.run_compare_models_evaluation import _aggregate_judge_scores, run_judge_rows
from src.agents.run_compare_models_inference import run_compare_models_inference
from src.providers.adapters import GenerationConfig, Message, ModelResponse
from src.schemas.db import JobConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_ROWS = [
    {"row_index": 0, "messages": [{"role": "user", "content": "Describe image A"}]},
    {"row_index": 1, "messages": [{"role": "user", "content": "Describe image B"}]},
    {"row_index": 2, "messages": [{"role": "user", "content": "Describe image C"}]},
]


def _make_client(content: str = "ok", input_tokens: int = 10, output_tokens: int = 5) -> MagicMock:
    client = MagicMock()
    client.complete.return_value = ModelResponse(
        content=content, input_tokens=input_tokens, output_tokens=output_tokens
    )
    return client


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _build_messages(row: dict) -> list[Message]:
    return [Message(role=m["role"], content=m["content"]) for m in row["messages"]]


def _parse_response(resp: ModelResponse, row: dict) -> dict:  # noqa: ARG001
    return {}


_CONFIG = GenerationConfig(temperature=0.0, top_p=1.0)


# ---------------------------------------------------------------------------
# Tests for run directly
# ---------------------------------------------------------------------------


def test_happy_path_all_rows_succeed(tmp_path: Path) -> None:
    client = _make_client("caption output")
    output_path = tmp_path / "out.jsonl"

    results = run(
        rows=SAMPLE_ROWS,
        output_path=output_path,
        build_messages=_build_messages,
        client=client,
        config=_CONFIG,
        parse_response=_parse_response,
    )

    assert len(results) == 3
    assert client.complete.call_count == 3
    written = _read_jsonl(output_path)
    assert len(written) == 3
    # Rows are in JSONL order (written sequentially)
    assert [r["row_index"] for r in written] == [0, 1, 2]
    assert all(r["output"] == "caption output" for r in written)


def test_partial_existing_file_retry_failed_false(tmp_path: Path) -> None:
    """Missing rows are filled in; successful rows untouched."""
    output_path = tmp_path / "out.jsonl"
    # Pre-write row 0 as successful
    output_path.write_text(
        json.dumps({"row_index": 0, "output": "existing", "input_tokens": 1, "output_tokens": 1, "latency_ms": 10})
        + "\n"
    )

    client = _make_client("new output")

    results = run(
        rows=SAMPLE_ROWS,
        output_path=output_path,
        build_messages=_build_messages,
        client=client,
        config=_CONFIG,
        parse_response=_parse_response,
        retry=False,
    )

    # client called only for rows 1 and 2
    assert client.complete.call_count == 2
    assert len(results) == 3
    row0 = next(r for r in results if r["row_index"] == 0)
    assert row0["output"] == "existing"
    row1 = next(r for r in results if r["row_index"] == 1)
    assert row1["output"] == "new output"


def test_error_rows_retry_failed_true(tmp_path: Path) -> None:
    """Error rows are re-run; successful retry replaces error record; permanent failures remain."""
    output_path = tmp_path / "out.jsonl"
    # Pre-write: row 0 success, row 1 error
    lines = [
        json.dumps({"row_index": 0, "output": "success", "input_tokens": 1, "output_tokens": 1, "latency_ms": 5}),
        json.dumps({"row_index": 1, "output": None, "error": "rate limit", "latency_ms": None}),
    ]
    output_path.write_text("\n".join(lines) + "\n")

    client = MagicMock()
    # row 1 retry succeeds
    client.complete.return_value = ModelResponse(content="retry ok", input_tokens=5, output_tokens=3)

    results = run(
        rows=SAMPLE_ROWS[:2],
        output_path=output_path,
        build_messages=_build_messages,
        client=client,
        config=_CONFIG,
        parse_response=_parse_response,
        retry=True,
    )

    assert client.complete.call_count == 1  # only row 1 re-run
    row1 = next(r for r in results if r["row_index"] == 1)
    assert row1["output"] == "retry ok"
    row0 = next(r for r in results if r["row_index"] == 0)
    assert row0["output"] == "success"


def test_rate_limit_retry_succeeds_on_third_attempt(tmp_path: Path) -> None:
    """First two attempts raise RateLimitError, third succeeds."""
    output_path = tmp_path / "out.jsonl"
    client = MagicMock()
    client.complete.side_effect = [
        openai.RateLimitError("rate limit", response=MagicMock(status_code=429), body={}),
        openai.RateLimitError("rate limit", response=MagicMock(status_code=429), body={}),
        ModelResponse(content="eventual success", input_tokens=8, output_tokens=4),
    ]

    with patch("src.agents.inference_runner.time.sleep"):
        results = run(
            rows=SAMPLE_ROWS[:1],
            output_path=output_path,
            build_messages=_build_messages,
            client=client,
            config=_CONFIG,
            parse_response=_parse_response,
        )

    assert client.complete.call_count == 3
    assert results[0]["output"] == "eventual success"
    written = _read_jsonl(output_path)
    assert written[0]["output"] == "eventual success"


def test_non_rate_limit_error_no_retry(tmp_path: Path) -> None:
    """Non-rate-limit error breaks immediately and writes error record."""
    output_path = tmp_path / "out.jsonl"
    client = MagicMock()
    client.complete.side_effect = ValueError("unexpected error")

    results = run(
        rows=SAMPLE_ROWS[:1],
        output_path=output_path,
        build_messages=_build_messages,
        client=client,
        config=_CONFIG,
        parse_response=_parse_response,
    )

    assert client.complete.call_count == 1  # no retry
    assert results[0]["output"] is None
    assert "unexpected error" in results[0]["error"]


# ---------------------------------------------------------------------------
# Integration: generate_golden_dataset refactored
# ---------------------------------------------------------------------------


@pytest.fixture
def db_engine_golden(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'golden.db'}", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)
    return engine


def test_generate_golden_dataset_uses_run(tmp_path: Path, db_engine_golden, monkeypatch) -> None:
    """generate_golden_dataset writes golden file via run."""
    from datetime import datetime, timezone

    input_path = tmp_path / "input.jsonl"
    rows = [
        {"row_index": 0, "messages": [{"role": "user", "content": "Caption this"}]},
        {"row_index": 1, "messages": [{"role": "user", "content": "Caption that"}]},
    ]
    input_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    job_id = "golden-test-job"
    with Session(db_engine_golden) as session:
        job = JobConfig(
            job_id=job_id,
            prompt_template="caption",
            sota_model="gemini/gemini-2.0-flash",
            input_file_jsonl_path=str(input_path),
            status="queued",
            created_at=datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat(),
        )
        session.add(job)
        session.commit()

    @contextmanager
    def _get_session():
        with Session(db_engine_golden) as session:
            yield session

    monkeypatch.setattr("src.agents.generate_golden_dataset.get_session", _get_session)
    monkeypatch.setattr("src.agents.pipeline_log.get_session", _get_session)
    monkeypatch.setattr("src.services.job_utils.get_session", _get_session)

    mock_client = _make_client("a golden caption")
    with patch("src.agents.generate_golden_dataset.get_client", return_value=mock_client):
        output_dir = tmp_path / "output" / job_id
        generate_golden_dataset(job_id, output_dir=output_dir)

    golden_path = output_dir / "golden_dataset.jsonl"
    assert golden_path.exists()
    written = _read_jsonl(golden_path)
    assert len(written) == 2
    assert all(r["output"] == "a golden caption" for r in written)
    # parse_response should include "input" field
    assert all("input" in r for r in written)


# ---------------------------------------------------------------------------
# Integration: run_compare_models_inference refactored
# ---------------------------------------------------------------------------


@pytest.fixture
def db_engine_inference(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'inference.db'}", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)
    return engine


def test_run_inference_per_model_jsonl_written(tmp_path: Path, db_engine_inference, monkeypatch) -> None:
    """run_compare_models_inference writes one JSONL per model under inference/."""
    from datetime import datetime, timezone

    input_path = tmp_path / "input.jsonl"
    rows = [
        {"row_index": 0, "messages": [{"role": "user", "content": "Hello"}]},
        {"row_index": 1, "messages": [{"role": "user", "content": "World"}]},
    ]
    input_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    job_id = "inference-test-job"
    with Session(db_engine_inference) as session:
        job = JobConfig(
            job_id=job_id,
            prompt_template="hello",
            compare_models=[
                {"model": "gemini/gemini-2.0-flash", "params": {}},
                {"model": "openai/gpt-4o-mini", "params": {}},
            ],
            input_file_jsonl_path=str(input_path),
            status="queued",
            created_at=datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat(),
        )
        session.add(job)
        session.commit()

    @contextmanager
    def _get_session():
        with Session(db_engine_inference) as session:
            yield session

    monkeypatch.setattr("src.agents.run_compare_models_inference.get_session", _get_session)
    monkeypatch.setattr("src.agents.pipeline_log.get_session", _get_session)
    monkeypatch.setattr("src.services.job_utils.get_session", _get_session)

    mock_client = _make_client("inf output")
    with patch("src.agents.run_compare_models_inference.get_client", return_value=mock_client):
        output_dir = tmp_path / "output" / job_id
        run_compare_models_inference(job_id, output_dir=output_dir)

    inf_dir = output_dir / "inference"
    assert (inf_dir / "gemini__gemini-2.0-flash.jsonl").exists()
    assert (inf_dir / "openai__gpt-4o-mini.jsonl").exists()
    for f in inf_dir.glob("*.jsonl"):
        rows_out = _read_jsonl(f)
        assert len(rows_out) == 2
        assert all(r["output"] == "inf output" for r in rows_out)


def test_run_inference_skip_complete_model(tmp_path: Path, db_engine_inference, monkeypatch) -> None:
    """Per-model skip fires when JSONL already complete with no errors."""
    from datetime import datetime, timezone

    input_path = tmp_path / "input.jsonl"
    row = {"row_index": 0, "messages": [{"role": "user", "content": "hi"}]}
    input_path.write_text(json.dumps(row) + "\n")

    job_id = "skip-test-job"
    model_string = "gemini/gemini-2.0-flash"
    with Session(db_engine_inference) as session:
        job = JobConfig(
            job_id=job_id,
            prompt_template="hi",
            compare_models=[{"model": model_string, "params": {}}],
            input_file_jsonl_path=str(input_path),
            status="queued",
            created_at=datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat(),
        )
        session.add(job)
        session.commit()

    @contextmanager
    def _get_session():
        with Session(db_engine_inference) as session:
            yield session

    monkeypatch.setattr("src.agents.run_compare_models_inference.get_session", _get_session)
    monkeypatch.setattr("src.agents.pipeline_log.get_session", _get_session)
    monkeypatch.setattr("src.services.job_utils.get_session", _get_session)

    # Pre-write a complete file
    output_dir = tmp_path / "output" / job_id
    inf_dir = output_dir / "inference"
    inf_dir.mkdir(parents=True, exist_ok=True)
    existing_file = inf_dir / "gemini__gemini-2.0-flash.jsonl"
    existing_file.write_text(
        json.dumps({"row_index": 0, "output": "cached", "input_tokens": 5, "output_tokens": 3, "latency_ms": 10}) + "\n"
    )

    mock_client = _make_client("should not be called")
    with patch("src.agents.run_compare_models_inference.get_client", return_value=mock_client):
        run_compare_models_inference(job_id, output_dir=output_dir)

    mock_client.complete.assert_not_called()


# ---------------------------------------------------------------------------
# Integration: evaluate judge path refactored
# ---------------------------------------------------------------------------


def test_run_judge_rows_writes_judge_jsonl(tmp_path: Path) -> None:
    """run_judge_rows writes judge JSONL under evaluation/judge/."""
    golden_map = {
        0: {"row_index": 0, "output": "A golden caption"},
        1: {"row_index": 1, "output": "Another golden caption"},
    }
    inference_map = {
        0: {"row_index": 0, "output": "A candidate caption"},
        1: {"row_index": 1, "output": "Another candidate caption"},
    }
    criteria = [{"name": "faithfulness", "description": "Accurate."}]

    judge_output_path = tmp_path / "evaluation" / "judge" / "gemini__gemini-2.0-flash.jsonl"
    judge_output_path.parent.mkdir(parents=True, exist_ok=True)

    mock_client = MagicMock()
    mock_client.complete.return_value = ModelResponse(content='{"faithfulness": 4}', input_tokens=20, output_tokens=5)

    with patch("src.agents.run_compare_models_evaluation.get_client", return_value=mock_client):
        rows = run_judge_rows(
            golden_map=golden_map,
            inference_map=inference_map,
            criteria=criteria,
            sota_model="gemini/gemini-2.0-flash",
            output_path=judge_output_path,
        )

    assert judge_output_path.exists()
    assert mock_client.complete.call_count == 2
    assert len(rows) == 2
    assert all(r.get("scores", {}).get("faithfulness") == 4 for r in rows)


def test_aggregate_judge_scores_computes_means() -> None:
    """_aggregate_judge_scores produces correct means from pre-written rows."""
    criteria = [
        {"name": "faithfulness", "description": "Accurate."},
        {"name": "completeness", "description": "Complete."},
    ]
    judge_rows = [
        {
            "row_index": 0,
            "output": '{"faithfulness": 4, "completeness": 3}',
            "scores": {"faithfulness": 4, "completeness": 3},
            "input_tokens": 10,
            "output_tokens": 5,
            "latency_ms": 100,
        },
        {
            "row_index": 1,
            "output": '{"faithfulness": 5, "completeness": 4}',
            "scores": {"faithfulness": 5, "completeness": 4},
            "input_tokens": 10,
            "output_tokens": 5,
            "latency_ms": 100,
        },
    ]

    result = _aggregate_judge_scores(judge_rows, criteria)

    assert result["mean_scores"]["faithfulness"] == pytest.approx(4.5)
    assert result["mean_scores"]["completeness"] == pytest.approx(3.5)
    assert result["overall_mean"] == pytest.approx(4.0)
    assert result["evaluated_rows"] == 2
    assert result["failed_rows"] == 0
    assert result["judge_input_tokens"] == 20
    assert result["judge_output_tokens"] == 10
