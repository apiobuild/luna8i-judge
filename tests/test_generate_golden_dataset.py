"""
T10 tests — generate_golden_dataset

Tests cover:
  - happy path: blob has one entry per input row with output + token counts
  - rate-limit on row 2 (all retries fail): row marked skipped, others complete, no exception
  - golden_dataset.jsonl written to DATA_DIR/output/{job_id}/
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from sqlmodel import Session, SQLModel, create_engine

import src.schemas.db  # noqa: F401 — register table models
from src.agents.generate_golden_dataset import generate_golden_dataset
from src.providers.adapters import ModelResponse
from src.schemas.db import JobConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_ROWS = [
    {"messages": [{"role": "user", "content": "Extract the name from: John Smith, 30"}]},
    {"messages": [{"role": "user", "content": "Extract the name from: Jane Doe, 25"}]},
    {"messages": [{"role": "user", "content": "Extract the name from: Bob Lee, 40"}]},
]


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
    """Redirect get_session in the agent to use the test DB."""

    @contextmanager
    def _get_session():
        with Session(db_engine) as session:
            yield session

    monkeypatch.setattr("src.agents.generate_golden_dataset.get_session", _get_session)
    monkeypatch.setattr("src.agents.pipeline_log.get_session", _get_session)
    return db_session


def _make_job(session: Session, rows: list[dict], tmp_path, sota_model: str = "gemini/gemini-2.0-flash") -> str:
    import json
    from datetime import datetime, timezone

    input_path = tmp_path / "input.jsonl"
    input_path.write_text("\n".join(json.dumps(r) for r in rows))

    now = datetime.now(timezone.utc).isoformat()
    job = JobConfig(
        job_id="test-job-123",
        status="queued",
        sota_model=sota_model,
        input_file_jsonl_path=str(input_path),
        created_at=now,
        updated_at=now,
    )
    session.add(job)
    session.commit()
    return job.job_id


def _mock_client(outputs: Sequence[str | Exception]) -> MagicMock:
    client = MagicMock()
    responses = []
    for out in outputs:
        if isinstance(out, Exception):
            responses.append(out)
        else:
            responses.append(ModelResponse(content=out, input_tokens=10, output_tokens=5))

    call_count = [0]

    def _complete(messages, config=None):
        val = responses[call_count[0]]
        call_count[0] += 1
        if isinstance(val, Exception):
            raise val
        return val

    client.complete.side_effect = _complete
    return client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_happy_path_produces_one_entry_per_row(patched_session, tmp_path, monkeypatch):
    """All rows succeed; blob has correct structure and token counts."""
    job_id = _make_job(patched_session, SAMPLE_ROWS, tmp_path)

    outputs = ["John Smith", "Jane Doe", "Bob Lee"]
    mock_client = _mock_client(outputs)

    monkeypatch.setattr(
        "src.agents.generate_golden_dataset.settings",
        MagicMock(
            DEFAULT_SOTA_MODEL="gemini/gemini-2.0-flash",
            DATA_DIR=str(tmp_path),
        ),
    )

    with patch("src.agents.generate_golden_dataset.get_client", return_value=mock_client):
        generate_golden_dataset(job_id)

    patched_session.refresh(patched_session.get(JobConfig, job_id))
    job = patched_session.get(JobConfig, job_id)

    assert job.status == "generate_golden_dataset"
    assert job.generated_golden_dataset_path is not None
    blob = [json.loads(line) for line in Path(job.generated_golden_dataset_path).read_text().strip().splitlines()]
    assert len(blob) == 3
    for i, entry in enumerate(blob):
        assert entry["row_index"] == i
        assert entry["output"] == outputs[i]
        assert entry["input_tokens"] == 10
        assert entry["output_tokens"] == 5
        assert "error" not in entry


def test_rate_limit_on_row_marks_skipped_and_continues(patched_session, tmp_path, monkeypatch):
    """Row 1 (0-indexed) fails all retries; other rows complete; no exception raised."""
    import openai

    job_id = _make_job(patched_session, SAMPLE_ROWS, tmp_path)

    rate_limit_err = openai.RateLimitError("rate limit", response=MagicMock(status_code=429), body={})
    # row 0: success, row 1: fail all 3 retry attempts, row 2: success
    outputs = [
        ModelResponse(content="John Smith", input_tokens=10, output_tokens=5),
        rate_limit_err,
        rate_limit_err,
        rate_limit_err,
        ModelResponse(content="Bob Lee", input_tokens=10, output_tokens=5),
    ]
    client = MagicMock()
    call_count = [0]

    def _complete(messages, config=None):
        val = outputs[call_count[0]]
        call_count[0] += 1
        if isinstance(val, Exception):
            raise val
        return val

    client.complete.side_effect = _complete

    monkeypatch.setattr(
        "src.agents.generate_golden_dataset.settings",
        MagicMock(
            DEFAULT_SOTA_MODEL="gemini/gemini-2.0-flash",
            DATA_DIR=str(tmp_path),
        ),
    )

    with (
        patch("src.agents.generate_golden_dataset.get_client", return_value=client),
        patch("src.agents.inference_runner.time.sleep"),
    ):
        generate_golden_dataset(job_id)  # must not raise

    job = patched_session.get(JobConfig, job_id)
    assert job.generated_golden_dataset_path is not None
    blob = [json.loads(line) for line in Path(job.generated_golden_dataset_path).read_text().strip().splitlines()]
    assert len(blob) == 3

    assert blob[0]["output"] == "John Smith"
    assert blob[0].get("error") is None

    assert blob[1]["output"] is None
    assert "error" in blob[1]
    assert "rate limit" in blob[1]["error"]

    assert blob[2]["output"] == "Bob Lee"
    assert blob[2].get("error") is None


def test_golden_jsonl_written_to_data_dir(patched_session, tmp_path, monkeypatch):
    """golden_dataset.jsonl is written to DATA_DIR/output/{job_id}/golden_dataset.jsonl."""
    job_id = _make_job(patched_session, SAMPLE_ROWS[:1], tmp_path)

    mock_client = _mock_client(["John Smith"])

    monkeypatch.setattr(
        "src.agents.generate_golden_dataset.settings",
        MagicMock(
            DEFAULT_SOTA_MODEL="gemini/gemini-2.0-flash",
            DATA_DIR=str(tmp_path),
        ),
    )

    with patch("src.agents.generate_golden_dataset.get_client", return_value=mock_client):
        generate_golden_dataset(job_id)

    golden_path = tmp_path / "output" / job_id / "golden_dataset.jsonl"
    assert golden_path.exists()

    lines = golden_path.read_text().strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["row_index"] == 0
    assert entry["output"] == "John Smith"
