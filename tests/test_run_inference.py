"""
T11a tests — run_compare_models_inference

Tests cover:
  - happy path: 2 run entries → JSONL files written per model, blob has summaries
  - rate limit on row 1 of entry 0 (all retries fail) → row marked failed in file, others complete, no exception
  - empty compare_models → blob is [], status advances to "running"
  - extra params forwarded to client and stored in output rows
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
from src.agents.run_compare_models_inference import run_compare_models_inference
from src.providers.adapters import ModelResponse
from src.schemas.db import JobConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_ROWS = [
    {"row_index": 0, "messages": [{"role": "user", "content": "Extract name from: Alice, 30"}]},
    {"row_index": 1, "messages": [{"role": "user", "content": "Extract name from: Bob, 25"}]},
    {"row_index": 2, "messages": [{"role": "user", "content": "Extract name from: Carol, 40"}]},
]

COMPARE_MODELS = [
    {"model": "gemini/gemini-2.0-flash", "params": {}},
    {"model": "openai/gpt-4o-mini", "params": {"temperature": 0.5}},
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
    @contextmanager
    def _get_session():
        with Session(db_engine) as session:
            yield session

    monkeypatch.setattr("src.agents.run_compare_models_inference.get_session", _get_session)
    monkeypatch.setattr("src.agents.pipeline_log.get_session", _get_session)
    monkeypatch.setattr("src.services.job_utils.get_session", _get_session)
    return db_session


def _make_job(
    session: Session,
    tmp_path: Path,
    rows: list[dict],
    compare_models: list[dict] | None = None,
) -> str:
    input_path = tmp_path / "input.jsonl"
    input_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    now = datetime.now(timezone.utc).isoformat()
    job = JobConfig(
        job_id="test-inference-job",
        status="generate_golden_dataset",
        sota_model="gemini/gemini-2.0-flash",
        input_file_jsonl_path=str(input_path),
        compare_models=compare_models or [],
        created_at=now,
        updated_at=now,
    )
    session.add(job)
    session.commit()
    return job.job_id


def _read_inference_file(tmp_path: Path, model_string: str) -> list[dict]:
    filename = model_string.replace("/", "__") + ".jsonl"
    path = tmp_path / "output" / "test-inference-job" / "inference" / filename
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _always_succeed(content: str = "result") -> MagicMock:
    client = MagicMock()
    client.complete.return_value = ModelResponse(content=content, input_tokens=80, output_tokens=30)
    return client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_happy_path_two_entries(patched_session, tmp_path, monkeypatch):
    """2 run entries each get a JSONL file with one row per input; blob has summaries."""
    job_id = _make_job(patched_session, tmp_path, SAMPLE_ROWS, COMPARE_MODELS)

    monkeypatch.setattr(
        "src.services.job_utils.settings",
        MagicMock(DATA_DIR=str(tmp_path)),
    )

    gemini_client = _always_succeed("gemini-output")
    openai_client = _always_succeed("openai-output")

    def _get_client(model_string: str, **kwargs):
        if "gemini" in model_string:
            return gemini_client
        return openai_client

    with patch("src.agents.run_compare_models_inference.get_client", side_effect=_get_client):
        run_compare_models_inference(job_id)

    patched_session.refresh(patched_session.get(JobConfig, job_id))
    job = patched_session.get(JobConfig, job_id)

    assert job.status == "running"

    for model_info in COMPARE_MODELS:
        file_rows = _read_inference_file(tmp_path, model_info["model"])
        assert len(file_rows) == len(SAMPLE_ROWS)
        for row in file_rows:
            assert row["output"] is not None
            assert "error" not in row
            assert row["latency_ms"] is not None


def test_rate_limit_marks_row_failed_others_complete(patched_session, tmp_path, monkeypatch):
    """Row 1 of entry 0 fails all 3 retries; others succeed; node does not raise."""
    job_id = _make_job(patched_session, tmp_path, SAMPLE_ROWS, COMPARE_MODELS[:1])

    monkeypatch.setattr(
        "src.services.job_utils.settings",
        MagicMock(DATA_DIR=str(tmp_path)),
    )

    import openai as _openai

    rate_limit_exc = _openai.RateLimitError("rate limited", response=MagicMock(status_code=429), body={})
    # row 0: ok, row 1: rate-limit x3 (exhausts retries), row 2: ok
    call_seq = [
        ModelResponse(content="Alice", input_tokens=80, output_tokens=30),
        rate_limit_exc,
        rate_limit_exc,
        rate_limit_exc,
        ModelResponse(content="Carol", input_tokens=80, output_tokens=30),
    ]
    client = MagicMock()
    idx = [0]

    def _complete(messages, config=None):
        val = call_seq[idx[0]]
        idx[0] += 1
        if isinstance(val, Exception):
            raise val
        return val

    client.complete.side_effect = _complete

    with patch("src.agents.run_compare_models_inference.get_client", return_value=client):
        with patch("src.agents.inference_runner.time") as mock_time:
            mock_time.monotonic.side_effect = [0.0, 1.0] * 10
            mock_time.sleep = MagicMock()
            run_compare_models_inference(job_id)  # must not raise

    job = patched_session.get(JobConfig, job_id)
    assert job.status == "running"

    rows = _read_inference_file(tmp_path, COMPARE_MODELS[0]["model"])
    failed = sum(1 for r in rows if r.get("output") is None)
    assert failed / len(rows) == pytest.approx(1 / 3)
    assert rows[0]["output"] == "Alice"
    assert rows[0].get("error") is None

    assert rows[1]["output"] is None
    assert "error" in rows[1]

    assert rows[2]["output"] == "Carol"
    assert rows[2].get("error") is None


def test_empty_compare_models_produces_empty_blob(patched_session, tmp_path, monkeypatch):
    """No compare_models → inference_output=[], status still advances to 'running'."""
    job_id = _make_job(patched_session, tmp_path, SAMPLE_ROWS, compare_models=[])

    monkeypatch.setattr(
        "src.services.job_utils.settings",
        MagicMock(DATA_DIR=str(tmp_path)),
    )

    with patch("src.agents.run_compare_models_inference.get_client") as mock_factory:
        run_compare_models_inference(job_id)
        mock_factory.assert_not_called()

    job = patched_session.get(JobConfig, job_id)
    assert job.status == "running"


def test_extra_params_forwarded_to_client_and_stored_in_output(patched_session, tmp_path, monkeypatch):
    """Extra params (e.g. max_tokens, stop) are forwarded to the client and stored in row results."""
    compare_models = [
        {"model": "openai/gpt-4o-mini", "params": {"temperature": 0.5, "max_tokens": 128, "stop": ["END"]}},
    ]
    job_id = _make_job(patched_session, tmp_path, SAMPLE_ROWS, compare_models)

    monkeypatch.setattr(
        "src.services.job_utils.settings",
        MagicMock(DATA_DIR=str(tmp_path)),
    )

    captured_configs: list = []

    def _complete(messages, config=None):
        captured_configs.append(config)
        return ModelResponse(content="ok", input_tokens=10, output_tokens=5)

    client = MagicMock()
    client.complete.side_effect = _complete

    with patch("src.agents.run_compare_models_inference.get_client", return_value=client):
        run_compare_models_inference(job_id)

    job = patched_session.get(JobConfig, job_id)
    assert job.status == "running"

    file_rows = _read_inference_file(tmp_path, "openai/gpt-4o-mini")
    assert len(file_rows) == len(SAMPLE_ROWS)
    for row in file_rows:
        assert row["extra_params"] == {"max_tokens": 128, "stop": ["END"]}

    assert len(captured_configs) == len(SAMPLE_ROWS)
    for cfg in captured_configs:
        assert cfg.extra_params == {"max_tokens": 128, "stop": ["END"]}


def test_skips_model_with_existing_output_file(patched_session, tmp_path, monkeypatch):
    """If a model's output JSONL already exists, run_inference skips it and uses the existing summary."""
    job_id = _make_job(patched_session, tmp_path, SAMPLE_ROWS, COMPARE_MODELS)

    monkeypatch.setattr(
        "src.services.job_utils.settings",
        MagicMock(DATA_DIR=str(tmp_path)),
    )

    # Pre-write gemini output file as if inference already ran
    inference_dir = tmp_path / "output" / job_id / "inference"
    inference_dir.mkdir(parents=True)
    gemini_file = inference_dir / "gemini__gemini-2.0-flash.jsonl"
    existing_rows = [
        {
            "model": "gemini/gemini-2.0-flash",
            "params": {},
            "row_index": i,
            "output": f"cached-{i}",
            "input_tokens": 10,
            "output_tokens": 5,
            "latency_ms": 100,
        }
        for i in range(len(SAMPLE_ROWS))
    ]
    gemini_file.write_text("\n".join(json.dumps(r) for r in existing_rows) + "\n")

    openai_client = _always_succeed("openai-output")

    def _get_client(model_string: str, **kwargs):
        if "gemini" in model_string:
            raise AssertionError("gemini client should not be called — output file exists")
        return openai_client

    with patch("src.agents.run_compare_models_inference.get_client", side_effect=_get_client):
        run_compare_models_inference(job_id)

    job = patched_session.get(JobConfig, job_id)
    assert job.status == "running"

    # gemini was skipped — existing file unchanged
    gemini_rows = _read_inference_file(tmp_path, "gemini/gemini-2.0-flash")
    assert all(r.get("output") is not None for r in gemini_rows)

    # openai file was written fresh
    openai_rows = _read_inference_file(tmp_path, "openai/gpt-4o-mini")
    assert len(openai_rows) == len(SAMPLE_ROWS)
