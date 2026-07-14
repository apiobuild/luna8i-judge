"""
T7a tests — PATCH /api/jobs/{job_id}/compare_models

Covers:
  - op="add": appends new entries, re-enqueues pipeline
  - op="add": duplicate entry is deduplicated, still re-enqueues
  - op="remove": removes subset, clears inference blobs, re-enqueues
  - op="remove": removing all entries → 400
  - job in running state → 409
  - unknown job → 404
"""

from __future__ import annotations

import io
import json
from contextlib import contextmanager
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from src.main import app

PATCH_COMPLETE = "src.providers.client.complete_chat"
PATCH_PIPELINE = "src.routers.jobs.run_pipeline"


@pytest.fixture
def client(tmp_db, monkeypatch, tmp_path):
    session: Session = tmp_db

    @contextmanager
    def patched_get_session():
        yield session

    monkeypatch.setattr("src.services.jobs.get_session", patched_get_session)
    monkeypatch.setattr("src.services.jobs.settings", type("S", (), {"DATA_DIR": str(tmp_path)})())
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def _jsonl(*rows: dict) -> bytes:
    return b"\n".join(json.dumps(r).encode() for r in rows)


def _text_row() -> dict:
    return {"messages": [{"role": "user", "content": "Extract the invoice number from: INV-001"}]}


def _upload_and_submit(client, status_override: str | None = None) -> str:
    """Submit a job and return its job_id. Optionally force a specific status via DB."""
    files = {"input_file_jsonl": ("test.jsonl", io.BytesIO(_jsonl(_text_row())), "application/octet-stream")}
    resp = client.post("/api/jobs/upload", files=files)
    upload_id = resp.json()["upload_id"]

    llm_response = json.dumps({"workload_type": "extraction", "confidence": "high", "confidence_note": "ok"})
    with patch(PATCH_COMPLETE, return_value=llm_response), patch(PATCH_PIPELINE):
        resp = client.post(
            "/api/jobs",
            json={
                "prompt_template": r"Extract .+",
                "upload_id": upload_id,
                "compare_models": [{"model": "gemini/gemini-2.0-flash", "params": {}}],
            },
        )
    assert resp.status_code == 200, resp.json()
    job_id = resp.json()["job_id"]

    from src.schemas.db import JobConfig
    from src.services.jobs import get_session as _gs

    with _gs() as s:
        from sqlmodel import select

        job = s.exec(select(JobConfig).where(JobConfig.job_id == job_id)).first()

        assert job is not None, "job should not be none"
        job.status = status_override if status_override is not None else "done"
        s.add(job)
        s.commit()

    return job_id


# ---------------------------------------------------------------------------
# Test 1 — op="add": appends new entry, re-enqueues
# ---------------------------------------------------------------------------


def test_add_new_entry(client):
    job_id = _upload_and_submit(client)
    with patch(PATCH_PIPELINE) as mock_pipeline:
        resp = client.patch(
            f"/api/jobs/{job_id}/compare_models",
            json={"op": "add", "compare_models": [{"model": "openai/gpt-4o", "params": {}}]},
        )
    assert resp.status_code == 200
    assert resp.json()["status"] == "queued"
    mock_pipeline.assert_called_once_with(job_id)

    # Verify merged list in DB
    from src.schemas.db import JobConfig
    from src.services.jobs import get_session as _gs

    with _gs() as s:
        job = s.get(JobConfig, job_id)

    assert job is not None, "job should not be none"
    assert job.compare_models is not None, "compare_models should not be none"
    models = [e["model"] for e in job.compare_models]
    assert "gemini/gemini-2.0-flash" in models
    assert "openai/gpt-4o" in models


# ---------------------------------------------------------------------------
# Test 2 — op="add": duplicate is deduplicated
# ---------------------------------------------------------------------------


def test_add_duplicate_deduplicates(client):
    job_id = _upload_and_submit(client)
    with patch(PATCH_PIPELINE) as mock_pipeline:
        resp = client.patch(
            f"/api/jobs/{job_id}/compare_models",
            json={"op": "add", "compare_models": [{"model": "gemini/gemini-2.0-flash", "params": {}}]},
        )
    assert resp.status_code == 200
    mock_pipeline.assert_called_once_with(job_id)

    from src.schemas.db import JobConfig
    from src.services.jobs import get_session as _gs

    with _gs() as s:
        job = s.get(JobConfig, job_id)
    assert job is not None, "job should not be none"
    assert job.compare_models is not None, "compare_models should not be none"
    assert len(job.compare_models) == 1


# ---------------------------------------------------------------------------
# Test 2b — op="add": existing inference blobs are preserved
# ---------------------------------------------------------------------------


def test_add_preserves_inference_blobs(client):
    """Adding a new model must not clear existing inference output — the new model will be skipped
    by run_compare_models_inference if its file already exists, so existing summaries should remain intact."""
    job_id = _upload_and_submit(client)

    from src.schemas.db import JobConfig
    from src.services.jobs import get_session as _gs

    with _gs() as s:
        job = s.get(JobConfig, job_id)
        assert job is not None
        job.evaluating_inference_output_path = "/fake/eval.jsonl"
        s.add(job)
        s.commit()

    with patch(PATCH_PIPELINE):
        resp = client.patch(
            f"/api/jobs/{job_id}/compare_models",
            json={"op": "add", "compare_models": [{"model": "openai/gpt-4o", "params": {}}]},
        )
    assert resp.status_code == 200

    with _gs() as s:
        job = s.get(JobConfig, job_id)

    assert job is not None, "job should not be none"
    assert job.evaluating_inference_output_path is not None, "evaluating path should not be cleared on add"


# ---------------------------------------------------------------------------
# Test 3 — op="remove": removes subset, clears blobs, re-enqueues
# ---------------------------------------------------------------------------


def test_remove_subset(client, tmp_db):
    job_id = _upload_and_submit(client)

    # Seed a second model and fake inference blob
    from src.schemas.db import JobConfig
    from src.services.jobs import get_session as _gs

    with _gs() as s:
        job = s.get(JobConfig, job_id)

        assert job is not None, "job should not be none"
        job.compare_models = [
            {"model": "gemini/gemini-2.0-flash", "params": {}},
            {"model": "openai/gpt-4o", "params": {}},
        ]
        job.evaluating_inference_output_path = "/fake/eval.jsonl"
        s.add(job)
        s.commit()

    with patch(PATCH_PIPELINE) as mock_pipeline:
        resp = client.patch(
            f"/api/jobs/{job_id}/compare_models",
            json={"op": "remove", "compare_models": [{"model": "openai/gpt-4o", "params": {}}]},
        )
    assert resp.status_code == 200
    mock_pipeline.assert_called_once_with(job_id)

    with _gs() as s:
        job = s.get(JobConfig, job_id)

    assert job is not None, "job should not be none"
    assert job.compare_models is not None, "compare_models should not be none"
    assert len(job.compare_models) == 1
    assert job.compare_models[0]["model"] == "gemini/gemini-2.0-flash"
    assert job.evaluating_inference_output_path is None


# ---------------------------------------------------------------------------
# Test 4 — op="remove": remove all → 400
# ---------------------------------------------------------------------------


def test_remove_all_returns_400(client):
    job_id = _upload_and_submit(client)
    with patch(PATCH_PIPELINE):
        resp = client.patch(
            f"/api/jobs/{job_id}/compare_models",
            json={"op": "remove", "compare_models": [{"model": "gemini/gemini-2.0-flash", "params": {}}]},
        )
    assert resp.status_code == 400
    assert "at least one" in resp.json()["error"].lower()


# ---------------------------------------------------------------------------
# Test 5 — job in running state → 409
# ---------------------------------------------------------------------------


def test_running_job_returns_409(client):
    job_id = _upload_and_submit(client, status_override="running")
    with patch(PATCH_PIPELINE):
        resp = client.patch(
            f"/api/jobs/{job_id}/compare_models",
            json={"op": "add", "compare_models": [{"model": "openai/gpt-4o", "params": {}}]},
        )
    assert resp.status_code == 409


# ---------------------------------------------------------------------------
# Test 6 — unknown job → 404
# ---------------------------------------------------------------------------


def test_unknown_job_returns_404(client):
    resp = client.patch(
        "/api/jobs/nonexistent-id/compare_models",
        json={"op": "add", "compare_models": [{"model": "openai/gpt-4o", "params": {}}]},
    )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Test 7 — op="remove": blocked when inference output file exists
# ---------------------------------------------------------------------------


@pytest.fixture
def client_with_data_dir(tmp_db, monkeypatch, tmp_path):
    """Like `client` but also yields tmp_path so tests can write inference files."""
    session: Session = tmp_db

    @contextmanager
    def patched_get_session():
        yield session

    monkeypatch.setattr("src.services.jobs.get_session", patched_get_session)
    monkeypatch.setattr("src.services.jobs.settings", type("S", (), {"DATA_DIR": str(tmp_path)})())
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c, tmp_path


def test_remove_blocked_when_output_file_exists(client_with_data_dir):
    """Removing a model whose inference JSONL is already on disk returns 400."""
    client, tmp_path = client_with_data_dir
    job_id = _upload_and_submit(client)

    inference_dir = tmp_path / "output" / job_id / "inference"
    inference_dir.mkdir(parents=True)
    (inference_dir / "gemini__gemini-2.0-flash.jsonl").write_text('{"row_index": 0}\n')

    with patch(PATCH_PIPELINE):
        resp = client.patch(
            f"/api/jobs/{job_id}/compare_models",
            json={"op": "remove", "compare_models": [{"model": "gemini/gemini-2.0-flash", "params": {}}]},
        )
    assert resp.status_code == 400
    assert "inference output already exists" in resp.json()["error"]
