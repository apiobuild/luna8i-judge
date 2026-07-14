"""
T17 tests — GET /api/jobs/{job_id}/report
"""

from __future__ import annotations

import io
import json
from contextlib import contextmanager
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.main import app
from src.services.job_constants import JobStatusStr

PATCH_COMPLETE = "src.providers.client.complete_chat"


@pytest.fixture
def client(tmp_db, monkeypatch, tmp_path):
    session = tmp_db

    @contextmanager
    def patched_get_session():
        yield session

    monkeypatch.setattr("src.services.jobs.get_session", patched_get_session)
    monkeypatch.setattr("src.services.jobs.settings", type("S", (), {"DATA_DIR": str(tmp_path)})())
    monkeypatch.setattr("src.services.job_utils.settings", type("S", (), {"DATA_DIR": str(tmp_path)})())
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c, tmp_path


# ---------------------------------------------------------------------------
# Test 1 — unknown job → 404
# ---------------------------------------------------------------------------


def test_report_unknown_job(client):
    c, _ = client
    resp = c.get("/api/jobs/does-not-exist/report")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Test 2 — job not yet completed → 409
# ---------------------------------------------------------------------------


def test_report_job_not_completed(client):
    c, tmp_path = client
    rows = [{"messages": [{"role": "user", "content": "Extract the invoice number from: INV-001"}]}]
    files = {
        "input_file_jsonl": (
            "test.jsonl",
            io.BytesIO(b"\n".join(json.dumps(r).encode() for r in rows)),
            "application/octet-stream",
        )
    }
    upload_resp = c.post("/api/jobs/upload", files=files)
    upload_id = upload_resp.json()["upload_id"]

    llm_response = json.dumps({"workload_type": "extraction", "confidence": "high", "confidence_note": "OK."})
    with patch(PATCH_COMPLETE, return_value=llm_response):
        resp = c.post("/api/jobs", json={"prompt_template": r"Extract .+", "upload_id": upload_id})
    job_id = resp.json()["job_id"]

    resp = c.get(f"/api/jobs/{job_id}/report")
    assert resp.status_code == 409
    assert "not ready" in resp.json()["detail"]["error"].lower()


# ---------------------------------------------------------------------------
# Test 3 — completed job → returns evaluation results
# ---------------------------------------------------------------------------


def test_report_happy_path(tmp_db, monkeypatch, tmp_path):
    session = tmp_db

    @contextmanager
    def patched_get_session():
        yield session

    monkeypatch.setattr("src.services.jobs.get_session", patched_get_session)
    monkeypatch.setattr("src.agents.job_runner.get_session", patched_get_session)
    monkeypatch.setattr("src.agents.pipeline_log.get_session", patched_get_session)
    monkeypatch.setattr("src.services.jobs.settings", type("S", (), {"DATA_DIR": str(tmp_path)})())
    monkeypatch.setattr("src.services.job_utils.settings", type("S", (), {"DATA_DIR": str(tmp_path)})())

    with TestClient(app, raise_server_exceptions=False) as c:
        rows = [{"messages": [{"role": "user", "content": "Extract the invoice number from: INV-001"}]}]
        files = {
            "input_file_jsonl": (
                "test.jsonl",
                io.BytesIO(b"\n".join(json.dumps(r).encode() for r in rows)),
                "application/octet-stream",
            )
        }
        upload_resp = c.post("/api/jobs/upload", files=files)
        upload_id = upload_resp.json()["upload_id"]

        llm_response = json.dumps({"workload_type": "extraction", "confidence": "high", "confidence_note": "OK."})
        with patch(PATCH_COMPLETE, return_value=llm_response):
            submit_resp = c.post("/api/jobs", json={"prompt_template": r"Extract .+", "upload_id": upload_id})
        job_id = submit_resp.json()["job_id"]

        # Write the combined report JSON (what the endpoint reads)
        report_dir = tmp_path / "output" / job_id
        report_dir.mkdir(parents=True, exist_ok=True)
        report_payload = {
            "job_id": job_id,
            "results": [
                {
                    "model": "openai/gpt-4o-mini",
                    "params": {},
                    "workload_type": "extraction",
                    "metrics": {"accuracy": 0.9},
                    "inference_usage": {},
                    "failure_rate": 0.0,
                }
            ],
        }
        (report_dir / "scale_and_cost_report.json").write_text(json.dumps(report_payload))

        # Force status to completed via the patched session
        from sqlmodel import select

        from src.schemas.db import JobConfig

        job = session.exec(select(JobConfig).where(JobConfig.job_id == job_id)).first()
        job.status = JobStatusStr.COMPLETED
        session.add(job)
        session.commit()

        resp = c.get(f"/api/jobs/{job_id}/report")
        assert resp.status_code == 200
        body = resp.json()
        assert body["job_id"] == job_id
        assert len(body["results"]) == 1
        assert body["results"][0]["model"] == "openai/gpt-4o-mini"
        assert body["results"][0]["metrics"]["accuracy"] == pytest.approx(0.9)
