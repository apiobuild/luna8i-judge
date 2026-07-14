"""
T6 tests — POST /api/uploads then POST /api/jobs

Tests cover:
  - happy path: submission returns queued job with detected_workload_details
  - workload_type override: skips LLM detection, sets confidence="override"
  - prompt consistency error (row mismatch) → 400
  - modality mismatch → 400
  - alias conflict → 400
"""

from __future__ import annotations

import io
import json
from contextlib import contextmanager
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from httpx import Response
from sqlmodel import Session

from src.main import app

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PATCH_COMPLETE = "src.providers.client.complete_chat"


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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _jsonl(*rows: dict) -> bytes:
    return b"\n".join(json.dumps(r).encode() for r in rows)


def _text_row(content: str = "Extract the invoice number from: INV-001") -> dict:
    return {"messages": [{"role": "user", "content": content}]}


def _image_row() -> dict:
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
                    {"type": "text", "text": "Describe."},
                ],
            }
        ]
    }


def _upload(client, rows: list[dict]) -> str:
    files = {"input_file_jsonl": ("test.jsonl", io.BytesIO(_jsonl(*rows)), "application/octet-stream")}
    resp = client.post("/api/jobs/upload", files=files)
    assert resp.status_code == 200, f"Upload failed: {resp.json()}"
    return resp.json()["upload_id"]


def _submit(client, rows: list[dict], extra_fields: dict | None = None, prompt: str = r"Extract .+") -> Response:
    upload_id = _upload(client, rows)
    body = {"prompt_template": prompt, "upload_id": upload_id}
    if extra_fields:
        body.update(extra_fields)
    return client.post("/api/jobs", json=body)


# ---------------------------------------------------------------------------
# Test 1 — happy path: returns queued job with detected_workload_details
# ---------------------------------------------------------------------------


def test_submit_happy_path(client):
    llm_response = json.dumps(
        {"workload_type": "extraction", "confidence": "high", "confidence_note": "Clear extraction prompt."}
    )
    with patch(PATCH_COMPLETE, return_value=llm_response):
        resp = _submit(client, [_text_row()])

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "queued"
    assert "job_id" in body
    details = body["detected_workload_details"]
    assert details["workload_type"] == "extraction"
    assert details["confidence"] == "high"
    assert details["modality"] == "text"


# ---------------------------------------------------------------------------
# Test 2 — workload_type override: skips detection, sets confidence="override"
# ---------------------------------------------------------------------------


def test_submit_workload_type_override(client):
    with patch(PATCH_COMPLETE) as mock_llm:
        resp = _submit(client, [_text_row()], extra_fields={"workload_type": "classification"})

    mock_llm.assert_not_called()
    assert resp.status_code == 200
    body = resp.json()
    details = body["detected_workload_details"]
    assert details["workload_type"] == "classification"
    assert details["confidence"] == "override"


# ---------------------------------------------------------------------------
# Test 3 — prompt consistency error → 400
# ---------------------------------------------------------------------------


def test_submit_prompt_consistency_error(client):
    mismatched_row = _text_row("Summarize this article.")
    with patch(PATCH_COMPLETE) as mock_llm:
        resp = _submit(client, [mismatched_row], prompt=r"Extract .+ from:")

    mock_llm.assert_not_called()
    assert resp.status_code == 400
    assert "Row 0" in resp.json()["error"]


# ---------------------------------------------------------------------------
# Test 4 — modality mismatch → 400
# ---------------------------------------------------------------------------


def test_submit_modality_mismatch(client):
    rows = [_text_row(), _image_row()]
    with patch(PATCH_COMPLETE):
        resp = _submit(client, rows)

    assert resp.status_code == 400
    assert "Modality mismatch" in resp.json()["error"]


# ---------------------------------------------------------------------------
# Test 5 — duplicate alias allowed → both 200
# ---------------------------------------------------------------------------


def test_submit_duplicate_alias_allowed(client):
    llm_response = json.dumps({"workload_type": "extraction", "confidence": "high", "confidence_note": "OK."})
    with patch(PATCH_COMPLETE, return_value=llm_response):
        r1 = _submit(client, [_text_row()], extra_fields={"alias": "invoice-v2"})
    assert r1.status_code == 200

    with patch(PATCH_COMPLETE, return_value=llm_response):
        r2 = _submit(client, [_text_row()], extra_fields={"alias": "invoice-v2"})

    assert r2.status_code == 200
    assert r1.json()["job_id"] != r2.json()["job_id"]


# ---------------------------------------------------------------------------
# T19 — POST /api/jobs/{job_id}/cancel
# ---------------------------------------------------------------------------


def _create_queued_job(client: TestClient) -> str:
    llm_response = json.dumps({"workload_type": "classification", "confidence": "high", "confidence_note": "OK."})
    with patch(PATCH_COMPLETE, return_value=llm_response):
        resp = _submit(client, [_text_row()])
    assert resp.status_code == 200
    return resp.json()["job_id"]


def test_cancel_queued_job(client):
    """Cancelling a queued job sets status='cancelled'."""
    job_id = _create_queued_job(client)
    resp = client.post(f"/api/jobs/{job_id}/cancel")
    assert resp.status_code == 200
    data = resp.json()
    assert data["job_id"] == job_id
    assert data["status"] == "cancelled"


def test_cancel_unknown_job(client):
    """Cancelling a non-existent job returns 404."""
    resp = client.post("/api/jobs/does-not-exist/cancel")
    assert resp.status_code == 404


def test_cancel_terminal_job_returns_409(client):
    """Cancelling a job already in a terminal state returns 409."""
    from src.services.jobs import cancel_job

    job_id = _create_queued_job(client)
    cancel_job(job_id)  # first cancel succeeds

    # Second cancel on an already-cancelled (terminal) job → 409
    resp = client.post(f"/api/jobs/{job_id}/cancel")
    assert resp.status_code == 409


# ---------------------------------------------------------------------------
# GET /api/jobs/{job_id}
# ---------------------------------------------------------------------------


def test_get_job_returns_config_fields(client):
    """GET /api/jobs/{job_id} returns job config without blob/path fields or runtime state."""
    job_id = _create_queued_job(client)
    resp = client.get(f"/api/jobs/{job_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["job_id"] == job_id
    # runtime state lives at /status, not here
    assert "status" not in data
    # blob/path fields must be excluded
    for excluded in (
        "input_file_jsonl_path",
        "pipeline_log",
        "generated_golden_dataset_path",
        "inference_output",
        "evaluating_inference_output_path",
        "scale_and_cost_projection_report_path",
    ):
        assert excluded not in data, f"'{excluded}' should be excluded"


def test_get_job_not_found(client):
    """GET /api/jobs/{job_id} returns 404 for unknown job."""
    resp = client.get("/api/jobs/does-not-exist")
    assert resp.status_code == 404
