import pytest
from pydantic import ValidationError

from src.schemas.jobs import BenchmarkJobReport, JobStatus, RunEntry, SubmissionRequest


def test_run_entry_defaults_params():
    entry = RunEntry(model="gemini/gemini-2.0-flash")
    assert entry.params == {}


def test_run_entry_accepts_params():
    entry = RunEntry(model="gemini/gemini-2.0-flash", params={"temperature": 0.5})
    assert entry.params["temperature"] == 0.5


def test_submission_request_requires_prompt_template():
    with pytest.raises(ValidationError):
        SubmissionRequest()


def test_submission_request_minimal():
    req = SubmissionRequest(prompt_template="extract the name")
    assert req.sota_model is None
    assert req.compare_models == []


def test_job_status_round_trips():
    s = JobStatus(job_id="abc", status="queued")
    assert JobStatus.model_validate(s.model_dump()).job_id == "abc"


def test_job_status_optional_fields_default_none():
    s = JobStatus(job_id="x", status="done")
    assert s.error_message is None
    assert s.detected_workload_details is None


def test_benchmark_job_report_fields():
    blob = BenchmarkJobReport(job_id="x")
    assert blob.workload_type is None
    assert blob.exec_summary is None
