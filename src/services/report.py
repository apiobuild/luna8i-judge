"""Assemble the final evaluation report payload from on-disk artifacts."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path

from src.services.job_utils import (
    job_evaluation_dir,
    job_evaluation_rows_path,
    job_golden_path,
    job_inference_dir,
    job_scale_and_cost_report_path,
)
from src.utils.jsonl import model_filename, read_rows

logger = logging.getLogger(__name__)

_BUNDLE = Path(__file__).parent.parent / "render.bundle.js"


def render_html_report(report_json: Path, html_output: Path) -> None:
    """Render report.html from an already-written report JSON via the Node bundle.

    Skips silently if Node.js or the bundle is absent.
    """
    if not shutil.which("node") or not _BUNDLE.exists():
        return

    if not report_json.exists():
        raise FileNotFoundError(f"Report JSON not found: {report_json}")

    try:
        subprocess.run(
            ["node", str(_BUNDLE), str(report_json), "null", str(html_output)],
            check=True,
            timeout=60,
        )
        logger.info("Report HTML written to %s", html_output)
    except Exception as exc:
        logger.warning("HTML render failed: %s", exc)


def _extract_row_fields(raw: dict, judge_map: dict[int, dict], workload_type: str) -> dict:
    row: dict = {"row_index": raw["row_index"], "failed": raw.get("output") is None}
    it = raw.get("input_tokens")
    ot = raw.get("output_tokens")
    lat = raw.get("latency_ms")
    if it is not None:
        row["input_tokens"] = int(it)
    if ot is not None:
        row["output_tokens"] = int(ot)
    if lat is not None:
        row["latency_ms"] = int(lat)
    row_index = raw["row_index"]
    if workload_type in ("summarization", "captioning") and row_index in judge_map:
        scores = judge_map[row_index].get("scores")
        if scores:
            row["scores"] = scores
            row["overall_score"] = round(sum(scores.values()) / len(scores), 4)
    return row


def _build_scatter_rows(
    job_id: str, model_string: str, workload_type: str, output_dir: Path | None = None
) -> list[dict]:
    inference_path = job_inference_dir(job_id, output_dir) / model_filename(model_string)
    if not inference_path.exists():
        return []

    inference_map: dict[int, dict] = {r["row_index"]: r for r in read_rows(inference_path)}

    judge_map: dict[int, dict] = {}
    judge_path = job_evaluation_rows_path(job_id, model_string, output_dir)
    if judge_path.exists():
        for r in read_rows(judge_path):
            judge_map[r["row_index"]] = r

    return [_extract_row_fields(inference_map[i], judge_map, workload_type) for i in sorted(inference_map)]


def build_report_payload(
    job_id: str,
    status: object,
    output_dir: Path | None = None,
    scale_and_cost: dict | None = None,
    job: dict | None = None,
) -> dict:
    """Assemble the full report payload, write report.json, and return the dict."""
    from src.services.jobs import get_job_status

    if status is None:
        status = get_job_status(job_id)

    eval_dir = job_evaluation_dir(job_id, output_dir)
    results = []

    golden_path = job_golden_path(job_id, output_dir)
    sota_model = getattr(status, "sota_model", None)
    workload_type = getattr(status, "workload_type", None) or ""

    if sota_model and not golden_path.exists():
        raise FileNotFoundError(f"Golden dataset not found: {golden_path}")

    if golden_path.exists() and sota_model:
        golden_rows = list(read_rows(golden_path))
        sota_rows = [_extract_row_fields(r, {}, workload_type) for r in golden_rows]

        successful = sum(1 for r in sota_rows if not r["failed"])
        input_tokens_list = [r["input_tokens"] for r in sota_rows if "input_tokens" in r]
        output_tokens_list = [r["output_tokens"] for r in sota_rows if "output_tokens" in r]
        latency_list = sorted(r["latency_ms"] for r in sota_rows if "latency_ms" in r)

        inference_usage: dict = {"successful_rows": successful}
        if input_tokens_list:
            inference_usage["total_input_tokens"] = sum(input_tokens_list)
            inference_usage["mean_input_tokens"] = sum(input_tokens_list) / len(input_tokens_list)
        if output_tokens_list:
            inference_usage["total_output_tokens"] = sum(output_tokens_list)
            inference_usage["mean_output_tokens"] = sum(output_tokens_list) / len(output_tokens_list)
        if latency_list:
            n = len(latency_list)
            inference_usage["latency_ms"] = {
                "mean": sum(latency_list) / n,
                "p50": latency_list[int(n * 0.50)],
                "p95": latency_list[int(n * 0.95)],
            }

        sota_entry: dict = {
            "model": sota_model,
            "params": {},
            "workload_type": workload_type,
            "is_sota": True,
            "inference_usage": inference_usage,
            "failure_rate": (len(sota_rows) - successful) / len(sota_rows) if sota_rows else 0,
            "rows": sota_rows,
        }
        results.append(sota_entry)

    if not eval_dir.exists():
        raise FileNotFoundError(f"Evaluation directory not found: {eval_dir}")

    for path in sorted(eval_dir.glob("*_evaluation_result.json")):
        entry = json.loads(path.read_text(encoding="utf-8"))
        model_string = entry.get("model", "")
        wt = entry.get("workload_type", "")
        entry["rows"] = _build_scatter_rows(job_id, model_string, wt, output_dir)
        results.append(entry)

    payload: dict = {"job_id": job_id, "results": results}
    if scale_and_cost is not None:
        payload["scale_and_cost"] = scale_and_cost
    if job is not None:
        payload["job"] = job

    out = job_scale_and_cost_report_path(job_id, output_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Combined report written to %s", out)

    return payload
