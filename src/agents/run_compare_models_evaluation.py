"""
T11b — Scoring / evaluation.

Reads inference_output JSONL files and the golden dataset, scores each run
entry according to workload_type, and writes evaluation results to
output/{job_id}/evaluation.jsonl.

Scoring paths:
  classification  → accuracy, per-class F1, macro/weighted F1, confusion matrix
  extraction      → field-level accuracy, missing-field rate, semantic matching
  summarization   → LLM-as-judge via sota_model (also used for captioning)
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

from sqlmodel import select

from src.agents.inference_runner import run
from src.agents.pipeline_log import _log
from src.db import get_session
from src.env import settings
from src.providers.adapters import GenerationConfig, Message, ModelResponse
from src.providers.client import GetManagedModelProviderAPIKeyFunc, get_client, get_key_from_db
from src.schemas.db import JobConfig
from src.services.job_constants import JobStatusStr
from src.services.job_utils import (
    job_evaluation_dir,
    job_evaluation_result_path,
    job_evaluation_rows_path,
    job_inference_dir,
    write_job_progress,
)
from src.utils.jsonl import load_inference, read_rows

logger = logging.getLogger(__name__)

_DEFAULT_JUDGE_CRITERIA = [
    {"name": "faithfulness", "description": "The response accurately reflects the source content without fabrication."},
    {"name": "completeness", "description": "The response covers all key points from the source."},
    {
        "name": "conciseness",
        "description": "The response is appropriately brief without omitting important information.",
    },
    {
        "name": "instruction following",
        "description": "The response follows the format and constraints specified in the prompt.",
    },
]

_JUDGE_SYSTEM_PROMPT = (
    "You are an impartial evaluator scoring a candidate LLM output"
    " against a reference (golden) output.\n"
    "Score each criterion on a 1–5 integer scale:\n"
    "  1 = very poor, 2 = poor, 3 = acceptable, 4 = good, 5 = excellent\n"
    "\n"
    "Respond with a JSON object only, no extra text. Example:\n"
    '{"faithfulness": 4, "completeness": 3, "conciseness": 5, "instruction following": 4}'
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_golden_dataset(golden_dataset_path: str) -> dict[int, dict]:
    """Return {row_index: golden_entry} from golden_dataset.jsonl."""
    rows = read_rows(Path(golden_dataset_path))
    return {r["row_index"]: r for r in rows}


def _parse_output(raw: Any) -> Any:
    """Attempt to parse a JSON string; return as-is if already a dict/list."""
    if isinstance(raw, (dict, list)):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return raw
    return raw


# ---------------------------------------------------------------------------
# Semantic matching for extraction
# ---------------------------------------------------------------------------

_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*\.?\d*")


def _normalise_str(v: Any) -> str:
    if v is None:
        return ""
    return str(v).strip().lower()


def _semantic_match(candidate: Any, golden: Any) -> bool:
    """Loose equality for extraction fields (dates, numbers, names)."""
    c, g = _normalise_str(candidate), _normalise_str(golden)
    if c == g:
        return True
    # Numeric comparison: strip formatting
    cn = _NUMBER_RE.sub(lambda m: m.group().replace(",", ""), c)
    gn = _NUMBER_RE.sub(lambda m: m.group().replace(",", ""), g)
    if cn and gn and cn == gn:
        return True
    # Very short strings (≤3 chars): require exact match (already checked above)
    if len(c) <= 3 or len(g) <= 3:
        return False
    # Substring containment for names / partial matches
    if c in g or g in c:
        return True
    return False


# ---------------------------------------------------------------------------
# Token + latency stats
# ---------------------------------------------------------------------------


def _compute_usage_stats(inference_map: dict[int, dict]) -> dict[str, Any]:
    """Aggregate token counts and latency percentiles across successful inference rows."""
    input_tokens_list: list[int] = []
    output_tokens_list: list[int] = []
    latency_list: list[int] = []

    for row in inference_map.values():
        if row.get("output") is None:
            continue
        it = row.get("input_tokens")
        ot = row.get("output_tokens")
        lat = row.get("latency_ms")
        if it is not None:
            input_tokens_list.append(int(it))
        if ot is not None:
            output_tokens_list.append(int(ot))
        if lat is not None:
            latency_list.append(int(lat))

    def _percentile(data: list[int], p: int) -> int | None:
        if not data:
            return None
        sorted_data = sorted(data)
        idx = max(0, int(len(sorted_data) * p / 100) - 1)
        return sorted_data[idx]

    def _mean(data: list[int]) -> float | None:
        return round(sum(data) / len(data), 1) if data else None

    total_input = sum(input_tokens_list) if input_tokens_list else None
    total_output = sum(output_tokens_list) if output_tokens_list else None

    result: dict = {
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "mean_input_tokens": _mean(input_tokens_list),
        "mean_output_tokens": _mean(output_tokens_list),
        "successful_rows": len(latency_list),
    }
    if latency_list:
        result["latency_ms"] = {
            "mean": _mean(latency_list),
            "p50": _percentile(latency_list, 50),
            "p95": _percentile(latency_list, 95),
        }
    return result


# ---------------------------------------------------------------------------
# Classification scoring
# ---------------------------------------------------------------------------


def _score_classification(
    golden_map: dict[int, dict],
    inference_map: dict[int, dict],
) -> dict[str, Any]:
    labels: list[str] = []
    preds: list[str] = []

    for row_index, golden_row in golden_map.items():
        if golden_row.get("output") is None:
            continue
        inf_row = inference_map.get(row_index)
        if inf_row is None or inf_row.get("output") is None:
            continue
        labels.append(_normalise_str(golden_row["output"]))
        preds.append(_normalise_str(inf_row["output"]))

    if not labels:
        return {"accuracy": None, "macro_f1": None, "weighted_f1": None, "per_class": {}, "confusion_matrix": {}}

    classes = sorted(set(labels) | set(preds))

    # Confusion matrix: {actual: {predicted: count}}
    cm: dict[str, dict[str, int]] = {c: {d: 0 for d in classes} for c in classes}
    correct = 0
    for g, p in zip(labels, preds):
        cm[g][p] += 1
        if g == p:
            correct += 1

    accuracy = correct / len(labels)

    # Per-class precision, recall, F1
    per_class: dict[str, Any] = {}
    weighted_f1_sum = 0.0
    macro_f1_sum = 0.0
    for cls in classes:
        tp = cm[cls][cls]
        fp = sum(cm[other][cls] for other in classes if other != cls)
        fn = sum(cm[cls][other] for other in classes if other != cls)
        support = tp + fn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class[cls] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }
        macro_f1_sum += f1
        weighted_f1_sum += f1 * support

    macro_f1 = macro_f1_sum / len(classes) if classes else 0.0
    weighted_f1 = weighted_f1_sum / len(labels) if labels else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class": per_class,
        "confusion_matrix": cm,
        "evaluated_rows": len(labels),
    }


# ---------------------------------------------------------------------------
# Extraction scoring
# ---------------------------------------------------------------------------


def _score_extraction(
    golden_map: dict[int, dict],
    inference_map: dict[int, dict],
    eval_fields: list[str],
) -> dict[str, Any]:
    field_correct: dict[str, int] = defaultdict(int)
    field_total: dict[str, int] = defaultdict(int)
    field_missing: dict[str, int] = defaultdict(int)
    evaluated_rows = 0

    for row_index, golden_row in golden_map.items():
        if golden_row.get("output") is None:
            continue
        inf_row = inference_map.get(row_index)
        if inf_row is None or inf_row.get("output") is None:
            continue

        golden_parsed = _parse_output(golden_row["output"])
        candidate_parsed = _parse_output(inf_row["output"])

        if not isinstance(golden_parsed, dict):
            continue
        if not isinstance(candidate_parsed, dict):
            # Entire row output is unparseable — count all fields as missing
            for field in eval_fields:
                if field in golden_parsed:
                    field_total[field] += 1
                    field_missing[field] += 1
            evaluated_rows += 1
            continue

        evaluated_rows += 1
        for field in eval_fields:
            if field not in golden_parsed:
                continue
            field_total[field] += 1
            if field not in candidate_parsed:
                field_missing[field] += 1
            else:
                if _semantic_match(candidate_parsed[field], golden_parsed[field]):
                    field_correct[field] += 1

    fields_accuracy: dict[str, float | None] = {}
    missing_rate: dict[str, float | None] = {}
    for field in eval_fields:
        total = field_total[field]
        if total == 0:
            fields_accuracy[field] = None
            missing_rate[field] = None
        else:
            correct = field_correct[field]
            missing = field_missing[field]
            fields_accuracy[field] = round(correct / total, 4)
            missing_rate[field] = round(missing / total, 4)

    all_correct = sum(field_correct.values())
    all_total = sum(field_total.values())
    overall_accuracy = round(all_correct / all_total, 4) if all_total > 0 else None

    return {
        "overall_accuracy": overall_accuracy,
        "fields": fields_accuracy,
        "missing_field_rate": missing_rate,
        "evaluated_rows": evaluated_rows,
    }


# ---------------------------------------------------------------------------
# LLM-as-judge scoring (summarization / captioning)
# ---------------------------------------------------------------------------


def _build_judge_prompt(candidate_output: str, golden_output: str, criteria: list[dict]) -> str:
    criteria_lines = "\n".join(f"- {c['name']}: {c['description']}" for c in criteria)
    criteria_keys = ", ".join(f'"{c["name"]}"' for c in criteria)
    return (
        f"## Reference output (golden)\n{golden_output}\n\n"
        f"## Candidate output\n{candidate_output}\n\n"
        f"## Criteria\n{criteria_lines}\n\n"
        f"Score each criterion 1–5. Return JSON with keys: {criteria_keys}."
    )


def _extract_image_parts(golden_row: dict) -> list[dict]:
    """Return image_url content parts from the golden row's input messages, if any."""
    input_data = golden_row.get("input") or {}
    messages = input_data.get("messages") or []
    parts: list[dict] = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    parts.append(part)
    return parts


def _parse_judge_response(raw: str, criteria: list[dict]) -> dict[str, int]:
    """Parse and validate a judge LLM response into {criterion: score} dict."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[^\n]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)

    try:
        scores = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        m = re.search(r"\{[^{}]+\}", raw, re.DOTALL)
        if m:
            scores = json.loads(m.group())
        else:
            raise ValueError(f"Judge returned unparseable response: {raw[:200]}")

    parsed: dict[str, int] = {}
    for c in criteria:
        name = c["name"]
        raw_score = scores.get(name)
        try:
            parsed[name] = max(1, min(5, int(float(raw_score))))
        except (TypeError, ValueError):
            parsed[name] = 3  # neutral fallback
    return parsed


def _aggregate_judge_scores(judge_rows: list[dict], criteria: list[dict]) -> dict[str, Any]:
    """Compute per-criterion and overall mean scores from judge JSONL rows."""
    criterion_scores: dict[str, list[int]] = defaultdict(list)
    judge_input_tokens = 0
    judge_output_tokens = 0
    evaluated_rows = 0
    failed_rows = 0

    for row in judge_rows:
        scores = row.get("scores")
        if scores is None or row.get("output") is None:
            failed_rows += 1
            continue
        for name, score in scores.items():
            try:
                criterion_scores[name].append(int(score))
            except (TypeError, ValueError):
                pass
        judge_input_tokens += row.get("input_tokens", 0) or 0
        judge_output_tokens += row.get("output_tokens", 0) or 0
        evaluated_rows += 1

    mean_scores: dict[str, float | None] = {}
    for c in criteria:
        name = c["name"]
        vals = criterion_scores.get(name, [])
        mean_scores[name] = round(sum(vals) / len(vals), 4) if vals else None

    overall_vals = [v for vals in criterion_scores.values() for v in vals]
    overall_mean = round(sum(overall_vals) / len(overall_vals), 4) if overall_vals else None

    return {
        "mean_scores": mean_scores,
        "overall_mean": overall_mean,
        "evaluated_rows": evaluated_rows,
        "failed_rows": failed_rows,
        "judge_input_tokens": judge_input_tokens,
        "judge_output_tokens": judge_output_tokens,
    }


def run_judge_rows(
    golden_map: dict[int, dict],
    inference_map: dict[int, dict],
    criteria: list[dict],
    sota_model: str,
    output_path: Path,
    on_progress: Any = None,
    retry_failed: bool = False,
    log: Any = None,
    include_images: bool = False,
    get_managed_model_provider_api_key_func: GetManagedModelProviderAPIKeyFunc = get_key_from_db,
) -> list[dict]:
    """
    Run the judge LLM over all (candidate, golden) pairs and stream-write results.
    Returns a list of judge row dicts for aggregation.

    When include_images=True the original image content parts from the golden row's
    input are prepended to the judge user message so the judge can verify the caption
    against the actual image rather than comparing text only.
    """
    judge_rows_input: list[dict] = []
    for row_index, golden_row in sorted(golden_map.items()):
        if golden_row.get("output") is None:
            continue
        inf_row = inference_map.get(row_index)
        if inf_row is None or inf_row.get("output") is None:
            continue
        row: dict[str, Any] = {
            "row_index": row_index,
            "_candidate_output": str(inf_row["output"]),
            "_golden_output": str(golden_row["output"]),
        }
        if include_images:
            row["_image_parts"] = _extract_image_parts(golden_row)
        judge_rows_input.append(row)

    client = get_client(sota_model, get_managed_model_provider_api_key_func=get_managed_model_provider_api_key_func)
    config = GenerationConfig(temperature=0.0, top_p=1.0)

    def build_messages(row: dict) -> list[Message]:
        prompt = _build_judge_prompt(row["_candidate_output"], row["_golden_output"], criteria)
        image_parts: list[dict] = row.get("_image_parts") or []
        if image_parts:
            user_content: list[Any] = [*image_parts, {"type": "text", "text": prompt}]
            return [Message(role="system", content=_JUDGE_SYSTEM_PROMPT), Message(role="user", content=user_content)]
        return [Message(role="system", content=_JUDGE_SYSTEM_PROMPT), Message(role="user", content=prompt)]

    def parse_response(resp: ModelResponse, row: dict) -> dict:  # noqa: ARG001
        scores = _parse_judge_response(resp.content, criteria)
        return {"scores": scores}

    return run(
        rows=judge_rows_input,
        output_path=output_path,
        build_messages=build_messages,
        client=client,
        config=config,
        parse_response=parse_response,
        on_progress=on_progress,
        retry=retry_failed,
        log=log,
    )


def _score_summarization(
    golden_map: dict[int, dict],
    inference_map: dict[int, dict],
    criteria: list[dict],
    sota_model: str,
    job_id: str,
    output_path: Path,
    include_images: bool = False,
    on_progress: Any = None,
    get_managed_model_provider_api_key_func: GetManagedModelProviderAPIKeyFunc = get_key_from_db,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Returns (metrics, evaluation_usage)."""
    _named_log = partial(_log, job_id, "run_compare_models_evaluation", logger=logger, log_only=True)

    judge_rows = run_judge_rows(
        golden_map=golden_map,
        inference_map=inference_map,
        criteria=criteria,
        sota_model=sota_model,
        output_path=output_path,
        on_progress=on_progress,
        log=_named_log,
        include_images=include_images,
        get_managed_model_provider_api_key_func=get_managed_model_provider_api_key_func,
    )
    metrics = _aggregate_judge_scores(judge_rows, criteria)
    evaluation_usage = {
        "judge_input_tokens": metrics.get("judge_input_tokens", 0),
        "judge_output_tokens": metrics.get("judge_output_tokens", 0),
    }
    return metrics, evaluation_usage


# ---------------------------------------------------------------------------
# Result entry builder
# ---------------------------------------------------------------------------


def _write_result(path: Path, entry: dict[str, Any]) -> None:
    path.write_text(json.dumps(entry, indent=2), encoding="utf-8")


def _summarize_model_inference_output(
    model: str,
    params: dict,
    workload_type: str | None,
    model_inference_output: dict[int, dict],
) -> dict[str, Any]:
    rows = model_inference_output.values()
    total = len(rows)  # type: ignore[arg-type]
    failed = sum(1 for r in rows if r.get("output") is None)
    failure_rate = round(failed / total, 4) if total > 0 else 0.0
    return {
        "model": model,
        "params": params,
        "workload_type": workload_type,
        "inference_usage": _compute_usage_stats(model_inference_output),
        "failure_rate": failure_rate,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_compare_models_evaluation(
    job_id: str,
    output_dir: Path | None = None,
    on_model_start: Any = None,
    on_model_complete: Any = None,
    on_progress: Any = None,
    get_managed_model_provider_api_key_func: GetManagedModelProviderAPIKeyFunc = get_key_from_db,
) -> None:
    with get_session() as session:
        job = session.exec(select(JobConfig).where(JobConfig.job_id == job_id)).first()
        if job is None:
            raise ValueError(f"Job '{job_id}' not found.")

        workload_type = job.workload_type
        golden_dataset_path = job.generated_golden_dataset_path
        eval_fields: list[str] = job.evaluation_fields or []
        judge_criteria_raw: list[dict] | None = job.judge_criteria
        sota_model: str = job.sota_model or settings.DEFAULT_SOTA_MODEL
        compare_models: list[dict] = job.compare_models or []
        input_modality: str | None = job.input_modality

    _named_log = partial(_log, job_id, "run_compare_models_evaluation", logger=logger)

    total_models = len(compare_models)
    model_progress: list[dict] = [{"model": e["model"], "status": "pending", "done": False} for e in compare_models]

    def _write_progress(model: str, model_idx: int, done: bool = False) -> None:
        model_progress[model_idx]["status"] = "done" if done else "running"
        model_progress[model_idx]["done"] = done
        write_job_progress(
            job_id,
            "evaluation_progress",
            {
                "current_model": model,
                "model_index": model_idx,
                "total_models": total_models,
                "models": list(model_progress),
            },
        )

    if not golden_dataset_path:
        raise ValueError(f"Job '{job_id}' has no generated_golden_dataset_path — run golden generation first.")

    _named_log(f"Starting evaluation: workload_type={workload_type}, {len(compare_models)} model(s)")

    golden_dataset = _load_golden_dataset(golden_dataset_path)

    inference_dir = job_inference_dir(job_id, output_dir)
    eval_dir = job_evaluation_dir(job_id, output_dir)

    if judge_criteria_raw is None:
        judge_criteria = _DEFAULT_JUDGE_CRITERIA
    else:
        judge_criteria = judge_criteria_raw

    # Feed the original image to the judge when the job is image-modal
    include_images = input_modality == "image"

    eval_dir.mkdir(parents=True, exist_ok=True)

    for model_idx, entry in enumerate(compare_models):
        model_string = entry["model"]
        params = entry.get("params", {})

        _write_progress(model_string, model_idx, done=False)
        if on_model_start:
            on_model_start(model_string, model_idx, total_models)

        inference_data = load_inference(inference_dir, model_string)

        total_inference_rows = len(inference_data)
        failed_inference = sum(1 for r in inference_data.values() if r.get("output") is None)
        failure_rate = failed_inference / total_inference_rows if total_inference_rows > 0 else 0.0

        _named_log(
            f"Scoring model {model_string} ({total_inference_rows} inference rows, failure_rate={failure_rate:.2%})"
        )

        evaluation_usage: dict[str, Any] | None = None

        if workload_type in ("summarization", "captioning"):
            if not judge_criteria:
                raise ValueError(f"workload_type='{workload_type}' requires judge_criteria but none were provided.")
            judge_output_path = job_evaluation_rows_path(job_id, model_string, output_dir)

            def _on_judge_progress(completed: int, total: int, _ms: str = model_string) -> None:
                if on_progress:
                    on_progress(completed, total, _ms)

            metrics, evaluation_usage = _score_summarization(
                golden_dataset,
                inference_data,
                judge_criteria,
                sota_model,
                job_id,
                judge_output_path,
                include_images=include_images,
                on_progress=_on_judge_progress,
                get_managed_model_provider_api_key_func=get_managed_model_provider_api_key_func,
            )
        elif workload_type == "classification":
            metrics = _score_classification(golden_dataset, inference_data)
        elif workload_type == "extraction":
            if not eval_fields:
                sample_output = next(
                    (r["output"] for r in golden_dataset.values() if isinstance(_parse_output(r.get("output")), dict)),
                    None,
                )
                if sample_output is not None:
                    eval_fields = list(_parse_output(sample_output).keys())
            metrics = _score_extraction(golden_dataset, inference_data, eval_fields)
        else:
            raise ValueError(
                f"Unsupported workload_type '{workload_type}'. Must be one of: classification, extraction, "
                "summarization, captioning."
            )

        result_entry = _summarize_model_inference_output(model_string, params, workload_type, inference_data)
        result_entry["metrics"] = metrics
        if evaluation_usage is not None:
            result_entry["evaluation_usage"] = evaluation_usage

        model_result_path = job_evaluation_result_path(job_id, model_string, output_dir)
        _write_result(model_result_path, result_entry)

        _write_progress(model_string, model_idx, done=True)
        if on_model_complete:
            on_model_complete(model_string, model_idx, total_models, failed_inference, total_inference_rows)
        _named_log(f"Model {model_string} scored → {model_result_path}")

    _named_log(f"Wrote evaluation results to {eval_dir}")

    now = datetime.now(timezone.utc).isoformat()
    with get_session() as session:
        job = session.exec(select(JobConfig).where(JobConfig.job_id == job_id)).first()
        if job is None:
            raise ValueError(f"Job '{job_id}' not found.")
        job.evaluating_inference_output_path = str(eval_dir)
        job.status = JobStatusStr.EVALUATING_INFERENCE_OUTPUT
        job.updated_at = now
        session.add(job)
        session.commit()

    _named_log("Done")
