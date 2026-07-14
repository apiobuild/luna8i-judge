"""
Shared job submission logic — used by both the HTTP router and the CLI.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.agents.detect_workload import PromptConsistencyError, check_modality_consistency, detect_workload
from src.schemas.jobs import CustomManagedProviderPricing, CustomSelfHostedProviderPricing, SubmissionRequest
from src.services.jobs import get_upload_raw
from src.utils.jsonl import parse_rows


class SubmissionError(ValueError):
    """Validation or parsing error during job submission."""


@dataclass
class ParsedSubmission:
    rows: list[dict]
    prompt_template: str
    sota_model: str | None
    alias: str | None
    workload_type: str | None
    input_modality: str
    detected_workload_details: dict
    output_json_schema: dict | None
    eval_fields: list[str] | None
    compare_models: list[dict] | None
    judge_criteria: list[dict] | None
    projection_by_num_records: list[int] | None
    target_sla_hours: float | None
    hosting_preference: str | None
    managed_provider_custom_pricing: CustomManagedProviderPricing | None
    self_hosted_provider_custom_pricing: CustomSelfHostedProviderPricing | None
    golden_generation_config: dict | None


def parse_submission_from_upload_id(upload_id: str, req: SubmissionRequest) -> ParsedSubmission:
    raw = get_upload_raw(upload_id)
    if raw is None:
        raise SubmissionError(f"upload_id '{upload_id}' not found.")
    return parse_submission(raw, req)


def parse_submission(raw: bytes, req: SubmissionRequest) -> ParsedSubmission:
    """
    Validate raw JSONL bytes + SubmissionRequest fields and resolve workload
    detection. Raises SubmissionError on any validation failure.
    """
    try:
        rows = parse_rows(raw)
    except ValueError as exc:
        raise SubmissionError(str(exc)) from exc

    if not rows:
        raise SubmissionError("Uploaded file is empty.")

    try:
        check_modality_consistency(rows)
    except ValueError as exc:
        raise SubmissionError(str(exc)) from exc

    eval_fields = req.eval_fields
    if eval_fields is None and req.output_json_schema:
        eval_fields = list(req.output_json_schema.get("properties", req.output_json_schema).keys())

    if req.workload_type:
        detected_workload_details: dict = {
            "workload_type": req.workload_type,
            "modality": req.modality or "text",
            "confidence": "override",
            "confidence_note": "Workload type set explicitly by user.",
        }
        resolved_workload = req.workload_type
        resolved_modality = req.modality or "text"
    else:
        try:
            detection = detect_workload(req.prompt_template, rows[:5], model=req.sota_model)
        except PromptConsistencyError as exc:
            raise SubmissionError(str(exc)) from exc

        detected_workload_details = {
            "workload_type": detection.workload_type,
            "modality": detection.modality,
            "confidence": detection.confidence,
            "confidence_note": detection.confidence_note,
        }
        resolved_workload = detection.workload_type
        resolved_modality = req.modality or detection.modality

    compare_models = [e.model_dump() for e in req.compare_models] if req.compare_models else None
    judge_criteria = [c.model_dump() for c in req.judge_criteria] if req.judge_criteria else None

    return ParsedSubmission(
        rows=rows,
        prompt_template=req.prompt_template,
        sota_model=req.sota_model,
        alias=req.alias,
        workload_type=resolved_workload,
        input_modality=resolved_modality,
        detected_workload_details=detected_workload_details,
        output_json_schema=req.output_json_schema,
        eval_fields=eval_fields,
        compare_models=compare_models,
        judge_criteria=judge_criteria,
        projection_by_num_records=req.projection_by_num_records,
        target_sla_hours=req.target_sla_hours,
        hosting_preference=req.hosting_preference,
        managed_provider_custom_pricing=req.managed_provider_custom_pricing,
        self_hosted_provider_custom_pricing=req.self_hosted_provider_custom_pricing,
        golden_generation_config=req.golden_generation_config.model_dump() if req.golden_generation_config else None,
    )
