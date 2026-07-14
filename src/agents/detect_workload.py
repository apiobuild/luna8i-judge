"""
Workload type detection — pure library functions.

Two-step detection:
  1. validate_prompt_consistency — regex-checks each row's last user message
     against the declared prompt_template; raises ValueError on first mismatch.
  2. detect_workload — LLM-classifies prompt_template into a workload type.

detect_workload is provider-agnostic; it accepts an optional `llm_call`
injectable for testing. Production callers pass no override and get the
DEFAULT_SOTA_MODEL from settings.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from src.providers.client import GetManagedModelProviderAPIKeyFunc, get_key_from_db

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class PromptConsistencyError(ValueError):
    """Raised when a row's user message doesn't match the prompt_template regex."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"{reason}.")
        self.reason = reason


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

SUPPORTED_WORKLOAD_TYPES = ("classification", "extraction", "summarization", "captioning")


@dataclass
class DetectionResult:
    workload_type: str | None
    modality: str
    confidence: str  # "high" | "low" | "override"
    confidence_note: str


# ---------------------------------------------------------------------------
# Internal LLM helpers
# ---------------------------------------------------------------------------

_CLASSIFICATION_SYSTEM = """\
You are a workload classifier for an LLM benchmarking tool.
You will be given a prompt_template. Classify it into exactly one of:
  - classification  (model assigns a label / category)
  - extraction      (model extracts structured fields from text)
  - summarization   (model writes a condensed free-text summary)
  - captioning      (model generates a caption or description for an image or media asset)

Reply ONLY with valid JSON in this exact shape:
{"workload_type": "<type>", "confidence": "high"|"low", "confidence_note": "<one sentence>"}

Use "low" confidence when the prompt is ambiguous, mixes types, or matches
none of the categories above. In that case set workload_type to null.
"""


# ---------------------------------------------------------------------------
# Modality detection
# ---------------------------------------------------------------------------


def _detect_modality(rows: list[dict]) -> str:
    """Scan content fields for non-text assets; return dominant modality."""
    for row in rows:
        for msg in row.get("messages", []):
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                t = part.get("type", "")
                if t == "image_url" or t == "image":
                    return "image"
                if t in ("audio", "audio_url"):
                    return "audio"
                if t in ("video", "video_url"):
                    return "video"
    return "text"


def _row_modality(row: dict) -> str:
    return _detect_modality([row])


def check_modality_consistency(rows: list[dict]) -> None:
    """Raise ValueError if rows mix text-only and multimodal content."""
    first_modality: str | None = None
    for i, row in enumerate(rows):
        modality = _row_modality(row)
        if first_modality is None:
            first_modality = modality
        elif modality != first_modality:
            raise ValueError(f"Modality mismatch: row 1 is {first_modality}, row {i + 1} contains {modality} content.")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_prompt_consistency(
    prompt_template: str,
    rows: list[dict],
) -> None:
    """
    Regex-check each row's last user message against prompt_template.
    Raises ValueError("Row N: ...") on the first row that doesn't match.
    Skips rows with no user messages.
    """
    if not prompt_template or not rows:
        return

    pattern = re.compile(prompt_template)

    for i, row in enumerate(rows):
        msgs = row.get("messages", [])
        user_msgs = [m for m in msgs if m.get("role") == "user"]
        if not user_msgs:
            continue
        content = user_msgs[-1].get("content", "")
        if isinstance(content, list):
            text_parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
            content = " ".join(text_parts)
        if not pattern.search(content):
            truncated = content[:200] + ("…" if len(content) > 200 else "")
            raise PromptConsistencyError(
                f"input message does not match the prompt template pattern.\n"
                f"  Row {i} message: {truncated!r}"
                f"  Prompt template:  {prompt_template}\n",
            )


def detect_workload(
    prompt_template: str,
    rows: list[dict],
    model: str | None = None,
    get_managed_model_provider_api_key_func: GetManagedModelProviderAPIKeyFunc = get_key_from_db,
) -> DetectionResult:
    """
    Full two-step detection:
      1. validate_prompt_consistency — raises PromptConsistencyError on mismatch
      2. LLM classification of prompt_template → DetectionResult

    rows are used for modality detection and consistency validation only;
    they are NOT sent to the classification LLM.

    model: model string from the job payload; falls back to DEFAULT_SOTA_MODEL from settings.
    """
    from src.env import settings
    from src.providers.client import complete_chat

    modality = _detect_modality(rows)
    resolved_model = model or settings.DEFAULT_SOTA_MODEL

    validate_prompt_consistency(prompt_template, rows)

    if not prompt_template:
        return DetectionResult(
            workload_type=None,
            modality=modality,
            confidence="low",
            confidence_note="No prompt template provided; cannot classify workload type.",
        )

    from src.providers.adapters import Message

    try:
        raw = complete_chat(
            resolved_model,
            [Message(role="user", content=prompt_template)],
            system=_CLASSIFICATION_SYSTEM,
            get_managed_model_provider_api_key_func=get_managed_model_provider_api_key_func,
        )
    except RuntimeError as exc:
        return DetectionResult(
            workload_type=None,
            modality=modality,
            confidence="low",
            confidence_note=f"Could not classify workload: {exc}",
        )

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        return DetectionResult(
            workload_type=None,
            modality=modality,
            confidence="low",
            confidence_note="LLM returned non-JSON; cannot classify workload type.",
        )

    workload_type = result.get("workload_type")
    if workload_type not in SUPPORTED_WORKLOAD_TYPES:
        workload_type = None

    return DetectionResult(
        workload_type=workload_type,
        modality=modality,
        confidence=result.get("confidence", "low"),
        confidence_note=result.get("confidence_note", ""),
    )
