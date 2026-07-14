"""
T5 tests — five required cases. Consistency check is pure regex; LLM calls
are intercepted by patching src.providers.client.complete_chat.
"""

import json
from unittest.mock import patch

import pytest

from src.agents.detect_workload import detect_workload, validate_prompt_consistency

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

TEXT_ROW = {"messages": [{"role": "user", "content": "Extract the invoice number from: INV-001"}]}
IMAGE_ROW = {
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]
}

PATCH = "src.providers.client.complete_chat"


# ---------------------------------------------------------------------------
# Test 1: consistent rows + clear extraction prompt → high confidence extraction
# ---------------------------------------------------------------------------


def test_detect_extraction_high_confidence():
    response = json.dumps(
        {"workload_type": "extraction", "confidence": "high", "confidence_note": "Clear extraction prompt."}
    )
    with patch(PATCH, return_value=response):
        result = detect_workload(r"Extract .+ from:", [TEXT_ROW])

    assert result.workload_type == "extraction"
    assert result.confidence == "high"
    assert result.modality == "text"


# ---------------------------------------------------------------------------
# Test 2: inconsistent row → ValueError, classification not called
# ---------------------------------------------------------------------------


def test_validate_prompt_consistency_passes_on_match():
    rows = [{"messages": [{"role": "user", "content": "Extract the invoice number from: INV-001"}]}] * 5
    validate_prompt_consistency(r"Extract the invoice number from: [A-Z]+-\d+", rows)  # must not raise


def test_validate_prompt_consistency_raises_on_mismatch():
    rows = [{"messages": [{"role": "user", "content": "Summarize this article."}]}] * 5
    with pytest.raises(ValueError, match="Row 0"):
        validate_prompt_consistency(r"Extract the invoice number from: [A-Z]+-\d+", rows)


def test_detect_workload_raises_before_classification_on_mismatch():
    rows = [{"messages": [{"role": "user", "content": "Summarize this article."}]}] * 5
    with patch(PATCH) as mock_complete:
        with pytest.raises(ValueError, match="Row 0"):
            detect_workload("Extract fields from", rows)
    mock_complete.assert_not_called()


# ---------------------------------------------------------------------------
# Test 3: ambiguous prompt → confidence="low", workload_type=None
# ---------------------------------------------------------------------------


def test_detect_ambiguous_prompt_low_confidence():
    response = json.dumps({"workload_type": None, "confidence": "low", "confidence_note": "Mixes types."})
    with patch(PATCH, return_value=response):
        result = detect_workload(".", [TEXT_ROW])

    assert result.workload_type is None
    assert result.confidence == "low"


# ---------------------------------------------------------------------------
# Test 4: rows with image URLs → modality="image"
# ---------------------------------------------------------------------------


def test_detect_image_modality():
    response = json.dumps(
        {"workload_type": "extraction", "confidence": "high", "confidence_note": "Extraction from image."}
    )
    with patch(PATCH, return_value=response):
        result = detect_workload("Describe", [IMAGE_ROW])

    assert result.modality == "image"


# ---------------------------------------------------------------------------
# Test 5: empty template → confidence="low", no LLM call
# ---------------------------------------------------------------------------


def test_detect_empty_template_low_confidence():
    with patch(PATCH) as mock_complete:
        result = detect_workload("", [TEXT_ROW])

    assert result.confidence == "low"
    assert result.workload_type is None
    mock_complete.assert_not_called()
