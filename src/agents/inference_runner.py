"""
T20 — Generic inference core.

run() is the shared inference loop used by:
  generate_golden_dataset.py  — golden generation
  run_compare_models_inference.py  — compare_models inference
  evaluate.py                 — LLM-as-judge scoring
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import openai
from tenacity import retry as tenacity_retry
from tenacity import retry_if_exception_type, stop_after_attempt, wait_exponential

from src.providers.adapters import GenerationConfig, Message, ModelClient, ModelResponse
from src.utils.jsonl import append_row, open_jsonl_writer, read_rows

_MAX_RETRIES = 3


@tenacity_retry(
    retry=retry_if_exception_type(openai.RateLimitError),
    stop=stop_after_attempt(_MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=1, max=9),
    reraise=True,
)
def _call_with_retry(client: ModelClient, messages: list[Message], config: GenerationConfig) -> ModelResponse:
    return client.complete(messages, config)


def run(
    rows: list[dict],
    output_path: Path,
    build_messages: Callable[[dict], list[Message]],
    client: ModelClient,
    config: GenerationConfig,
    parse_response: Callable[[ModelResponse, dict], dict],
    on_progress: Callable[[int, int], None] | None = None,
    retry: bool = False,
    log: Callable[[str, str], None] | None = None,
) -> list[dict]:
    """
    Run a model over a list of rows, stream-writing results to output_path.

    Args:
        rows: Input rows; each must have a "row_index" key.
        output_path: JSONL file to write results into.
        build_messages: Converts one row to a list of Messages for the model.
        client: ModelClient instance to call.
        config: GenerationConfig (temperature, top_p, response_format, etc.).
        parse_response: Maps (ModelResponse, row) → extra fields merged into the row output.
        on_progress: Called with (completed, total) after each row.
        retry: If True and output_path already exists, re-run rows that have errors.
        log: Optional log(message, level) callback.

    Returns:
        List of written row dicts (one per input row, in input order).
    """

    def _log(msg: str, level: str = "info") -> None:
        if log:
            log(msg, level)

    # Load existing rows if file exists
    existing: dict[int, dict] = {}
    if output_path.exists():
        for r in read_rows(output_path):
            existing[r["row_index"]] = r

    # Decide which rows need to be run
    rows_to_run: list[dict] = []
    for row in rows:
        idx = row.get("row_index", rows.index(row))
        if idx in existing and (existing[idx].get("output") is not None or not retry):
            continue
        rows_to_run.append(row)

    total = len(rows)
    completed = total - len(rows_to_run)

    # Build a mutable results index from existing rows
    results: dict[int, dict] = dict(existing)

    rows_to_run.sort(key=lambda r: r.get("row_index", rows.index(r)))

    if rows_to_run:
        out_f = (
            output_path.open("a", encoding="utf-8", buffering=1)
            if output_path.exists()
            else open_jsonl_writer(output_path)
        )

        try:
            for row in rows_to_run:
                row_index = row.get("row_index", rows.index(row))
                output = None
                error = None
                input_tokens = 0
                output_tokens = 0
                latency_ms = None

                messages = build_messages(row)

                try:
                    t0 = time.monotonic()
                    resp = _call_with_retry(client, messages, config)
                    latency_ms = int((time.monotonic() - t0) * 1000)
                    output = resp.content
                    input_tokens = resp.input_tokens
                    output_tokens = resp.output_tokens
                except Exception as exc:
                    error = str(exc)
                    _log(f"Row {row_index} failed: {exc}", "warning")

                if output is None:
                    row_result: dict = {
                        "row_index": row_index,
                        "output": None,
                        "error": error or "unknown error",
                        "latency_ms": None,
                    }
                else:
                    extra = parse_response(resp, row)  # type: ignore[arg-type]
                    row_result = {
                        "row_index": row_index,
                        "output": output,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "latency_ms": latency_ms,
                        **extra,
                    }

                results[row_index] = row_result
                append_row(out_f, row_result)
                completed += 1

                if on_progress:
                    on_progress(completed, total)
        finally:
            out_f.close()
    else:
        # All rows already done — still call progress for each
        if on_progress:
            for i in range(1, completed + 1):
                on_progress(i, total)

    return [results[row.get("row_index", i)] for i, row in enumerate(rows) if row.get("row_index", i) in results]
