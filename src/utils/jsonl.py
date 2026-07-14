from __future__ import annotations

import json
from pathlib import Path


def model_filename(model_string: str, prefix: str = "") -> str:
    """Convert 'provider/model-name' to '{prefix}provider__model-name.jsonl'."""
    return prefix + model_string.replace("/", "__") + ".jsonl"


def parse_rows(raw: bytes) -> list[dict]:
    """Parse UTF-8 JSONL bytes into a list of row dicts."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Input file is not valid UTF-8.") from exc

    rows: list[dict] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            raise ValueError(f"Row {lineno}: invalid JSON.")

        if not isinstance(row.get("messages"), list):
            raise ValueError(f"Row {lineno}: missing or invalid 'messages' array.")

        for msg in row["messages"]:
            if "role" not in msg or "content" not in msg:
                raise ValueError(f"Row {lineno}: each message must have 'role' and 'content'.")

        if "row_index" not in row:
            row["row_index"] = lineno - 1
        rows.append(row)

    return rows


def read_rows(path: Path) -> list[dict]:
    path = Path(path)
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_rows(path: Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def open_jsonl_writer(path: Path):
    """Return an open file handle for streaming row-by-row JSONL writes.

    Caller is responsible for closing (or using as a context manager).
    Creates parent directories and truncates any existing file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("w", encoding="utf-8", buffering=1)  # line-buffered


def append_row(f, row: dict) -> None:
    """Write a single JSON row to an open JSONL file handle."""
    f.write(json.dumps(row) + "\n")


def load_inference(inference_dir: Path, model_string: str) -> dict[int, dict]:
    """Return {row_index: inference_row} for one model's output file."""
    path = inference_dir / model_filename(model_string)
    if not path.exists():
        return {}
    rows = read_rows(path)
    return {r["row_index"]: r for r in rows}
