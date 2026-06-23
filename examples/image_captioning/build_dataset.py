#!/usr/bin/env python3
"""
Curates the image captioning demo dataset from VizWiz-Captions val split (CC-BY 4.0).

Annotations: https://vizwiz.cs.colorado.edu/VizWiz_final/caption/annotations.zip
Images served per-file from: https://vizwiz.cs.colorado.edu/VizWiz_visualization_img/

Outputs:
  examples/image_captioning/input.jsonl
  examples/image_captioning/ground_truth.jsonl
  examples/image_captioning/images/   (downloaded image files, <1 MB each)

Usage:
  pip install requests
  python examples/image_captioning/build_dataset.py

Images in input.jsonl are referenced via raw GitHub URL:
  https://raw.githubusercontent.com/apiobuild/luna8i-judge/main/examples/image_captioning/images/<filename>
These URLs resolve once the repo is main and the CD workflow has pushed to the `main` branch.

Only images under 1 MB are downloaded; rows with larger images are skipped.
50 rows total: 40 good-quality + 10 poor-quality (is_precanned).
"""

import io
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = REPO_ROOT / "examples"
GITHUB_RAW_BASE = (
    "https://raw.githubusercontent.com/apiobuild/luna8i-judge/main/examples"
)

ANNOTATIONS_URL = "https://vizwiz.cs.colorado.edu/VizWiz_final/caption/annotations.zip"
IMAGE_BASE_URL = "https://vizwiz.cs.colorado.edu/VizWiz_visualization_img"

MAX_IMAGE_BYTES = 1 * 1024 * 1024  # 1 MB
TARGET_GOOD = 40
TARGET_POOR = 10


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  wrote {len(rows)} rows → {path.relative_to(REPO_ROOT)}")


def load_val_annotations(
    session: requests.Session,
) -> tuple[list[dict], dict[int, list[str]], set[int]]:
    """Download annotations zip and return (images list, image_id -> captions dict)."""
    print("Downloading VizWiz-Captions annotations …")
    resp = session.get(ANNOTATIONS_URL, timeout=60)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        with zf.open("annotations/val.json") as f:
            data = json.load(f)

    images = data["images"]

    # Build image_id -> non-rejected captions list; skip precanned (poor quality marker)
    captions_by_image: dict[int, list[str]] = defaultdict(list)
    for ann in data["annotations"]:
        if not ann["is_rejected"] and not ann["is_precanned"]:
            captions_by_image[ann["image_id"]].append(ann["caption"])

    # Mark images as poor-quality if all their annotations are precanned
    precanned_image_ids = set()
    all_ann_by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in data["annotations"]:
        all_ann_by_image[ann["image_id"]].append(ann)
    for img in images:
        anns = all_ann_by_image[img["id"]]
        if anns and all(a["is_precanned"] for a in anns):
            precanned_image_ids.add(img["id"])

    return images, captions_by_image, precanned_image_ids


def fetch_image(session: requests.Session, filename: str) -> bytes | None:
    url = f"{IMAGE_BASE_URL}/{filename}"
    try:
        head = session.head(url, timeout=10, allow_redirects=True)
        if int(head.headers.get("content-length", 0)) > MAX_IMAGE_BYTES:
            return None
        resp = session.get(url, timeout=30, allow_redirects=True)
        resp.raise_for_status()
        if len(resp.content) > MAX_IMAGE_BYTES:
            return None
        return resp.content
    except Exception as e:
        print(f"    skip (download error for {filename}): {e}")
        return None


def main() -> None:
    images_dir = EXAMPLES_DIR / "image_captioning" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers["User-Agent"] = "luna8i-judge-curation/1.0"

    images, captions_by_image, precanned_image_ids = load_val_annotations(session)

    good_rows: list[dict] = []
    poor_rows: list[dict] = []
    skipped = 0

    for img in images:
        if len(good_rows) >= TARGET_GOOD and len(poor_rows) >= TARGET_POOR:
            break

        image_id = img["id"]
        filename = img["file_name"]
        is_poor = image_id in precanned_image_ids

        if is_poor and len(poor_rows) >= TARGET_POOR:
            continue
        if not is_poor and len(good_rows) >= TARGET_GOOD:
            continue

        # Skip if no usable captions for good images
        if not is_poor and not captions_by_image[image_id]:
            skipped += 1
            continue

        data = fetch_image(session, filename)
        if data is None:
            skipped += 1
            continue

        (images_dir / filename).write_bytes(data)
        github_url = f"{GITHUB_RAW_BASE}/image_captioning/images/{filename}"

        if is_poor:
            ground_truth_output = "unanswerable"
        else:
            ground_truth_output = captions_by_image[image_id][0]

        entry = {
            "input_row": {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": github_url}},
                            {
                                "type": "text",
                                "text": 'Describe this image in one to two sentences. If the image quality is too poor to make out any content, respond with {"caption": "unanswerable"}.',
                            },
                        ],
                    }
                ]
            },
            "ground_truth_output": ground_truth_output,
        }

        (poor_rows if is_poor else good_rows).append(entry)

        total = len(good_rows) + len(poor_rows)
        if total % 10 == 0:
            print(
                f"    {len(good_rows)} good-quality, {len(poor_rows)} poor-quality (skipped {skipped})"
            )

    all_rows = good_rows + poor_rows
    print(
        f"\n  total: {len(all_rows)} rows ({len(good_rows)} good, {len(poor_rows)} poor), skipped {skipped}"
    )

    write_jsonl(
        EXAMPLES_DIR / "image_captioning" / "input.jsonl",
        [r["input_row"] for r in all_rows],
    )
    write_jsonl(
        EXAMPLES_DIR / "image_captioning" / "ground_truth.jsonl",
        [{"output": r["ground_truth_output"]} for r in all_rows],
    )
    print("\nDone. Run `git add examples/image_captioning/` to stage the outputs.")


if __name__ == "__main__":
    main()
