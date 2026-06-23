# Image Captioning Example

Benchmarks candidate models on image captioning using 50 images from the
[VizWiz-Captions](https://vizwiz.org/tasks-and-tools/image-captioning/) val split (CC-BY 4.0).

## Dataset

| | Count |
|---|---|
| Good-quality images | 40 |
| Poor-quality images (unanswerable) | 10 |
| **Total** | **50** |

Poor-quality images have `"output": "unanswerable"` in `ground_truth.jsonl` — these are images
whose quality was too low for human annotators to describe.

## Files

| File | Description |
|---|---|
| `input.jsonl` | 50 rows, each an OpenAI `messages` payload with one image URL + caption prompt |
| `ground_truth.jsonl` | First human caption per image; `"unanswerable"` for poor-quality images |
| `output_schema.json` | JSON Schema for the expected `{"caption": "..."}` response format |
| `images/` | Downloaded image files (≤ 1 MB each) |
| `build_dataset.py` | Curation script — re-run to regenerate |

## Input format

Each row in `input.jsonl` is a self-contained `messages` payload:

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "https://raw.githubusercontent.com/apiobuild/luna8i-judge/main/examples/image_captioning/images/VizWiz_val_00000000.jpg"}},
        {"type": "text", "text": "Describe this image in one to two sentences. If the image quality is too poor to make out any content, respond with {\"caption\": \"unanswerable\"}."}
      ]
    }
  ]
}
```

Images are referenced via raw GitHub URLs and resolve once the files are pushed to `main`.

## Output schema

Models are expected to return a JSON object matching `output_schema.json`:

```json
{"caption": "A computer screen shows a repair prompt on the screen."}
```

For poor-quality images the expected output is `{"caption": "unanswerable"}`. Pass `output_schema.json`
as `output_json_schema` when submitting this job so luna8i-judge enforces the format and scores the
`caption` field.

## Regenerating

```bash
pip install requests
python examples/image_captioning/build_dataset.py
```

Downloads annotations from `vizwiz.cs.colorado.edu` and fetches images on demand (no login required).
