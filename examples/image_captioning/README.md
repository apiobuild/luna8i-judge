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
| `golden_dataset.jsonl` | State-of-the-art model outputs used as the quality ceiling for candidate scoring |
| `output_schema.json` | JSON Schema for the expected `{"caption": "..."}` response format |
| `images/` | Downloaded image files (≤ 1 MB each) |
| `build_dataset.py` | Curation script — re-run to regenerate |

## Input format

Each row in `input.jsonl` is a self-contained `messages` payload:

```json
{
  "row_index": 0,
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

## Prerequisites

Install `luna8i-judge` and add your Gemini API key:

```bash
pip install luna8i-judge
export GEMINI_API_KEY=...
```

## Running golden dataset generation

***Note: A pre-generated `golden_dataset.jsonl` is committed to this directory so you can skip directly to inference if you prefer.***

Generate a sample `golden_dataset.jsonl` on the first 5 rows to verify everything looks right:

```bash
luna8i-judge job create \
  --file input.jsonl \
  --output-json-schema output_schema.json \
  --prompt-template "Describe this image in one to two sentences\. If the image quality is too poor to make out any content, respond with \{\\\"caption\\\": \\\"unanswerable\\\"\}\." \
  --sota-model gemini/gemini-3.1-flash-lite \
  --output ./ \
  --limit 5 \
  --run
```

After the job finishes, check the job details to confirm workload detection:

```bash
luna8i-judge job get <job-id>
```

Expected output:

```json
{
  "detected_workload_details": {
    "workload_type": "captioning",
    "modality": "image",
    "confidence": "high",
    "confidence_note": "The prompt explicitly asks to describe an image, which aligns directly with the captioning category."
  }
}
```

Once you're happy with the sample, generate the full `golden_dataset.jsonl` (this step can take a few minutes for all 50 rows):

```bash
luna8i-judge job create \
  --file input.jsonl \
  --output-json-schema output_schema.json \
  --prompt-template "Describe this image in one to two sentences\. If the image quality is too poor to make out any content, respond with \{\\\"caption\\\": \\\"unanswerable\\\"\}\." \
  --sota-model gemini/gemini-3.1-flash-lite \
  --output ./ \
  --run
```

## Running inference against benchmarking models

### Running inference with (Local) Ollama

***Note: Pre-generated `ollama__llava.jsonl` and `ollama__minicpm-v.jsonl` are committed to this directory so you can skip running inference if you prefer.***

Run local vision-capable models via [Ollama](https://ollama.com). Start Ollama before running:

```bash
luna8i-judge models ollama start
```

These two models run well on a MacBook Pro (Apple Silicon or Intel) with 18 GB unified memory:

| Model string | Description | Disk / RAM |
|---|---|---|
| `ollama/llava` | LLaVA 7B — the standard vision baseline | ~5 GB |
| `ollama/minicpm-v` | MiniCPM-V 2.6 — best quality of the three | ~6 GB |

> **Note:** The input rows reference raw GitHub URLs. Make sure the images are pushed to `main` before running, or Ollama will fail to fetch them.

### Model 1: ollama/llava

Pull each model, append its results to the same job, then unload before the next.

```bash
# LLaVA 7B
luna8i-judge models ollama pull llava
luna8i-judge job run <job-id> \
  --step running_inference \
  --golden-dataset-path golden_dataset.jsonl \
  --compare-models '[{"model": "ollama/llava"}]' \
  --force
luna8i-judge models ollama unload llava

# MiniCPM-V 2.6
luna8i-judge models ollama pull minicpm-v
luna8i-judge job run <job-id> \ a8c21611-fa34-43c4-9590-22a8240e91b2
  --step running_inference \
  --golden-dataset-path golden_dataset.jsonl \
  --compare-models '[{"model": "ollama/minicpm-v"}]' \
  --force
luna8i-judge models ollama unload minicpm-v
```

`--force` appends the new model's results to the existing job without regenerating the golden dataset.
