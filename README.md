![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)

# luna8i-judge

Benchmark tool for LLM workloads. Upload a prompt and sample inputs, get a report comparing cost and quality across candidate models vs. a high-tier SOTA baseline.

---

## Quick start

### Option 1 — Docker (full app + UI)

```bash
# 1. Copy the example env file and fill in your API keys
curl -O https://raw.githubusercontent.com/apiobuild/luna8i-judge/main/.env.example
cp .env.example .env
# edit .env

# 2. Run
docker run \
  --env-file .env \
  -p 8080:8080 \
  -v "$PWD/data:/data" \
  apiobuild/luna8i-judge:latest
```

Opens at [http://localhost:8080](http://localhost:8080).

### Option 2 — CLI only

```bash
pip install luna8i-judge
luna8i-judge job create --help
```

No Docker or Node.js required.

---

## API key setup

**Docker users:** open the app and go to **LLM Providers** in the sidebar to add or update keys through the UI. No env vars required.

**CLI users** must supply keys via env vars (no UI available):

```
GEMINI_API_KEY=       # Google Gemini
OPENAI_API_KEY=       # OpenAI
ANTHROPIC_API_KEY=    # Anthropic
TOGETHER_API_KEY=     # Together AI
FIREWORKS_API_KEY=    # Fireworks AI
DASHSCOPE_API_KEY=    # Alibaba Cloud (Qwen)
```

See [`.env.example`](.env.example) for all options including local GPU (vLLM / Ollama) settings.

---

## Example datasets

Ready-to-run datasets are in [`examples/`](examples/):

| Folder | Workload | Modality |
|--------|----------|----------|
| `examples/classification/` | Sentiment — Movie Reviews | text |
| `examples/extraction/` | Receipt Field Extraction | text |
| `examples/summarization/` | News Summarization | text |
| `examples/image_captioning/` | Image Captioning | text + image |
| `examples/contract_extraction/` | Contract Clause Extraction (CUAD) | text |

---

## License

MIT
