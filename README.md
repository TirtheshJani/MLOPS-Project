# Clinical Note Summarizer — MLOps on GKE

[![CI](https://github.com/TirtheshJani/MLOPS-Project/actions/workflows/ci.yaml/badge.svg)](https://github.com/TirtheshJani/MLOPS-Project/actions/workflows/ci.yaml)
[![CD](https://github.com/TirtheshJani/MLOPS-Project/actions/workflows/cd.yaml/badge.svg)](https://github.com/TirtheshJani/MLOPS-Project/actions/workflows/cd.yaml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-261230.svg)](https://github.com/astral-sh/ruff)

> **End-to-end MLOps for healthcare NLP.** A FLAN-T5 clinical-note summarization
> service with a FastAPI backend, React frontend, Docker image, Kubernetes
> manifests, and GitHub Actions pipelines for GKE Autopilot.

> ⚠️ **Disclaimer:** Informational demo only. Not a medical device and not
> intended for use with real PHI.

---

## Table of Contents

- [Highlights](#highlights)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Quick Start](#quick-start)
- [API](#api)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Training Pipeline](#training-pipeline)
- [Deployment](#deployment)
- [CI/CD](#cicd)
- [Development](#development)
- [Dataset](#dataset)
- [Roadmap](#roadmap)
- [License](#license)

---

## Highlights

- **Production-style FastAPI service** with health/readiness probes, Pydantic
  validation, per-IP rate limiting, and a consistent JSON error envelope.
- **Graceful model fallback** — the service falls back to a stub summarizer
  if the model cannot be loaded so probes and the UI still work.
- **Multi-stage Dockerfile** with a non-root runtime user.
- **Kubernetes manifests** (Deployment, Service, HPA) tuned for GKE Autopilot.
- **GitHub Actions CI/CD** — lint, test, build frontend, push image to
  Artifact Registry via Workload Identity Federation, and roll out to GKE.
- **Reproducible training pipeline** for fine-tuning FLAN-T5 on MTS-Dialog.
- **Typed Python, ruff linting, pre-commit hooks, and coverage reports.**

## Architecture

```
 ┌──────────────┐     ┌──────────────┐     ┌───────────────────┐
 │  React (Vite)│────▶│   FastAPI    │────▶│  FLAN-T5 (HF)     │
 │    UI        │◀────│   Backend    │◀────│  Transformers     │
 └──────────────┘     └──────────────┘     └───────────────────┘
         │                  │
         └──── served from ─┘
                  │
                  ▼
         ┌────────────────┐
         │ Docker image   │  (multi-stage; API + built SPA)
         └────────────────┘
                  │
                  ▼
         ┌────────────────────────────────┐
         │ GKE Autopilot                  │
         │ Deployment · Service · HPA     │
         └────────────────────────────────┘
                  ▲
                  │
         ┌────────────────────────────────┐
         │ GitHub Actions                 │
         │ lint → test → build → deploy   │
         └────────────────────────────────┘
```

## Tech Stack

| Layer | Tools |
| --- | --- |
| Modeling | PyTorch · Transformers · FLAN-T5 · datasets · evaluate (ROUGE) |
| Backend  | FastAPI · Pydantic v2 · Uvicorn |
| Frontend | React 19 · Vite 7 |
| Infra    | Docker · Kubernetes · GKE Autopilot · Artifact Registry |
| CI/CD    | GitHub Actions · Workload Identity Federation |
| Quality  | pytest · pytest-cov · ruff · mypy · pre-commit |

## Quick Start

### Prerequisites
- Python **3.11+**
- Node.js **20+** (for the frontend)
- Docker (optional, for containerised runs)

### 1. Clone

```bash
git clone https://github.com/TirtheshJani/MLOPS-Project.git
cd MLOPS-Project
```

### 2. Backend

```bash
python -m venv .venv
source .venv/bin/activate                 # Windows: .\.venv\Scripts\Activate.ps1
pip install -r clinical-note-summarizer/requirements.txt

# Run against the public google/flan-t5-base checkpoint (default)
PYTHONPATH=clinical-note-summarizer \
  uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Smoke test:

```bash
curl -s -X POST http://localhost:8000/summarize \
  -H 'Content-Type: application/json' \
  -d '{"text": "Patient admitted for pneumonia. Treated and discharged."}'
```

### 3. Frontend

```bash
cd web
npm ci
npm run dev          # http://localhost:5173
```

### 4. Standalone Streamlit demo (optional)

For a zero-infra local walkthrough without FastAPI/React:

```bash
pip install -r demo_requirements.txt
streamlit run demo_app.py
```

### 5. Docker

```bash
# Build the SPA first so the image ships with static assets
( cd web && npm ci && VITE_API_BASE_URL="" npm run build )

docker build -t clinical-summarizer-app .
docker run --rm -p 8000:8000 clinical-summarizer-app
```

## API

OpenAPI docs are available at `/docs` (Swagger UI) and `/redoc`.

| Method | Path         | Description                                   |
| ------ | ------------ | --------------------------------------------- |
| GET    | `/health`    | Liveness probe (`status`, `model_loaded`)      |
| GET    | `/ready`     | Readiness probe                                |
| POST   | `/summarize` | Generate summary from a clinical note          |

### `POST /summarize`

```jsonc
// Request
{
  "text": "Patient presents with chest pain ...",
  "max_new_tokens": 256,        // 1..1024 (default 256)
  "temperature": 0.0            // 0.0..1.0 (default 0 → beam search)
}

// Response
{ "summary": "..." }

// Error envelope (all HTTPException responses)
{ "error": { "code": 413, "message": "Input too large. Max 10,000 characters." } }
```

## Configuration

All configuration is via environment variables.

| Variable | Default | Description |
| --- | --- | --- |
| `MODEL_DIR` | `models/flan-t5-bhc-summarizer` or `google/flan-t5-base` | Local path or HF repo id to load |
| `USE_FAST_TOKENIZER` | `""` | Set `true` to prefer fast tokenizers (useful in CI) |
| `CORS_ORIGINS` | `http://localhost:3000,http://localhost:5173` | Comma-separated allow-list |
| `RATE_LIMIT_MAX` | `30` | Max requests per window per IP |
| `RATE_LIMIT_WINDOW_SEC` | `60` | Window size in seconds |
| `MAX_INPUT_CHARS` | `10000` | Hard cap on `/summarize` input size |
| `FRONTEND_DIST` | `web/dist` | Location of the built SPA to serve |

## Project Structure

```
MLOPS-Project/
├── .github/workflows/          # CI, CD, auth-check pipelines
├── clinical-note-summarizer/
│   ├── app/main.py             # FastAPI service
│   ├── scripts/                # Training + preprocessing
│   ├── tests/                  # pytest suite
│   └── requirements.txt
├── web/                        # React + Vite frontend
├── kubernetes/                 # Deployment, Service, HPA, SA
├── scripts/                    # Dataset download + EDA
├── notebooks/                  # EDA notebooks
├── docs/                       # Dataset rationale + notes
├── Dockerfile                  # Multi-stage production image
├── pyproject.toml              # Ruff / pytest / mypy config
├── .pre-commit-config.yaml
└── LICENSE
```

## Training Pipeline

```
preprocess → tokenize (2048/256) → fine-tune FLAN-T5 → ROUGE eval → export
```

```bash
python clinical-note-summarizer/scripts/preprocess_t5.py \
  --input data/primary/mts-dialog \
  --output data/processed

python clinical-note-summarizer/scripts/train.py \
  --model google/flan-t5-base \
  --data data/processed \
  --output-dir models/flan-t5-bhc-summarizer
```

## Deployment

Apply manifests to your GKE cluster:

```bash
kubectl apply -f kubernetes/
kubectl rollout status deployment/clinical-summarizer-deployment
```

Update the running image (also automated by `cd.yaml` on every push to `main`):

```bash
kubectl set image deployment/clinical-summarizer-deployment \
  clinical-summarizer-app=<AR_HOST>/<PROJECT>/<REPO>/clinical-summarizer-app:<sha>
```

## CI/CD

`.github/workflows/ci.yaml` runs on every push/PR:

1. **Lint** — `ruff check` + `ruff format --check`.
2. **Backend tests** — `pytest` with a tiny HF model for speed, coverage enabled.
3. **Frontend** — `npm ci`, `npm run lint`, `npm run build`.

`.github/workflows/cd.yaml` on push to `main`:

1. Build the SPA.
2. Authenticate to GCP via **Workload Identity Federation** (no JSON keys).
3. Build and push the Docker image to Artifact Registry.
4. `kubectl set image` + `kubectl rollout status`.

## Development

```bash
# Install dev dependencies (ruff, mypy, pytest, pre-commit)
pip install -r clinical-note-summarizer/requirements.txt
pip install pre-commit

# Enable pre-commit hooks
pre-commit install

# Run the full check locally
ruff check .
ruff format --check .
MODEL_DIR=hf-internal-testing/tiny-random-t5 USE_FAST_TOKENIZER=true \
  PYTHONPATH=clinical-note-summarizer pytest -q
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for a fuller workflow.

## Dataset

**Microsoft MTS-Dialog** — public, de-identified clinician–patient dialogues
released under Creative Commons. See
[`docs/dataset_rationale_mts_dialog.md`](docs/dataset_rationale_mts_dialog.md)
for selection rationale and
[`docs/eda_summary_mts_dialog.md`](docs/eda_summary_mts_dialog.md) for EDA
highlights.

## Roadmap

- [ ] Structured JSON logs + request ids
- [ ] Prometheus `/metrics` endpoint and Grafana dashboard
- [ ] Batch inference endpoint
- [ ] Canary deploys via Argo Rollouts
- [ ] Distributed rate limiting backed by Redis

## License

MIT — see [LICENSE](LICENSE).

Contact: [LinkedIn](https://www.linkedin.com/in/tirthesh-jani) ·
[GitHub](https://github.com/TirtheshJani)
