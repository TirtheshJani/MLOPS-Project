# 🏥 Clinical Note Summarizer — End-to-End MLOps Project

A production-grade HealthTech application that summarizes complex clinical notes into concise, patient-friendly summaries.

Built with PyTorch, FastAPI, Docker, GKE (Autopilot), and GitHub Actions CI/CD. Includes a React (Vite) frontend served by the same backend.

---

## Overview
Complete MLOps workflow: Data → Model → API → Docker → K8s → CI/CD → Monitoring.

## Key Features
- FLAN‑T5 summarization (HF Transformers)
- FastAPI inference service with validation
- React UI (Vite), single-image deploy (API + UI)
- GKE Autopilot deployment (Service + HPA optional)
- CI/CD (GitHub Actions): test → build → push → deploy
- Workload Identity Federation for secure GCP auth

## Architecture
- React/Vite SPA → FastAPI `/summarize` → FLAN‑T5 (HF) → Docker → GKE (LB) → CI/CD

## Tech Stack
ML: PyTorch, Transformers, Datasets

Backend: FastAPI, Pydantic, Uvicorn

Frontend: React (Vite), Axios, React Hook Form

DevOps: Docker, GKE Autopilot, Artifact Registry, GitHub Actions

## Dataset
Chosen: Microsoft MTS-Dialog (public, de-identified, Creative Commons).
Rationale: see `docs/dataset_rationale_mts_dialog.md`.

## Model Training (summary)
Preprocess (prefix "summarize: "), tokenize (2048/256), fine‑tune FLAN‑T5, evaluate ROUGE, export to `models/` (not in Git).

## Backend API
- `POST /summarize` → `{ summary: string }`
- Optional params: `max_new_tokens` (default 256), `temperature` (default 0.0)
- Guards: per‑IP rate limit, max input size, consistent errors `{error:{code,message}}`

## Frontend UI
- One‑page app: textarea input, generation controls, examples, compare mode, copy/download (.txt/.md)
- Disclaimer & PHI warning visible
- Served from `web/dist` by FastAPI

## Containerization & Deployment
- Multi‑stage Dockerfile; final image serves API + UI
- Image in Artifact Registry
- K8s manifests: `kubernetes/deployment.yaml`, `service.yaml` (LB), optional `hpa.yaml`

## CI/CD Pipeline
- CI (`.github/workflows/ci.yaml`): install → build web → pytest
- CD (`.github/workflows/cd.yaml`): build web → docker build/push (SHA, latest) → `kubectl set image` → rollout status

## Local Development
### Quickstart (Windows PowerShell)

Prereqs: Python 3.11, Node.js 18+, Git, Docker Desktop (optional for container).

- Backend only (FastAPI):
```
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r clinical-note-summarizer/requirements.txt

# Optional: point to a local fine-tuned model (if you have one)
$env:MODEL_DIR = "models\\flan-t5-bhc-summarizer"
$env:USE_FAST_TOKENIZER = "1"

uvicorn clinical-note-summarizer.app.main:app --reload --host 0.0.0.0 --port 8000
```
Test:
```
Invoke-RestMethod -Method Post `
  -Uri http://localhost:8000/summarize `
  -ContentType 'application/json' `
  -Body (@{ text = 'Patient presents with...' } | ConvertTo-Json)
```

- Frontend dev (Vite):
```
cd web
$env:VITE_API_BASE_URL = "http://localhost:8000"
npm ci
npm run dev
```
Open `http://localhost:5173` (Vite) calling the API at `http://localhost:8000`.

Note: If PowerShell blocks script activation, run: `Set-ExecutionPolicy -Scope Process Bypass`.

### Docker (single image: API + built UI)

1) Build frontend assets so `web/dist/` exists:
```
cd web
npm ci
# Empty base URL means relative calls to the same host/port as the API
$env:VITE_API_BASE_URL = ""
npm run build
cd ..
```

2) Build the image:
```
docker build -t clinical-summarizer-app .
```

3) Run the container:
```
docker run --rm -p 8000:8000 clinical-summarizer-app
```
Open `http://localhost:8000`.

Optional: use a local fine-tuned model without baking it into the image (Windows PowerShell path example):
```
docker run --rm -p 8000:8000 `
  -e MODEL_DIR="/app/models/flan-t5-bhc-summarizer" `
  -v "${PWD}\models:/app/models" `
  clinical-summarizer-app
```
If no local model is provided, the service falls back to `google/flan-t5-base` on first request (downloads on demand).

## Production Deployment (GKE)
New image (example):
```
docker tag clinical-summarizer-app \
  northamerica-northeast2-docker.pkg.dev/<PROJECT_ID>/clinical-summarizer-repo/clinical-summarizer-app:latest
docker push northamerica-northeast2-docker.pkg.dev/<PROJECT_ID>/clinical-summarizer-repo/clinical-summarizer-app:latest

kubectl set image deployment/clinical-summarizer-deployment \
  clinical-summarizer-app=northamerica-northeast2-docker.pkg.dev/<PROJECT_ID>/clinical-summarizer-repo/clinical-summarizer-app:latest
kubectl rollout status deployment/clinical-summarizer-deployment
```

## Disclaimer
Informational demo only; not a medical device. Do not paste PHI or real patient identifiers.

