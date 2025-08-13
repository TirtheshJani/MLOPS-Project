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
Backend only:
```
uvicorn clinical-note-summarizer.app.main:app --reload --host 0.0.0.0 --port 8000
```

Frontend dev:
```
cd web
echo VITE_API_BASE_URL=http://localhost:8000 > .env
npm ci
npm run dev
```

Single container (UI + API):
```
cd web && npm run build && cd ..
docker build -t clinical-summarizer-app .
docker run --rm -p 8000:8000 clinical-summarizer-app
```
Open http://localhost:8000

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

