# 🏥 MLOps Project: Clinical Note Summarizer

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-326CE5?style=for-the-badge&logo=kubernetes&logoColor=white)](https://kubernetes.io/)
[![Google Cloud](https://img.shields.io/badge/Google_Cloud-4285F4?style=for-the-badge&logo=google-cloud&logoColor=white)](https://cloud.google.com/)
[![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)](https://reactjs.org/)

> **End-to-End MLOps for Healthcare NLP**  
> A production-grade HealthTech application that summarizes complex clinical notes into concise, patient-friendly summaries using FLAN-T5 and modern MLOps practices.

---

## 📊 Project Overview

This project demonstrates a complete **MLOps workflow** for deploying a clinical note summarization service. It bridges the gap between ML research and production deployment, featuring:

- 🤖 **FLAN-T5** fine-tuned for medical summarization
- ⚡ **FastAPI** backend with async inference
- 🎨 **React frontend** for intuitive user experience
- 🐳 **Docker containerization** for consistency
- ☁️ **GKE Autopilot deployment** for scalability
- 🔄 **CI/CD with GitHub Actions** for automation

### Complete ML Pipeline
```
Data → Model Training → API → Docker → Kubernetes → CI/CD → Monitoring
```

---

## 🛠️ Tech Stack

### Machine Learning
| Tool | Purpose |
|------|---------|
| PyTorch | Deep learning framework |
| Transformers | Hugging Face model library |
| Datasets | Data processing |

### Backend
| Tool | Purpose |
|------|---------|
| FastAPI | High-performance API framework |
| Pydantic | Data validation |
| Uvicorn | ASGI server |

### Frontend
| Tool | Purpose |
|------|---------|
| React (Vite) | Modern UI framework |
| Axios | HTTP client |
| React Hook Form | Form management |

### DevOps
| Tool | Purpose |
|------|---------|
| Docker | Containerization |
| GKE Autopilot | Managed Kubernetes |
| GitHub Actions | CI/CD automation |
| Artifact Registry | Container storage |

---

## 🏗️ Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────────┐
│   React UI  │────▶│  FastAPI    │────▶│  FLAN-T5 Model  │
│   (Vite)    │◀────│   Backend   │◀────│  (HF Transform) │
└─────────────┘     └─────────────┘     └─────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│                  Docker Container                        │
│         (Single image: API + Built UI)                  │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│              GKE Autopilot Cluster                       │
│         (Load Balancer + HPA + Rollouts)                │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│              GitHub Actions CI/CD                        │
│    (Test → Build → Push → Deploy → Verify)              │
└─────────────────────────────────────────────────────────┘
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- Node.js 18+
- Docker Desktop
- Google Cloud SDK (for deployment)
- kubectl

### Quick Start

#### 1. Clone and Setup
```bash
git clone https://github.com/TirtheshJani/MLOPS-Project.git
cd MLOPS-Project
```

#### 2. Backend Only (Development)
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r clinical-note-summarizer/requirements.txt

# Run development server
uvicorn clinical-note-summarizer.app.main:app --reload --host 0.0.0.0 --port 8000
```

Test the API:
```bash
curl -X POST http://localhost:8000/summarize \
  -H "Content-Type: application/json" \
  -d '{"text": "Patient presents with chest pain..."}'
```

#### 3. Frontend Development
```bash
cd web
npm ci
npm run dev
```

#### 4. Docker Build
```bash
# Build frontend assets
cd web
npm ci
VITE_API_BASE_URL="" npm run build
cd ..

# Build Docker image
docker build -t clinical-summarizer-app .

# Run container
docker run --rm -p 8000:8000 clinical-summarizer-app
```

---

## 📁 Repository Structure

```
MLOPS-Project/
├── .github/
│   └── workflows/
│       ├── ci.yaml              # Continuous Integration
│       └── cd.yaml              # Continuous Deployment
├── clinical-note-summarizer/
│   ├── app/
│   │   └── main.py              # FastAPI application
│   ├── models/                  # Model artifacts (gitignored)
│   └── requirements.txt
├── web/                         # React frontend
│   ├── src/
│   ├── dist/                    # Build output
│   └── package.json
├── kubernetes/
│   ├── deployment.yaml
│   ├── service.yaml
│   └── hpa.yaml
├── notebooks/                   # Training notebooks
├── scripts/                     # Utility scripts
├── docs/                        # Documentation
├── Dockerfile
├── README.md
└── LICENSE
```

---

## 💡 Key Features

### API Capabilities
- `POST /summarize` - Generate clinical summaries
- Rate limiting per IP
- Configurable generation parameters
- Consistent error handling

### Frontend Features
- Clean, intuitive interface
- Example clinical notes
- Compare mode (input vs. output)
- Export results (txt, md)
- PHI warnings

### Production Features
- Health check endpoints
- Horizontal Pod Autoscaling
- Rolling deployments
- Workload Identity Federation

---

## 📊 Dataset

**Microsoft MTS-Dialog Dataset**
- Public, de-identified medical dialogues
- Creative Commons license
- Suitable for clinical summarization training

See `docs/dataset_rationale_mts_dialog.md` for selection rationale.

---

## 🔧 Model Training

### Training Pipeline
1. **Preprocessing**: Add "summarize: " prefix
2. **Tokenization**: 2048 input / 256 output tokens
3. **Fine-tuning**: FLAN-T5 on medical dialogues
4. **Evaluation**: ROUGE metrics
5. **Export**: Save to `models/`

### Training Command
```bash
python scripts/train.py \
  --model google/flan-t5-base \
  --dataset mts-dialog \
  --output-dir models/flan-t5-bhc-summarizer
```

---

## 🚢 Deployment

### GKE Deployment
```bash
# Tag and push image
docker tag clinical-summarizer-app \
  gcr.io/PROJECT_ID/clinical-summarizer-app:latest
docker push gcr.io/PROJECT_ID/clinical-summarizer-app:latest

# Update deployment
kubectl set image deployment/clinical-summarizer-deployment \
  clinical-summarizer-app=gcr.io/PROJECT_ID/clinical-summarizer-app:latest

# Verify rollout
kubectl rollout status deployment/clinical-summarizer-deployment
```

---

## 🔄 CI/CD Pipeline

### Continuous Integration
```yaml
# .github/workflows/ci.yaml
1. Checkout code
2. Setup Python
3. Install dependencies
4. Build frontend
5. Run pytest
```

### Continuous Deployment
```yaml
# .github/workflows/cd.yaml
1. Build frontend
2. Docker build & push
3. Update GKE deployment
4. Verify rollout status
```

---

## 📈 Monitoring

### Health Endpoints
- `GET /health` - Service health check
- `GET /ready` - Readiness probe
- Metrics via Prometheus (optional)

### Logging
- Structured JSON logging
- Cloud Logging integration
- Request tracing

---

## 🔒 Security

- **No PHI in demos** - Synthetic data only
- **Rate limiting** - Prevent abuse
- **Input validation** - Pydantic models
- **GCP Workload Identity** - Secure authentication

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Disclaimer**: This is an informational demo only; not a medical device. Do not process real PHI.

---

## 📧 Contact

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/tirthesh-jani)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/TirtheshJani)

---

<p align="center">
  <i>Production-grade MLOps for healthcare AI 🏥🤖</i>
</p>
