from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Dict, List
import time
import logging

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


DEFAULT_MODEL_DIR = Path("models/flan-t5-bhc-summarizer")
FALLBACK_MODEL_NAME = "google/flan-t5-base"


class InputData(BaseModel):
    text: str = Field(..., min_length=1, description="Full clinical note to summarize")
    max_new_tokens: Optional[int] = 256
    temperature: Optional[float] = 0.0


class PredictionResponse(BaseModel):
    summary: str


def _resolve_model_source() -> str:
    env_path = os.getenv("MODEL_DIR")
    if env_path:
        return env_path
    if (DEFAULT_MODEL_DIR / "config.json").exists():
        return str(DEFAULT_MODEL_DIR)
    return FALLBACK_MODEL_NAME


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_src = _resolve_model_source()
    device = _get_device()

    app.state.tokenizer = None
    app.state.model = None
    app.state.device = device
    app.state.stub_mode = False

    # Prefer fast tokenizers in CI/tiny model scenarios to avoid sentencepiece issues
    use_fast_env = os.getenv("USE_FAST_TOKENIZER", "").lower() in {"1", "true", "yes"}
    try:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_src, use_fast=use_fast_env)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(model_src, use_fast=not use_fast_env)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_src)
        model.to(device)
        model.eval()

        app.state.tokenizer = tokenizer
        app.state.model = model
    except Exception:
        # As a last resort, enable stub mode so the service can start and respond
        app.state.stub_mode = True

    try:
        yield
    finally:
        app.state.model = None
        app.state.tokenizer = None


app = FastAPI(title="BHC Summarization Service", version="0.1.0", lifespan=lifespan)

# CORS for frontend/browser usage (tighten allow_origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Basic in-memory rate limiter (demo only; not shared across replicas)
RATE_LIMIT_MAX = int(os.getenv("RATE_LIMIT_MAX", "30"))  # requests
RATE_LIMIT_WINDOW_SEC = int(os.getenv("RATE_LIMIT_WINDOW_SEC", "60"))
app.state.rate_buckets: Dict[str, List[float]] = {}

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("summarizer")

# Serve built frontend if available (web/dist)
FRONTEND_DIST = Path("web/dist")
if FRONTEND_DIST.exists():
    assets_dir = FRONTEND_DIST / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/")
    def index() -> FileResponse:
        index_file = FRONTEND_DIST / "index.html"
        return FileResponse(str(index_file))


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/summarize", response_model=PredictionResponse)
def summarize(payload: InputData, request: Request) -> PredictionResponse:
    # Simple per-IP rate limit
    now = time.time()
    ip = request.client.host if request.client else "unknown"
    bucket = app.state.rate_buckets.get(ip, [])
    # prune
    bucket = [t for t in bucket if now - t < RATE_LIMIT_WINDOW_SEC]
    if len(bucket) >= RATE_LIMIT_MAX:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please try again later.")
    bucket.append(now)
    app.state.rate_buckets[ip] = bucket

    tokenizer: AutoTokenizer = app.state.tokenizer
    model: AutoModelForSeq2SeqLM = app.state.model
    device: torch.device = app.state.device
    if tokenizer is None or model is None:
        # Stub summarization: return a trimmed version of the input
        text = payload.text.strip()
        words = text.split()
        trimmed = " ".join(words[:50])
        return PredictionResponse(summary=trimmed if trimmed else text)

    text = payload.text.strip()
    # Enforce max input size (defense in depth in addition to Pydantic max)
    if len(text) > 10000:
        raise HTTPException(status_code=413, detail="Input too large. Max 10,000 characters.")

    prompt = f"summarize: {text}".strip()

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    max_new = payload.max_new_tokens or 256
    temp = payload.temperature or 0.0
    do_sample = temp > 0.0

    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new,
            num_beams=(1 if do_sample else 4),
            do_sample=do_sample,
            temperature=(temp if do_sample else None),
            length_penalty=1.0,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )

    summary_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    latency_ms = int((time.perf_counter() - start) * 1000)
    logger.info(
        "summary_generated",
        extra={
            "ip": ip,
            "input_chars": len(text),
            "summary_chars": len(summary_text),
            "latency_ms": latency_ms,
            "do_sample": do_sample,
            "temperature": temp,
            "max_new_tokens": max_new,
        },
    )
    return PredictionResponse(summary=summary_text)


# Consistent error response shape
from fastapi.responses import JSONResponse
from starlette.requests import Request as StarletteRequest


@app.exception_handler(HTTPException)
def http_exc_handler(_: StarletteRequest, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"error": {"code": exc.status_code, "message": exc.detail}})


# Optional: serve full SPA (static + catch-all) if built assets exist
if FRONTEND_DIST.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIST), html=False), name="static")

    @app.get("/{full_path:path}")
    def spa_catch_all(full_path: str) -> FileResponse:
        index_file = FRONTEND_DIST / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
        raise HTTPException(status_code=404, detail="Not found")
