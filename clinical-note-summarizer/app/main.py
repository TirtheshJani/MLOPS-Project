from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


DEFAULT_MODEL_DIR = Path("models/flan-t5-bhc-summarizer")
FALLBACK_MODEL_NAME = "google/flan-t5-base"


class InputData(BaseModel):
    text: str = Field(..., min_length=1, description="Full clinical note to summarize")


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


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/summarize", response_model=PredictionResponse)
def summarize(payload: InputData) -> PredictionResponse:
    tokenizer: AutoTokenizer = app.state.tokenizer
    model: AutoModelForSeq2SeqLM = app.state.model
    device: torch.device = app.state.device
    if tokenizer is None or model is None:
        # Stub summarization: return a trimmed version of the input
        text = payload.text.strip()
        words = text.split()
        trimmed = " ".join(words[:50])
        return PredictionResponse(summary=trimmed if trimmed else text)

    prompt = f"summarize: {payload.text}".strip()

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            num_beams=4,
            length_penalty=1.0,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )

    summary_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return PredictionResponse(summary=summary_text)
