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

    tokenizer = AutoTokenizer.from_pretrained(model_src, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_src)
    model.to(device)
    model.eval()

    app.state.tokenizer = tokenizer
    app.state.model = model
    app.state.device = device

    try:
        yield
    finally:
        # Free resources
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
        raise HTTPException(status_code=503, detail="Model not loaded")

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
