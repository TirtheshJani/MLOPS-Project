"""Unit tests for module-level helpers that don't require a running app."""

from __future__ import annotations

import importlib
import os
from pathlib import Path

import pytest

from .context import ensure_path

ensure_path()

import app.main as main_module  # noqa: E402  (path setup must happen first)


@pytest.fixture(autouse=True)
def _restore_env():
    """Save and restore environment variables touched by these tests."""
    keys = ["MODEL_DIR", "CORS_ORIGINS"]
    saved = {k: os.environ.get(k) for k in keys}
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def test_resolve_model_source_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_DIR", "some/custom-model")
    importlib.reload(main_module)
    assert main_module._resolve_model_source() == "some/custom-model"


def test_resolve_model_source_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("MODEL_DIR", raising=False)
    # Point DEFAULT_MODEL_DIR at a non-existent path for deterministic fallback.
    monkeypatch.setattr(main_module, "DEFAULT_MODEL_DIR", tmp_path / "nowhere")
    assert main_module._resolve_model_source() == main_module.FALLBACK_MODEL_NAME


def test_parse_cors_origins_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CORS_ORIGINS", raising=False)
    origins = main_module._parse_cors_origins()
    assert "http://localhost:3000" in origins
    assert "http://localhost:5173" in origins


def test_parse_cors_origins_custom(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CORS_ORIGINS", "https://a.example.com, https://b.example.com")
    origins = main_module._parse_cors_origins()
    assert origins == ["https://a.example.com", "https://b.example.com"]


def test_get_device_returns_torch_device() -> None:
    import torch

    device = main_module._get_device()
    assert isinstance(device, torch.device)
    assert device.type in {"cpu", "cuda", "mps"}
