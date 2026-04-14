"""End-to-end tests for the summarization API.

CI runs these against the ``hf-internal-testing/tiny-random-t5`` model for
speed. They assert shape, status codes, and error-handling semantics
rather than summary quality.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from .context import get_app

app = get_app()


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Module-scoped TestClient so the model is loaded only once."""
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Health / readiness
# ---------------------------------------------------------------------------


def test_health(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "model_loaded" in body


def test_ready(client: TestClient) -> None:
    resp = client.get("/ready")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] in {"ready", "degraded"}
    assert isinstance(body["model_loaded"], bool)


def test_openapi_schema(client: TestClient) -> None:
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    schema = resp.json()
    assert "/summarize" in schema["paths"]
    assert "/health" in schema["paths"]


# ---------------------------------------------------------------------------
# Summarize: happy path
# ---------------------------------------------------------------------------


def test_summarize_returns_text(client: TestClient) -> None:
    payload = {"text": "Patient admitted for pneumonia. Treated with antibiotics and discharged."}
    resp = client.post("/summarize", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert "summary" in body
    assert isinstance(body["summary"], str)
    assert len(body["summary"].strip()) > 0


def test_summarize_respects_max_new_tokens(client: TestClient) -> None:
    payload = {
        "text": "76-year-old with HTN and OA presents for med check. BP stable.",
        "max_new_tokens": 32,
    }
    resp = client.post("/summarize", json=payload)
    assert resp.status_code == 200
    assert "summary" in resp.json()


def test_summarize_with_sampling(client: TestClient) -> None:
    payload = {
        "text": "Patient with chronic back pain on NSAIDs. Follow-up in 2 weeks.",
        "temperature": 0.7,
        "max_new_tokens": 64,
    }
    resp = client.post("/summarize", json=payload)
    assert resp.status_code == 200
    assert isinstance(resp.json().get("summary"), str)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_summarize_empty_text_rejected(client: TestClient) -> None:
    resp = client.post("/summarize", json={"text": ""})
    assert resp.status_code == 422


def test_summarize_missing_text_rejected(client: TestClient) -> None:
    resp = client.post("/summarize", json={})
    assert resp.status_code == 422


def test_summarize_invalid_temperature_rejected(client: TestClient) -> None:
    resp = client.post(
        "/summarize",
        json={"text": "hello world", "temperature": 5.0},
    )
    assert resp.status_code == 422


def test_summarize_invalid_max_tokens_rejected(client: TestClient) -> None:
    resp = client.post(
        "/summarize",
        json={"text": "hello world", "max_new_tokens": 99999},
    )
    assert resp.status_code == 422


def test_summarize_oversize_input_rejected(client: TestClient) -> None:
    # Exceed MAX_INPUT_CHARS (default 10_000)
    payload = {"text": "x" * 10_001}
    resp = client.post("/summarize", json=payload)
    assert resp.status_code == 413
    body = resp.json()
    assert body["error"]["code"] == 413
    assert "Input too large" in body["error"]["message"]


# ---------------------------------------------------------------------------
# Error envelope
# ---------------------------------------------------------------------------


def test_error_envelope_shape(client: TestClient) -> None:
    """413 errors should return the documented ``{"error": {...}}`` shape."""
    resp = client.post("/summarize", json={"text": "x" * 10_001})
    assert resp.status_code == 413
    body = resp.json()
    assert set(body["error"].keys()) == {"code", "message"}
