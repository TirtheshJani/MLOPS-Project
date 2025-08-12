from fastapi.testclient import TestClient
try:
    # Prefer package import when running locally or in CI with correct PYTHONPATH
    from app.main import app  # type: ignore
except ModuleNotFoundError:
    # Fallback to dynamic sys.path adjustment
    from .context import get_app
    app = get_app()


client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json().get("status") == "ok"


def test_summarize_returns_text():
    payload = {"text": "Patient admitted for pneumonia. Treated with antibiotics and discharged."}
    resp = client.post("/summarize", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert "summary" in body
    assert isinstance(body["summary"], str)
    assert len(body["summary"].strip()) > 0


