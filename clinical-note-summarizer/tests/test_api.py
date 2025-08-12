import pytest
from fastapi.testclient import TestClient
from .context import get_app

app = get_app()


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json().get("status") == "ok"


def test_summarize_returns_text(client):
    payload = {"text": "Patient admitted for pneumonia. Treated with antibiotics and discharged."}
    resp = client.post("/summarize", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert "summary" in body
    assert isinstance(body["summary"], str)
    assert len(body["summary"].strip()) > 0


