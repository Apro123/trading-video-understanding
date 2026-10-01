import pytest
from fastapi.testclient import TestClient
from video_search_and_summarization.server import app

client = TestClient(app)

def test_ready_and_live_probes():
    res = client.get("/v1/ready")
    assert res.status_code == 200
    assert res.json()["status"] == "ready"

    res = client.get("/v1/live")
    assert res.status_code == 200
    assert res.json()["status"] == "live"

def test_healthz():
    res = client.get("/v1/healthz")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["model"] == "nvidia/Cosmos3-Nano"

def test_list_models():
    res = client.get("/models")
    assert res.status_code == 200
    data = res.json()
    assert "data" in data
    assert any(m["id"] == "nvidia/Cosmos3-Nano" for m in data["data"])
