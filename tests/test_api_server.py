import pytest
from fastapi.testclient import TestClient
try:
    from server import app
except ImportError:
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

def test_ui_root_endpoint():
    res = client.get("/")
    assert res.status_code == 200
    assert "NVIDIA Video Search & Summarization" in res.text
    assert "Semantic Video Search" in res.text

def test_api_videos_list():
    res = client.get("/api/videos")
    assert res.status_code == 200
    data = res.json()
    assert isinstance(data, list)
    assert len(data) >= 2
    video_ids = [v["video_id"] for v in data]
    assert "video_1" in video_ids
    assert "video_2" in video_ids

def test_api_video_status():
    res = client.get("/api/videos/status/video_1")
    assert res.status_code == 200
    data = res.json()
    assert data["video_id"] == "video_1"
    assert data["is_downloaded"] is True
    assert data["has_index"] is True

def test_api_semantic_search():
    res = client.post("/api/search", json={
        "video_id": "video_1",
        "query": "breakout",
        "top_k": 3
    })
    assert res.status_code == 200
    data = res.json()
    assert data["total_chunks"] > 0
    assert len(data["results"]) > 0
    assert "timestamp_range" in data["results"][0]

