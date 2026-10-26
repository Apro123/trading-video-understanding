import pytest
from fastapi.testclient import TestClient
from server import app

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
    assert "Strategy Breakdown" in res.text
    assert "Semantic Video Search" in res.text
    assert "Trade Strategy Spec" not in res.text
    assert "Chronological Milestones" not in res.text

def test_api_videos_stream_range_and_head():
    # Test HEAD request
    head_res = client.head("/api/videos/stream/video_1")
    assert head_res.status_code == 200
    assert head_res.headers.get("accept-ranges") == "bytes"
    assert "content-length" in head_res.headers

    # Test GET with Range header (HTTP 206)
    get_res = client.get("/api/videos/stream/video_1", headers={"Range": "bytes=0-1024"})
    assert get_res.status_code == 206
    assert "bytes 0-1024/" in get_res.headers.get("content-range", "")
    assert len(get_res.content) == 1025

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

def test_api_video_strategies():
    res = client.get("/api/videos/strategies/video_2")
    assert res.status_code == 200
    data = res.json()
    assert data["video_id"] == "video_2"
    assert data["total_strategies"] >= 1
    assert len(data["strategies"]) == data["total_strategies"]
    s = data["strategies"][0]
    assert "name" in s
    assert "timestamp_range" in s
    assert "directional_bias" in s
    assert "core_thesis" in s
    assert "entry_trigger" in s
    assert "invalidation_stop" in s
    assert len(s["step_by_step_rules"]) > 0

    res2 = client.get("/api/videos/strategies/video_1")
    assert res2.status_code == 200
    data2 = res2.json()
    assert data2["video_id"] == "video_1"
    assert data2["total_strategies"] >= 1


def test_api_video_grounding():
    res = client.get("/api/videos/grounding/video_1")
    assert res.status_code == 200
    data = res.json()
    assert "video_id" in data
    assert "levels" in data


def test_api_video_frame_overlay():
    res = client.get("/api/videos/overlay/clip_1?t=5.0")
    assert res.status_code == 200
    assert res.headers["content-type"] == "image/png"
    assert len(res.content) > 100



