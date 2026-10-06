"""
Unit and integration tests for DBOS-inspired Ingestion & Analysis Orchestrator.
"""

import os
import time
import pytest
import tempfile
from fastapi.testclient import TestClient
from server import app
from orchestrator import (
    SQLiteWorkflowStore,
    IngestionAnalysisOrchestrator,
    WorkflowRecord,
    WorkflowStatus,
    StepStatus,
    WorkflowStepRecord
)

client = TestClient(app)

def test_sqlite_workflow_store_crud():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_workflows.sqlite3")
        store = SQLiteWorkflowStore(db_path=db_path)

        rec = WorkflowRecord(
            workflow_id="wf-test-123",
            video_id="clip_1",
            mode="async",
            status=WorkflowStatus.PENDING,
            current_step="Init",
            current_step_index=0,
            total_steps=6,
            progress_percent=0.0
        )
        store.save(rec)

        # Retrieve
        fetched = store.get("wf-test-123")
        assert fetched is not None
        assert fetched.workflow_id == "wf-test-123"
        assert fetched.video_id == "clip_1"
        assert fetched.status == WorkflowStatus.PENDING

        # Update
        fetched.status = WorkflowStatus.RUNNING
        fetched.progress_percent = 50.0
        fetched.current_step = "VLM Perception"
        store.save(fetched)

        updated = store.get("wf-test-123")
        assert updated.status == WorkflowStatus.RUNNING
        assert updated.progress_percent == 50.0
        assert updated.current_step == "VLM Perception"

        # List
        all_recs = store.list()
        assert len(all_recs) == 1
        assert all_recs[0].workflow_id == "wf-test-123"

def test_orchestrator_initialization_and_listeners():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_workflows.sqlite3")
        orch = IngestionAnalysisOrchestrator(db_path=db_path)

        rec = orch.create_workflow_record("clip_1", mode="sync")
        assert rec.total_steps == 6
        assert len(rec.steps) == 6
        assert rec.status == WorkflowStatus.PENDING

        events_received = []
        def listener(r):
            events_received.append(r.progress_percent)

        orch.subscribe(rec.workflow_id, listener)
        orch.update_step(rec, 1, StepStatus.COMPLETED, progress_percent=15.0, summary="Subtitles synced")

        assert len(events_received) == 1
        assert events_received[0] == 15.0

        orch.unsubscribe(rec.workflow_id, listener)
        orch.update_step(rec, 2, StepStatus.RUNNING, progress_percent=30.0)
        assert len(events_received) == 1  # Unsubscribed

def test_orchestrator_cancellation():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_workflows.sqlite3")
        orch = IngestionAnalysisOrchestrator(db_path=db_path)
        rec = orch.create_workflow_record("clip_1", mode="async")
        
        success = orch.cancel(rec.workflow_id)
        assert success is True
        
        cancelled = orch.store.get(rec.workflow_id)
        assert cancelled.status == WorkflowStatus.CANCELLED
        assert "cancelled" in cancelled.error.lower()

def test_api_videos_data_endpoint():
    res = client.get("/api/videos/data/clip_1")
    assert res.status_code == 200
    data = res.json()
    assert data["video_id"] == "clip_1"
    assert "metadata" in data
    assert "subtitles" in data
    assert len(data["subtitles"]) > 0
    assert "chunks" in data
    assert len(data["chunks"]) > 0
    assert "milestones" in data
    assert len(data["milestones"]) > 0
    assert "detected_price_levels" in data["chunks"][0]

def test_api_orchestration_run_sync():
    # Run sync on clip_1 with skip_model=True for quick verification
    res = client.post("/api/orchestrate/run", json={
        "video_id": "clip_1",
        "mode": "sync",
        "chunk_duration_s": 35.0,
        "chunk_overlap_s": 5.0,
        "skip_model": True,
        "skip_llm": True
    })
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "SUCCESS"
    assert data["progress_percent"] == 100.0
    assert data["current_step"] == "Execution Complete"
    assert "result" in data
    assert data["result"]["total_chunks"] > 0

def test_api_orchestration_run_async():
    # Run async on clip_2 with skip_model=True
    res = client.post("/api/orchestrate/run", json={
        "video_id": "clip_2",
        "mode": "async",
        "chunk_duration_s": 35.0,
        "chunk_overlap_s": 5.0,
        "skip_model": True,
        "skip_llm": True
    })
    assert res.status_code == 202
    data = res.json()
    assert data["mode"] == "async"
    assert "workflow_id" in data
    assert "poll_url" in data
    wf_id = data["workflow_id"]

    # Poll status endpoint
    poll_res = client.get(f"/api/orchestrate/jobs/{wf_id}")
    assert poll_res.status_code == 200
    poll_data = poll_res.json()
    assert poll_data["workflow_id"] == wf_id
    assert poll_data["total_steps"] == 6

    # Wait briefly for completion
    for _ in range(10):
        time.sleep(0.3)
        check = client.get(f"/api/orchestrate/jobs/{wf_id}").json()
        if check["status"] in ("SUCCESS", "FAILED"):
            break

    final = client.get(f"/api/orchestrate/jobs/{wf_id}").json()
    assert final["status"] in ("SUCCESS", "RUNNING")
