"""
DBOS-Inspired Durable Workflow Orchestrator for Trading Video Ingestion & Analysis.
Provides both Synchronous and Asynchronous execution modes with durable checkpointing
backed by SQLite (zero external infrastructure required) and optional DBOS Python integration.

Reference: https://github.com/dbos-inc/dbos-transact-py
"""

import os
import time
import uuid
import json
import sqlite3
import logging
import threading
from enum import Enum
from typing import Dict, Any, Optional, List, Callable
from pydantic import BaseModel, Field

try:
    from dbos import DBOS
    HAS_DBOS = True
except ImportError:
    HAS_DBOS = False

logger = logging.getLogger(__name__)

# Workflow and Step Status Enums
class WorkflowStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

class StepStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"

class WorkflowStepRecord(BaseModel):
    step_index: int
    name: str
    status: StepStatus = StepStatus.PENDING
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    duration_s: float = 0.0
    progress_percent: float = 0.0
    summary: str = ""
    error: Optional[str] = None

class WorkflowRecord(BaseModel):
    workflow_id: str
    video_id: str
    mode: str = "sync"  # "sync" or "async"
    status: WorkflowStatus = WorkflowStatus.PENDING
    current_step: str = ""
    current_step_index: int = 0
    total_steps: int = 6
    progress_percent: float = 0.0
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    completed_at: Optional[float] = None
    duration_s: float = 0.0
    steps: List[WorkflowStepRecord] = Field(default_factory=list)
    params: Dict[str, Any] = Field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# Default Pipeline Step Definitions
DEFAULT_PIPELINE_STEPS = [
    {"index": 1, "name": "Subtitle Ingestion & 0:00 Normalization"},
    {"index": 2, "name": "Video Chunking & Keyframe Sampling"},
    {"index": 3, "name": "VLM Perception & Hybrid Visual Grounding"},
    {"index": 4, "name": "Multimodal Vector & Keyword Search Indexing"},
    {"index": 5, "name": "Long Video Summary & Milestone Formulation"},
    {"index": 6, "name": "Algorithmic Strategy Specification Synthesis"},
]

class SQLiteWorkflowStore:
    """
    Lightweight, ACID-compliant durable storage for workflow executions.
    Mirrors DBOS durable execution patterns using SQLite WAL mode.
    """

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            db_path = os.path.join(base_dir, "workflows.sqlite3")
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=15.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self):
        with self._lock:
            with self._get_conn() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS workflows (
                        workflow_id TEXT PRIMARY KEY,
                        video_id TEXT NOT NULL,
                        mode TEXT NOT NULL,
                        status TEXT NOT NULL,
                        progress_percent REAL NOT NULL,
                        current_step TEXT NOT NULL,
                        current_step_index INTEGER NOT NULL,
                        total_steps INTEGER NOT NULL,
                        data_json TEXT NOT NULL,
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL,
                        completed_at REAL
                    );
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_workflows_video 
                    ON workflows(video_id, created_at DESC);
                """)
                conn.commit()

    def save(self, record: WorkflowRecord) -> None:
        with self._lock:
            record.updated_at = time.time()
            if record.completed_at and record.created_at:
                record.duration_s = round(record.completed_at - record.created_at, 2)
            elif record.created_at:
                record.duration_s = round(time.time() - record.created_at, 2)

            data_json = record.model_dump_json()
            with self._get_conn() as conn:
                conn.execute("""
                    INSERT INTO workflows (
                        workflow_id, video_id, mode, status, progress_percent,
                        current_step, current_step_index, total_steps, data_json,
                        created_at, updated_at, completed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(workflow_id) DO UPDATE SET
                        status=excluded.status,
                        progress_percent=excluded.progress_percent,
                        current_step=excluded.current_step,
                        current_step_index=excluded.current_step_index,
                        data_json=excluded.data_json,
                        updated_at=excluded.updated_at,
                        completed_at=excluded.completed_at;
                """, (
                    record.workflow_id,
                    record.video_id,
                    record.mode,
                    record.status.value,
                    record.progress_percent,
                    record.current_step,
                    record.current_step_index,
                    record.total_steps,
                    data_json,
                    record.created_at,
                    record.updated_at,
                    record.completed_at
                ))
                conn.commit()

    def get(self, workflow_id: str) -> Optional[WorkflowRecord]:
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT data_json FROM workflows WHERE workflow_id = ?",
                (workflow_id,)
            ).fetchone()
            if row:
                data = json.loads(row["data_json"])
                return WorkflowRecord(**data)
        return None

    def list(self, video_id: Optional[str] = None, limit: int = 20) -> List[WorkflowRecord]:
        with self._get_conn() as conn:
            if video_id:
                rows = conn.execute(
                    "SELECT data_json FROM workflows WHERE video_id = ? ORDER BY created_at DESC LIMIT ?",
                    (video_id, limit)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT data_json FROM workflows ORDER BY created_at DESC LIMIT ?",
                    (limit,)
                ).fetchall()

            records = []
            for r in rows:
                try:
                    records.append(WorkflowRecord(**json.loads(r["data_json"])))
                except Exception as e:
                    logger.warning(f"Error parsing workflow record: {e}")
            return records


class IngestionAnalysisOrchestrator:
    """
    Orchestration coordinator providing synchronous and asynchronous execution
    of the trading video ingestion, VLM visual grounding, and strategy synthesis pipeline.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.store = SQLiteWorkflowStore(db_path=db_path)
        self._cancellation_events: Dict[str, threading.Event] = {}
        self._listeners: Dict[str, List[Callable[[WorkflowRecord], None]]] = {}
        self._listeners_lock = threading.Lock()

    def create_workflow_record(
        self,
        video_id: str,
        mode: str = "sync",
        params: Optional[Dict[str, Any]] = None
    ) -> WorkflowRecord:
        """Initializes a new workflow record with pre-configured step metadata."""
        workflow_id = str(uuid.uuid4())
        steps = [
            WorkflowStepRecord(
                step_index=item["index"],
                name=item["name"],
                status=StepStatus.PENDING
            )
            for item in DEFAULT_PIPELINE_STEPS
        ]
        record = WorkflowRecord(
            workflow_id=workflow_id,
            video_id=video_id,
            mode=mode,
            status=WorkflowStatus.PENDING,
            current_step="Initialized",
            current_step_index=0,
            total_steps=len(steps),
            progress_percent=0.0,
            steps=steps,
            params=params or {}
        )
        self.store.save(record)
        return record

    def subscribe(self, workflow_id: str, callback: Callable[[WorkflowRecord], None]):
        """Subscribes a listener to receive real-time workflow status updates."""
        with self._listeners_lock:
            if workflow_id not in self._listeners:
                self._listeners[workflow_id] = []
            self._listeners[workflow_id].append(callback)

    def unsubscribe(self, workflow_id: str, callback: Callable[[WorkflowRecord], None]):
        """Unsubscribes a listener from workflow updates."""
        with self._listeners_lock:
            if workflow_id in self._listeners and callback in self._listeners[workflow_id]:
                self._listeners[workflow_id].remove(callback)

    def _notify_listeners(self, record: WorkflowRecord):
        """Dispatches record updates to registered event listeners."""
        with self._listeners_lock:
            callbacks = list(self._listeners.get(record.workflow_id, []))
        for cb in callbacks:
            try:
                cb(record)
            except Exception as e:
                logger.warning(f"Error in workflow listener: {e}")

    def update_step(
        self,
        record: WorkflowRecord,
        step_index: int,
        status: StepStatus,
        progress_percent: Optional[float] = None,
        summary: str = "",
        error: Optional[str] = None
    ) -> None:
        """Checkpoints the progress of an individual workflow step into durable storage."""
        now = time.time()
        for step in record.steps:
            if step.step_index == step_index:
                step.status = status
                if status == StepStatus.RUNNING and not step.started_at:
                    step.started_at = now
                elif status in (StepStatus.COMPLETED, StepStatus.FAILED):
                    step.completed_at = now
                    if step.started_at:
                        step.duration_s = round(now - step.started_at, 2)
                if summary:
                    step.summary = summary
                if error:
                    step.error = error

        record.current_step_index = step_index
        step_obj = next((s for s in record.steps if s.step_index == step_index), None)
        if step_obj:
            record.current_step = step_obj.name

        if progress_percent is not None:
            record.progress_percent = min(100.0, max(0.0, round(progress_percent, 1)))

        self.store.save(record)
        self._notify_listeners(record)

    def cancel(self, workflow_id: str) -> bool:
        """Signals cancellation to an active workflow execution."""
        if workflow_id in self._cancellation_events:
            self._cancellation_events[workflow_id].set()

        record = self.store.get(workflow_id)
        if record and record.status in (WorkflowStatus.PENDING, WorkflowStatus.RUNNING):
            record.status = WorkflowStatus.CANCELLED
            record.completed_at = time.time()
            record.error = "Workflow cancelled by user request"
            self.store.save(record)
            self._notify_listeners(record)
            return True
        return False

    def is_cancelled(self, workflow_id: str) -> bool:
        event = self._cancellation_events.get(workflow_id)
        return bool(event and event.is_set())

    def _execute_pipeline_core(
        self,
        record: WorkflowRecord,
        cancel_event: threading.Event
    ) -> Dict[str, Any]:
        """
        Executes the 6-stage VSS pipeline while updating durable step checkpoints.
        """
        try:
            from pipeline import run_vss_pipeline
            from downloader import get_video_file_path, get_subtitle_file_path
            from config import DEFAULT_COSMOS3_PATH, DEFAULT_LLM_BASE_URL, DEFAULT_LLM_MODEL
        except ImportError:
            from .pipeline import run_vss_pipeline
            from .downloader import get_video_file_path, get_subtitle_file_path
            from .config import DEFAULT_COSMOS3_PATH, DEFAULT_LLM_BASE_URL, DEFAULT_LLM_MODEL

        params = record.params
        video_id = record.video_id

        # Resolve paths
        video_path = params.get("video_path") or get_video_file_path(video_id)
        subtitles_path = params.get("subtitles_path") or get_subtitle_file_path(video_id)
        output_dir = params.get("output_dir") or os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"outputs_{video_id}_grounded" if "clip" in video_id else f"outputs_{video_id}"
        )

        if not video_path or not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found for video_id='{video_id}' (path={video_path})")
        if not subtitles_path or not os.path.exists(subtitles_path):
            raise FileNotFoundError(f"Subtitles file not found for video_id='{video_id}' (path={subtitles_path})")

        # Progress tracking callback passed to pipeline
        def on_pipeline_progress(
            step_idx: int,
            step_name: str,
            pct: float,
            details: Optional[Dict[str, Any]] = None
        ):
            if cancel_event.is_set():
                raise RuntimeError(f"Workflow {record.workflow_id} was cancelled")

            summary_text = ""
            if details:
                if "subtitles_count" in details:
                    summary_text = f"Normalized {details['subtitles_count']} subtitles anchored to 0:00"
                elif "chunks_count" in details:
                    summary_text = f"Sliced timeline into {details['chunks_count']} chunks"
                elif "chunk_id" in details:
                    summary_text = f"Processed chunk #{details['chunk_id']}"
                elif "milestones_count" in details:
                    summary_text = f"Formulated {details['milestones_count']} milestone events"
                elif "artifacts" in details:
                    summary_text = f"Exported {len(details['artifacts'])} artifacts"

            status = StepStatus.RUNNING if pct < 100.0 else StepStatus.COMPLETED
            self.update_step(
                record=record,
                step_index=step_idx,
                status=status,
                progress_percent=pct,
                summary=summary_text
            )

        # Mark Step 1 as Running
        self.update_step(record, 1, StepStatus.RUNNING, progress_percent=5.0, summary="Starting subtitle sync")

        result = run_vss_pipeline(
            video_path=video_path,
            subtitles_path=subtitles_path,
            output_dir=output_dir,
            model_path=params.get("model_path", DEFAULT_COSMOS3_PATH),
            skip_model=params.get("skip_model", False),
            chunk_duration_s=params.get("chunk_duration_s", 35.0),
            chunk_overlap_s=params.get("chunk_overlap_s", 5.0),
            frames_per_chunk=params.get("frames_per_chunk", 4),
            device=params.get("device", "cuda"),
            llm_base_url=params.get("llm_base_url", DEFAULT_LLM_BASE_URL),
            llm_model=params.get("llm_model", DEFAULT_LLM_MODEL),
            skip_llm=params.get("skip_llm", False),
            progress_callback=on_pipeline_progress
        )

        # Mark all steps completed
        now = time.time()
        for step in record.steps:
            if step.status != StepStatus.COMPLETED:
                step.status = StepStatus.COMPLETED
                step.completed_at = now

        result_dict = {
            "video_path": result.video_path,
            "subtitles_path": result.subtitles_path,
            "duration_s": result.duration_s,
            "total_chunks": result.total_chunks,
            "generated_files": result.generated_files,
            "hardware_metrics": result.hardware_metrics,
            "output_dir": output_dir,
            "milestones_count": len(result.summary_response.key_events) if result.summary_response else 0
        }
        return result_dict

    def run_sync(
        self,
        video_id: str,
        params: Optional[Dict[str, Any]] = None
    ) -> WorkflowRecord:
        """
        Executes the workflow synchronously, blocking the caller until completion
        while updating durable step records in SQLite.
        """
        record = self.create_workflow_record(video_id, mode="sync", params=params)
        cancel_event = threading.Event()
        self._cancellation_events[record.workflow_id] = cancel_event

        record.status = WorkflowStatus.RUNNING
        self.store.save(record)
        self._notify_listeners(record)

        try:
            result_dict = self._execute_pipeline_core(record, cancel_event)
            record.status = WorkflowStatus.SUCCESS
            record.progress_percent = 100.0
            record.current_step = "Execution Complete"
            record.result = result_dict
            record.completed_at = time.time()
            self.store.save(record)
            self._notify_listeners(record)
        except Exception as e:
            record.status = WorkflowStatus.FAILED if not cancel_event.is_set() else WorkflowStatus.CANCELLED
            record.error = str(e)
            record.completed_at = time.time()
            self.store.save(record)
            self._notify_listeners(record)
            logger.error(f"Synchronous workflow {record.workflow_id} failed: {e}", exc_info=True)
            raise
        finally:
            self._cancellation_events.pop(record.workflow_id, None)

        return record

    def run_async(
        self,
        video_id: str,
        params: Optional[Dict[str, Any]] = None
    ) -> WorkflowRecord:
        """
        Initiates the workflow asynchronously in a background worker thread.
        Immediately returns the initial WorkflowRecord with status=RUNNING and workflow_id.
        """
        record = self.create_workflow_record(video_id, mode="async", params=params)
        cancel_event = threading.Event()
        self._cancellation_events[record.workflow_id] = cancel_event

        record.status = WorkflowStatus.RUNNING
        record.current_step = "Starting background execution"
        record.progress_percent = 2.0
        self.store.save(record)
        self._notify_listeners(record)

        def _worker():
            try:
                res = self._execute_pipeline_core(record, cancel_event)
                record.status = WorkflowStatus.SUCCESS
                record.progress_percent = 100.0
                record.current_step = "Execution Complete"
                record.result = res
                record.completed_at = time.time()
                self.store.save(record)
                self._notify_listeners(record)
            except Exception as e:
                is_cancelled = cancel_event.is_set()
                record.status = WorkflowStatus.CANCELLED if is_cancelled else WorkflowStatus.FAILED
                record.error = "Cancelled by user" if is_cancelled else str(e)
                record.completed_at = time.time()
                self.store.save(record)
                self._notify_listeners(record)
                logger.error(f"Asynchronous workflow {record.workflow_id} error: {e}", exc_info=True)
            finally:
                self._cancellation_events.pop(record.workflow_id, None)

        thread = threading.Thread(target=_worker, name=f"workflow-{record.workflow_id[:8]}", daemon=True)
        thread.start()

        return record

# Global Default Orchestrator Instance
global_orchestrator = IngestionAnalysisOrchestrator()
