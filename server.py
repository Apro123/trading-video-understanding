"""
VSS REST API & Web UI Microservice.
Implements the NVIDIA Video Search and Summarization (VSS) REST API specification:
https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization

Endpoints:
- GET /: Interactive Web UI for Video video selection, playback sync, and semantic search
- GET /v1/ready: Readiness probe (HTTP 200)
- GET /v1/live: Liveness probe (HTTP 200)
- GET /v1/healthz: Health check with GPU VRAM profiling
- GET /models: List available VSS foundation models
- POST /v1/summarize: Execute long video summarization on a video clip + subtitles
- POST /v1/search: Execute semantic vector and keyword search across indexed video chunks
- POST /v1/trade_idea: Synthesize formal algorithmic trade idea specification

Interactive Web UI Endpoints:
- GET /api/videos: List available Video trading videos and their statuses
- POST /api/videos/download: Download video and subtitles via yt-dlp (highest quality video-only non-premium)
- POST /api/videos/analyze: Run VSS indexing and strategy synthesis
- GET /api/videos/status/{video_id}: Retrieve video status, trade spec, and summary
- POST /api/search: Execute semantic search on selected video
- GET /api/videos/stream/{video_id}: Stream video file with HTTP Range support for seeking
- GET /api/videos/subtitles/{video_id}: Stream WebVTT subtitles for HTML5 video track
"""

import os
import re
import time
import json
import logging
import asyncio
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException, status, BackgroundTasks, Request, Response
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse, StreamingResponse
from pydantic import BaseModel

try:
    from .config import DEFAULT_COSMOS3_PATH, DEFAULT_OUTPUT_DIR, VSSVLMConfig
    from .schemas import (
        SummarizationQuery,
        CompletionResponse,
        SearchQuery,
        SearchResponse,
        TradeIdeaSpec,
        MultiStrategyBreakdown,
        StrategySegment
    )
    from .pipeline import run_vss_pipeline
    from .search_engine import VSSSearchIndex
    from .vlm_reasoner import VSSVLMReasoner
    from .subtitle_sync import SubtitleSyncManager, seconds_to_time_str
    from .chunk_engine import VideoChunkManager
    from .summarizer import VSSLVSSummarizer
    from .trade_strategy_agent import VSSTradeStrategyAgent
    from .strategy_breakdown import MultiStrategyBreakdownEngine
    from .downloader import (
        list_available_videos,
        download_video,
        get_video_status,
        get_video_file_path,
        get_subtitle_file_path,
        KNOWN_VIDEOS
    )
    from .utils import safe_load_json
    from .orchestrator import (
        global_orchestrator,
        WorkflowRecord,
        WorkflowStatus,
        StepStatus
    )
except (ImportError, ValueError):
    from config import DEFAULT_COSMOS3_PATH, DEFAULT_OUTPUT_DIR, VSSVLMConfig
    from schemas import (
        SummarizationQuery,
        CompletionResponse,
        SearchQuery,
        SearchResponse,
        TradeIdeaSpec,
        MultiStrategyBreakdown,
        StrategySegment
    )
    from pipeline import run_vss_pipeline
    from search_engine import VSSSearchIndex
    from vlm_reasoner import VSSVLMReasoner
    from subtitle_sync import SubtitleSyncManager, seconds_to_time_str
    from chunk_engine import VideoChunkManager
    from summarizer import VSSLVSSummarizer
    from trade_strategy_agent import VSSTradeStrategyAgent
    from strategy_breakdown import MultiStrategyBreakdownEngine
    from downloader import (
        list_available_videos,
        download_video,
        get_video_status,
        get_video_file_path,
        get_subtitle_file_path,
        KNOWN_VIDEOS
    )
    from utils import safe_load_json
    from orchestrator import (
        global_orchestrator,
        WorkflowRecord,
        WorkflowStatus,
        StepStatus
    )

from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

# In-memory search index cache per video_id
video_search_indices: Dict[str, VSSSearchIndex] = {}
active_search_index: Optional[VSSSearchIndex] = None
active_reasoner: Optional[VSSVLMReasoner] = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

class VideoSearchRequest(BaseModel):
    video_id: str
    query: str
    top_k: int = 10
    min_score: float = 0.0

class VideoActionRequest(BaseModel):
    video_id: str

class OrchestrationRequest(BaseModel):
    video_id: str
    mode: str = "async"  # "sync" or "async"
    chunk_duration_s: float = 35.0
    chunk_overlap_s: float = 5.0
    skip_model: bool = False
    skip_llm: bool = False

def get_or_load_search_index(video_id: str) -> Optional[VSSSearchIndex]:
    """Retrieves cached in-memory search index or loads from outputs_{video_id}/vss_search_index.json."""
    if video_id in video_search_indices:
        return video_search_indices[video_id]

    # Check video-specific output dir or default
    candidate_paths = [
        os.path.join(BASE_DIR, f"outputs_vss_{video_id}_grounded", "vss_search_index.json"),
        os.path.join(BASE_DIR, f"outputs_vss_{video_id}", "vss_search_index.json"),
        os.path.join(BASE_DIR, f"outputs_{video_id}", "vss_search_index.json"),
        os.path.join(DEFAULT_OUTPUT_DIR, "vss_search_index.json")
    ]

    for p in candidate_paths:
        if os.path.exists(p):
            try:
                idx = VSSSearchIndex.from_file(p)
                video_search_indices[video_id] = idx
                return idx
            except Exception as e:
                logger.warning(f"Error loading search index from {p}: {e}")

    return None

def analyze_and_index_video(video_id: str) -> Dict[str, Any]:
    """Runs chunking, search indexing, summarization, and trade idea synthesis on a video."""
    vpath = get_video_file_path(video_id)
    spath = get_subtitle_file_path(video_id)

    if not vpath or not os.path.exists(vpath):
        raise FileNotFoundError(f"Video file not found for {video_id}")
    if not spath or not os.path.exists(spath):
        raise FileNotFoundError(f"Subtitle file not found for {video_id}")

    out_dir = os.path.join(BASE_DIR, f"outputs_{video_id}")
    os.makedirs(out_dir, exist_ok=True)

    # 1. Chunking & Subtitle Synchronization
    sub_mgr = SubtitleSyncManager(spath)
    chunk_mgr = VideoChunkManager(vpath, subtitle_mgr=sub_mgr, chunk_duration_s=35.0, chunk_overlap_s=5.0)
    chunks = chunk_mgr.generate_chunks()

    # 2. Build & Persist Search Index
    search_idx = VSSSearchIndex(video_id=video_id)
    search_idx.index_chunks(chunks, captions=[])
    idx_data = search_idx.export_index_json()
    with open(os.path.join(out_dir, "vss_search_index.json"), "w", encoding="utf-8") as f:
        json.dump(idx_data, f, indent=2)
    video_search_indices[video_id] = search_idx

    # 3. Summarization
    summarizer = VSSLVSSummarizer()
    comp = summarizer.summarize(chunks, captions=[])
    with open(os.path.join(out_dir, "trading_summary.json"), "w", encoding="utf-8") as f:
        json.dump(comp.model_dump(), f, indent=2)

    # 4. Trade Strategy Synthesis
    agent = VSSTradeStrategyAgent()
    spec = agent.synthesize_trade_idea(comp, chunks=chunks)
    with open(os.path.join(out_dir, "trading_strategy_spec.json"), "w", encoding="utf-8") as f:
        json.dump(spec.model_dump(), f, indent=2)
    with open(os.path.join(out_dir, "trading_strategy_spec.md"), "w", encoding="utf-8") as f:
        f.write(agent.export_spec_markdown(spec))

    # 5. Multi-Strategy Breakdown Across Video Intervals
    breakdown_engine = MultiStrategyBreakdownEngine()
    breakdown = breakdown_engine.breakdown_video_strategies(video_id)

    return {
        "status": "success",
        "video_id": video_id,
        "chunks_indexed": len(chunks),
        "total_strategies": breakdown.total_strategies,
        "output_dir": out_dir
    }

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes search indices and preloads models."""
    global active_reasoner
    vlm_config = VSSVLMConfig(
        model_path=os.environ.get("COSMOS3_MODEL_PATH", DEFAULT_COSMOS3_PATH),
        skip_model=os.environ.get("SKIP_MODEL", "false").lower() in ("true", "1")
    )
    active_reasoner = VSSVLMReasoner(vlm_config)
    try:
        active_reasoner.load_model()
    except Exception as e:
        logger.info(f"[VSS Server] Model preloading note: {e}")

    # Pre-cache known search indices
    for vid in KNOWN_VIDEOS:
        get_or_load_search_index(vid)
    yield

app = FastAPI(
    title="NVIDIA Video Search and Summarization (VSS) Trading Service",
    description="GPU-accelerated reference microservice for trading video understanding, semantic search, and strategy synthesis.",
    version="1.0.0",
    lifespan=lifespan
)

# ==========================================
# Web UI Endpoints
# ==========================================

@app.get("/", response_class=HTMLResponse, tags=["Web UI"])
def get_ui():
    """Serves the interactive single-page trading video intelligence dashboard."""
    index_path = os.path.join(TEMPLATES_DIR, "index.html")
    if not os.path.exists(index_path):
        return HTMLResponse("<h2>UI template not found</h2>", status_code=404)
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.get("/api/videos", tags=["Video Management"])
def list_videos():
    """Returns available Video trading videos and their download/indexing status."""
    return list_available_videos()

@app.post("/api/videos/download", tags=["Video Management"])
def download_video_endpoint(req: VideoActionRequest):
    """Downloads a Video video and subtitles using yt-dlp according to non-premium rules."""
    try:
        status_info = download_video(req.video_id)
        return status_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/videos/analyze", tags=["Video Management"])
def analyze_video_endpoint(req: VideoActionRequest, background_tasks: BackgroundTasks):
    """Runs VSS chunking, semantic indexing, summarization, and strategy synthesis."""
    try:
        res = analyze_and_index_video(req.video_id)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/videos/status/{video_id}", tags=["Video Management"])
def get_video_status_endpoint(video_id: str):
    """Retrieves full status, trading strategy spec, and video summary for a video."""
    meta = get_video_status(video_id)
    out_dir = meta.get("output_dir", os.path.join(BASE_DIR, f"outputs_{video_id}"))

    return {
        **meta,
        "trade_spec": safe_load_json(os.path.join(out_dir, "trading_strategy_spec.json")),
        "summary": safe_load_json(os.path.join(out_dir, "trading_summary.json")),
        "strategies": safe_load_json(os.path.join(out_dir, "multi_strategy_breakdown.json"))
    }

@app.get("/api/videos/data/{video_id}", tags=["Video Management"])
def get_video_unified_data_endpoint(video_id: str):
    """
    Returns unified playback and timeline synchronization data:
    - Subtitle entries with millisecond intervals for live teleprompter sync
    - Grounded chunks with bounding boxes, calibrated price levels, and deictic anchors
    - Chronological milestone events
    - Algorithmic trade strategy spec & multi-strategy breakdown
    """
    meta = get_video_status(video_id)
    out_dir = meta.get("output_dir", os.path.join(BASE_DIR, f"outputs_{video_id}"))

    # 1. Subtitles
    spath = get_subtitle_file_path(video_id)
    subtitles = []
    if spath and os.path.exists(spath):
        try:
            sub_mgr = SubtitleSyncManager(spath)
            subtitles = [
                {
                    "index": e.index,
                    "start_s": round(e.clip_timestamp_s, 2),
                    "end_s": round(e.clip_end_s, 2),
                    "start_str": e.clip_timestamp_str,
                    "end_str": seconds_to_time_str(e.clip_end_s),
                    "text": e.text
                }
                for e in sub_mgr.entries
            ]
        except Exception as e:
            logger.warning(f"Error loading subtitles for {video_id}: {e}")

    # 2. Summary & Milestones
    summary_data = safe_load_json(os.path.join(out_dir, "trading_summary.json"), default={})
    milestones = summary_data.get("key_events", [])
    chunk_captions = summary_data.get("chunk_captions", [])

    # 3. Grounded Chunks
    chunks_data = []
    for c in chunk_captions:
        grounding = c.get("grounding") or c.get("visual_observations", {}).get("grounded_coordinates") or {}
        chunks_data.append({
            "chunk_id": c.get("chunk_id", 0),
            "start_s": c.get("start_s", 0.0),
            "end_s": c.get("end_s", 0.0),
            "start_time": c.get("start_time", "00:00"),
            "end_time": c.get("end_time", "00:00"),
            "content": c.get("content", ""),
            "signals": c.get("detected_trade_signals", []),
            "detected_price_levels": grounding.get("detected_price_levels", []),
            "deictic_anchors": grounding.get("deictic_anchors", []),
            "chart_regions": grounding.get("chart_regions", []),
            "price_range": grounding.get("price_range", [100.0, 200.0])
        })

    # 4. Strategy Specs
    trade_spec = safe_load_json(os.path.join(out_dir, "trading_strategy_spec.json"), default={})
    strategies = safe_load_json(os.path.join(out_dir, "multi_strategy_breakdown.json"), default={})

    return {
        "video_id": video_id,
        "metadata": meta,
        "subtitles": subtitles,
        "chunks": chunks_data,
        "milestones": milestones,
        "summary": summary_data,
        "trade_spec": trade_spec,
        "strategies": strategies
    }

@app.get("/api/videos/strategies/{video_id}", response_model=MultiStrategyBreakdown, tags=["Trading Strategy"])
def get_video_strategies_endpoint(video_id: str):
    """Retrieves all discrete trading strategies identified across video intervals."""
    meta = get_video_status(video_id)
    out_dir = meta.get("output_dir", os.path.join(BASE_DIR, f"outputs_{video_id}"))
    strat_path = os.path.join(out_dir, "multi_strategy_breakdown.json")
    cached = safe_load_json(strat_path)
    if cached:
        try:
            return MultiStrategyBreakdown(**cached)
        except Exception as e:
            logger.warning(f"Error parsing cached strategies: {e}")

    engine = MultiStrategyBreakdownEngine()
    try:
        breakdown = engine.breakdown_video_strategies(video_id)
        return breakdown
    except Exception as e:
        logger.error(f"Failed to generate strategy breakdown for {video_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/search", response_model=SearchResponse, tags=["Semantic Search"])
def search_video_api(req: VideoSearchRequest):
    """Executes semantic and keyword search across indexed chunks of the selected video."""
    index = get_or_load_search_index(req.video_id)
    if index is None:
        raise HTTPException(
            status_code=404,
            detail=f"No search index found for video '{req.video_id}'. Please run analysis first."
        )

    results = index.search(req.query, top_k=req.top_k, min_score=req.min_score)
    return results

@app.api_route("/api/videos/stream/{video_id}", methods=["GET", "HEAD"], tags=["Video Streaming"])
def stream_video(video_id: str):
    """
    High-performance video streaming endpoint with HTTP 206 Partial Content Range and HEAD support.
    Leverages faststart-optimized MP4 files for instant browser seeking and near-zero CPU usage.
    """
    vpath = get_video_file_path(video_id)
    if not vpath or not os.path.exists(vpath):
        raise HTTPException(status_code=404, detail="Video file not found")
    return FileResponse(vpath, media_type="video/mp4")

def format_vtt_timestamp(seconds: float) -> str:
    secs_int = int(seconds)
    millis = int((seconds - secs_int) * 1000)
    h = secs_int // 3600
    m = (secs_int % 3600) // 60
    s = secs_int % 60
    return f"{h:02d}:{m:02d}:{s:02d}.{millis:03d}"

@app.get("/api/videos/subtitles/{video_id}", tags=["Video Streaming"])
def stream_subtitles(video_id: str):
    """Serves the WebVTT subtitle track for browser closed captioning."""
    spath = get_subtitle_file_path(video_id)
    if not spath or not os.path.exists(spath):
        raise HTTPException(status_code=404, detail="Subtitles not found")

    if spath.endswith(".vtt"):
        return FileResponse(spath, media_type="text/vtt")

    # Dynamically convert raw subtitle formats into valid WebVTT on the fly
    try:
        sub_mgr = SubtitleSyncManager(spath)
        lines = ["WEBVTT\n"]
        for idx, e in enumerate(sub_mgr.entries):
            st = format_vtt_timestamp(e.clip_timestamp_s)
            en = format_vtt_timestamp(e.clip_end_s)
            lines.append(f"{idx+1}\n{st} --> {en}\n{e.text}\n")
        return Response(content="\n".join(lines), media_type="text/vtt")
    except Exception as e:
        logger.warning(f"Error converting subtitles to WebVTT for {video_id}: {e}")
        return FileResponse(spath, media_type="text/plain")

# ==========================================
# Ingestion & Analysis Orchestration (DBOS)
# ==========================================

@app.post("/api/orchestrate/run", tags=["Orchestration"])
def run_orchestration_endpoint(req: OrchestrationRequest):
    """
    Executes the ingestion, VLM visual grounding, and strategy synthesis pipeline
    in either Synchronous or Asynchronous mode with durable SQLite checkpointing.
    """
    params = {
        "chunk_duration_s": req.chunk_duration_s,
        "chunk_overlap_s": req.chunk_overlap_s,
        "skip_model": req.skip_model,
        "skip_llm": req.skip_llm,
    }

    if req.mode == "sync":
        try:
            record = global_orchestrator.run_sync(req.video_id, params=params)
            # Invalidate search index cache so freshly analyzed video index is reloaded
            video_search_indices.pop(req.video_id, None)
            return record.model_dump()
        except Exception as e:
            logger.error(f"Synchronous orchestration failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    else:
        record = global_orchestrator.run_async(req.video_id, params=params)
        video_search_indices.pop(req.video_id, None)
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={
                **record.model_dump(),
                "poll_url": f"/api/orchestrate/jobs/{record.workflow_id}",
                "stream_url": f"/api/orchestrate/jobs/{record.workflow_id}/stream"
            }
        )

@app.get("/api/orchestrate/jobs/{workflow_id}", tags=["Orchestration"])
def get_job_status_endpoint(workflow_id: str):
    """Retrieves real-time status and durable step history for a workflow job."""
    record = global_orchestrator.store.get(workflow_id)
    if not record:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return record.model_dump()

@app.get("/api/orchestrate/jobs", tags=["Orchestration"])
def list_jobs_endpoint(video_id: Optional[str] = None, limit: int = 20):
    """Lists recent workflow orchestration jobs from durable SQLite storage."""
    records = global_orchestrator.store.list(video_id=video_id, limit=limit)
    return [r.model_dump() for r in records]

@app.post("/api/orchestrate/jobs/{workflow_id}/cancel", tags=["Orchestration"])
def cancel_job_endpoint(workflow_id: str):
    """Cancels a running workflow job."""
    success = global_orchestrator.cancel(workflow_id)
    if not success:
        raise HTTPException(status_code=400, detail="Unable to cancel workflow (may be already finished or not found)")
    return {"status": "cancelled", "workflow_id": workflow_id}

@app.get("/api/orchestrate/jobs/{workflow_id}/stream", tags=["Orchestration"])
async def stream_workflow_progress_endpoint(workflow_id: str, request: Request):
    """
    Server-Sent Events (SSE) stream pushing live step-by-step progress
    directly to the frontend UI as video frames and chunks are analyzed.
    """
    record = global_orchestrator.store.get(workflow_id)
    if not record:
        raise HTTPException(status_code=404, detail="Workflow not found")

    async def event_generator():
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def listener(rec: WorkflowRecord):
            loop.call_soon_threadsafe(queue.put_nowait, rec)

        global_orchestrator.subscribe(workflow_id, listener)
        try:
            init_rec = global_orchestrator.store.get(workflow_id)
            if init_rec:
                yield f"event: progress\ndata: {init_rec.model_dump_json()}\n\n"

            while True:
                if await request.is_disconnected():
                    break
                try:
                    rec = await asyncio.wait_for(queue.get(), timeout=1.5)
                    yield f"event: progress\ndata: {rec.model_dump_json()}\n\n"
                    if rec.status in (WorkflowStatus.SUCCESS, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED):
                        break
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
                    latest = global_orchestrator.store.get(workflow_id)
                    if latest and latest.status in (WorkflowStatus.SUCCESS, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED):
                        yield f"event: progress\ndata: {latest.model_dump_json()}\n\n"
                        break
        finally:
            global_orchestrator.unsubscribe(workflow_id, listener)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

# ==========================================
# Standard NVIDIA VSS Blueprint REST APIs
# ==========================================

@app.get("/v1/ready", tags=["Probes"])
def ready_probe():
    """VSS Readiness probe. HTTP 200 means the service is ready to accept requests."""
    return {"status": "ready", "timestamp": int(time.time())}

@app.get("/v1/live", tags=["Probes"])
def live_probe():
    """VSS Liveness probe."""
    return {"status": "live", "timestamp": int(time.time())}

@app.get("/v1/healthz", tags=["Probes"])
def health_check():
    """VSS Health status including GPU memory statistics."""
    vram = active_reasoner.get_vram_stats() if active_reasoner else {}
    return {
        "status": "healthy",
        "service": "vss-trading-service",
        "model": "nvidia/Cosmos3-Nano",
        "gpu_vram": vram
    }

@app.get("/models", tags=["Models"])
def list_models():
    """Lists available vision-language foundation models."""
    return {
        "object": "list",
        "data": [
            {
                "id": "nvidia/Cosmos3-Nano",
                "object": "model",
                "created": 1717405636,
                "owned_by": "nvidia",
                "capabilities": ["multimodal_video_reasoning", "dense_captioning", "text_synthesis"]
            }
        ]
    }

@app.post("/v1/summarize", response_model=CompletionResponse, tags=["Summarization"])
def summarize_video(query: SummarizationQuery):
    """
    Summarizes a video file and associated subtitles into a VSS CompletionResponse.
    """
    global active_search_index
    video_path = query.video_path or query.url
    subtitles_path = query.subtitles_path

    if not video_path or not os.path.exists(video_path):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Video file not found or invalid path: {video_path}"
        )
    if not subtitles_path or not os.path.exists(subtitles_path):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Subtitles file not found or invalid path: {subtitles_path}"
        )

    try:
        pipeline_result = run_vss_pipeline(
            video_path=video_path,
            subtitles_path=subtitles_path,
            output_dir=DEFAULT_OUTPUT_DIR,
            chunk_duration_s=query.chunk_duration,
            chunk_overlap_s=query.chunk_overlap_duration,
            skip_model=not query.enable_reasoning
        )
        return pipeline_result.summary_response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"VSS summarization pipeline failed: {str(e)}"
        )

@app.post("/v1/search", response_model=SearchResponse, tags=["Search"])
def search_video(query: SearchQuery):
    """
    Performs semantic vector and keyword search across indexed video chunks.
    """
    global active_search_index
    index_file = os.path.join(DEFAULT_OUTPUT_DIR, "vss_search_index.json")

    if active_search_index is None:
        if os.path.exists(index_file):
            try:
                active_search_index = VSSSearchIndex.from_file(index_file)
            except Exception as e:
                logger.warning(f"Failed to load search index from {index_file}: {e}")

        if active_search_index is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No search index found. Please run /v1/summarize on a video first to build the index."
            )

    results = active_search_index.search(query.query, top_k=query.top_k, min_score=query.min_score)
    return results

@app.post("/v1/trade_idea", response_model=TradeIdeaSpec, tags=["Trading Strategy"])
def extract_trade_idea(query: SummarizationQuery):
    """
    Synthesizes formal algorithmic trade rules and TradeIdeaSpec.
    """
    video_path = query.video_path or query.url
    subtitles_path = query.subtitles_path

    if not video_path or not os.path.exists(video_path):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Video file not found: {video_path}"
        )
    if not subtitles_path or not os.path.exists(subtitles_path):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Subtitles file not found: {subtitles_path}"
        )

    pipeline_result = run_vss_pipeline(
        video_path=video_path,
        subtitles_path=subtitles_path,
        output_dir=DEFAULT_OUTPUT_DIR,
        skip_model=not query.enable_reasoning
    )
    return pipeline_result.trade_spec
