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
import time
import json
import logging
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException, status, BackgroundTasks
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
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
    from .subtitle_sync import SubtitleSyncManager
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
    from subtitle_sync import SubtitleSyncManager
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

def get_or_load_search_index(video_id: str) -> Optional[VSSSearchIndex]:
    """Retrieves cached in-memory search index or loads from outputs_{video_id}/vss_search_index.json."""
    if video_id in video_search_indices:
        return video_search_indices[video_id]

    # Check video-specific output dir or default
    candidate_paths = [
        os.path.join(BASE_DIR, f"outputs_{video_id}", "vss_search_index.json"),
        os.path.join(DEFAULT_OUTPUT_DIR, "vss_search_index.json")
    ]

    for p in candidate_paths:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                idx = VSSSearchIndex(video_id=data.get("video_id", video_id))
                idx.chunks_data = data.get("chunks", [])
                idx._total_docs = len(idx.chunks_data)
                for d in idx.chunks_data:
                    d["tokens"] = idx._tokenize(f"{d.get('transcript','')} {d.get('caption','')} {d.get('reasoning','')}")
                    for t in set(d["tokens"]):
                        idx._doc_frequencies[t] = idx._doc_frequencies.get(t, 0) + 1
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
    out_dir = os.path.join(BASE_DIR, f"outputs_{video_id}")

    spec_data = None
    spec_path = os.path.join(out_dir, "trading_strategy_spec.json")
    if os.path.exists(spec_path):
        try:
            with open(spec_path, "r", encoding="utf-8") as f:
                spec_data = json.load(f)
        except Exception:
            pass

    summary_data = None
    summary_path = os.path.join(out_dir, "trading_summary.json")
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary_data = json.load(f)
        except Exception:
            pass

    strat_data = None
    strat_path = os.path.join(out_dir, "multi_strategy_breakdown.json")
    if os.path.exists(strat_path):
        try:
            with open(strat_path, "r", encoding="utf-8") as f:
                strat_data = json.load(f)
        except Exception:
            pass

    return {
        **meta,
        "trade_spec": spec_data,
        "summary": summary_data,
        "strategies": strat_data
    }

@app.get("/api/videos/strategies/{video_id}", response_model=MultiStrategyBreakdown, tags=["Trading Strategy"])
def get_video_strategies_endpoint(video_id: str):
    """Retrieves all discrete trading strategies identified across video intervals."""
    out_dir = os.path.join(BASE_DIR, f"outputs_{video_id}")
    strat_path = os.path.join(out_dir, "multi_strategy_breakdown.json")
    if os.path.exists(strat_path):
        try:
            with open(strat_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return MultiStrategyBreakdown(**data)
        except Exception as e:
            logger.warning(f"Error reading cached strategies: {e}")

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

@app.get("/api/videos/stream/{video_id}", tags=["Video Streaming"])
def stream_video(video_id: str):
    """Streams the local MP4 video file with HTTP 206 Partial Content Range support."""
    vpath = get_video_file_path(video_id)
    if not vpath or not os.path.exists(vpath):
        raise HTTPException(status_code=404, detail="Video file not found")
    return FileResponse(vpath, media_type="video/mp4")

@app.get("/api/videos/subtitles/{video_id}", tags=["Video Streaming"])
def stream_subtitles(video_id: str):
    """Serves the WebVTT subtitle track for browser closed captioning."""
    spath = get_subtitle_file_path(video_id)
    if not spath or not os.path.exists(spath):
        raise HTTPException(status_code=404, detail="Subtitles not found")
    return FileResponse(spath, media_type="text/vtt")

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
            with open(index_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            active_search_index = VSSSearchIndex(video_id=data.get("video_id", "default_video"))
            active_search_index.chunks_data = data.get("chunks", [])
            active_search_index._total_docs = len(active_search_index.chunks_data)
            for d in active_search_index.chunks_data:
                d["tokens"] = active_search_index._tokenize(f"{d.get('transcript','')} {d.get('caption','')} {d.get('reasoning','')}")
                for t in set(d["tokens"]):
                    active_search_index._doc_frequencies[t] = active_search_index._doc_frequencies.get(t, 0) + 1
        else:
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
