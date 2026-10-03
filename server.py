"""
VSS REST API Microservice.
Implements the NVIDIA Video Search and Summarization (VSS) REST API specification:
https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization

Endpoints:
- GET /v1/ready: Readiness probe (HTTP 200)
- GET /v1/live: Liveness probe (HTTP 200)
- GET /v1/healthz: Health check with GPU VRAM profiling
- GET /models: List available VSS foundation models
- POST /v1/summarize: Execute long video summarization on a video clip + subtitles
- POST /v1/search: Execute semantic vector and keyword search across indexed video chunks
- POST /v1/trade_idea: Synthesize formal algorithmic trade idea specification
"""

import os
import time
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse

try:
    from .config import DEFAULT_COSMOS3_PATH, DEFAULT_OUTPUT_DIR, VSSVLMConfig
    from .schemas import (
        SummarizationQuery,
        CompletionResponse,
        SearchQuery,
        SearchResponse,
        TradeIdeaSpec
    )
    from .pipeline import run_vss_pipeline
    from .search_engine import VSSSearchIndex
    from .vlm_reasoner import VSSVLMReasoner
except (ImportError, ValueError):
    from config import DEFAULT_COSMOS3_PATH, DEFAULT_OUTPUT_DIR, VSSVLMConfig
    from schemas import (
        SummarizationQuery,
        CompletionResponse,
        SearchQuery,
        SearchResponse,
        TradeIdeaSpec
    )
    from pipeline import run_vss_pipeline
    from search_engine import VSSSearchIndex
    from vlm_reasoner import VSSVLMReasoner

from contextlib import asynccontextmanager

# Shared in-memory search index and reasoner
active_search_index: Optional[VSSSearchIndex] = None
active_reasoner: Optional[VSSVLMReasoner] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes the VSS reasoning engine upon server startup."""
    global active_reasoner
    vlm_config = VSSVLMConfig(
        model_path=os.environ.get("COSMOS3_MODEL_PATH", DEFAULT_COSMOS3_PATH),
        skip_model=os.environ.get("SKIP_MODEL", "false").lower() in ("true", "1")
    )
    active_reasoner = VSSVLMReasoner(vlm_config)
    try:
        active_reasoner.load_model()
    except Exception as e:
        print(f"[VSS Server Warning] Could not preload model: {e}")
    yield

app = FastAPI(
    title="NVIDIA Video Search and Summarization (VSS) Trading Service",
    description="GPU-accelerated reference microservice for trading video understanding, semantic search, and strategy synthesis.",
    version="1.0.0",
    lifespan=lifespan
)

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

    # If in-memory index not loaded, attempt to restore from disk
    if active_search_index is None:
        if os.path.exists(index_file):
            import json
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
