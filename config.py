"""
Configuration settings for the NVIDIA VSS (Video Search and Summarization) Trading Pipeline.
Adheres to the configuration specifications in NVIDIA AI Blueprints:
https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List

# Default model path on NVIDIA DGX Spark GPU server
DEFAULT_COSMOS3_PATH = os.environ.get(
    "COSMOS3_MODEL_PATH",
    "/mnt/edrive/huggingface/hub/models--nvidia--Cosmos3-Nano/snapshots/e59a53c25979a090fa8706c9acc0c254a6e89b92"
)

# Nemotron 3.5 Lightning reasoning endpoint on NVIDIA DGX Spark
DEFAULT_LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1")
DEFAULT_LLM_MODEL = os.environ.get("LLM_MODEL", "nemotron-3.5-lightning")

DEFAULT_OUTPUT_DIR = os.environ.get("VSS_OUTPUT_DIR", "/workspace/outputs_vss")

@dataclass
class VSSChunkConfig:
    """Configuration for video chunking and temporal windowing."""
    chunk_duration_s: float = 20.0          # Duration of each video chunk in seconds
    chunk_overlap_s: float = 3.0           # Overlap between consecutive chunks
    frames_per_chunk: int = 4              # Number of frames sampled per chunk for VLM
    frame_max_dimension: int = 1280        # Max resolution dimension for frame downscaling
    adaptive_subtitle_chunking: bool = True # Align chunk boundaries with subtitle pauses

@dataclass
class VSSVLMConfig:
    """Configuration for the Vision-Language Model (Cosmos 3 Nano)."""
    model_name: str = "nvidia/Cosmos3-Nano"
    model_path: str = DEFAULT_COSMOS3_PATH
    device: str = "cuda"
    torch_dtype: str = "bfloat16"
    enable_reasoning: bool = True
    max_new_tokens: int = 768
    temperature: float = 0.2
    top_p: float = 0.8
    skip_model: bool = False               # Fallback mode when GPU/model is unavailable

@dataclass
class VSSSearchConfig:
    """Configuration for semantic vector search across video chunks."""
    embedding_dimension: int = 384         # Lightweight dense embedding dimension
    top_k: int = 3                         # Default top k retrieval results
    similarity_metric: str = "cosine"      # Similarity metric: cosine or dot
    enable_bm25_hybrid: bool = True        # Hybrid search combining sparse BM25 + dense vectors

@dataclass
class VSSSummarizationConfig:
    """Configuration for Long Video Summarization (LVS) and Strategy Synthesis."""
    scenario: str = "trading_technical_analysis"
    custom_events: List[str] = field(default_factory=lambda: [
        "trend_reversal",
        "breakout",
        "market_structure_shift",
        "retest_buy_zone",
        "rsi_divergence",
        "entry_trigger",
        "invalidation_stop"
    ])
    max_summary_tokens: int = 1536
    extract_trade_spec: bool = True
    llm_base_url: str = DEFAULT_LLM_BASE_URL
    llm_model: str = DEFAULT_LLM_MODEL
    enable_llm_reasoning: bool = True

@dataclass
class VSSPipelineConfig:
    """Master configuration encapsulating all VSS sub-components."""
    chunking: VSSChunkConfig = field(default_factory=VSSChunkConfig)
    vlm: VSSVLMConfig = field(default_factory=VSSVLMConfig)
    search: VSSSearchConfig = field(default_factory=VSSSearchConfig)
    summarization: VSSSummarizationConfig = field(default_factory=VSSSummarizationConfig)
    output_dir: str = DEFAULT_OUTPUT_DIR
    llm_base_url: str = DEFAULT_LLM_BASE_URL
    llm_model: str = DEFAULT_LLM_MODEL
    api_port: int = int(os.environ.get("VSS_PORT", "38111"))
    api_host: str = os.environ.get("VSS_HOST", "0.0.0.0")
