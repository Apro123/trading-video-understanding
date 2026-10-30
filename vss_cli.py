"""
VSS Command-Line Interface (CLI).
Implements the CLI tool suite for NVIDIA Video Search and Summarization:
https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization

Commands:
- vss summarize: Run multimodal video chunking, VLM reasoning, and LVS summarization.
- vss search: Query an indexed video archive or clip with natural language.
- vss trade-idea: Extract formal algorithmic trade rules and TradeIdeaSpec.
- vss serve: Launch the VSS FastAPI microservice on the designated port.
"""

import os
import sys
import json
import argparse

try:
    from .config import DEFAULT_COSMOS3_PATH, DEFAULT_OUTPUT_DIR
    from .pipeline import run_vss_pipeline
    from .search_engine import VSSSearchIndex
except (ImportError, ValueError):
    from config import DEFAULT_COSMOS3_PATH, DEFAULT_OUTPUT_DIR
    from pipeline import run_vss_pipeline
    from search_engine import VSSSearchIndex

def cmd_summarize(args):
    """Executes VSS summarization and outputs artifacts."""
    run_vss_pipeline(
        video_path=args.video,
        subtitles_path=args.subtitles,
        output_dir=args.output_dir,
        model_path=args.model_path,
        skip_model=args.skip_model,
        chunk_duration_s=args.chunk_duration,
        chunk_overlap_s=args.chunk_overlap,
        frames_per_chunk=args.frames_per_chunk,
        device=args.device,
        llm_base_url=args.llm_base_url,
        llm_model=args.llm_model,
        skip_llm=args.skip_llm
    )

def cmd_search(args):
    """Searches an existing VSS search index."""
    if not os.path.exists(args.index):
        print(f"Error: Search index file not found at {args.index}")
        sys.exit(1)

    idx = VSSSearchIndex.from_file(args.index)

    res = idx.search(args.query, top_k=args.top_k)
    print(f"\n🔍 Search Query: \"{args.query}\"")
    print(f"Matched {len(res.results)} video segments across {res.total_chunks} chunks:\n")

    for i, item in enumerate(res.results, 1):
        print(f"#{i} [{item.timestamp_range}] (Relevance Score: {item.score:.3f})")
        print(f"   Context:    {item.trade_context}")
        print(f"   Caption:    {item.matched_caption}")
        if item.matched_transcript:
            print(f"   Transcript: \"{item.matched_transcript[:80]}...\"")
        print("-" * 60)

def cmd_trade_idea(args):
    """Synthesizes structured trade idea specification."""
    res = run_vss_pipeline(
        video_path=args.video,
        subtitles_path=args.subtitles,
        output_dir=args.output_dir,
        model_path=args.model_path,
        skip_model=args.skip_model,
        device=args.device,
        llm_base_url=args.llm_base_url,
        llm_model=args.llm_model,
        skip_llm=args.skip_llm
    )
    spec = res.trade_spec
    print("\n" + "=" * 60)
    print(f"🎯 TRADE IDEA: {spec.asset} ({spec.timeframe})")
    print(f"Directional Bias: {spec.directional_bias.upper()}")
    print(f"Core Thesis:      {spec.core_thesis}")
    print(f"Entry Trigger:    {spec.entry_trigger}")
    print(f"Stop Loss:        {spec.invalidation_stop}")
    print(f"Risk/Reward:      {spec.risk_reward_ratio}")
    print("=" * 60)

def cmd_serve(args):
    """Starts the VSS FastAPI microservice via uvicorn."""
    import uvicorn
    print(f"Starting NVIDIA VSS Trading Microservice on {args.host}:{args.port}...")
    uvicorn.run("server:app", host=args.host, port=args.port, reload=args.reload)

def main():
    parser = argparse.ArgumentParser(description="NVIDIA VSS Trading CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Summarize sub-command
    p_sum = subparsers.add_parser("summarize", help="Summarize video clip with subtitles")
    p_sum.add_argument("--video", default="/workspace/trading_lesson.clip.mp4", help="Video file path")
    p_sum.add_argument("--subtitles", default="/workspace/videos/video_1.vtt", help="Subtitles file path")
    p_sum.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, help="Output directory")
    p_sum.add_argument("--model_path", default=DEFAULT_COSMOS3_PATH, help="Cosmos 3 Nano snapshot path")
    p_sum.add_argument("--skip_model", action="store_true", help="Deterministic fallback without GPU model")
    p_sum.add_argument("--chunk_duration", type=float, default=20.0, help="Chunk duration in seconds")
    p_sum.add_argument("--chunk_overlap", type=float, default=3.0, help="Overlap duration in seconds")
    p_sum.add_argument("--frames_per_chunk", type=int, default=4, help="Frames sampled per chunk")
    p_sum.add_argument("--device", default="cuda", help="cuda or cpu")
    p_sum.add_argument("--llm_base_url", default=None, help="Base URL for Nemotron reasoning LLM")
    p_sum.add_argument("--llm_model", default="nemotron-3.5-lightning", help="Model name for Nemotron reasoning LLM")
    p_sum.add_argument("--skip_llm", action="store_true", help="Skip LLM reasoning and use algorithmic fallback")
    p_sum.set_defaults(func=cmd_summarize)

    # Search sub-command
    p_search = subparsers.add_parser("search", help="Search video chunks by natural language")
    p_search.add_argument("--query", required=True, help="Natural language search query")
    p_search.add_argument("--index", default=os.path.join(DEFAULT_OUTPUT_DIR, "vss_search_index.json"), help="Index file path")
    p_search.add_argument("--top_k", type=int, default=3, help="Max results to return")
    p_search.set_defaults(func=cmd_search)

    # Trade-idea sub-command
    p_idea = subparsers.add_parser("trade-idea", help="Extract trade idea and execution rules")
    p_idea.add_argument("--video", default="/workspace/trading_lesson.clip.mp4", help="Video file path")
    p_idea.add_argument("--subtitles", default="/workspace/videos/video_1.vtt", help="Subtitles file path")
    p_idea.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, help="Output directory")
    p_idea.add_argument("--model_path", default=DEFAULT_COSMOS3_PATH, help="Cosmos 3 Nano snapshot path")
    p_idea.add_argument("--skip_model", action="store_true", help="Deterministic fallback without GPU model")
    p_idea.add_argument("--device", default="cuda", help="cuda or cpu")
    p_idea.add_argument("--llm_base_url", default=None, help="Base URL for Nemotron reasoning LLM")
    p_idea.add_argument("--llm_model", default="nemotron-3.5-lightning", help="Model name for Nemotron reasoning LLM")
    p_idea.add_argument("--skip_llm", action="store_true", help="Skip LLM reasoning and use algorithmic fallback")
    p_idea.set_defaults(func=cmd_trade_idea)

    # Serve sub-command
    p_serve = subparsers.add_parser("serve", help="Launch VSS REST API microservice")
    p_serve.add_argument("--port", type=int, default=38111, help="Port to listen on (default: 38111)")
    p_serve.add_argument("--host", default="0.0.0.0", help="Host interface (default: 0.0.0.0)")
    p_serve.add_argument("--reload", action="store_true", help="Enable auto-reload")
    p_serve.set_defaults(func=cmd_serve)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
