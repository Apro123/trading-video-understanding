"""
Master End-to-End Pipeline for Trading Video Understanding with NVIDIA VSS Blueprint.
Implements the reference architecture from:
https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization

Replaces the custom pipeline with the standardized NVIDIA Video Search and Summarization (VSS)
workflow extended for financial trading clips and subtitles:
- Dynamic 0:00 subtitle synchronization
- Multimodal chunk-based video perception using NVIDIA Cosmos 3 Nano
- In-memory vector and keyword semantic search indexing
- Long Video Summarization (LVS) with chronological milestone extraction
- Algorithmic trade idea and strategy specification synthesis
- Peak GPU VRAM tracking on NVIDIA DGX Spark (NVIDIA GB10 GPU)
"""

import os
import sys
import time
import json
import argparse
from typing import Dict, Any, Optional, Callable

try:
    from .config import (
        DEFAULT_COSMOS3_PATH,
        DEFAULT_LLM_BASE_URL,
        DEFAULT_LLM_MODEL,
        DEFAULT_OUTPUT_DIR,
        VSSPipelineConfig,
        VSSChunkConfig,
        VSSVLMConfig,
        VSSSearchConfig,
        VSSSummarizationConfig
    )
    from .schemas import (
        VideoChunk,
        VlmCaptionResponse,
        CompletionResponse,
        TradeIdeaSpec,
        VSSPipelineResult
    )
    from .subtitle_sync import SubtitleSyncManager
    from .chunk_engine import VideoChunkManager, get_video_duration_ffprobe
    from .vlm_reasoner import VSSVLMReasoner
    from .nemotron_reasoner import NemotronReasoner
    from .search_engine import VSSSearchIndex
    from .summarizer import VSSLVSSummarizer
    from .trade_strategy_agent import VSSTradeStrategyAgent
except (ImportError, ValueError):
    from config import (
        DEFAULT_COSMOS3_PATH,
        DEFAULT_LLM_BASE_URL,
        DEFAULT_LLM_MODEL,
        DEFAULT_OUTPUT_DIR,
        VSSPipelineConfig,
        VSSChunkConfig,
        VSSVLMConfig,
        VSSSearchConfig,
        VSSSummarizationConfig
    )
    from schemas import (
        VideoChunk,
        VlmCaptionResponse,
        CompletionResponse,
        TradeIdeaSpec,
        VSSPipelineResult
    )
    from subtitle_sync import SubtitleSyncManager
    from chunk_engine import VideoChunkManager, get_video_duration_ffprobe
    from vlm_reasoner import VSSVLMReasoner
    from nemotron_reasoner import NemotronReasoner
    from search_engine import VSSSearchIndex
    from summarizer import VSSLVSSummarizer
    from trade_strategy_agent import VSSTradeStrategyAgent

def run_vss_pipeline(
    video_path: str = "/workspace/trading_lesson.clip.mp4",
    subtitles_path: str = "/workspace/subtitles_clip",
    output_dir: str = DEFAULT_OUTPUT_DIR,
    model_path: str = DEFAULT_COSMOS3_PATH,
    skip_model: bool = False,
    chunk_duration_s: float = 20.0,
    chunk_overlap_s: float = 3.0,
    frames_per_chunk: int = 4,
    device: str = "cuda",
    llm_base_url: str = DEFAULT_LLM_BASE_URL,
    llm_model: str = DEFAULT_LLM_MODEL,
    skip_llm: bool = False,
    progress_callback: Optional[Callable[[int, str, float, Optional[Dict[str, Any]]], None]] = None
) -> VSSPipelineResult:
    """
    Executes the end-to-end NVIDIA VSS pipeline on a trading video clip and subtitle file.
    """
    start_total_time = time.time()
    os.makedirs(output_dir, exist_ok=True)

    nemotron = None if skip_llm else NemotronReasoner(base_url=llm_base_url, model_name=llm_model)
    llm_status = "Disabled" if skip_llm else (f"{nemotron.base_url} (Active)" if (nemotron and nemotron.is_available()) else f"{llm_base_url} (Offline / Fallback)")

    print("=" * 80)
    print("🚀 NVIDIA AI BLUEPRINT: VIDEO SEARCH & SUMMARIZATION (VSS) TRADING PIPELINE")
    print("=" * 80)
    print(f"Input Video:      {video_path}")
    print(f"Input Subtitles:  {subtitles_path}")
    print(f"Output Directory: {output_dir}")
    print(f"VLM Model:        {model_path} ({'CPU/Fallback' if skip_model else 'GPU Cosmos-3 Nano'})")
    print(f"LLM Reasoner:     {llm_status} [{llm_model}]")
    print("-" * 80)

    # -------------------------------------------------------------
    # STAGE 1: Video Inspection & Subtitle Normalization (0:00 Origin)
    # -------------------------------------------------------------
    print("\n>>> [STAGE 1/6] Video Inspection & Subtitle Normalization")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not os.path.exists(subtitles_path):
        raise FileNotFoundError(f"Subtitles file not found: {subtitles_path}")

    video_duration = get_video_duration_ffprobe(video_path)
    print(f"✓ Video duration: {video_duration:.2f}s (ffprobe)")

    sub_mgr = SubtitleSyncManager(subtitles_path, clip_duration_s=video_duration)
    print(f"✓ Normalized {len(sub_mgr.entries)} subtitle intervals anchored to 0:00.")
    if sub_mgr.entries:
        print(f"  First Anchor: [{sub_mgr.entries[0].orig_timestamp_str} -> {sub_mgr.entries[0].clip_timestamp_str}] \"{sub_mgr.entries[0].text[:50]}...\"")
        print(f"  Last Anchor:  [{sub_mgr.entries[-1].orig_timestamp_str} -> {sub_mgr.entries[-1].clip_timestamp_str}] \"{sub_mgr.entries[-1].text[:50]}...\"")

    if progress_callback:
        progress_callback(1, "Subtitle Ingestion & 0:00 Normalization", 15.0, {"subtitles_count": len(sub_mgr.entries)})

    # -------------------------------------------------------------
    # STAGE 2: Video Chunking & Keyframe Extraction via FFmpeg Pipe
    # -------------------------------------------------------------
    print(f"\n>>> [STAGE 2/6] Video Chunking & Frame Sampling (Window: {chunk_duration_s}s, Overlap: {chunk_overlap_s}s)")
    chunk_mgr = VideoChunkManager(
        video_path=video_path,
        subtitle_mgr=sub_mgr,
        chunk_duration_s=chunk_duration_s,
        chunk_overlap_s=chunk_overlap_s,
        frames_per_chunk=frames_per_chunk
    )
    chunks = chunk_mgr.generate_chunks()
    print(f"✓ Generated {len(chunks)} temporal chunks across {video_duration:.1f}s.")

    if progress_callback:
        progress_callback(2, "Video Chunking & Keyframe Sampling", 30.0, {"chunks_count": len(chunks)})

    # -------------------------------------------------------------
    # STAGE 3: Multimodal Video Reasoning with NVIDIA Cosmos 3 Nano
    # -------------------------------------------------------------
    print("\n>>> [STAGE 3/6] Multimodal Video Perception via NVIDIA Cosmos 3 Nano")
    vlm_config = VSSVLMConfig(
        model_path=model_path,
        device=device,
        skip_model=skip_model
    )
    reasoner = VSSVLMReasoner(vlm_config)
    reasoner.load_model()

    chunk_captions = []
    total_chunks = len(chunks)
    for idx, chunk in enumerate(chunks):
        frames = chunk_mgr.extract_chunk_frames(chunk)
        if progress_callback:
            chunk_pct = 30.0 + (35.0 * (idx / max(1, total_chunks)))
            progress_callback(3, f"VLM Perception & Visual Grounding (Chunk {idx+1}/{total_chunks})", chunk_pct, {"chunk_id": chunk.chunk_id})

        print(f"  -> Processing Chunk #{chunk.chunk_id} [{chunk.start_str} - {chunk.end_str}] ({len(frames)} frames):")
        if chunk.synchronized_transcript:
            print(f"     Transcript: \"{chunk.synchronized_transcript[:60]}...\"")
        cap = reasoner.reason_over_chunk(chunk, frames)
        chunk_captions.append(cap)
        summary_snippet = cap.content.replace("\n", " ")[:80]
        print(f"     VLM Caption: {summary_snippet}...")
        if cap.detected_trade_signals:
            print(f"     Signals:     {', '.join(cap.detected_trade_signals)}")

    if progress_callback:
        progress_callback(3, "VLM Perception & Visual Grounding", 65.0, {"chunks_processed": len(chunk_captions)})

    # -------------------------------------------------------------
    # STAGE 4: Semantic Search Indexing
    # -------------------------------------------------------------
    print("\n>>> [STAGE 4/6] Constructing VSS Semantic Video Search Index")
    search_index = VSSSearchIndex(video_id=os.path.basename(video_path))
    search_index.index_chunks(chunks, chunk_captions)
    print(f"✓ Indexed {len(chunks)} chunks with multimodal embeddings and BM25 keywords.")

    if progress_callback:
        progress_callback(4, "Multimodal Vector & Keyword Search Indexing", 75.0, {"indexed_chunks": len(chunks)})

    # Execute representative trading search queries
    demo_queries = [
        "Where does the breakout or trend reversal happen?",
        "Where is the buy zone and accumulation consolidation retest?",
        "What is the stop loss or protected swing low?"
    ]
    for q in demo_queries:
        res = search_index.search(q, top_k=1)
        if res.results:
            top_hit = res.results[0]
            print(f"  Query: \"{q}\" -> Match: [{top_hit.timestamp_range}] (Score: {top_hit.score:.3f})")

    # -------------------------------------------------------------
    # STAGE 5: Long Video Summarization (LVS) & Milestone Events
    # -------------------------------------------------------------
    print("\n>>> [STAGE 5/6] Synthesizing Long Video Summary (LVS) & Milestone Events")
    summarizer = VSSLVSSummarizer(reasoner=reasoner, nemotron=nemotron)
    trade_agent = VSSTradeStrategyAgent(reasoner=reasoner, nemotron=nemotron)

    # Initial completion synthesis
    temp_completion = CompletionResponse(
        video_id=os.path.basename(video_path),
        summary="",
        chunk_captions=chunk_captions
    )
    trade_spec = trade_agent.synthesize_trade_idea(temp_completion, chunks=chunks)
    completion = summarizer.summarize(chunks, chunk_captions, trade_spec=trade_spec, video_id=os.path.basename(video_path))
    print(f"✓ Formulated {len(completion.key_events)} chronological milestone events.")

    if progress_callback:
        progress_callback(5, "Long Video Summary & Milestone Formulation", 90.0, {"milestones_count": len(completion.key_events)})

    # -------------------------------------------------------------
    # STAGE 6: Output Artifact Generation
    # -------------------------------------------------------------
    print("\n>>> [STAGE 6/6] Generating Output Artifacts")
    vram_stats = reasoner.get_vram_stats()

    # 1. trading_summary.json (Standard VSS Completion Response)
    summary_json_path = os.path.join(output_dir, "trading_summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(completion.model_dump(mode="json"), f, indent=2)

    # 2. trading_summary.md (Polished Markdown Summary & Event Timeline)
    summary_md_path = os.path.join(output_dir, "trading_summary.md")
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write(summarizer.export_summary_markdown(completion))

    # 3. trading_strategy_spec.json (Machine-readable Algorithmic Trade Spec)
    spec_json_path = os.path.join(output_dir, "trading_strategy_spec.json")
    spec_data = trade_spec.model_dump(mode="json")
    spec_data["hardware_metrics"] = vram_stats
    with open(spec_json_path, "w", encoding="utf-8") as f:
        json.dump(spec_data, f, indent=2)

    # 4. trading_strategy_spec.md (Markdown Trade Plan)
    spec_md_path = os.path.join(output_dir, "trading_strategy_spec.md")
    with open(spec_md_path, "w", encoding="utf-8") as f:
        f.write(trade_agent.export_spec_markdown(trade_spec))

    # 5. vss_search_index.json (Searchable Vector/Keyword Index)
    search_index_path = os.path.join(output_dir, "vss_search_index.json")
    with open(search_index_path, "w", encoding="utf-8") as f:
        json.dump(search_index.export_index_json(), f, indent=2)

    # 6. augmented_subtitles.json & .md (Backward-compatible Grounded Subtitles)
    aug_subs = []
    for chunk in chunks:
        cap = next((c for c in chunk_captions if c.chunk_id == chunk.chunk_id), None)
        for s in chunk.subtitle_entries:
            aug_subs.append({
                "clip_timestamp": s.clip_timestamp_str,
                "orig_timestamp": s.orig_timestamp_str,
                "spoken_text": s.text,
                "visual_confluence": cap.content if cap else "",
                "signals": cap.detected_trade_signals if cap else []
            })
    aug_json_path = os.path.join(output_dir, "augmented_subtitles.json")
    with open(aug_json_path, "w", encoding="utf-8") as f:
        json.dump(aug_subs, f, indent=2)

    aug_md_path = os.path.join(output_dir, "augmented_subtitles.md")
    with open(aug_md_path, "w", encoding="utf-8") as f:
        f.write("# 📝 Multimodal Grounded Subtitles & Visual Confluences\n\n")
        f.write("| Clip Time | Original Time | Spoken Subtitle | Grounded Visual Confluence | Signals |\n")
        f.write("|---|---|---|---|---|\n")
        for entry in aug_subs:
            sp = entry["spoken_text"].replace("|", "/")
            vc = entry["visual_confluence"].replace("|", "/")
            sig = ", ".join(entry["signals"])
            f.write(f"| `{entry['clip_timestamp']}` | `{entry['orig_timestamp']}` | {sp} | {vc} | `{sig}` |\n")

    elapsed = time.time() - start_total_time
    generated_files = {
        "summary_json": summary_json_path,
        "summary_md": summary_md_path,
        "spec_json": spec_json_path,
        "spec_md": spec_md_path,
        "search_index": search_index_path,
        "augmented_subtitles_json": aug_json_path,
        "augmented_subtitles_md": aug_md_path
    }
    if progress_callback:
        progress_callback(6, "Algorithmic Strategy Specification Synthesis", 100.0, {"artifacts": list(generated_files.keys())})

    print("\n" + "=" * 80)
    print(f"✅ VSS PIPELINE EXECUTION COMPLETE ({elapsed:.1f}s)")
    print(f"📊 Peak GPU VRAM Allocated: {vram_stats['peak_allocated_gb']} GB")
    print(f"📊 Peak GPU VRAM Reserved:  {vram_stats['peak_reserved_gb']} GB")
    print(f"📊 System GPU Memory Used:  {vram_stats['system_used_gb']} GB")
    print("\nGenerated Artifacts:")
    for k, v in generated_files.items():
        print(f"  • {k}: {v}")
    print("=" * 80)

    return VSSPipelineResult(
        video_path=video_path,
        subtitles_path=subtitles_path,
        duration_s=video_duration,
        total_chunks=len(chunks),
        summary_response=completion,
        trade_spec=trade_spec,
        search_index_size=len(chunks),
        hardware_metrics=vram_stats,
        generated_files=generated_files
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NVIDIA VSS Trading Video Understanding Pipeline")
    parser.add_argument("--video", default="/workspace/trading_lesson.clip.mp4", help="Path to video clip")
    parser.add_argument("--subtitles", default="/workspace/subtitles_clip", help="Path to subtitle file")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, help="Output directory for VSS artifacts")
    parser.add_argument("--model_path", default=DEFAULT_COSMOS3_PATH, help="Path to Cosmos 3 Nano model snapshot")
    parser.add_argument("--skip_model", action="store_true", help="Run with deterministic fallback (no model load)")
    parser.add_argument("--chunk_duration", type=float, default=20.0, help="Duration of each chunk in seconds")
    parser.add_argument("--chunk_overlap", type=float, default=3.0, help="Overlap duration in seconds")
    parser.add_argument("--frames_per_chunk", type=int, default=4, help="Frames sampled per chunk")
    parser.add_argument("--device", default="cuda", help="Execution device (cuda or cpu)")
    parser.add_argument("--llm_base_url", default=DEFAULT_LLM_BASE_URL, help="Base URL for Nemotron reasoning LLM")
    parser.add_argument("--llm_model", default=DEFAULT_LLM_MODEL, help="Model name for Nemotron reasoning LLM")
    parser.add_argument("--skip_llm", action="store_true", help="Skip LLM reasoning and use algorithmic fallback")

    args = parser.parse_args()
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
