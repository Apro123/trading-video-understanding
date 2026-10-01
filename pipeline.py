"""
Master End-to-End Pipeline for Trading Video Understanding with NVIDIA Cosmos 3 Nano.
Executes inside the Docker container on the NVIDIA DGX Spark GPU system.
Features:
- Dynamic subtitle timestamp subtraction (0:00 origin)
- Generic multimodal chart reasoning across arbitrary video clips (AV1, VP9, H.264)
- Cosmos-3 Nano dual-role visual perception & text LLM strategy synthesis
- Precise peak VRAM tracking on NVIDIA GB10 GPU
"""

import os
import sys
import argparse
import json
import time
import subprocess
import torch
import numpy as np

from subtitle_parser import parse_subtitles
from ocr_extractor import TradingChartOCRExtractor
from cosmos_reasoner import CosmosTradingReasoner, extract_frame_ffmpeg, DEFAULT_COSMOS3_PATH
from subtitle_augmenter import SubtitleAugmenter
from strategy_synthesizer import StrategySynthesizer

def get_system_vram_usage() -> float:
    """Queries nvidia-smi for total used GPU memory in GB."""
    try:
        cmd = "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits"
        out = subprocess.check_output(cmd, shell=True).decode("utf-8").strip()
        return round(float(out.split("\n")[0]) / 1024.0, 3)
    except Exception:
        return 0.0

def get_video_duration(video_path: str) -> float:
    """Gets video duration in seconds via ffprobe."""
    try:
        cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", video_path]
        out = subprocess.check_output(cmd).decode("utf-8").strip()
        return float(out)
    except Exception as e:
        print(f"[Warning] Could not get duration via ffprobe: {e}")
        return 0.0

def run_pipeline(
    video_path: str = "/workspace/trading_lesson.clip.mp4",
    subtitles_path: str = "/workspace/subtitles_clip",
    output_dir: str = "/workspace/outputs",
    model_path: str = DEFAULT_COSMOS3_PATH,
    skip_model: bool = False,
    num_samples: int = 3
):
    print("=" * 75)
    print("📈 DYNAMIC TRADING VIDEO UNDERSTANDING PIPELINE (COSMOS-3 NANO)")
    print("=" * 75)
    start_total_time = time.time()
    os.makedirs(output_dir, exist_ok=True)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # STAGE 1: Video Inspection & Subtitle Parsing
    print(f"\n>>> STAGE 1: Inspecting Video and Normalizing Subtitles")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not os.path.exists(subtitles_path):
        raise FileNotFoundError(f"Subtitles file not found: {subtitles_path}")

    video_duration = get_video_duration(video_path)
    print(f"Video Duration: {video_duration:.2f}s (verified via FFprobe)")

    # Extract initial anchor frame via FFmpeg
    initial_frame_pil = extract_frame_ffmpeg(video_path, min(5.0, max(0.5, video_duration / 4.0)))
    initial_frame = np.array(initial_frame_pil) if initial_frame_pil is not None else None

    # Automatically parse subtitles with 0:00 start offset
    subtitle_entries = parse_subtitles(subtitles_path, clip_duration_s=video_duration)
    print(f"Parsed {len(subtitle_entries)} subtitle intervals.")
    if subtitle_entries:
        print(f"  First Anchor: [{subtitle_entries[0]['orig_timestamp_str']} -> {subtitle_entries[0]['clip_timestamp_str']}] '{subtitle_entries[0]['text'][:50]}...'")
        print(f"  Last Anchor:  [{subtitle_entries[-1]['orig_timestamp_str']} -> {subtitle_entries[-1]['clip_timestamp_str']}] '{subtitle_entries[-1]['text'][:50]}...'")

    # STAGE 2: OCR Chart UI Anchoring
    print("\n>>> STAGE 2: Generic Chart Metadata Extraction via EasyOCR")
    chart_meta = {
        "detected_header_lines": [],
        "ticker_candidates": [],
        "timeframe_candidates": [],
        "indicators_detected": [],
        "visible_price_levels": [],
        "visible_dates": []
    }
    if initial_frame is not None:
        try:
            ocr_extractor = TradingChartOCRExtractor(use_gpu=True)
            chart_meta = ocr_extractor.extract_chart_metadata(initial_frame)
            print(f"OCR Detected Indicators: {chart_meta['indicators_detected']}")
            print(f"OCR Detected Tickers:    {chart_meta['ticker_candidates']}")
            print(f"OCR Detected Timeframes: {chart_meta['timeframe_candidates']}")
        except Exception as e:
            print(f"[Pipeline Warning] OCR extraction note: {e}")

    # STAGE 3: Multimodal Video Reasoning via Cosmos-3 Nano
    print("\n>>> STAGE 3: Multimodal Video Reasoning with NVIDIA Cosmos-3 Nano")
    cosmos_results = []
    reasoner = None

    if not skip_model:
        try:
            reasoner = CosmosTradingReasoner(model_path=model_path)
            reasoner.load_model()

            # Select evenly spaced milestone segments across the video
            n_entries = len(subtitle_entries)
            if num_samples > 0 and n_entries > 0:
                step = max(1, n_entries // num_samples)
                target_indices = [min(i * step, n_entries - 1) for i in range(num_samples)]
                target_indices = sorted(list(set(target_indices)))
            else:
                target_indices = list(range(n_entries))

            print(f"Running Cosmos-3 Nano multimodal inference on {len(target_indices)} key video intervals...")
            for idx in target_indices:
                sub = subtitle_entries[idx]
                print(f"  -> Processing Interval #{idx} [{sub['clip_timestamp_str']}] ({sub['duration_s']:.1f}s): \"{sub['text'][:45]}...\"")
                res = reasoner.reason_over_segment(
                    video_path=video_path,
                    start_s=sub["clip_timestamp_s"],
                    end_s=sub["clip_end_s"],
                    spoken_text=sub["text"],
                    segment_id=sub["index"]
                )
                cosmos_results.append(res)
                summary_snippet = res['cosmos_reasoning'].replace('\n', ' ')[:90]
                print(f"     Cosmos Output: {summary_snippet}...")

        except Exception as e:
            print(f"[Pipeline Warning] Cosmos-3 Nano inference encountered: {e}")
    else:
        print("[Pipeline] Skipping model inference as requested.")

    # STAGE 4: Dynamic Subtitle Augmentation
    print("\n>>> STAGE 4: Fusing Visual Observations with Spoken Subtitles")
    augmenter = SubtitleAugmenter(reasoner=reasoner)
    augmented_timeline = augmenter.augment(
        subtitles=subtitle_entries,
        cosmos_outputs=cosmos_results,
        chart_metadata=chart_meta
    )
    print(f"Successfully synthesized {len(augmented_timeline)} augmented timeline records.")

    # Record Peak VRAM Usage
    vram_stats = reasoner.get_vram_stats() if reasoner else {
        "current_allocated_gb": 0.0,
        "current_reserved_gb": 0.0,
        "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / (1024**3), 3) if torch.cuda.is_available() else 0.0,
        "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / (1024**3), 3) if torch.cuda.is_available() else 0.0
    }
    vram_stats["system_total_used_gb"] = get_system_vram_usage()

    # STAGE 5: Synthesis into LLM-Ready Formats
    print("\n>>> STAGE 5: Synthesizing Trading Strategy & Generating Artifacts")
    synthesizer = StrategySynthesizer(output_dir=output_dir, reasoner=reasoner)
    generated_files = synthesizer.generate_outputs(
        augmented_timeline=augmented_timeline,
        chart_metadata=chart_meta,
        vram_stats=vram_stats
    )

    elapsed = time.time() - start_total_time
    print("\n" + "=" * 75)
    print(f"✅ PIPELINE EXECUTION COMPLETE ({elapsed:.1f}s)")
    print(f"📊 PEAK GPU VRAM ALLOCATED: {vram_stats['peak_allocated_gb']} GB")
    print(f"📊 PEAK GPU VRAM RESERVED:  {vram_stats['peak_reserved_gb']} GB")
    print(f"📊 TOTAL SYSTEM GPU USAGE:  {vram_stats['system_total_used_gb']} GB")
    print(f"1. Augmented Subtitles (JSON): {generated_files['json']}")
    print(f"2. Augmented Subtitles (Markdown): {generated_files['markdown']}")
    print(f"3. Trading Strategy Spec (JSON): {generated_files['spec']}")
    print(f"4. POC Evaluation Summary (Markdown): {generated_files['summary']}")
    print("=" * 75)

    return generated_files, vram_stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trading Video Understanding Pipeline")
    parser.add_argument("--video", default="/workspace/trading_lesson.clip.mp4", help="Path to video clip")
    parser.add_argument("--subtitles", default="/workspace/subtitles_clip", help="Path to subtitle file")
    parser.add_argument("--output_dir", default="/workspace/outputs", help="Directory for output artifacts")
    parser.add_argument("--model_path", default=DEFAULT_COSMOS3_PATH, help="Path to Cosmos 3 Nano snapshot")
    parser.add_argument("--skip_model", action="store_true", help="Skip model inference")
    parser.add_argument("--num_samples", type=int, default=3, help="Number of key video segments to sample")

    args = parser.parse_args()
    run_pipeline(
        video_path=args.video,
        subtitles_path=args.subtitles,
        output_dir=args.output_dir,
        model_path=args.model_path,
        skip_model=args.skip_model,
        num_samples=args.num_samples
    )
