"""
Video Downloader & Metadata Engine for Trading Video Understanding.
Follows strict rule:
- Highest quality video-only format (excluding premium formats).
- Always downloads or aligns subtitles (WebVTT/SRT).
- Caches and detects locally existing video files and subtitles.
"""

import os
import re
import json
import shutil
import logging
import subprocess
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

try:
    from .utils import safe_load_json
except (ImportError, ValueError):
    from utils import safe_load_json

VIDEOS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "videos")
os.makedirs(VIDEOS_DIR, exist_ok=True)

# Pre-registered trading videos & local test clips
KNOWN_VIDEOS: Dict[str, Dict[str, Any]] = {
    "clip_1": {
        "video_id": "clip_1",
        "title": "NVDA Market Structure Shift & Breakout Retest (Trading Lesson 1)",
        "url": "local:trading_lesson.clip.mp4",
        "asset": "NVDA",
        "timeframe": "1D (Daily)",
        "duration_s": 157.02,
        "duration_str": "2m 37s",
        "clip_ref": "trading_lesson.clip.mp4",
        "subtitles_ref": "videos/video_1.vtt",
        "output_dir_ref": "outputs_vss_clip_1_grounded",
        "description": "Bullish change of character (CHoCH), pivot level breakout validation at $1927, demand zone retesting, and protected swing stop placement."
    },
    "clip_2": {
        "video_id": "clip_2",
        "title": "RSI Momentum vs Reversal & Divergence Confluence (Trading Lesson 2)",
        "url": "local:trading_lesson_2.clip.mp4",
        "asset": "BTC/USD & Multi-Asset",
        "timeframe": "1D / 4H",
        "duration_s": 113.0,
        "duration_str": "1m 53s",
        "clip_ref": "trading_lesson_2.clip.mp4",
        "subtitles_ref": "videos/video_2.vtt",
        "output_dir_ref": "outputs_vss_clip_2_grounded",
        "description": "RSI indicator dynamics: overcoming common overbought/oversold misconceptions, exploiting momentum divergence at key resistance, and candlestick confirmation."
    },
    "video_1": {
        "video_id": "video_1",
        "title": "Market Structure: Swing Trading & Long-Term Investing",
        "url": "local:video_1.mp4",
        "asset": "NVDA",
        "timeframe": "1W (Weekly)",
        "duration_s": 2369.0,
        "duration_str": "39m 29s",
        "output_dir_ref": "outputs_video_1",
        "description": "Comprehensive institutional market structure breakdown, multi-year consolidation breakouts, and buy zone pullback entries on NVIDIA weekly charts."
    },
    "video_2": {
        "video_id": "video_2",
        "title": "Technical Analysis Course: Indicator & Price Action Dynamics",
        "url": "local:video_2.mp4",
        "asset": "Forex (AUDCAD / EURUSD)",
        "timeframe": "1D (Daily)",
        "duration_s": 4655.0,
        "duration_str": "1h 17m 35s",
        "output_dir_ref": "outputs_video_2",
        "description": "Complete technical analysis masterclass covering RSI momentum vs reversal, RSI divergence, structural support/resistance, and candlestick confirmation."
    }
}

def get_video_file_path(video_id: str) -> Optional[str]:
    """Finds the local MP4 or WebM video file for a video_id."""
    root_dir = os.path.dirname(VIDEOS_DIR)
    
    # 1. Check if direct file path was passed
    if os.path.exists(video_id) and os.path.getsize(video_id) > 1024:
        return os.path.abspath(video_id)
        
    # 2. Standard naming conventions (full video prioritized)
    candidates = [
        os.path.join(VIDEOS_DIR, f"{video_id}.mp4"),
        os.path.join(VIDEOS_DIR, f"{video_id}.webm"),
        os.path.join(VIDEOS_DIR, f"{video_id}.mkv"),
        os.path.join(root_dir, f"{video_id}.mp4"),
        os.path.join(root_dir, f"{video_id}.clip.mp4"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate) and os.path.getsize(candidate) > 1024:
            return os.path.abspath(candidate)

    # 3. Check clip_ref in KNOWN_VIDEOS (fallback for test fixtures)
    if video_id in KNOWN_VIDEOS and "clip_ref" in KNOWN_VIDEOS[video_id]:
        clip_name = KNOWN_VIDEOS[video_id]["clip_ref"]
        for base in [root_dir, VIDEOS_DIR]:
            p = os.path.join(base, clip_name)
            if os.path.exists(p) and os.path.getsize(p) > 1024:
                return os.path.abspath(p)

    # 4. Search directory contents
    for d in [VIDEOS_DIR, root_dir]:
        if not os.path.exists(d):
            continue
        for fname in os.listdir(d):
            if video_id in fname and fname.endswith((".mp4", ".webm", ".mkv")):
                p = os.path.join(d, fname)
                if os.path.getsize(p) > 1024:
                    return os.path.abspath(p)
    return None

def get_subtitle_file_path(video_id: str) -> Optional[str]:
    """Finds the local subtitle file (.vtt or .srt or raw subtitle file) for a video_id."""
    root_dir = os.path.dirname(VIDEOS_DIR)

    # 1. Check if direct path passed
    if os.path.exists(video_id) and os.path.getsize(video_id) > 50:
        return os.path.abspath(video_id)

    # 2. Standard candidate paths (full video subtitles prioritized)
    candidates = [
        os.path.join(VIDEOS_DIR, f"{video_id}.vtt"),
        os.path.join(VIDEOS_DIR, f"subtitles_{video_id}.en.vtt"),
        os.path.join(VIDEOS_DIR, f"{video_id}.en.vtt"),
        os.path.join(VIDEOS_DIR, f"{video_id}.srt"),
        os.path.join(root_dir, f"subtitles_{video_id}"),
        os.path.join(root_dir, f"subtitles_{video_id}.vtt"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate) and os.path.getsize(candidate) > 50:
            return os.path.abspath(candidate)

    # 3. Check subtitles_ref in KNOWN_VIDEOS (fallback)
    if video_id in KNOWN_VIDEOS and "subtitles_ref" in KNOWN_VIDEOS[video_id]:
        sub_name = KNOWN_VIDEOS[video_id]["subtitles_ref"]
        for base in [root_dir, VIDEOS_DIR]:
            p = os.path.join(base, sub_name)
            if os.path.exists(p) and os.path.getsize(p) > 50:
                return os.path.abspath(p)

    # 4. Search directory contents
    for fname in os.listdir(root_dir):
        if video_id in fname and fname.endswith((".vtt", ".srt")):
            p = os.path.join(root_dir, fname)
            if os.path.getsize(p) > 50:
                return os.path.abspath(p)
    return None



def get_video_status(video_id: str) -> Dict[str, Any]:
    """Returns the download and indexing status of a trading video or local clip."""
    meta = KNOWN_VIDEOS.get(video_id, {
        "video_id": video_id,
        "title": f"Video [{video_id}]",
        "url": f"local:{video_id}",
        "duration_s": 0.0,
        "duration_str": "Unknown",
        "asset": "Unknown",
        "timeframe": "Unknown"
    })

    video_path = get_video_file_path(video_id)
    sub_path = get_subtitle_file_path(video_id)

    video_size_mb = round(os.path.getsize(video_path) / (1024 * 1024), 1) if video_path else 0.0
    sub_size_kb = round(os.path.getsize(sub_path) / 1024, 1) if sub_path else 0.0

    # Check for existing analysis output directory
    root_dir = os.path.dirname(VIDEOS_DIR)
    candidate_out_dirs = [
        os.path.join(root_dir, f"outputs_vss_{video_id}_grounded"),
        os.path.join(root_dir, f"outputs_vss_{video_id}"),
        os.path.join(root_dir, f"outputs_{video_id}"),
    ]
    if "output_dir_ref" in meta:
        candidate_out_dirs.insert(0, os.path.join(root_dir, meta["output_dir_ref"]))

    output_dir = candidate_out_dirs[0]
    for d in candidate_out_dirs:
        if os.path.exists(os.path.join(d, "vss_search_index.json")) or os.path.exists(os.path.join(d, "trading_summary.json")):
            output_dir = d
            break

    index_file = os.path.join(output_dir, "vss_search_index.json")
    spec_file = os.path.join(output_dir, "trading_strategy_spec.json")

    has_index = os.path.exists(index_file)
    has_spec = os.path.exists(spec_file)

    indexed_chunks = 0
    if has_index:
        d = safe_load_json(index_file, default={})
        indexed_chunks = d.get("total_chunks", len(d.get("chunks", [])))

    return {
        **meta,
        "is_downloaded": video_path is not None,
        "video_path": video_path,
        "video_size_mb": video_size_mb,
        "has_subtitles": sub_path is not None,
        "subtitles_path": sub_path,
        "subtitles_size_kb": sub_size_kb,
        "output_dir": output_dir,
        "has_index": has_index,
        "indexed_chunks": indexed_chunks,
        "has_spec": has_spec
    }

def get_video_chapters(video_id: str) -> List[Dict[str, Any]]:
    """Retrieves chapter metadata for a video, caching locally."""
    cache_path = os.path.join(VIDEOS_DIR, f"{video_id}_chapters.json")
    cached = safe_load_json(cache_path)
    if cached is not None:
        return cached
    return []

def list_available_videos() -> List[Dict[str, Any]]:
    """Lists status for all supported trading videos."""
    return [get_video_status(vid) for vid in KNOWN_VIDEOS]

def download_video(video_id: str) -> Dict[str, Any]:
    """
    Ensures video and subtitles are available locally.
    """
    if video_id not in KNOWN_VIDEOS:
        raise ValueError(f"Unknown video ID: {video_id}. Allowed: {list(KNOWN_VIDEOS.keys())}")

    target_video = os.path.join(VIDEOS_DIR, f"{video_id}.mp4")
    target_sub = os.path.join(VIDEOS_DIR, f"{video_id}.vtt")

    # If both exist and have healthy sizes, reuse
    if os.path.exists(target_video) and os.path.getsize(target_video) > 1024 * 1024 and \
       os.path.exists(target_sub) and os.path.getsize(target_sub) > 500:
        logger.info(f"Video {video_id} and subtitles already available.")
        return get_video_status(video_id)

    meta = KNOWN_VIDEOS.get(video_id, {})
    url = meta.get("url", "")
    if url.startswith("http"):
        # Download Subtitles first if missing
        if not os.path.exists(target_sub) or os.path.getsize(target_sub) < 500:
            logger.info(f"Downloading subtitles for {video_id}...")
            sub_cmd = [
                "yt-dlp",
                "--write-auto-subs",
                "--write-subs",
                "--sub-langs", "en.*,en",
                "--skip-download",
                "--no-playlist",
                "-o", os.path.join(VIDEOS_DIR, f"subtitles_{video_id}.%(ext)s"),
                url
            ]
            subprocess.run(sub_cmd, capture_output=True, text=True)
            for f in os.listdir(VIDEOS_DIR):
                if f.startswith(f"subtitles_{video_id}") and f.endswith(".vtt"):
                    shutil.copyfile(os.path.join(VIDEOS_DIR, f), target_sub)
                    break

        # Download Video
        if not os.path.exists(target_video) or os.path.getsize(target_video) < 1024 * 1024:
            format_selector = "bestvideo[vcodec^=avc1][format_note!*=Premium][format_note!*=premium]/bestvideo[ext=mp4]/bestvideo"
            vid_cmd = [
                "yt-dlp",
                "-f", format_selector,
                "--no-playlist",
                "-o", target_video,
                url
            ]
            res = subprocess.run(vid_cmd, capture_output=True, text=True)
            if res.returncode == 0 and os.path.exists(target_video):
                try:
                    faststart_temp = f"{target_video}.faststart.mp4"
                    ff_res = subprocess.run([
                        "ffmpeg", "-i", target_video, "-c", "copy", "-movflags", "+faststart", faststart_temp, "-y"
                    ], capture_output=True, text=True)
                    if ff_res.returncode == 0 and os.path.exists(faststart_temp) and os.path.getsize(faststart_temp) > 1024:
                        shutil.move(faststart_temp, target_video)
                except Exception as e:
                    logger.warning(f"Faststart optimization note: {e}")

    return get_video_status(video_id)
