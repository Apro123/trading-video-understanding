"""
Video Video Downloader & Metadata Engine for Trading Video Understanding.
Follows strict rule:
- Highest quality video-only format (excluding premium formats).
- Always downloads subtitles (WebVTT/SRT).
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

VIDEOS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "videos")
os.makedirs(VIDEOS_DIR, exist_ok=True)

# Pre-registered Video trading videos from videos.md
KNOWN_VIDEOS: Dict[str, Dict[str, Any]] = {
    "video_1": {
        "video_id": "video_1",
        "title": "Master Technical Charts: Swing Trading & Long-Term Investing with Market Structure",
        "url": "local:video_1.mp4",
        "asset": "NVDA",
        "timeframe": "1W (Weekly)",
        "duration_s": 2369.0,
        "duration_str": "39m 29s",
        "clip_ref": "trading_lesson.clip.mp4",
        "subtitles_ref": "subtitles_clip",
        "description": "Comprehensive institutional market structure breakdown, multi-year consolidation breakouts, and buy zone pullback entries on NVIDIA weekly charts."
    },
    "video_2": {
        "video_id": "video_2",
        "title": "The Only Technical Analysis Video You Will Ever Need... (Full Course: Beginner To Advanced)",
        "url": "local:video_2.mp4",
        "asset": "Forex (AUDCAD / EURUSD)",
        "timeframe": "1D (Daily)",
        "duration_s": 4655.0,
        "duration_str": "1h 17m 35s",
        "clip_ref": "trading_lesson_2.clip.mp4",
        "subtitles_ref": "subtitles_clip_2",
        "description": "Complete technical analysis masterclass covering RSI momentum vs reversal, RSI divergence, structural support/resistance, and candlestick confirmation."
    }
}

def get_video_file_path(video_id: str) -> Optional[str]:
    """Finds the local MP4 or WebM video file for a video_id."""
    candidates = [
        os.path.join(VIDEOS_DIR, f"{video_id}.mp4"),
        os.path.join(VIDEOS_DIR, f"{video_id}.webm"),
        os.path.join(VIDEOS_DIR, f"{video_id}.mkv"),
    ]
    # Also search for files containing the video ID in VIDEOS_DIR and project root
    for candidate in candidates:
        if os.path.exists(candidate) and os.path.getsize(candidate) > 1024:
            return candidate

    for d in [VIDEOS_DIR, os.path.dirname(VIDEOS_DIR)]:
        if not os.path.exists(d):
            continue
        for fname in os.listdir(d):
            if video_id in fname and fname.endswith((".mp4", ".webm", ".mkv")):
                p = os.path.join(d, fname)
                if os.path.getsize(p) > 1024:
                    return p
    return None

def get_subtitle_file_path(video_id: str) -> Optional[str]:
    """Finds the local subtitle file (.vtt or .srt) for a video_id."""
    candidates = [
        os.path.join(VIDEOS_DIR, f"{video_id}.vtt"),
        os.path.join(VIDEOS_DIR, f"subtitles_{video_id}.en.vtt"),
        os.path.join(VIDEOS_DIR, f"{video_id}.en.vtt"),
        os.path.join(VIDEOS_DIR, f"{video_id}.srt"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate) and os.path.getsize(candidate) > 100:
            return candidate

    root_dir = os.path.dirname(VIDEOS_DIR)
    for fname in os.listdir(root_dir):
        if video_id in fname and fname.endswith((".vtt", ".srt")):
            p = os.path.join(root_dir, fname)
            if os.path.getsize(p) > 100:
                return p
    return None

def get_video_status(video_id: str) -> Dict[str, Any]:
    """Returns the download and indexing status of a Video video."""
    meta = KNOWN_VIDEOS.get(video_id, {
        "video_id": video_id,
        "title": f"Video Video [{video_id}]",
        "url": f"https://example.com/watch?v={video_id}",
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
    output_dir = os.path.join(os.path.dirname(VIDEOS_DIR), f"outputs_{video_id}")
    index_file = os.path.join(output_dir, "vss_search_index.json")
    spec_file = os.path.join(output_dir, "trading_strategy_spec.json")

    has_index = os.path.exists(index_file)
    has_spec = os.path.exists(spec_file)

    indexed_chunks = 0
    if has_index:
        try:
            with open(index_file, "r", encoding="utf-8") as f:
                d = json.load(f)
                indexed_chunks = d.get("total_chunks", len(d.get("chunks", [])))
        except Exception:
            pass

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

def list_available_videos() -> List[Dict[str, Any]]:
    """Lists status for all supported Video trading videos."""
    return [get_video_status(vid) for vid in KNOWN_VIDEOS]

def download_video(video_id: str) -> Dict[str, Any]:
    """
    Downloads Video video & subtitles using yt-dlp.
    Rules:
    - Highest quality possible for video only (excluding premium).
    - Downloads subtitles.
    """
    if video_id not in KNOWN_VIDEOS:
        raise ValueError(f"Unknown video ID: {video_id}. Allowed: {list(KNOWN_VIDEOS.keys())}")

    target_video = os.path.join(VIDEOS_DIR, f"{video_id}.mp4")
    target_sub = os.path.join(VIDEOS_DIR, f"{video_id}.vtt")

    # If both exist and have healthy sizes, reuse
    if os.path.exists(target_video) and os.path.getsize(target_video) > 1024 * 1024 and \
       os.path.exists(target_sub) and os.path.getsize(target_sub) > 500:
        logger.info(f"Video {video_id} and subtitles already downloaded.")
        return get_video_status(video_id)

    url = f"https://example.com/watch?v={video_id}"

    # 1. Download Subtitles first if missing
    if not os.path.exists(target_sub) or os.path.getsize(target_sub) < 500:
        logger.info(f"Downloading subtitles for {video_id}...")
        sub_cmd = [
            "yt-dlp",
            "--write-auto-subs",
            "--write-subs",
            "--sub-langs", "en.*,en",
            "--skip-download",
            "--extractor-args", "video:player_client=android,web",
            "--no-playlist",
            "-o", os.path.join(VIDEOS_DIR, f"subtitles_{video_id}.%(ext)s"),
            url
        ]
        res = subprocess.run(sub_cmd, capture_output=True, text=True)
        # Check generated files and normalize to target_sub
        for f in os.listdir(VIDEOS_DIR):
            if f.startswith(f"subtitles_{video_id}") and f.endswith(".vtt"):
                shutil.copyfile(os.path.join(VIDEOS_DIR, f), target_sub)
                break

    # 2. Download Highest Quality Video-Only (Non-Premium)
    if not os.path.exists(target_video) or os.path.getsize(target_video) < 1024 * 1024:
        logger.info(f"Downloading highest quality video for {video_id}...")
        format_selector = "bestvideo[format_note!*=Premium][format_note!*=premium][ext=mp4]/137/299/623/bestvideo"
        vid_cmd = [
            "yt-dlp",
            "-f", format_selector,
            "--extractor-args", "video:player_client=android,web",
            "--no-playlist",
            "-o", target_video,
            url
        ]
        res = subprocess.run(vid_cmd, capture_output=True, text=True)
        if res.returncode != 0:
            logger.warning(f"yt-dlp video download returned code {res.returncode}: {res.stderr}")

    return get_video_status(video_id)
