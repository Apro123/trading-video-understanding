"""
Subtitle Parser for Trading Video Understanding.
Parses subtitle blocks with timestamps in MM:SS or HH:MM:SS format into structured intervals.
Automatically calculates relative time offsets by subtracting the first timestamp
so that the clip starts at 0:00.
"""

import os
from typing import List, Dict, Any, Optional

def time_str_to_seconds(t_str: str) -> float:
    """Converts MM:SS or HH:MM:SS to total seconds."""
    parts = t_str.strip().split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    elif len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    raise ValueError(f"Invalid timestamp format: {t_str}")

def seconds_to_time_str(secs: float) -> str:
    """Converts seconds to MM:SS format."""
    m = int(secs) // 60
    s = int(secs) % 60
    return f"{m:02d}:{s:02d}"

def parse_subtitles(
    subtitle_file: str,
    clip_start_str: Optional[str] = None,
    clip_duration_s: Optional[float] = None
) -> List[Dict[str, Any]]:
    """
    Parses subtitle file where lines alternate between timestamp and transcript text.
    If clip_start_str is not provided, automatically detects the first timestamp
    in the file as the 0:00 starting point.
    """
    if not os.path.exists(subtitle_file):
        raise FileNotFoundError(f"Subtitle file not found: {subtitle_file}")

    entries = []
    with open(subtitle_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    raw_blocks = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # Check if line is a timestamp (MM:SS)
        if ":" in line and all(p.isdigit() for p in line.split(":")):
            ts_str = line
            text_lines = []
            i += 1
            while i < len(lines) and not (":" in lines[i] and all(p.isdigit() for p in lines[i].split(":"))):
                text_lines.append(lines[i])
                i += 1
            raw_blocks.append((ts_str, " ".join(text_lines)))
        else:
            i += 1

    if not raw_blocks:
        return []

    # Determine origin timestamp
    if clip_start_str is not None:
        clip_origin_s = time_str_to_seconds(clip_start_str)
    else:
        # Automatically use first timestamp in file as 00:00 anchor
        clip_origin_s = time_str_to_seconds(raw_blocks[0][0])

    for idx, (ts_str, text) in enumerate(raw_blocks):
        orig_s = time_str_to_seconds(ts_str)
        clip_s = max(0.0, orig_s - clip_origin_s)
        entries.append({
            "index": idx,
            "orig_timestamp_str": ts_str,
            "orig_timestamp_s": orig_s,
            "clip_timestamp_s": clip_s,
            "clip_timestamp_str": seconds_to_time_str(clip_s),
            "text": text
        })

    # Compute end times for each entry
    for idx in range(len(entries)):
        if idx + 1 < len(entries):
            entries[idx]["clip_end_s"] = entries[idx + 1]["clip_timestamp_s"]
            entries[idx]["orig_end_s"] = entries[idx + 1]["orig_timestamp_s"]
        else:
            if clip_duration_s is not None and clip_duration_s > entries[idx]["clip_timestamp_s"]:
                entries[idx]["clip_end_s"] = clip_duration_s
                entries[idx]["orig_end_s"] = entries[idx]["orig_timestamp_s"] + (clip_duration_s - entries[idx]["clip_timestamp_s"])
            else:
                entries[idx]["clip_end_s"] = entries[idx]["clip_timestamp_s"] + 8.0
                entries[idx]["orig_end_s"] = entries[idx]["orig_timestamp_s"] + 8.0
        entries[idx]["duration_s"] = max(1.0, entries[idx]["clip_end_s"] - entries[idx]["clip_timestamp_s"])

    return entries

if __name__ == "__main__":
    for f in ["subtitles_clip", "subtitles_clip_2"]:
        if os.path.exists(f):
            print(f"=== Testing {f} ===")
            res = parse_subtitles(f)
            print(f"Parsed {len(res)} entries. First: {res[0]['orig_timestamp_str']} -> {res[0]['clip_timestamp_str']}, Last: {res[-1]['orig_timestamp_str']} -> {res[-1]['clip_timestamp_str']}")
