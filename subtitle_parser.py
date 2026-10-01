"""
Subtitle Parser for Trading Video Understanding POC.
Parses subtitle blocks with timestamps in MM:SS format into structured intervals
relative to the start of the video clip (18:39).
"""

import os
from typing import List, Dict, Any

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
    clip_start_str: str = "18:39",
    clip_end_str: str = "21:16"
) -> List[Dict[str, Any]]:
    """
    Parses subtitle file where lines alternate between timestamp and transcript text.
    Computes both original video timestamp and clip-relative timestamp (in seconds).
    """
    clip_origin_s = time_str_to_seconds(clip_start_str)
    clip_end_s = time_str_to_seconds(clip_end_str)
    
    entries = []
    with open(subtitle_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

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
            
            orig_s = time_str_to_seconds(ts_str)
            clip_s = max(0.0, orig_s - clip_origin_s)
            
            text = " ".join(text_lines)
            entries.append({
                "index": len(entries),
                "orig_timestamp_str": ts_str,
                "orig_timestamp_s": orig_s,
                "clip_timestamp_s": clip_s,
                "clip_timestamp_str": seconds_to_time_str(clip_s),
                "text": text
            })
        else:
            i += 1

    # Compute end times for each entry based on the next entry or clip end
    for idx in range(len(entries)):
        if idx + 1 < len(entries):
            entries[idx]["clip_end_s"] = entries[idx + 1]["clip_timestamp_s"]
            entries[idx]["orig_end_s"] = entries[idx + 1]["orig_timestamp_s"]
        else:
            entries[idx]["clip_end_s"] = clip_end_s - clip_origin_s
            entries[idx]["orig_end_s"] = clip_end_s
        entries[idx]["duration_s"] = entries[idx]["clip_end_s"] - entries[idx]["clip_timestamp_s"]

    return entries

if __name__ == "__main__":
    sub_path = "subtitles_clip"
    if os.path.exists(sub_path):
        parsed = parse_subtitles(sub_path)
        print(f"Parsed {len(parsed)} subtitle segments:")
        for e in parsed:
            print(f"[{e['orig_timestamp_str']} / {e['clip_timestamp_str']}] ({e['duration_s']:.1f}s) {e['text']}")
