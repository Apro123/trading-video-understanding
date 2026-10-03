"""
Subtitle Synchronization and Dynamic Alignment Engine.
Supports clipping, dynamic timestamp anchoring (0:00 normalization),
temporal interval overlap matching, and deictic token detection.
"""

import os
import re
from typing import List, Dict, Any, Optional, Tuple
try:
    from .schemas import SubtitleEntry
except (ImportError, ValueError):
    from schemas import SubtitleEntry

DEICTIC_KEYWORDS = [
    "here", "this candle", "this level", "this point", "boom boom",
    "at this particular point", "right here", "broke above", "buy zone",
    "swing high", "swing low", "this line", "retest", "look at this", "in between"
]

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

class SubtitleSyncManager:
    """
    Manages parsing, timestamp normalization, and temporal chunk synchronization
    for arbitrary trading clips and subtitle transcripts.
    """

    def __init__(self, subtitle_file: str, clip_duration_s: Optional[float] = None):
        self.subtitle_file = subtitle_file
        self.clip_duration_s = clip_duration_s
        self.entries: List[SubtitleEntry] = []
        self._origin_offset_s: float = 0.0
        self.load_and_normalize()

    def load_and_normalize(self) -> List[SubtitleEntry]:
        """Parses the raw subtitle file and normalizes timestamps relative to 0:00."""
        if not os.path.exists(self.subtitle_file):
            raise FileNotFoundError(f"Subtitle file not found: {self.subtitle_file}")

        with open(self.subtitle_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        raw_blocks: List[Tuple[str, str]] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            # Detect timestamp lines like "18:40" or "01:18:40"
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

        # First timestamp is the anchor origin (e.g. 18:40 or 39:46)
        self._origin_offset_s = time_str_to_seconds(raw_blocks[0][0])

        parsed_entries = []
        for idx, (ts_str, text) in enumerate(raw_blocks):
            orig_s = time_str_to_seconds(ts_str)
            clip_s = max(0.0, orig_s - self._origin_offset_s)
            parsed_entries.append({
                "index": idx,
                "orig_timestamp_str": ts_str,
                "orig_timestamp_s": orig_s,
                "clip_timestamp_s": clip_s,
                "clip_timestamp_str": seconds_to_time_str(clip_s),
                "clip_end_s": clip_s + 5.0,  # placeholder, updated below
                "duration_s": 5.0,
                "text": text
            })

        # Calculate accurate end times and durations
        for idx in range(len(parsed_entries)):
            if idx + 1 < len(parsed_entries):
                parsed_entries[idx]["clip_end_s"] = parsed_entries[idx + 1]["clip_timestamp_s"]
            else:
                if self.clip_duration_s is not None and self.clip_duration_s > parsed_entries[idx]["clip_timestamp_s"]:
                    parsed_entries[idx]["clip_end_s"] = self.clip_duration_s
                else:
                    parsed_entries[idx]["clip_end_s"] = parsed_entries[idx]["clip_timestamp_s"] + 8.0

            dur = max(1.0, parsed_entries[idx]["clip_end_s"] - parsed_entries[idx]["clip_timestamp_s"])
            parsed_entries[idx]["duration_s"] = dur

        self.entries = [SubtitleEntry(**item) for item in parsed_entries]
        return self.entries

    def get_entries_in_window(self, start_s: float, end_s: float) -> List[SubtitleEntry]:
        """Returns all subtitle entries overlapping with [start_s, end_s]."""
        overlapping = []
        for e in self.entries:
            if e.clip_timestamp_s < end_s and e.clip_end_s > start_s:
                overlapping.append(e)
        return overlapping

    def format_transcript_window(self, start_s: float, end_s: float) -> str:
        """Formats the spoken transcript within [start_s, end_s] for VLM prompts."""
        matching = self.get_entries_in_window(start_s, end_s)
        if not matching:
            return ""
        formatted = []
        for e in matching:
            formatted.append(f"[{seconds_to_time_str(e.clip_timestamp_s)} - {seconds_to_time_str(e.clip_end_s)}] {e.text}")
        return "\n".join(formatted)

    def extract_deictic_tokens(self, text: str) -> List[str]:
        """Identifies deictic or trading references requiring visual grounding."""
        detected = []
        lower = text.lower()
        for kw in DEICTIC_KEYWORDS:
            if re.search(r'\b' + re.escape(kw) + r'\b', lower):
                detected.append(kw)
        return list(set(detected))
