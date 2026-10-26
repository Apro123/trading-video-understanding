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
    """Converts MM:SS, HH:MM:SS, or timestamps with fractional seconds to total seconds."""
    t_clean = t_str.strip().replace(",", ".")
    parts = t_clean.split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    elif len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    raise ValueError(f"Invalid timestamp format: {t_str}")

def seconds_to_time_str(secs: float) -> str:
    """Converts seconds to MM:SS or HH:MM:SS format."""
    secs_int = int(secs)
    h = secs_int // 3600
    m = (secs_int % 3600) // 60
    s = secs_int % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

class SubtitleSyncManager:
    """
    Manages parsing, timestamp normalization, and temporal chunk synchronization
    for arbitrary trading clips, full Video videos, WebVTT, and SRT transcripts.
    """

    def __init__(self, subtitle_file: str, clip_duration_s: Optional[float] = None):
        self.subtitle_file = subtitle_file
        self.clip_duration_s = clip_duration_s
        self.entries: List[SubtitleEntry] = []
        self._origin_offset_s: float = 0.0
        self.load_and_normalize()

    def load_and_normalize(self) -> List[SubtitleEntry]:
        """Parses the raw subtitle file (WebVTT, SRT, or legacy format) and normalizes timestamps."""
        if not os.path.exists(self.subtitle_file):
            raise FileNotFoundError(f"Subtitle file not found: {self.subtitle_file}")

        with open(self.subtitle_file, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        raw_blocks: List[Tuple[float, float, str, str]] = []

        # 1. Check for WebVTT or SRT "-->" timestamp arrows
        vtt_pattern = re.compile(
            r"((?:\d{1,2}:)?\d{2}:\d{2}[\.,]\d{1,3})\s*-->\s*((?:\d{1,2}:)?\d{2}:\d{2}[\.,]\d{1,3})[^\n]*\n([\s\S]*?)(?=\n\s*(?:(?:\d{1,2}:)?\d{2}:\d{2}[\.,]\d{1,3}\s*-->|\Z))",
            re.MULTILINE
        )
        matches = list(vtt_pattern.finditer(content))
        if matches:
            last_plain = None
            for m in matches:
                start_s = time_str_to_seconds(m.group(1))
                end_s = time_str_to_seconds(m.group(2))
                raw_text = m.group(3)
                body = [l.strip() for l in raw_text.splitlines() if l.strip()]
                if not body:
                    continue

                # Handle Video rolling duplicate lines with word-level <c> tags
                tagged = [l for l in body if "<c>" in l]
                if tagged:
                    # The active newly spoken line has <c> tags
                    line = tagged[-1]
                    clean_text = re.sub(r"<[^>]+>", "", line).strip()
                else:
                    # Pick the newest line from the cue body
                    clean_text = re.sub(r"<[^>]+>", "", body[-1]).strip()

                if clean_text and clean_text != last_plain:
                    raw_blocks.append((start_s, end_s, seconds_to_time_str(start_s), clean_text))
                    last_plain = clean_text

        # 2. Fallback to legacy format: "18:40 \n text"
        if not raw_blocks:
            lines = [line.strip() for line in content.splitlines() if line.strip()]
            i = 0
            while i < len(lines):
                line = lines[i]
                if ":" in line and all(p.replace(".", "").isdigit() for p in line.split(":")):
                    ts_str = line
                    text_lines = []
                    i += 1
                    while i < len(lines) and not (":" in lines[i] and all(p.replace(".", "").isdigit() for p in lines[i].split(":"))):
                        text_lines.append(lines[i])
                        i += 1
                    try:
                        start_s = time_str_to_seconds(ts_str)
                        raw_blocks.append((start_s, start_s + 5.0, ts_str, " ".join(text_lines)))
                    except Exception:
                        pass
                else:
                    i += 1

        if not raw_blocks:
            return []

        # If clip starts with an offset (e.g. 18:40) and clip_duration_s is shorter than total offset,
        # normalize to 0:00. If it is a full video or starts near 0, offset is 0.0.
        first_start = raw_blocks[0][0]
        if self.clip_duration_s and self.clip_duration_s < first_start:
            self._origin_offset_s = first_start
        elif first_start > 60.0 and self.clip_duration_s is None:
            # If the first subtitle starts past 1 minute and no clip duration is specified,
            # this indicates an isolated clip excerpt anchored at original video time.
            self._origin_offset_s = first_start
        else:
            self._origin_offset_s = 0.0

        parsed_entries = []
        for idx, (start_s, end_s, orig_ts_str, text) in enumerate(raw_blocks):
            clip_start_s = max(0.0, start_s - self._origin_offset_s)
            clip_end_s = max(clip_start_s + 1.0, end_s - self._origin_offset_s)
            parsed_entries.append({
                "index": idx,
                "orig_timestamp_str": orig_ts_str,
                "orig_timestamp_s": start_s,
                "clip_timestamp_s": clip_start_s,
                "clip_timestamp_str": seconds_to_time_str(clip_start_s),
                "clip_end_s": clip_end_s,
                "duration_s": max(0.5, clip_end_s - clip_start_s),
                "text": text
            })

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
