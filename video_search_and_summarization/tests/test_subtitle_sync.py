import pytest
import os
import tempfile

from video_search_and_summarization.subtitle_sync import (
    SubtitleSyncManager,
    time_str_to_seconds,
    seconds_to_time_str
)

SAMPLE_SUBTITLE = """18:40
first subtitle block describing boom boom
18:46
second block practice market structure
19:11
price broke above here at 1927
19:42
January 30th 2023 Nvidia went bullish
"""

def test_time_conversions():
    assert time_str_to_seconds("00:00") == 0.0
    assert time_str_to_seconds("18:40") == 1120.0
    assert time_str_to_seconds("01:18:40") == 4720.0
    assert seconds_to_time_str(1120.0) == "18:40"
    assert seconds_to_time_str(0.0) == "00:00"

def test_subtitle_sync_normalization():
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt") as f:
        f.write(SAMPLE_SUBTITLE)
        tmp_path = f.name

    try:
        mgr = SubtitleSyncManager(tmp_path, clip_duration_s=100.0)
        entries = mgr.entries
        assert len(entries) == 4

        # Verify 0:00 anchor
        assert entries[0].orig_timestamp_str == "18:40"
        assert entries[0].orig_timestamp_s == 1120.0
        assert entries[0].clip_timestamp_s == 0.0
        assert entries[0].clip_timestamp_str == "00:00"

        # Verify second entry offset (18:46 - 18:40 = 6s)
        assert entries[1].orig_timestamp_str == "18:46"
        assert entries[1].clip_timestamp_s == 6.0
        assert entries[1].clip_timestamp_str == "00:06"

        # Window matching
        window = mgr.get_entries_in_window(0.0, 10.0)
        assert len(window) == 2
        assert window[0].index == 0
        assert window[1].index == 1

        # Deictic token extraction
        tokens = mgr.extract_deictic_tokens(entries[2].text)
        assert "here" in tokens
        assert "broke above" in tokens
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
