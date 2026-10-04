import pytest
from unittest.mock import MagicMock
from chunk_engine import VideoChunkManager
from subtitle_sync import SubtitleSyncManager
from schemas import SubtitleEntry

MODULE_PREFIX = "chunk_engine"

def test_chunk_generation_boundaries(monkeypatch):
    monkeypatch.setattr(
        f"{MODULE_PREFIX}.get_video_duration_ffprobe",
        lambda path: 60.0
    )

    mock_sub_mgr = MagicMock(spec=SubtitleSyncManager)
    mock_sub_mgr.get_entries_in_window.return_value = []
    mock_sub_mgr.format_transcript_window.return_value = "Sample transcript"

    mgr = VideoChunkManager(
        video_path="dummy.mp4",
        subtitle_mgr=mock_sub_mgr,
        chunk_duration_s=20.0,
        chunk_overlap_s=5.0,
        frames_per_chunk=3
    )
    chunks = mgr.generate_chunks()

    # Step size is 20 - 5 = 15s. Total duration = 60s
    # Chunks start at 0, 15, 30, 45
    assert len(chunks) == 4
    assert chunks[0].start_s == 0.0
    assert chunks[0].end_s == 20.0
    assert len(chunks[0].sampled_frame_timestamps) == 3

    assert chunks[1].start_s == 15.0
    assert chunks[1].end_s == 35.0

    assert chunks[3].start_s == 45.0
    assert chunks[3].end_s == 60.0
