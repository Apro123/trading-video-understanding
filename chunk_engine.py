"""
Video Chunking and Frame Extraction Engine.
Implements the chunking specification of NVIDIA Video Search and Summarization (VSS).
Extracts video frames via FFmpeg streaming pipes with native support for AV1, VP9, and H.264.
"""

import os
import io
import subprocess
import logging
from typing import List, Optional, Tuple
from PIL import Image

logger = logging.getLogger(__name__)

try:
    from .schemas import VideoChunk
    from .subtitle_sync import SubtitleSyncManager, seconds_to_time_str
except (ImportError, ValueError):
    from schemas import VideoChunk
    from subtitle_sync import SubtitleSyncManager, seconds_to_time_str

def get_video_duration_ffprobe(video_path: str) -> float:
    """Retrieves exact video duration in seconds via ffprobe."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode("utf-8").strip()
        return float(out)
    except Exception as e:
        logger.warning(f"ffprobe failed to get duration: {e}. Fallback to 120s.")
        return 120.0

def extract_frame_ffmpeg(
    video_path: str,
    timestamp_s: float,
    max_dimension: int = 1280
) -> Optional[Image.Image]:
    """
    Extracts a single RGB frame from a video file at a given timestamp using an in-memory FFmpeg pipe.
    Supports arbitrary codecs including AV1, VP9, and H.264.
    """
    try:
        cmd = [
            "ffmpeg", "-ss", f"{timestamp_s:.3f}",
            "-i", video_path,
            "-vframes", "1",
            "-f", "image2pipe",
            "-vcodec", "png",
            "-"
        ]
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=12
        )
        if proc.returncode == 0 and len(proc.stdout) > 0:
            img = Image.open(io.BytesIO(proc.stdout)).convert("RGB")
            # Downscale proportionally if larger than max_dimension
            w, h = img.size
            if max(w, h) > max_dimension:
                scale = max_dimension / float(max(w, h))
                new_size = (int(w * scale), int(h * scale))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            return img
    except Exception as e:
        logger.warning(f"Frame extraction failed at {timestamp_s:.2f}s: {e}")
    return None

class VideoChunkManager:
    """
    Generates video chunks aligned with temporal windows and synchronized subtitles.
    """

    def __init__(
        self,
        video_path: str,
        subtitle_mgr: SubtitleSyncManager,
        chunk_duration_s: float = 20.0,
        chunk_overlap_s: float = 3.0,
        frames_per_chunk: int = 4,
        max_dimension: int = 1280
    ):
        self.video_path = video_path
        self.subtitle_mgr = subtitle_mgr
        self.chunk_duration_s = max(5.0, chunk_duration_s)
        self.chunk_overlap_s = max(0.0, min(chunk_overlap_s, self.chunk_duration_s - 1.0))
        self.frames_per_chunk = max(1, frames_per_chunk)
        self.max_dimension = max_dimension
        self.video_duration_s = get_video_duration_ffprobe(video_path)

    def generate_chunks(self) -> List[VideoChunk]:
        """
        Slices the video timeline into overlapping chunks, associating each chunk
        with its synchronized subtitles and frame sample timestamps.
        """
        chunks: List[VideoChunk] = []
        step_s = self.chunk_duration_s - self.chunk_overlap_s
        start_s = 0.0
        chunk_idx = 0

        while start_s < self.video_duration_s:
            end_s = min(self.video_duration_s, start_s + self.chunk_duration_s)
            dur_s = end_s - start_s

            # Calculate frame sampling timestamps within the chunk
            if self.frames_per_chunk == 1:
                frame_ts = [start_s + (dur_s / 2.0)]
            else:
                step = dur_s / float(self.frames_per_chunk + 1)
                frame_ts = [round(start_s + step * (i + 1), 2) for i in range(self.frames_per_chunk)]

            # Synchronize subtitles
            matched_subs = self.subtitle_mgr.get_entries_in_window(start_s, end_s)
            transcript_text = self.subtitle_mgr.format_transcript_window(start_s, end_s)

            chunk = VideoChunk(
                chunk_id=chunk_idx,
                start_s=round(start_s, 2),
                end_s=round(end_s, 2),
                duration_s=round(dur_s, 2),
                start_str=seconds_to_time_str(start_s),
                end_str=seconds_to_time_str(end_s),
                sampled_frame_timestamps=frame_ts,
                subtitle_entries=matched_subs,
                synchronized_transcript=transcript_text
            )
            chunks.append(chunk)

            chunk_idx += 1
            start_s += step_s

            # Guard against tiny trailing sliver
            if self.video_duration_s - start_s < 2.0:
                break

        return chunks

    def extract_chunk_frames(self, chunk: VideoChunk) -> List[Image.Image]:
        """Extracts PIL Images for all sampled frames in a chunk."""
        frames = []
        for ts in chunk.sampled_frame_timestamps:
            img = extract_frame_ffmpeg(self.video_path, ts, max_dimension=self.max_dimension)
            if img is not None:
                frames.append(img)
        return frames
