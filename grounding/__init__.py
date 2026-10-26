"""
Deterministic Pixel Computer Vision and Chart Grounding Engine.
Provides pixel-precise layout segmentation, line and zone detection,
axis calibration with regression fitting, and VLM crop reading.
"""

from .layout import Layout, detect_layout, crop
from .drawings import (
    HLine, Zone, PositionBox, Crosshair,
    find_hlines, find_zones, find_position_tools, find_crosshair, find_date_label
)
from .axis import (
    TextLine, TagSegment, AxisPixels, AxisCalibration, TagReading,
    analyze_axis_pixels, parse_axis_read, find_value_lattice, calibrate, cross_check_tags
)
from .ground import (
    GroundedLevel, PositionTool, GroundedFrame, ChartContext, ground_frame
)
from .viz import draw_overlay

__all__ = [
    "Layout", "detect_layout", "crop",
    "HLine", "Zone", "PositionBox", "Crosshair",
    "find_hlines", "find_zones", "find_position_tools", "find_crosshair", "find_date_label",
    "TextLine", "TagSegment", "AxisPixels", "AxisCalibration", "TagReading",
    "analyze_axis_pixels", "parse_axis_read", "find_value_lattice", "calibrate", "cross_check_tags",
    "GroundedLevel", "PositionTool", "GroundedFrame", "ChartContext", "ground_frame",
    "draw_overlay"
]
