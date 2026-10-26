"""
Unit tests for deterministic pixel computer vision grounding and caching.
"""

import pytest
import numpy as np
import cv2

from cache import DiskCache, stable_hash, extract_json
from grounding.layout import detect_layout, Layout, crop
from grounding.drawings import find_hlines, find_zones, find_position_tools, Zone
from grounding.axis import (
    find_value_lattice, calibrate, AxisPixels, TextLine, TagSegment, cross_check_tags
)
from grounding.ground import ground_frame, GroundedFrame
from grounding.viz import draw_overlay


def test_disk_cache_and_stable_hash():
    cache = DiskCache("test_ns")
    k = stable_hash("test_prompt", {"foo": "bar"}, 123)
    assert len(k) == 16

    data = {"status": "success", "val": 42.5}
    cache.set(k, data)

    cached = cache.get(k)
    assert cached is not None
    assert cached["val"] == 42.5
    assert cached["status"] == "success"


def test_extract_json_fenced_and_raw():
    fenced = '```json\n{"strategy": "breakout", "target": 65.5}\n```'
    parsed = extract_json(fenced)
    assert parsed["strategy"] == "breakout"
    assert parsed["target"] == 65.5

    prose = 'Here is the result: {"bias": "bullish"} thank you.'
    parsed_prose = extract_json(prose)
    assert parsed_prose["bias"] == "bullish"


def test_pixel_layout_detection():
    # Synthetic 1280x720 chart canvas
    img = np.full((720, 1280, 3), 245, dtype=np.uint8)
    # Add vertical divider for price axis at x=1050
    img[:, 1050:1052] = 0
    # Add horizontal divider for date axis at y=640
    img[640:642, :1050] = 0

    lay = detect_layout(img, t=0.0)
    assert isinstance(lay, Layout)
    assert lay.width == 1280
    assert lay.height == 720
    assert lay.theme in ("light", "dark")
    assert lay.price_axis[0] > 0


def test_find_value_lattice():
    # Evenly spaced price ticks
    values = [70.0, 65.0, 60.0, 55.0, 50.0]
    lat = find_value_lattice(values)
    assert lat is not None
    step, anchor, members = lat
    assert abs(step - 5.0) < 1e-4
    assert len(members) == 5


def test_axis_calibration_linear():
    # Grid bands at y = 100, 200, 300, 400
    bands = [
        TextLine(y=100.0, y0=95, y1=105),
        TextLine(y=200.0, y0=195, y1=205),
        TextLine(y=300.0, y0=295, y1=305),
        TextLine(y=400.0, y0=395, y1=405)
    ]
    pix = AxisPixels(grid_bands=bands, tags=[])
    values = [100.0, 80.0, 60.0, 40.0]  # tick step = 20, corresponding to dy = 100 px

    lay = Layout(
        width=1280, height=720, theme="light",
        pane=(50, 50, 1050, 640),
        price_axis=(1052, 50, 1150, 640),
        date_axis=(50, 640, 1050, 680),
        header=(50, 50, 500, 75),
        symbol_box=(48, 0, 350, 34),
        bg=(245, 245, 245)
    )

    cal, issues = calibrate(0.0, pix, values, lay)
    assert cal is not None
    assert cal.ok is True
    assert cal.scale == "linear"
    assert cal.r2 > 0.999

    # Test inverse and price calculation
    p100 = cal.price(100.0)
    assert abs(p100 - 100.0) < 1.0
    p400 = cal.price(400.0)
    assert abs(p400 - 40.0) < 1.0


def test_find_position_tools():
    # Long position: green zone above red zone sharing horizontal edge at y=300
    green_zone = Zone(color="green", x0=200, y0=200, x1=400, y1=300, fill=0.9)
    red_zone = Zone(color="red", x0=200, y0=300, x1=400, y1=400, fill=0.9)

    tools = find_position_tools([green_zone, red_zone])
    assert len(tools) == 1
    pt = tools[0]
    assert pt.direction == "long"
    assert pt.entry_y == 300.0
    assert pt.stop_y == 400.0
    assert pt.target_y == 200.0
    assert pt.edges_aligned is True


def test_ground_frame_and_draw_overlay():
    img = np.full((720, 1280, 3), 240, dtype=np.uint8)
    gf = ground_frame(img, t=10.0, video_id="test_video")
    assert isinstance(gf, GroundedFrame)
    assert gf.t == 10.0

    rendered = draw_overlay(img, gf)
    assert rendered.shape == img.shape
