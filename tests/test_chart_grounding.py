"""
Unit tests for Hybrid Visual Grounding Pipeline.
"""

import pytest
from PIL import Image
from schemas import VideoChunk, BoundingBox
from chart_grounding import (
    PriceCoordinateMapper,
    ChartLayoutSegmenter,
    DeicticGrounder,
    HybridChartGroundingPipeline
)
from vlm_reasoner import VSSVLMReasoner

def test_price_coordinate_mapper_conversions():
    mapper = PriceCoordinateMapper(min_price=100.0, max_price=200.0, canvas_ymin=0.10, canvas_ymax=0.70)
    assert mapper.price_range == 100.0

    # Top of canvas (ymin=0.10) maps to max_price (200.0)
    top_price = mapper.y_to_price(0.10)
    assert top_price == 200.0

    # Bottom of canvas (ymax=0.70) maps to min_price (100.0)
    bot_price = mapper.y_to_price(0.70)
    assert bot_price == 100.0

    # Midpoint of canvas (0.40) maps to mid price (150.0)
    mid_price = mapper.y_to_price(0.40)
    assert mid_price == 150.0

    # Reverse mapping: 150.0 maps back to 0.40
    y_norm = mapper.price_to_y(150.0)
    assert abs(y_norm - 0.40) < 1e-3

    # Clamping behavior
    assert mapper.y_to_price(0.0) == 200.0
    assert mapper.y_to_price(1.0) == 100.0

def test_price_to_bbox():
    mapper = PriceCoordinateMapper(min_price=100.0, max_price=200.0)
    bbox = mapper.price_to_bbox(150.0, height_padding=0.04)
    assert isinstance(bbox, BoundingBox)
    assert 0.0 <= bbox.ymin < bbox.ymax <= 1.0
    assert 0.0 <= bbox.xmin < bbox.xmax <= 1.0
    # Pixel conversion
    pixels = bbox.to_pixels(1280, 720)
    assert len(pixels) == 4
    assert pixels[0] < pixels[2]  # left < right
    assert pixels[1] < pixels[3]  # top < bottom

def test_chart_layout_segmentation():
    segmenter = ChartLayoutSegmenter()
    regions = segmenter.segment()
    assert len(regions) >= 5
    region_types = {r.region_type for r in regions}
    assert "header_ticker" in region_types
    assert "price_axis" in region_types
    assert "time_axis" in region_types
    assert "main_canvas" in region_types
    assert "indicator_pane" in region_types

    for r in regions:
        assert 0.0 <= r.bounding_box.ymin <= 1.0
        assert 0.0 <= r.bounding_box.xmin <= 1.0
        assert 0.0 <= r.bounding_box.ymax <= 1.0
        assert 0.0 <= r.bounding_box.xmax <= 1.0

def test_deictic_grounder_anchoring():
    mapper = PriceCoordinateMapper(min_price=150.0, max_price=220.0)
    pipeline = HybridChartGroundingPipeline()
    transcript = "Notice how price broke above this level at 195.50 and our stop loss is down here below 160.00 in the buy zone."
    levels = pipeline.extract_price_levels(transcript, mapper)
    grounder = DeicticGrounder()
    anchors = grounder.ground_transcript(transcript, mapper, levels)

    assert len(anchors) >= 1
    # Check that prices are within range
    for a in anchors:
        if a.grounded_price is not None:
            assert 150.0 <= a.grounded_price <= 220.0

def test_hybrid_grounding_pipeline_chunk_integration():
    pipeline = HybridChartGroundingPipeline()
    chunk = VideoChunk(
        chunk_id=1,
        start_s=10.0,
        end_s=30.0,
        duration_s=20.0,
        start_str="00:10",
        end_str="00:30",
        sampled_frame_timestamps=[15.0, 25.0],
        synchronized_transcript="ES chart Daily timeframe. We have a clear breakout above 5400 and stop below 5320 with RSI divergence."
    )
    dummy_frame = Image.new("RGB", (640, 360), color=(30, 30, 30))
    grounding = pipeline.ground_chunk(chunk, [dummy_frame])

    assert grounding.ticker == "ES"
    assert "Daily" in grounding.timeframe
    assert grounding.price_range is not None
    assert len(grounding.detected_price_levels) >= 2
    assert len(grounding.chart_regions) >= 5
    assert "rsi" in grounding.indicator_summary

    prompt_block = pipeline.format_grounding_prompt_block(grounding)
    assert "HYBRID VISUAL GROUNDING ANCHORS" in prompt_block
    assert "ES" in prompt_block
    assert "Daily" in prompt_block

    caption = "Bullish breakout observed."
    enriched = pipeline.enrich_caption_with_grounding(caption, grounding)
    assert "Grounded Coordinates:" in enriched

def test_vlm_reasoner_with_grounding():
    reasoner = VSSVLMReasoner()
    chunk = VideoChunk(
        chunk_id=0,
        start_s=0.0,
        end_s=20.0,
        duration_s=20.0,
        start_str="00:00",
        end_str="00:20",
        sampled_frame_timestamps=[5.0, 15.0],
        synchronized_transcript="Looking at BTC on the 4H chart. We see a retest of the buy zone at 62000."
    )
    dummy_frame = Image.new("RGB", (320, 240), color=(10, 10, 10))
    response = reasoner.reason_over_chunk(chunk, [dummy_frame])

    assert response.grounding is not None
    assert response.grounding.ticker == "BTC"
    assert "4H" in response.grounding.timeframe
    assert "grounded_coordinates" in response.visual_observations
    assert response.visual_observations["grounded_coordinates"]["ticker"] == "BTC"
    assert "Grounded Coordinates:" in response.content
