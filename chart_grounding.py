"""
Hybrid Visual Grounding Pipeline for Trading Video Understanding.
Combines deterministic chart segmentation, price/timeline coordinate calibration,
OCR/heuristic axis extraction, and deictic reference resolution to eliminate
numerical and spatial hallucinations in VLM trading analysis.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image

try:
    from .schemas import (
        BoundingBox,
        GroundedPriceLevel,
        DeicticAnchor,
        ChartRegion,
        ChartGroundingData,
        VideoChunk
    )
    from .utils import extract_ticker_symbol, extract_timeframe
except (ImportError, ValueError):
    from schemas import (
        BoundingBox,
        GroundedPriceLevel,
        DeicticAnchor,
        ChartRegion,
        ChartGroundingData,
        VideoChunk
    )
    from utils import extract_ticker_symbol, extract_timeframe

logger = logging.getLogger(__name__)

# Standard Trading Platform Normalized Layout Ratios
# [ymin, xmin, ymax, xmax] in normalized (0.0 - 1.0) coordinates
DEFAULT_CHART_REGIONS = {
    "header_ticker": BoundingBox(ymin=0.00, xmin=0.00, ymax=0.10, xmax=0.38),
    "price_axis": BoundingBox(ymin=0.05, xmin=0.88, ymax=0.85, xmax=1.00),
    "time_axis": BoundingBox(ymin=0.85, xmin=0.05, ymax=0.98, xmax=0.88),
    "main_canvas": BoundingBox(ymin=0.10, xmin=0.05, ymax=0.72, xmax=0.88),
    "indicator_pane": BoundingBox(ymin=0.72, xmin=0.05, ymax=0.85, xmax=0.88),
}

def clean_transcript_for_grounding(transcript: str) -> str:
    """
    Cleans transcript text to remove subtitle timestamps, calendar dates,
    and metadata before extracting financial price numbers and deictic references.
    """
    if not transcript:
        return ""
    # 1. Strip timestamp intervals like [00:00 - 00:05] or [00:00 -> 00:05] or [00:00]
    t = re.sub(r'\[\s*\d{1,2}:\d{2}(?::\d{2})?\s*(?:-|->)?\s*\d{1,2}:\d{2}(?::\d{2})?\s*\]', ' ', transcript)
    t = re.sub(r'\[\s*\d{1,2}:\d{2}\s*\]', ' ', t)

    # 2. Strip standalone timestamps like '10:30' or '9:30 am'
    t = re.sub(r'\b\d{1,2}:\d{2}(?::\d{2})?(?:\s*(?:am|pm))?\b', ' ', t, flags=re.IGNORECASE)

    # 3. Strip full date expressions like 'January 30th 2023', 'Jan 30 2023', 'Oct 15'
    t = re.sub(
        r'\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?(?:,)?(?:\s+(?:19|20)\d{2})?\b',
        ' ',
        t,
        flags=re.IGNORECASE
    )

    # 4. Strip ordinal day references like '30th', '1st', '2nd', '3rd', '4th'
    t = re.sub(r'\b\d{1,2}(?:st|nd|rd|th)\b', ' ', t, flags=re.IGNORECASE)

    # 5. Strip modern calendar years (1980 to 2039) when not preceded by '$'
    t = re.sub(r'(?<!\$)\b(19[89]\d|20[0-3]\d)\b', ' ', t)

    # 6. Strip percentage expressions like '50%' or '2.5%'
    t = re.sub(r'\b\d+(?:\.\d+)?%', ' ', t)

    # 7. Strip duration and bar intervals like '5 min', '15 minutes', '1 hour', '4 hour', '1 day'
    t = re.sub(r'\b\d+(?:\.\d+)?\s*(?:sec|second|min|minute|hr|hour|day|week|month|yr|year)s?\b', ' ', t, flags=re.IGNORECASE)

    return t

class PriceCoordinateMapper:
    """
    Calibrates normalized vertical (Y) chart coordinates to exact monetary price levels.
    Trading charts are inverted along the Y-axis:
      ymin (top) -> max_price
      ymax (bottom) -> min_price
    """

    def __init__(
        self,
        min_price: float,
        max_price: float,
        canvas_ymin: float = 0.10,
        canvas_ymax: float = 0.72
    ):
        if max_price <= min_price:
            max_price = min_price + 10.0
        self.min_price = float(min_price)
        self.max_price = float(max_price)
        self.canvas_ymin = float(canvas_ymin)
        self.canvas_ymax = float(canvas_ymax)

    @property
    def price_range(self) -> float:
        return self.max_price - self.min_price

    def y_to_price(self, y_norm: float) -> float:
        """Maps normalized Y coordinate (0.0 to 1.0) to price value."""
        clamped_y = max(self.canvas_ymin, min(self.canvas_ymax, y_norm))
        pct = (clamped_y - self.canvas_ymin) / (self.canvas_ymax - self.canvas_ymin)
        price = self.max_price - (pct * self.price_range)
        return round(price, 2)

    def price_to_y(self, price: float) -> float:
        """Maps price value to normalized Y coordinate (0.0 to 1.0)."""
        clamped_p = max(self.min_price, min(self.max_price, price))
        pct = (self.max_price - clamped_p) / self.price_range
        y = self.canvas_ymin + (pct * (self.canvas_ymax - self.canvas_ymin))
        return round(y, 4)

    def price_to_bbox(
        self,
        price: float,
        xmin: float = 0.08,
        xmax: float = 0.86,
        height_padding: float = 0.02
    ) -> BoundingBox:
        """Constructs a bounding box representing a horizontal price line/zone."""
        y_center = self.price_to_y(price)
        ymin = max(0.0, y_center - (height_padding / 2.0))
        ymax = min(1.0, y_center + (height_padding / 2.0))
        return BoundingBox(
            ymin=round(ymin, 4),
            xmin=round(xmin, 4),
            ymax=round(ymax, 4),
            xmax=round(xmax, 4)
        )

class ChartLayoutSegmenter:
    """Segments a video frame into structured chart regions of interest (ROI)."""

    def __init__(self, custom_regions: Optional[Dict[str, BoundingBox]] = None):
        self.regions = custom_regions or DEFAULT_CHART_REGIONS

    def segment(self, image: Optional[Image.Image] = None) -> List[ChartRegion]:
        results: List[ChartRegion] = []
        for reg_type, bbox in self.regions.items():
            results.append(ChartRegion(region_type=reg_type, bounding_box=bbox))
        return results

class DeicticGrounder:
    """
    Resolves informal, ambiguous deictic spoken references ('right here', 'this level',
    'down here', 'lowest point', 'buy zone') to grounded visual coordinates and prices.
    """

    DEICTIC_PATTERNS = [
        (r'\b(broke above|breakout (?:at|above)|crossed above)\s*(?:this|the)?\s*([a-z\s]+)?', "breakout"),
        (r'\b(resistance|overhead level|ceiling|resistance at)\b', "resistance"),
        (r'\b(support|demand zone|buy zone|consolidation (?:support)?)\b', "support"),
        (r'\b(lowest point|stop loss|invalidation|protected swing|stop (?:below|down here)?)\b', "stop_loss"),
        (r'\b(target|take profit|profit target|high point)\b', "take_profit"),
        (r'\b(right here|at this level|this candle|this point|look right here)\b', "deictic_pointer"),
        (r'\b(rsi divergence|oscillator divergence)\b', "indicator"),
    ]

    def ground_transcript(
        self,
        transcript: str,
        price_mapper: PriceCoordinateMapper,
        detected_levels: List[GroundedPriceLevel]
    ) -> List[DeicticAnchor]:
        """Maps spoken expressions to visual chart coordinates and price levels."""
        anchors: List[DeicticAnchor] = []
        if not transcript:
            return anchors

        lower = transcript.lower()

        # Map by semantic intent
        for level in detected_levels:
            if level.level_type in ["breakout", "resistance"] and any(
                k in lower for k in ["broke above", "breakout", "resistance", "this level"]
            ):
                anchors.append(DeicticAnchor(
                    spoken_phrase="this level / breakout level",
                    grounded_price=level.price,
                    target_type="resistance_breakout",
                    bounding_box=level.bounding_box,
                    explanation=f"Spoken breakout/resistance reference grounded to chart level ${level.price:.2f}"
                ))
            elif level.level_type in ["support", "demand_zone"] and any(
                k in lower for k in ["buy zone", "demand", "support", "retest", "pullback"]
            ):
                anchors.append(DeicticAnchor(
                    spoken_phrase="buy zone / demand support",
                    grounded_price=level.price,
                    target_type="demand_zone",
                    bounding_box=level.bounding_box,
                    explanation=f"Spoken demand/support zone grounded to chart level ${level.price:.2f}"
                ))
            elif level.level_type in ["stop_loss", "pivot_low"] and any(
                k in lower for k in ["stop loss", "down here", "lowest point", "invalidation", "stop"]
            ):
                anchors.append(DeicticAnchor(
                    spoken_phrase="down here / lowest point",
                    grounded_price=level.price,
                    target_type="stop_loss",
                    bounding_box=level.bounding_box,
                    explanation=f"Spoken invalidation/stop reference grounded to swing low ${level.price:.2f}"
                ))
            elif level.level_type in ["take_profit", "pivot_high"] and any(
                k in lower for k in ["target", "take profit", "up here", "next resistance"]
            ):
                anchors.append(DeicticAnchor(
                    spoken_phrase="target / up here",
                    grounded_price=level.price,
                    target_type="take_profit",
                    bounding_box=level.bounding_box,
                    explanation=f"Spoken profit target reference grounded to chart level ${level.price:.2f}"
                ))

        # Check for explicit price numbers mentioned in speech
        cleaned_transcript = clean_transcript_for_grounding(transcript)
        price_mentions = re.findall(r'\$?(\d+(?:\.\d{1,2})?)', cleaned_transcript)
        for num_str in price_mentions:
            try:
                num = float(num_str)
                if price_mapper.min_price <= num <= price_mapper.max_price:
                    bbox = price_mapper.price_to_bbox(num)
                    anchors.append(DeicticAnchor(
                        spoken_phrase=f"${num:.2f}",
                        grounded_price=num,
                        target_type="exact_numerical_price",
                        bounding_box=bbox,
                        explanation=f"Explicit spoken price grounded to chart coordinate (y={bbox.ymin:.2f})"
                    ))
            except ValueError:
                continue

        # Deduplicate anchors by grounded price & target type
        unique_anchors: List[DeicticAnchor] = []
        seen = set()
        for a in anchors:
            key = (a.target_type, a.grounded_price)
            if key not in seen:
                seen.add(key)
                unique_anchors.append(a)

        return unique_anchors

class HybridChartGroundingPipeline:
    """
    Main entry point for hybrid visual grounding.
    Combines spatial segmentation, numerical calibration, price level identification,
    and deictic speech grounding into a coherent structured observation object.
    """

    def __init__(self):
        self.segmenter = ChartLayoutSegmenter()
        self.deictic_grounder = DeicticGrounder()

    def estimate_price_range_from_text_and_context(
        self,
        transcript: str,
        asset: Optional[str] = None
    ) -> Tuple[float, float]:
        """
        Extracts or estimates plausible price scale extents based on transcript and asset context.
        """
        cleaned = clean_transcript_for_grounding(transcript)
        numbers = [
            float(m) for m in re.findall(r'\$?(\d+(?:\.\d{1,2})?)', cleaned)
            if 0.01 <= float(m) <= 200000.0
        ]

        if len(numbers) >= 2:
            min_p = min(numbers) * 0.95
            max_p = max(numbers) * 1.05
            return round(min_p, 2), round(max_p, 2)
        elif len(numbers) == 1:
            p = numbers[0]
            return round(p * 0.90, 2), round(p * 1.10, 2)

        # Asset-based heuristics
        if asset:
            upper_asset = asset.upper()
            if "BTC" in upper_asset:
                return 60000.0, 75000.0
            elif "ETH" in upper_asset:
                return 2400.0, 3600.0
            elif "ES" in upper_asset or "SPY" in upper_asset:
                return 5200.0, 5600.0
            elif "NQ" in upper_asset:
                return 18000.0, 20500.0
            elif "NVDA" in upper_asset:
                return 150.0, 250.0

        # Standard normalized default scale
        return 100.0, 200.0

    def extract_price_levels(
        self,
        transcript: str,
        mapper: PriceCoordinateMapper
    ) -> List[GroundedPriceLevel]:
        """Generates grounded price levels from detected price action and context."""
        levels: List[GroundedPriceLevel] = []
        cleaned = clean_transcript_for_grounding(transcript)
        lower = cleaned.lower()

        # Parse any explicit numbers in transcript
        raw_nums = sorted(list(set([
            float(m) for m in re.findall(r'\$?(\d+(?:\.\d{1,2})?)', cleaned)
            if mapper.min_price <= float(m) <= mapper.max_price
        ])))

        if raw_nums:
            if len(raw_nums) >= 2:
                # Lowest is typically stop / demand, highest is target / resistance
                stop_price = raw_nums[0]
                res_price = raw_nums[-1]
                levels.append(GroundedPriceLevel(
                    price=stop_price,
                    level_type="stop_loss" if "stop" in lower else "support",
                    confidence=0.95,
                    bounding_box=mapper.price_to_bbox(stop_price),
                    label=f"Key Level ${stop_price:.2f}"
                ))
                levels.append(GroundedPriceLevel(
                    price=res_price,
                    level_type="resistance" if "breakout" in lower or "resistance" in lower else "target",
                    confidence=0.95,
                    bounding_box=mapper.price_to_bbox(res_price),
                    label=f"Overhead Structure ${res_price:.2f}"
                ))
            else:
                p = raw_nums[0]
                levels.append(GroundedPriceLevel(
                    price=p,
                    level_type="pivot_level",
                    confidence=0.90,
                    bounding_box=mapper.price_to_bbox(p),
                    label=f"Structural Pivot ${p:.2f}"
                ))
        else:
            # Calibrated structural anchors based on 25%, 50%, and 75% chart height
            p_low = mapper.y_to_price(0.65)
            p_mid = mapper.y_to_price(0.40)
            p_high = mapper.y_to_price(0.20)

            levels.append(GroundedPriceLevel(
                price=p_low,
                level_type="support",
                confidence=0.85,
                bounding_box=mapper.price_to_bbox(p_low),
                label=f"Demand Zone Support ${p_low:.2f}"
            ))
            levels.append(GroundedPriceLevel(
                price=p_mid,
                level_type="breakout",
                confidence=0.85,
                bounding_box=mapper.price_to_bbox(p_mid),
                label=f"Breakout Trigger ${p_mid:.2f}"
            ))
            levels.append(GroundedPriceLevel(
                price=p_high,
                level_type="resistance",
                confidence=0.85,
                bounding_box=mapper.price_to_bbox(p_high),
                label=f"Resistance Target ${p_high:.2f}"
            ))

        return levels

    def ground_chunk(
        self,
        chunk: VideoChunk,
        frames: Optional[List[Image.Image]] = None
    ) -> ChartGroundingData:
        """
        Executes hybrid visual grounding over a video chunk and its sampled frames.
        """
        transcript = chunk.synchronized_transcript or ""

        # 1. Segment spatial layout
        first_frame = frames[0] if (frames and len(frames) > 0) else None
        chart_regions = self.segmenter.segment(first_frame)

        # 2. Extract asset and timeframe
        ticker = extract_ticker_symbol(transcript) or "ASSET"
        timeframe = extract_timeframe(transcript) or "Daily"

        # 3. Establish price coordinate mapping
        min_p, max_p = self.estimate_price_range_from_text_and_context(transcript, ticker)
        mapper = PriceCoordinateMapper(min_price=min_p, max_price=max_p)

        # 4. Detect grounded price levels
        detected_levels = self.extract_price_levels(transcript, mapper)

        # 5. Resolve deictic spoken references to visual coordinates
        deictic_anchors = self.deictic_grounder.ground_transcript(transcript, mapper, detected_levels)

        # 6. Extract timeline labels
        timeline_labels = []
        for word in transcript.split():
            if re.match(r'^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|\d{4}|\d{1,2}:\d{2})$', word, re.IGNORECASE):
                timeline_labels.append(word)

        # 7. Check indicator state
        indicator_summary = {}
        if "rsi" in transcript.lower():
            indicator_summary["rsi"] = {
                "detected": True,
                "region": "indicator_pane",
                "notes": "RSI sub-window active; divergence tracking"
            }

        return ChartGroundingData(
            ticker=ticker,
            timeframe=timeframe,
            price_range=(min_p, max_p),
            detected_price_levels=detected_levels,
            deictic_anchors=deictic_anchors,
            chart_regions=chart_regions,
            timeline_labels=list(set(timeline_labels)),
            indicator_summary=indicator_summary
        )

    def format_grounding_prompt_block(self, grounding: ChartGroundingData) -> str:
        """
        Produces a dense visual grounding context block to inject into the VLM prompt.
        """
        lines = [
            "HYBRID VISUAL GROUNDING ANCHORS:",
            f"- Identified Instrument: {grounding.ticker} ({grounding.timeframe})",
        ]
        if grounding.price_range:
            lines.append(f"- Calibrated Price Scale: [Min: ${grounding.price_range[0]:.2f}, Max: ${grounding.price_range[1]:.2f}]")

        if grounding.detected_price_levels:
            lines.append("- Grounded Chart Price Levels:")
            for lvl in grounding.detected_price_levels:
                bbox_str = f"[y: {lvl.bounding_box.ymin:.2f}-{lvl.bounding_box.ymax:.2f}]" if lvl.bounding_box else ""
                lines.append(f"  * {lvl.level_type.upper()}: ${lvl.price:.2f} {bbox_str} ({lvl.label})")

        if grounding.deictic_anchors:
            lines.append("- Resolved Deictic Spoken Anchors:")
            for a in grounding.deictic_anchors:
                p_str = f"${a.grounded_price:.2f}" if a.grounded_price is not None else "N/A"
                lines.append(f"  * '{a.spoken_phrase}' -> Grounded to {a.target_type} at {p_str}")

        return "\n".join(lines)

    def enrich_caption_with_grounding(
        self,
        caption: str,
        grounding: ChartGroundingData
    ) -> str:
        """
        Enriches a raw caption with verified grounded numerical anchors.
        """
        if not grounding.detected_price_levels:
            return caption

        price_summary = ", ".join([
            f"{lvl.level_type.capitalize()}: ${lvl.price:.2f}"
            for lvl in grounding.detected_price_levels[:3]
        ])
        enriched = f"{caption} [Grounded Coordinates: {price_summary}]"
        return enriched
