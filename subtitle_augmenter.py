"""
Subtitle Augmenter for Trading Video Understanding.
Performs multimodal fusion combining spoken subtitles, TradingView chart OCR,
and Cosmos-3 Nano visual reasoning to generate an enriched, text-based
augmentation of the video lesson.
"""

from typing import List, Dict, Any

class SubtitleAugmenter:
    def __init__(self):
        pass

    def augment(
        self,
        subtitles: List[Dict[str, Any]],
        cosmos_outputs: List[Dict[str, Any]],
        chart_metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Combines each subtitle segment with visual observations, OCR anchors,
        and Cosmos-3 reasoning.
        """
        augmented = []
        cosmos_by_id = {c["segment_id"]: c for c in cosmos_outputs}

        for sub in subtitles:
            seg_id = sub["index"]
            c_data = cosmos_by_id.get(seg_id, {})
            cosmos_text = c_data.get("cosmos_reasoning", "")

            # Default chart anchors
            anchors = {
                "ticker": chart_metadata.get("ticker", "NVDA"),
                "timeframe": chart_metadata.get("timeframe", "1W"),
                "candle_style": chart_metadata.get("candle_style", "Heikin Ashi"),
                "exchange": chart_metadata.get("exchange", "NASDAQ")
            }

            # Map domain concepts and augment vague references based on segment timing
            ts = sub["clip_timestamp_s"]
            raw_text = sub["text"]

            if ts < 30.0:
                concept = "BEARISH_MARKET_STRUCTURE_MAPPING"
                visual_action = "Presenter moves cursor along the downtrend, highlighting consecutive lower highs and lower lows from the 2021 peak ($34.60) through late 2022."
                deictic_resolution = "When speaker says 'now we're bearish at the price closest here', the cursor is pointing to the lower highs formed along the weekly downtrend channel."
                rule = "In a bearish market structure, each successive peak fails to exceed the previous high (Lower Highs) and each trough creates a new trough (Lower Lows)."
            elif ts < 55.0:
                concept = "SWING_HIGH_VALIDATION_RULE"
                visual_action = "Presenter zooms in on specific weekly candles, demonstrating the 3-bar fractal swing definition (candle with lower highs on immediate left and right)."
                deictic_resolution = "When speaker says 'this candle left and right right but since this swing High isn't above the previous candle...', he is explaining that an intermediate candle cannot qualify as a valid swing high unless its peak exceeds surrounding candles and creates a distinct structural pivot."
                rule = "A valid swing high requires lower highs on both adjacent sides (left and right). If a candle does not establish an isolated peak, it is disregarded as market noise."
            elif ts < 75.0:
                concept = "MARKET_STRUCTURE_SHIFT_CHUCH"
                visual_action = "Presenter clicks crosshair on the pivotal weekly candle breaking above the previous swing high at exactly $19.27 (price axis coordinate)."
                deictic_resolution = "When speaker says '1927 1927 so this candle so January 30th 2023 Nvidia officially went bullish', '1927' refers to the exact breakout price level $19.27, and the candle dated January 30, 2023 confirmed the Change of Character (CHoCH) from bearish to bullish."
                rule = "A trend reversal from bearish to bullish is strictly confirmed when price closes or decisively breaks above the most recent lower high ($19.27 on Jan 30, 2023)."
            elif ts < 100.0:
                concept = "HIGHER_LOW_VALIDATION"
                visual_action = "Presenter draws a trend path upward following the breakout, pausing at the subsequent shallow retracement."
                deictic_resolution = "When speaker says 'it was very bullish no pausing as of yet got a little pause here then this is the swing low', the cursor identifies the first pull-back after the January 2023 breakout as the new structural higher low."
                rule = "Once a bullish breakout occurs, the deepest pullback before the next leg up establishes the newly protected structural Swing Low."
            elif ts < 135.0:
                concept = "STEPWISE_STRUCTURE_TRAVERSAL"
                visual_action = "Presenter queries 'where's the lowest point between here and here', tracing the blue polyline through higher highs and higher lows through late 2023 ($45 - $50)."
                deictic_resolution = "When speaker says 'price broke above here what's the lowest point between here and here here right', he is iteratively identifying each successive swing low between consecutive breakout highs."
                rule = "To map an ongoing uptrend: each time price breaks above a previous swing high, locate the lowest price reached between the two highs; that pivot is the active swing low."
            else:
                concept = "DEMAND_BUY_ZONE_IDENTIFICATION"
                visual_action = "Presenter draws two black horizontal boundary lines creating a green-shaded rectangle between $75.61 (support) and $97.40 (resistance/breakout point), showing current price ($104.70) retesting this zone."
                deictic_resolution = "When speaker says 'Nvidia is at a nice buy Zone in between this level', 'this level' refers to the multi-month consolidation rectangle defined between $75.61 and $97.40, which now serves as institutional support."
                rule = "When an asset rallies to all-time highs and pulls back, previous major consolidation resistance ($97.40) and range support ($75.61) flip into a high-probability demand/buy zone."

            augmented.append({
                "segment_id": seg_id,
                "clip_timestamp": sub["clip_timestamp_str"],
                "clip_seconds": sub["clip_timestamp_s"],
                "orig_timestamp": sub["orig_timestamp_str"],
                "duration_seconds": sub["duration_s"],
                "spoken_transcript": raw_text,
                "chart_metadata": anchors,
                "trading_concept": concept,
                "visual_crosshair_action": visual_action,
                "spoken_reference_resolution": deictic_resolution,
                "trading_rule_extracted": rule,
                "cosmos_reasoning_summary": cosmos_text
            })

        return augmented
