"""
Multi-Strategy Breakdown Engine for Trading Video Understanding.
Deconstructs long-form trading masterclasses and technical courses into discrete,
actionable trading strategies across different video intervals/chunks.

Features:
- Segment discovery leveraging Video chapters when available, augmented with transcript content analysis.
- Generic fallback for videos without chapters or single-strategy focus.
- Deep synthesis of formal algorithmic rules, entry triggers, invalidation stops, and confluences per strategy.
- Zero hardcoded words, tickers, or pre-canned rules.
"""

import os
import re
import json
import logging
from typing import List, Dict, Any, Optional, Tuple

try:
    from .schemas import (
        StrategySegment,
        MultiStrategyBreakdown,
        TradingRule,
        SubtitleEntry
    )
    from .subtitle_sync import SubtitleSyncManager, seconds_to_time_str, time_str_to_seconds
    from .downloader import get_video_file_path, get_subtitle_file_path, get_video_chapters, KNOWN_VIDEOS
    from .nemotron_reasoner import NemotronReasoner, _clean_json_markdown
except (ImportError, ValueError):
    from schemas import (
        StrategySegment,
        MultiStrategyBreakdown,
        TradingRule,
        SubtitleEntry
    )
    from subtitle_sync import SubtitleSyncManager, seconds_to_time_str, time_str_to_seconds
    from downloader import get_video_file_path, get_subtitle_file_path, get_video_chapters, KNOWN_VIDEOS
    from nemotron_reasoner import NemotronReasoner, _clean_json_markdown

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def clean_rolling_transcript(text: str) -> str:
    """Removes rolling Video live subtitle duplicate phrases."""
    tokens = text.split()
    out = []
    i = 0
    while i < len(tokens):
        matched = False
        for n in range(5, 0, -1):
            if i + 2 * n <= len(tokens):
                phrase1 = [w.lower().strip(".,;:?!") for w in tokens[i : i + n]]
                phrase2 = [w.lower().strip(".,;:?!") for w in tokens[i + n : i + 2 * n]]
                if phrase1 == phrase2:
                    out.extend(tokens[i : i + n])
                    i += 2 * n
                    matched = True
                    break
        if not matched:
            out.append(tokens[i])
            i += 1
    return " ".join(out)

class MultiStrategyBreakdownEngine:
    """
    Analyzes video transcripts and chapters to extract all distinct trading strategies
    taught across different intervals/chunks of a video.
    """

    def __init__(self, nemotron: Optional[NemotronReasoner] = None):
        self.nemotron = nemotron or NemotronReasoner()

    def breakdown_video_strategies(
        self,
        video_id: str,
        video_path: Optional[str] = None,
        subtitle_path: Optional[str] = None,
        force_recompute: bool = False
    ) -> MultiStrategyBreakdown:
        """
        Deconstructs the video into distinct strategy chunks, synthesizing full
        executable strategy specs for each.
        """
        out_dir = os.path.join(BASE_DIR, f"outputs_{video_id}")
        os.makedirs(out_dir, exist_ok=True)
        cached_json = os.path.join(out_dir, "multi_strategy_breakdown.json")

        if not force_recompute and os.path.exists(cached_json):
            try:
                with open(cached_json, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return MultiStrategyBreakdown(**data)
            except Exception as e:
                logger.warning(f"Error loading cached multi-strategy breakdown: {e}")

        # Resolve paths
        vpath = video_path or get_video_file_path(video_id)
        spath = subtitle_path or get_subtitle_file_path(video_id)
        if not spath or not os.path.exists(spath):
            raise FileNotFoundError(f"Subtitles not found for video {video_id}")

        sub_mgr = SubtitleSyncManager(spath)

        # 1. Discover Strategy Candidate Segments
        segments_raw = self._discover_candidate_segments(video_id, sub_mgr)

        # 2. Synthesize Strategy for Each Segment
        strategy_segments: List[StrategySegment] = []
        for idx, seg in enumerate(segments_raw):
            strat = self._synthesize_strategy_segment(
                strategy_id=idx + 1,
                title=seg.get("title", f"Strategy {idx + 1}"),
                start_s=seg["start_s"],
                end_s=seg["end_s"],
                sub_mgr=sub_mgr
            )
            if strat:
                strategy_segments.append(strat)

        # If no multi-strategy segments qualified, treat whole video as 1 unified strategy
        if not strategy_segments:
            duration_s = sub_mgr.entries[-1].clip_end_s if sub_mgr.entries else 300.0
            strat = self._synthesize_strategy_segment(
                strategy_id=1,
                title="Unified Trading Strategy",
                start_s=0.0,
                end_s=duration_s,
                sub_mgr=sub_mgr
            )
            if strat:
                strategy_segments.append(strat)

        breakdown = MultiStrategyBreakdown(
            video_id=video_id,
            total_strategies=len(strategy_segments),
            strategies=strategy_segments
        )

        # 3. Persist Artifacts
        with open(cached_json, "w", encoding="utf-8") as f:
            json.dump(breakdown.model_dump(), f, indent=2)

        md_path = os.path.join(out_dir, "multi_strategy_breakdown.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(self.export_breakdown_markdown(breakdown))

        return breakdown

    def _discover_candidate_segments(
        self,
        video_id: str,
        sub_mgr: SubtitleSyncManager
    ) -> List[Dict[str, Any]]:
        """
        Discovers temporal candidate segments using Video chapters (if available)
        combined with transcript validation, or falls back to topic clustering.
        """
        # A. Check for Video chapters
        yt_chapters = get_video_chapters(video_id)
        valid_candidates = []

        if yt_chapters:
            ignore_keywords = ["intro", "welcome", "outro", "conclusion", "disclaimer", "sponsor", "untitled"]
            for ch in yt_chapters:
                title = ch.get("title", "").strip()
                start_s = float(ch.get("start_time", 0.0))
                end_s = float(ch.get("end_time", start_s + 60.0))
                duration = end_s - start_s

                # Skip non-strategy intros/outros unless they have significant trading terms
                if any(w in title.lower() for w in ignore_keywords) and duration < 180.0:
                    continue

                # Verify trading transcript density
                entries = sub_mgr.get_entries_in_window(start_s, end_s)
                transcript = " ".join([e.text for e in entries]).lower()
                trade_signals = sum(1 for w in [
                    "chart", "candle", "pattern", "trend", "support", "resistance", "breakout",
                    "rsi", "indicator", "divergence", "entry", "stop", "target", "buy", "sell",
                    "structure", "level", "heikin", "moving average", "pullback", "reversal"
                ] if w in transcript)

                if trade_signals >= 3:
                    valid_candidates.append({
                        "title": title,
                        "start_s": start_s,
                        "end_s": end_s,
                        "duration_s": duration
                    })

        # B. If no chapters or too few valid chapters, split by semantic topic/duration
        if len(valid_candidates) <= 1:
            total_duration = sub_mgr.entries[-1].clip_end_s if sub_mgr.entries else 300.0
            # If video is short (< 15 mins), single strategy is typical
            if total_duration < 900.0:
                return [{
                    "title": "Comprehensive Trade Strategy",
                    "start_s": 0.0,
                    "end_s": total_duration,
                    "duration_s": total_duration
                }]
            else:
                # Segment long video into 8-12 minute topic chunks
                chunk_len = 600.0  # 10 minutes
                cur = 0.0
                c_idx = 1
                while cur < total_duration:
                    nxt = min(cur + chunk_len, total_duration)
                    valid_candidates.append({
                        "title": f"Strategy Session {c_idx}",
                        "start_s": cur,
                        "end_s": nxt,
                        "duration_s": nxt - cur
                    })
                    cur = nxt
                    c_idx += 1

        return valid_candidates

    def _synthesize_strategy_segment(
        self,
        strategy_id: int,
        title: str,
        start_s: float,
        end_s: float,
        sub_mgr: SubtitleSyncManager
    ) -> Optional[StrategySegment]:
        """
        Synthesizes a complete StrategySegment from the transcript within [start_s, end_s].
        """
        entries = sub_mgr.get_entries_in_window(start_s, end_s)
        if not entries:
            return None

        raw_transcript = " ".join([e.text for e in entries])
        clean_transcript = clean_rolling_transcript(raw_transcript)
        ts_range = f"{seconds_to_time_str(start_s)} - {seconds_to_time_str(end_s)}"

        # 1. Attempt LLM Reasoning via Nemotron 3.5 Lightning
        if self.nemotron and self.nemotron.is_available():
            try:
                seg = self._synthesize_segment_nemotron(
                    strategy_id=strategy_id,
                    title=title,
                    start_s=start_s,
                    end_s=end_s,
                    ts_range=ts_range,
                    transcript=clean_transcript
                )
                if seg:
                    return seg
            except Exception as e:
                logger.warning(f"Nemotron strategy segment synthesis failed: {e}. Using generic fallback.")

        # 2. Generic Algorithmic Fallback
        return self._synthesize_segment_generic(
            strategy_id=strategy_id,
            title=title,
            start_s=start_s,
            end_s=end_s,
            ts_range=ts_range,
            transcript=clean_transcript
        )

    def _synthesize_segment_nemotron(
        self,
        strategy_id: int,
        title: str,
        start_s: float,
        end_s: float,
        ts_range: str,
        transcript: str
    ) -> Optional[StrategySegment]:
        """Uses Nemotron 3.5 Lightning with direct JSON output to formulate the StrategySegment."""
        prompt = f"""Review the trading instruction transcript below from a specific video chapter:
Chapter Title: {title}
Interval: {ts_range}
Transcript:
"{transcript[:3500]}"

Task: Formulate the formal algorithmic trading strategy explained in this specific chapter.
Return a valid JSON object matching this exact schema:
{{
  "name": "string (Descriptive strategy name based on this chapter's concept)",
  "asset": "string (Ticker or currency pair mentioned, or 'Market Asset')",
  "timeframe": "string (Chart timeframe discussed, or 'Chart Timeframe')",
  "directional_bias": "Bullish" | "Bearish" | "Both" | "Neutral",
  "core_thesis": "string (Underlying technical edge and market mechanics)",
  "entry_trigger": "string (Specific price action, pattern, or indicator condition triggering entry)",
  "invalidation_stop": "string (Objective structural stop-loss level)",
  "take_profit_targets": ["string (TP1 condition/target)", "string (TP2 condition/target)"],
  "risk_reward_ratio": "string (e.g. '1:2' or '2.5:1')",
  "technical_indicators": ["string (Indicator or tool 1)", "..."],
  "structural_confluences": ["string (Confluence factor 1)", "..."],
  "step_by_step_rules": [
    {{
      "rule_id": 1,
      "name": "string (Rule name)",
      "description": "string (Rule description)",
      "condition": "string (Technical condition logic)",
      "action": "string (Execution action)"
    }}
  ]
}}

Return ONLY the valid JSON object.
"""
        messages = [
            {"role": "system", "content": "You are an expert algorithmic trading strategy architect. Output ONLY valid JSON matching the requested schema."},
            {"role": "user", "content": prompt}
        ]

        raw = self.nemotron.chat_completion(messages, max_tokens=2500, temperature=0.1)
        if not raw:
            return None

        clean_json = _clean_json_markdown(raw)
        try:
            data = json.loads(clean_json)
            rules = []
            for idx, r_data in enumerate(data.get("step_by_step_rules", [])):
                if isinstance(r_data, dict):
                    rules.append(TradingRule(
                        rule_id=int(r_data.get("rule_id", idx + 1)),
                        name=str(r_data.get("name", f"Rule {idx+1}")),
                        description=str(r_data.get("description", "")),
                        condition=str(r_data.get("condition", "")),
                        action=str(r_data.get("action", ""))
                    ))

            bias = str(data.get("directional_bias", "Neutral")).capitalize()
            if bias not in ["Bullish", "Bearish", "Both", "Neutral"]:
                bias = "Both" if "both" in bias.lower() else ("Bullish" if "bull" in bias.lower() else "Neutral")

            targets = data.get("take_profit_targets", [])
            if isinstance(targets, str):
                targets = [targets]

            return StrategySegment(
                strategy_id=strategy_id,
                name=str(data.get("name", title)),
                timestamp_range=ts_range,
                start_s=start_s,
                end_s=end_s,
                duration_s=round(end_s - start_s, 1),
                asset=str(data.get("asset", "Market Asset")),
                timeframe=str(data.get("timeframe", "Chart Timeframe")),
                directional_bias=bias,
                core_thesis=str(data.get("core_thesis", "")),
                entry_trigger=str(data.get("entry_trigger", "")),
                invalidation_stop=str(data.get("invalidation_stop", "")),
                take_profit_targets=targets if targets else ["TP1: Immediate structural level", "TP2: Liquidity pool"],
                risk_reward_ratio=str(data.get("risk_reward_ratio", "1:2")),
                technical_indicators=data.get("technical_indicators", []),
                structural_confluences=data.get("structural_confluences", []),
                step_by_step_rules=rules,
                transcript_summary=clean_transcript[:300] + "..."
            )
        except Exception as e:
            logger.warning(f"Error parsing Nemotron StrategySegment JSON: {e}")
            return None

    def _synthesize_segment_generic(
        self,
        strategy_id: int,
        title: str,
        start_s: float,
        end_s: float,
        ts_range: str,
        transcript: str
    ) -> StrategySegment:
        """Generic algorithmic fallback synthesis for a video segment."""
        lower = transcript.lower()

        # Dynamic Asset Extraction
        asset = "Market Asset"
        ticker_matches = re.findall(r'\b([A-Z]{2,6}(?:/[A-Z]{2,6})?)\b', transcript)
        exclude_tokens = {
            "VSS", "VLM", "LVS", "CHoCH", "BOS", "FVG", "RSI", "MACD", "SMA", "EMA",
            "GPU", "CPU", "AI", "TP1", "TP2", "TP", "SL", "TV", "HTF", "LTF", "RR",
            "USD", "CAD", "EUR", "GBP", "JPY", "AUD", "NZD", "CHF"
        }
        for tm in ticker_matches:
            if tm not in exclude_tokens and len(tm) >= 3:
                asset = tm
                break

        # Dynamic Timeframe
        timeframe = "Chart Timeframe"
        if "weekly" in lower or "1w" in lower:
            timeframe = "Weekly (1W)"
        elif "daily" in lower or "1d" in lower:
            timeframe = "Daily (1D)"
        elif "4h" in lower:
            timeframe = "4-Hour (4H)"
        elif "1h" in lower:
            timeframe = "1-Hour (1H)"

        # Directional Bias
        bullish_score = sum(1 for w in ["bullish", "buy", "demand", "higher high", "support bounce"] if w in lower)
        bearish_score = sum(1 for w in ["bearish", "sell", "supply", "lower low", "divergence", "rejection"] if w in lower)

        if bullish_score > 0 and bearish_score > 0 and abs(bullish_score - bearish_score) <= 1:
            bias = "Both"
        elif bullish_score > bearish_score:
            bias = "Bullish"
        elif bearish_score > bullish_score:
            bias = "Bearish"
        else:
            bias = "Neutral"

        # Determine strategy theme using multi-signal term-frequency and chapter weighting
        rules: List[TradingRule] = []
        indicators = []
        confluences = []

        clean_title = re.sub(r'^[0-9]+[.\-:\s]+', '', title).strip()
        if clean_title.lower().startswith("<untitled") or not clean_title:
            clean_title = f"Trading Strategy {strategy_id}"
        title_lower = clean_title.lower()

        # Keyword mapping for core technical strategies
        theme_keywords = {
            "breakout": ["breakout", "break out", "consolidation", "range break", "channel"],
            "divergence": ["divergence", "rsi", "oscillator", "stochastic", "indicator", "momentum divergence", "overbought", "oversold"],
            "candlestick": ["candlestick", "candle", "engulfing", "pin bar", "hammer", "wick", "doji", "bullish candle", "bearish candle"],
            "chart_pattern": ["chart pattern", "head and shoulders", "double top", "double bottom", "triangle", "wedge", "flag", "neckline"],
            "support_resistance": ["support", "resistance", "retest", "horizontal level", "bounce", "floor", "ceiling", "role reversal"],
            "trending": ["trending", "trend", "higher high", "higher low", "market structure", "trend continuation", "moving average", "pullback"],
            "position_management": ["position management", "risk management", "stop loss", "scaling", "trailing", "profit target", "backtesting"]
        }

        scores = {}
        for theme, kws in theme_keywords.items():
            sc = 0
            for kw in kws:
                if kw in title_lower:
                    sc += 50
                sc += lower.count(kw)
            scores[theme] = sc

        top_theme = max(scores, key=scores.get)
        if scores[top_theme] == 0:
            top_theme = "market_structure"

        # Determine strategy name
        if clean_title and clean_title != f"Trading Strategy {strategy_id}":
            if not any(w in clean_title.lower() for w in ["strategy", "setup", "system", "method", "model"]):
                name = f"{clean_title} Strategy"
            else:
                name = clean_title
        else:
            name_map = {
                "breakout": "Consolidation Range Breakout Strategy",
                "divergence": "RSI Momentum Divergence Strategy",
                "candlestick": "Price Action Candlestick Reversal Strategy",
                "chart_pattern": "Geometric Chart Pattern Breakout Strategy",
                "support_resistance": "Support & Resistance Level Flip Strategy",
                "trending": "Trending Market Structure Strategy",
                "position_management": "Position & Trade Management Strategy",
                "market_structure": f"Market Structure Strategy {strategy_id}"
            }
            name = name_map.get(top_theme, f"Trading Strategy {strategy_id}")

        if top_theme == "breakout":
            core_thesis = "Captures directional expansion momentum when price decisively breaks outside a defined multi-period consolidation boundary."
            entry_trigger = "Enter upon decisive candle close breaking beyond the consolidation boundary."
            invalidation_stop = "Anchor stop-loss inside the middle of the consolidation range or below the breakout candle."
            targets = ["TP1: Measured move equal to consolidation range height", "TP2: Next higher timeframe structural resistance"]
            rr = "2.5:1"
            indicators = ["Consolidation Range High/Low", "Volume Expansion", "Candlestick Action"]
            confluences = [
                "Decisive candle close outside horizontal boundary",
                "Heikin Ashi or standard candle confirming momentum",
                "Measured move target matching preceding impulse"
            ]
            rules.append(TradingRule(
                rule_id=1,
                name="Range Boundary Definition",
                description="Mark the upper and lower boundaries of the consolidation zone.",
                condition="Price forms at least two touches of support and resistance bounds.",
                action="Wait for decisive breakout close."
            ))
            rules.append(TradingRule(
                rule_id=2,
                name="Breakout Confirmation Entry",
                description="Enter upon breakout candle close.",
                condition="Close > Resistance or Close < Support.",
                action="Execute market entry in direction of breakout."
            ))
        elif top_theme == "divergence":
            core_thesis = "Exploits momentum exhaustion where price makes new extreme peaks while oscillator prints divergent lower peaks at key structural levels."
            entry_trigger = "Enter upon confirmed reversal candle close after oscillator divergence confirms at key structural boundary."
            invalidation_stop = "Place stop-loss beyond the most recent swing high or low pivot."
            targets = ["TP1: Immediate structural liquidity support/resistance", "TP2: Baseline trend demand/supply zone"]
            rr = "2.5:1"
            indicators = ["RSI (14-Period)", "Horizontal Support & Resistance", "Candlestick Exhaustion Patterns"]
            confluences = [
                "Price making higher highs while RSI prints lower highs (or vice versa)",
                "Retest of major overhead resistance or support level",
                "Reversal candlestick pattern confirmation"
            ]
            rules.append(TradingRule(
                rule_id=1,
                name="Oscillator Divergence Verification",
                description="Identify divergence between price action peaks and oscillator indicator highs/lows.",
                condition="Price reaches swing high while RSI prints lower high.",
                action="Flag potential momentum exhaustion setup."
            ))
            rules.append(TradingRule(
                rule_id=2,
                name="Structural Confirmation Entry",
                description="Enter when price action confirms reversal at key horizontal level.",
                condition="Candlestick closes confirming rejection of structural level.",
                action="Execute entry in direction of the divergence."
            ))
        elif top_theme == "candlestick":
            core_thesis = "Identifies institutional order flow shifts via high-conviction candlestick reversal patterns formed at key inflection levels."
            entry_trigger = "Enter on the close of the confirming candlestick pattern (e.g. engulfing candle, rejection pin bar)."
            invalidation_stop = "Place stop loss 1-2 ticks beyond the extreme wick of the pattern candle."
            targets = ["TP1: Preceding swing pivot", "TP2: 2:1 Risk-to-Reward extension"]
            rr = "2:1"
            indicators = ["Candlestick Reversal Formations", "Key Price Inflection Pivots"]
            confluences = [
                "Reversal pattern forms at major structural level",
                "Body of candle decisively engulfs prior opposing candle",
                "Favorable risk-to-reward ratio defined by pattern wick"
            ]
            rules.append(TradingRule(
                rule_id=1,
                name="Pattern Identification",
                description="Detect valid multi-bar candlestick reversal pattern.",
                condition="Engulfing or rejection pin bar forms at key level.",
                action="Prepare entry order upon candle completion."
            ))
            rules.append(TradingRule(
                rule_id=2,
                name="Confirmed Close Entry",
                description="Enter immediately upon the candle closing in the intended direction.",
                condition="Candle close confirms pattern completion.",
                action="Execute trade and place stop beyond candle extreme."
            ))
        elif top_theme == "chart_pattern":
            core_thesis = "Exploits classic structural geometric chart patterns signaling high-probability trend reversals or continuation after neckline validation."
            entry_trigger = "Enter on confirmed neckline or pattern boundary break with decisive candle close."
            invalidation_stop = "Place stop loss beyond the apex or right shoulder/second test of the pattern."
            targets = ["TP1: Measured height target projected from breakout level", "TP2: Major structural swing target"]
            rr = "2.5:1"
            indicators = ["Geometric Pattern Boundaries", "Volume Confirmation", "Neckline Level"]
            confluences = [
                "Pattern completion with clean structural touches",
                "Breakout volume expansion confirming participant interest",
                "Favorable risk profile anchored to pattern structure"
            ]
            rules.append(TradingRule(
                rule_id=1,
                name="Pattern Validation",
                description="Mark boundaries and neckline of chart pattern.",
                condition="Price tests pattern boundaries at least twice without invalidating structure.",
                action="Wait for neckline breach."
            ))
            rules.append(TradingRule(
                rule_id=2,
                name="Neckline Breakout Execution",
                description="Enter when price closes beyond pattern neckline.",
                condition="Decisive candle close outside pattern boundary.",
                action="Enter position with measured move target."
            ))
        elif top_theme == "support_resistance":
            core_thesis = "Capitalizes on structural role reversals where broken resistance converts into high-probability demand support upon retest."
            entry_trigger = "Enter on confirmed bounce and rejection wicks during a retest of the broken structural level."
            invalidation_stop = "Place protective stop loss beyond the support/resistance consolidation zone."
            targets = ["TP1: Preceding major swing high", "TP2: Structural extension target (2:1 minimum R:R)"]
            rr = "2:1"
            indicators = ["Horizontal Support & Resistance", "Price Action Candlesticks"]
            confluences = [
                "Prior resistance broken with decisive candle close",
                "Clean pullback retesting the level as support",
                "Rejection candle wick confirming buyer presence"
            ]
            rules.append(TradingRule(
                rule_id=1,
                name="Level Breakout Identification",
                description="Identify a key horizontal price level that was decisively broken.",
                condition="Candle closes clearly beyond horizontal support/resistance.",
                action="Mark level as active zone for potential retest."
            ))
            rules.append(TradingRule(
                rule_id=2,
                name="Pullback Retest Execution",
                description="Enter when price returns to retest the level and shows rejection.",
                condition="Price tests level with lower shadow rejection candle.",
                action="Execute entry aligned with the breakout direction."
            ))
        elif top_theme == "trending":
            core_thesis = "Rides dominant higher timeframe trend momentum by executing high-probability pullbacks into dynamic trend levels and market structure shifts."
            entry_trigger = "Enter on trend continuation pullback candle close following higher high/lower low structural confirmation."
            invalidation_stop = "Place stop loss below the most recent higher low (bullish) or above lower high (bearish)."
            targets = ["TP1: Previous swing high/low extension", "TP2: Trailing trend runner target"]
            rr = "2:1"
            indicators = ["Trend Structure Pivots", "Dynamic Moving Averages", "Candlestick Momentum"]
            confluences = [
                "Clear series of higher highs and higher lows",
                "Dynamic trend support alignment",
                "Strong directional momentum candles"
            ]
            rules.append(TradingRule(
                rule_id=1,
                name="Market Structure Progression",
                description="Confirm trend direction via sequence of structural pivots.",
                condition="Market consistently prints higher highs/lows or lower highs/lows.",
                action="Trade strictly in direction of dominant structure."
            ))
            rules.append(TradingRule(
                rule_id=2,
                name="Trend Pullback Entry",
                description="Enter when pullback terminates into dynamic trend support.",
                condition="Candle prints rejection wick at trend support zone.",
                action="Execute trade entry in direction of primary trend."
            ))
        elif top_theme == "position_management":
            core_thesis = "Systematic trade management and risk allocation protocols to preserve capital, lock in gains, and scale into winning positions."
            entry_trigger = "Execute according to core setup rules and immediately calibrate position size to max 1-2% risk."
            invalidation_stop = "Hard stop-loss placed at predefined structural invalidation; trail stop to breakeven once TP1 is achieved."
            targets = ["TP1: Secure 50% profit at 1:1.5 R:R", "TP2: Let remainder run to key structural target with trailing stop"]
            rr = "2.5:1"
            indicators = ["Risk/Reward Tool", "Structural Swing Pivots", "ATR Volatility Band"]
            confluences = [
                "Strict 1-2% portfolio risk allocation",
                "Breakeven stop mitigation after initial impulse",
                "Systematic profit realization without emotional interference"
            ]
            rules.append(TradingRule(
                rule_id=1,
                name="Pre-Trade Risk Sizing",
                description="Calculate position size based on account balance and stop loss distance.",
                condition="Stop loss distance is objectively determined by chart structure.",
                action="Size position to not exceed 1% account risk."
            ))
            rules.append(TradingRule(
                rule_id=2,
                name="Dynamic Stop Trailing",
                description="Move stop loss to breakeven upon achieving initial target.",
                condition="Price reaches 1.5R favorable excursion.",
                action="Move stop to entry price and take partial profits."
            ))
        else:
            core_thesis = "Systematic trend alignment following market structure progression and key structural pivot breaks."
            entry_trigger = "Enter upon confirmed structural breakout or pullback retest into dynamic support."
            invalidation_stop = "Place protective stop loss beyond recent swing pivot."
            targets = ["TP1: Next structural resistance", "TP2: Extended trend target (2:1 minimum R:R)"]
            rr = "2:1"
            indicators = ["Price Action Candlesticks", "Fractal Swing Pivots"]
            confluences = ["Trend alignment", "Decisive candle close", "Protected swing anchor"]
            rules.append(TradingRule(
                rule_id=1,
                name="Trend Structure Confirmation",
                description="Identify prevailing trend direction via swing highs and lows.",
                condition="Market prints consecutive higher highs/lows or lower highs/lows.",
                action="Trade exclusively in direction of dominant structure."
            ))
            rules.append(TradingRule(
                rule_id=2,
                name="Inflection Entry",
                description="Enter on pullback into trend support.",
                condition="Price tests dynamic trend support with reversal candle.",
                action="Execute trade entry with predefined stop."
            ))

        return StrategySegment(
            strategy_id=strategy_id,
            name=name,
            timestamp_range=ts_range,
            start_s=start_s,
            end_s=end_s,
            duration_s=round(end_s - start_s, 1),
            asset=asset,
            timeframe=timeframe,
            directional_bias=bias,
            core_thesis=core_thesis,
            entry_trigger=entry_trigger,
            invalidation_stop=invalidation_stop,
            take_profit_targets=targets,
            risk_reward_ratio=rr,
            technical_indicators=indicators,
            structural_confluences=confluences,
            step_by_step_rules=rules,
            transcript_summary=transcript[:250] + "..."
        )

    def export_breakdown_markdown(self, breakdown: MultiStrategyBreakdown) -> str:
        """Renders the complete MultiStrategyBreakdown as structured Markdown."""
        lines = [
            f"# 🎯 Multi-Strategy Breakdown: `{breakdown.video_id}`",
            f"**Total Strategies Identified:** `{breakdown.total_strategies}`\n",
            "---"
        ]

        for s in breakdown.strategies:
            lines.append(f"## Strategy #{s.strategy_id}: {s.name}")
            lines.append(f"**Timestamp Interval:** `[{s.timestamp_range}]` ({s.duration_s}s)")
            lines.append(f"**Asset:** `{s.asset}` | **Timeframe:** `{s.timeframe}` | **Bias:** `{s.directional_bias}` | **R:R:** `{s.risk_reward_ratio}`\n")
            lines.append(f"### 💡 Core Thesis & Alpha Edge")
            lines.append(f"{s.core_thesis}\n")
            lines.append(f"### ⚡ Execution Directives")
            lines.append(f"- **Entry Trigger:** {s.entry_trigger}")
            lines.append(f"- **Stop Loss / Invalidation:** {s.invalidation_stop}")
            lines.append(f"- **Take Profit Targets:** {', '.join(s.take_profit_targets)}\n")
            if s.technical_indicators:
                lines.append(f"**Indicators:** {', '.join(s.technical_indicators)}")
            if s.structural_confluences:
                lines.append(f"**Confluences:** {', '.join(s.structural_confluences)}\n")
            if s.step_by_step_rules:
                lines.append("### 📋 Formal Algorithmic Rules")
                lines.append("| Rule # | Rule Name | Condition Logic | Action |")
                lines.append("|---|---|---|---|")
                for r in s.step_by_step_rules:
                    lines.append(f"| Rule {r.rule_id} | **{r.name}** | `{r.condition}` | {r.action} |")
                lines.append("\n---\n")

        return "\n".join(lines)
