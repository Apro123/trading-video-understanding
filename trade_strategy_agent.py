"""
VSS Trading Strategy Synthesis Agent.
Implements the Report Generation and Trade Idea Synthesis agent workflow of the NVIDIA VSS Blueprint:
https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization

Translates multimodal video observations and long video summaries into formal,
machine-executable algorithmic trading rules and structured trade idea specifications.
Uses Nemotron 3.5 Lightning reasoning when available, with clean generic fallback.
"""

import os
import re
import json
import logging
from typing import List, Dict, Any, Optional

try:
    from .schemas import (
        TradeIdeaSpec,
        TradingRule,
        CompletionResponse,
        VlmCaptionResponse,
        VideoChunk
    )
    from .vlm_reasoner import VSSVLMReasoner
    from .nemotron_reasoner import NemotronReasoner
    from .utils import extract_ticker_symbol, extract_timeframe, extract_directional_bias
except (ImportError, ValueError):
    from schemas import (
        TradeIdeaSpec,
        TradingRule,
        CompletionResponse,
        VlmCaptionResponse,
        VideoChunk
    )
    from vlm_reasoner import VSSVLMReasoner
    from nemotron_reasoner import NemotronReasoner
    from utils import extract_ticker_symbol, extract_timeframe, extract_directional_bias

logger = logging.getLogger(__name__)

class VSSTradeStrategyAgent:
    """
    Agent responsible for synthesizing structured trading strategy specifications
    from VSS multimodal video captions and summaries.
    """

    def __init__(
        self,
        reasoner: Optional[VSSVLMReasoner] = None,
        nemotron: Optional[NemotronReasoner] = None
    ):
        self.reasoner = reasoner
        self.nemotron = nemotron or NemotronReasoner()

    def synthesize_trade_idea(
        self,
        completion: CompletionResponse,
        chunks: Optional[List[VideoChunk]] = None,
        video_metadata: Optional[Dict[str, Any]] = None
    ) -> TradeIdeaSpec:
        """
        Formulates a comprehensive TradeIdeaSpec based on VSS video captions,
        reasoning notes, and spoken subtitles.
        Prioritizes Nemotron 3.5 reasoning, with generic algorithmic fallback.
        """
        # 1. Attempt LLM Reasoning via Nemotron 3.5 Lightning
        if self.nemotron and self.nemotron.is_available():
            try:
                effective_chunks = chunks or [
                    VideoChunk(
                        chunk_id=c.chunk_id,
                        start_s=c.start_s,
                        end_s=c.end_s,
                        duration_s=c.end_s - c.start_s,
                        start_str=c.start_time,
                        end_str=c.end_time,
                        synchronized_transcript=c.content
                    ) for c in completion.chunk_captions
                ]
                spec = self.nemotron.synthesize_trade_idea(
                    chunks=effective_chunks,
                    chunk_captions=completion.chunk_captions,
                    key_events=completion.key_events
                )
                if spec is not None:
                    return spec
            except Exception as e:
                logger.warning(f"Nemotron trade idea synthesis failed: {e}. Using generic fallback.")

        # 2. Generic Rule-Based Algorithmic Fallback (Zero hardcoded tickers or prices)
        return self._generic_rule_based_synthesis(completion, video_metadata)

    def _generic_rule_based_synthesis(
        self,
        completion: CompletionResponse,
        video_metadata: Optional[Dict[str, Any]] = None
    ) -> TradeIdeaSpec:
        """Completely generic synthesis without clip-specific keywords or hardcoded values."""
        text_parts = [completion.summary]
        for c in completion.chunk_captions:
            text_parts.append(c.content)
            text_parts.append(c.reasoning_description)
        for ev in completion.key_events:
            text_parts.append(ev.action_or_concept)
            text_parts.append(ev.chart_evidence)
            text_parts.append(ev.spoken_rationale)
        if video_metadata:
            text_parts.append(str(video_metadata))

        all_text = " ".join(text_parts)
        lower = all_text.lower()

        # Dynamic Asset, Timeframe, and Directional Bias Extraction
        asset = extract_ticker_symbol(all_text, default="Market Asset Under Analysis")
        timeframe = extract_timeframe(all_text, default="Chart Timeframe")
        bias = extract_directional_bias(all_text, default="Neutral")

        # Dynamic Rules Formulation
        rules: List[TradingRule] = []

        if bias == "Bearish Reversal" or ("rsi" in lower and "divergence" in lower):
            core_thesis = "Momentum exhaustion reversal identified via oscillator divergence against price extremes at key overhead structure."
            entry_trigger = "Enter short upon confirmed close below the reversal trigger candle at key overhead structure."
            invalidation_stop = "Place stop loss beyond the recent extreme swing high pivot."
            targets = ["TP1: Immediate structural liquidity support", "TP2: Baseline demand zone"]
            rr = "2.5:1"
            indicators = ["Momentum Oscillator (RSI/MACD)", "Key Overhead Resistance / Supply Level", "Candlestick Exhaustion Patterns"]
            confluences = [
                "Oscillator printing lower highs while price prints higher highs",
                "Rejection wick at structural resistance",
                "Breakdown of immediate candle low confirming reversal"
            ]

            rules.append(TradingRule(
                rule_id=1,
                name="Momentum Divergence Confirmation",
                description="Verify divergence between price highs and oscillator indicator peaks.",
                condition="Price_High[t] > Price_High[t-n] and Indicator_High[t] < Indicator_High[t-n]",
                action="Flag high-probability momentum exhaustion zone."
            ))
            rules.append(TradingRule(
                rule_id=2,
                name="Resistance Level Test",
                description="Price must test recognized overhead structural supply level.",
                condition="Price tests major horizontal resistance",
                action="Prepare execution trigger on confirmation."
            ))
            rules.append(TradingRule(
                rule_id=3,
                name="Short Entry & Stop Loss Execution",
                description="Enter short on confirmation with protective stop placed above extreme swing high.",
                condition="Close < Trigger_Candle_Low",
                action="Execute short position and set invalidation stop above swing high."
            ))

        else:
            core_thesis = "Systematic market structure trend progression confirmed via structural pivot breakout and demand retest."
            entry_trigger = "Enter on confirmed structural breakout above resistance or upon pullback retest into the prior consolidation demand zone."
            invalidation_stop = "Place protective stop loss anchored strictly below the lowest structural swing low pivot."
            targets = ["TP1: Preceding major structural swing high", "TP2: Extension / continuation target (2:1 minimum R:R)"]
            rr = "2.5:1"
            indicators = ["Price Action Candlesticks", "Fractal Swing High / Low Pivots", "Consolidation Support Boundaries"]
            confluences = [
                "Decisive candle close breaking above prior structural swing high",
                "Valid multi-bar fractal swing pivot formation",
                "Previous consolidation resistance acting as dynamic demand support",
                "Protected swing low anchor defining objective risk boundary"
            ]

            rules.append(TradingRule(
                rule_id=1,
                name="Structural Pivot Validation",
                description="Identify and validate key swing high or low structural reference levels.",
                condition="High[t] > High[t-1] and High[t] > High[t+1]",
                action="Mark horizontal price coordinate as structural reference pivot."
            ))
            rules.append(TradingRule(
                rule_id=2,
                name="Trend Reversal / Breakout Confirmation",
                description="Decisive close beyond previous structural pivot confirms Change of Character or continuation.",
                condition="Close > Previous_Swing_High",
                action="Confirm directional market structure bias."
            ))
            rules.append(TradingRule(
                rule_id=3,
                name="Pullback Retest Entry",
                description="Enter upon price retesting the breakout level or consolidation support zone.",
                condition="Price retests support zone with reversal candlestick confirmation",
                action="Execute long entry with calculated position sizing."
            ))
            rules.append(TradingRule(
                rule_id=4,
                name="Protected Invalidation Stop",
                description="Anchor stop loss below the lowest structural trough between breakout impulses.",
                condition="Stop_Loss = min(Price) between breakout impulses",
                action="Set protected stop loss to limit downside risk."
            ))

        return TradeIdeaSpec(
            asset=asset,
            timeframe=timeframe,
            directional_bias=bias,
            market_context=f"Market regime analyzed from video intervals with multimodal VLM perception.",
            core_thesis=core_thesis,
            entry_trigger=entry_trigger,
            invalidation_stop=invalidation_stop,
            take_profit_targets=targets,
            risk_reward_ratio=rr,
            technical_indicators=indicators,
            structural_confluences=confluences,
            step_by_step_rules=rules
        )

    def export_spec_markdown(self, spec: TradeIdeaSpec) -> str:
        """Renders the TradeIdeaSpec into a structured markdown report."""
        md = []
        md.append(f"# 🎯 NVIDIA VSS Algorithmic Trade Strategy Specification")
        md.append("")
        md.append(f"| Parameter | Specification |")
        md.append(f"|---|---|")
        md.append(f"| **Asset / Ticker** | `{spec.asset}` |")
        md.append(f"| **Chart Timeframe** | `{spec.timeframe}` |")
        md.append(f"| **Directional Bias** | **{spec.directional_bias.upper()}** |")
        md.append(f"| **Risk / Reward Ratio** | `{spec.risk_reward_ratio}` |")
        md.append("")
        md.append("---")
        md.append("")
        md.append("## 💡 Core Strategy Thesis")
        md.append(spec.core_thesis)
        md.append("")
        md.append("## ⚡ Execution Rules")
        md.append(f"- **Entry Trigger:** {spec.entry_trigger}")
        md.append(f"- **Invalidation / Stop Loss:** {spec.invalidation_stop}")
        md.append("- **Take Profit Targets:**")
        for tp in spec.take_profit_targets:
            md.append(f"  - {tp}")
        md.append("")
        md.append("---")
        md.append("")
        md.append("## 🔍 Technical Indicators & Structural Confluences")
        md.append("")
        md.append("### Technical Indicators:")
        for ind in spec.technical_indicators:
            md.append(f"- {ind}")
        md.append("")
        md.append("### Structural Confluences:")
        for conf in spec.structural_confluences:
            md.append(f"- {conf}")
        md.append("")
        md.append("---")
        md.append("")
        md.append("## 📋 Formal Algorithmic Rule Engine")
        md.append("")
        md.append("| Rule # | Rule Name | Description | Condition Logic | Action |")
        md.append("|---|---|---|---|---|")
        for r in spec.step_by_step_rules:
            desc = r.description.replace("|", "/")
            cond = r.condition.replace("|", "/")
            act = r.action.replace("|", "/")
            md.append(f"| Rule {r.rule_id} | **{r.name}** | {desc} | `{cond}` | {act} |")
        md.append("")

        return "\n".join(md)
