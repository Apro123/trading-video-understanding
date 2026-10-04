"""
Generic Nemotron 3.5 Lightning Reasoning Engine for Trading Video Understanding.
Interfaces with Nemotron 3.5 Lightning running via vLLM / OpenAI-compatible endpoint.

Provides zero-shot, prompt-driven algorithmic trading reasoning:
- Extraction of structured TradeIdeaSpec and formal executable rules
- Extraction of chronological TradingEvent milestones
- Synthesis of executive narrative summaries with deictic chart grounding

Completely generic: contains NO hardcoded ticker names, dates, or clip-specific logic.
"""

import os
import re
import json
import logging
from typing import List, Dict, Any, Optional

try:
    import httpx
except ImportError:
    httpx = None

try:
    from .schemas import (
        VideoChunk,
        VlmCaptionResponse,
        TradingEvent,
        TradeIdeaSpec,
        TradingRule
    )
    from .config import DEFAULT_LLM_BASE_URL, DEFAULT_LLM_MODEL
except (ImportError, ValueError):
    from schemas import (
        VideoChunk,
        VlmCaptionResponse,
        TradingEvent,
        TradeIdeaSpec,
        TradingRule
    )
    from config import DEFAULT_LLM_BASE_URL, DEFAULT_LLM_MODEL

logger = logging.getLogger(__name__)

SYSTEM_REASONING_PROMPT = (
    "You are an expert financial market analyst and algorithmic trading strategy architect. "
    "You analyze multimodal video observations, candlestick chart data, and spoken transcripts "
    "to extract objective market structure, formal trading rules, and actionable trade setups. "
    "Be direct, precise, and concise. Output ONLY valid JSON matching the requested schema."
)

def _clean_json_markdown(text: str) -> str:
    """Strips markdown code fences and returns pure JSON text."""
    text = text.strip()
    match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text, re.IGNORECASE)
    if match:
        text = match.group(1).strip()
    # Strip illegal trailing commas in JSON objects and arrays
    text = re.sub(r',\s*([\]}])', r'\1', text)
    return text

class NemotronReasoner:
    """
    OpenAI-compatible reasoning client connecting to Nemotron 3.5 Lightning.
    Auto-detects the host gateway whether running natively or inside Docker.
    """

    _cached_base_url: Optional[str] = None
    _cached_available: Optional[bool] = None

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_name: str = DEFAULT_LLM_MODEL,
        timeout_s: float = 120.0
    ):
        self.model_name = model_name
        self.timeout_s = timeout_s
        self.base_url = self._resolve_base_url(base_url)

    @classmethod
    def _resolve_base_url(cls, explicit_url: Optional[str]) -> str:
        """Resolves the accessible base URL for the LLM server with caching."""
        if explicit_url:
            return explicit_url.rstrip("/")

        if cls._cached_base_url is not None:
            return cls._cached_base_url

        env_url = os.environ.get("LLM_BASE_URL")
        if env_url:
            cls._cached_base_url = env_url.rstrip("/")
            return cls._cached_base_url

        # Candidate endpoints to check:
        candidates = [
            DEFAULT_LLM_BASE_URL.rstrip("/"),
            "http://localhost:8000/v1",        # NVIDIA DGX Spark GPU server
            "http://172.17.0.1:8000/v1",  # Docker standard bridge to Linux host
            "http://127.0.0.1:8000/v1",  # Localhost / --network host
            "http://localhost:8000/v1"
        ]

        if httpx:
            for url in candidates:
                try:
                    r = httpx.get(f"{url}/models", timeout=0.4)
                    if r.status_code == 200:
                        cls._cached_base_url = url
                        cls._cached_available = True
                        return url
                except Exception:
                    continue

        cls._cached_base_url = candidates[0]
        cls._cached_available = False
        return cls._cached_base_url

    def is_available(self) -> bool:
        """Checks if the Nemotron server is reachable."""
        if not httpx:
            return False
        if NemotronReasoner._cached_available is not None:
            return NemotronReasoner._cached_available

        try:
            r = httpx.get(f"{self.base_url}/models", timeout=0.8)
            NemotronReasoner._cached_available = (r.status_code == 200)
        except Exception:
            NemotronReasoner._cached_available = False

        return NemotronReasoner._cached_available

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 5000,
        temperature: float = 0.1
    ) -> Optional[str]:
        """Sends a chat completion request to Nemotron."""
        if not self.is_available() or not httpx:
            return None

        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "chat_template_kwargs": {"enable_thinking": False}
        }

        try:
            r = httpx.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                timeout=self.timeout_s
            )
            if r.status_code == 200:
                data = r.json()
                choices = data.get("choices", [])
                if choices:
                    msg = choices[0].get("message", {})
                    content = msg.get("content")
                    if not content:
                        # Fallback if content was placed in reasoning
                        content = msg.get("reasoning") or msg.get("reasoning_content")
                    if content:
                        return content.strip()
            else:
                logger.warning(f"Nemotron API error {r.status_code}: {r.text[:200]}")
        except Exception as e:
            logger.warning(f"Nemotron request failed: {e}")

        return None

    def synthesize_trade_idea(
        self,
        chunks: List[VideoChunk],
        chunk_captions: List[VlmCaptionResponse],
        key_events: Optional[List[TradingEvent]] = None
    ) -> Optional[TradeIdeaSpec]:
        """
        Synthesizes a complete TradeIdeaSpec from video chunks, VLM captions,
        and chronological milestones using Nemotron reasoning.
        """
        if not self.is_available():
            return None

        # Assemble video context
        timeline_blocks = []
        cap_map = {c.chunk_id: c for c in chunk_captions}

        for chunk in chunks:
            cap = cap_map.get(chunk.chunk_id)
            block = [f"VIDEO CHUNK #{chunk.chunk_id} [{chunk.start_str} - {chunk.end_str}]:"]
            if chunk.synchronized_transcript:
                block.append(f'  Spoken Transcript: "{chunk.synchronized_transcript}"')
            if cap:
                block.append(f'  VLM Visual Observation: "{cap.content}"')
                if cap.reasoning_description:
                    block.append(f'  Chart Visual Details: "{cap.reasoning_description[:300]}..."')
                if cap.detected_trade_signals:
                    block.append(f'  Detected Signals: {", ".join(cap.detected_trade_signals)}')
            timeline_blocks.append("\n".join(block))

        timeline_text = "\n\n".join(timeline_blocks)

        prompt = f"""Review the sequential video timeline, candlestick chart observations, and spoken subtitles below from a technical trading lesson:

{timeline_text}

Task: Formulate the formal algorithmic trading idea and execution strategy explained in this video.
Return a valid JSON object matching this exact schema:
{{
  "asset": "string (Ticker symbol or currency pair identified from chart/speech)",
  "timeframe": "string (Chart timeframe identified from chart header or speech)",
  "directional_bias": "Bullish" | "Bearish" | "Neutral",
  "market_context": "string (Current market structure, trend regime, and key technical context)",
  "core_thesis": "string (Underlying trading thesis and edge)",
  "entry_trigger": "string (Specific price action, breakout, or retest condition that triggers entry)",
  "invalidation_stop": "string (Objective structural stop-loss / invalidation level)",
  "take_profit_targets": ["string (TP1 level/condition)", "string (TP2 level/condition)"],
  "risk_reward_ratio": "string (Estimated risk-to-reward ratio, e.g. '1:2' or '2.5:1')",
  "technical_indicators": ["string (Technical indicator or tool 1)", "..."],
  "structural_confluences": ["string (Technical chart confluence 1)", "..."],
  "step_by_step_rules": [
    {{
      "rule_id": 1,
      "name": "string (Descriptive rule name)",
      "description": "string (Detailed rule explanation)",
      "condition": "string (Technical condition)",
      "action": "string (Execution action)"
    }}
  ]
}}

Ensure all rules, prices, and anchors are derived strictly from the video evidence. Return ONLY valid JSON.
"""

        messages = [
            {"role": "system", "content": SYSTEM_REASONING_PROMPT},
            {"role": "user", "content": prompt}
        ]

        raw = self.chat_completion(messages, max_tokens=6000, temperature=0.1)
        if not raw:
            return None

        clean_json = _clean_json_markdown(raw)
        try:
            data = json.loads(clean_json)
            # Validate rules defensively
            rules = []
            raw_rules = data.get("step_by_step_rules", [])
            for idx, r_data in enumerate(raw_rules):
                if not isinstance(r_data, dict):
                    continue
                rules.append(TradingRule(
                    rule_id=int(r_data.get("rule_id", idx + 1)),
                    name=str(r_data.get("name", f"Rule {idx+1}")),
                    description=str(r_data.get("description", "")),
                    condition=str(r_data.get("condition", r_data.get("condition_logic", ""))),
                    action=str(r_data.get("action", r_data.get("execution_action", "")))
                ))

            # Normalize list and string fields
            targets = data.get("take_profit_targets", [])
            if isinstance(targets, str):
                targets = [targets]
            indicators = data.get("technical_indicators", [])
            if isinstance(indicators, str):
                indicators = [indicators]
            confluences = data.get("structural_confluences", [])
            if isinstance(confluences, str):
                confluences = [confluences]

            bias = str(data.get("directional_bias", "Neutral")).capitalize()
            if bias not in ["Bullish", "Bearish", "Neutral"]:
                bias = "Bullish" if "bull" in bias.lower() else ("Bearish" if "bear" in bias.lower() else "Neutral")

            return TradeIdeaSpec(
                asset=str(data.get("asset", "Unknown")),
                timeframe=str(data.get("timeframe", "Unknown")),
                directional_bias=bias,
                market_context=str(data.get("market_context", "")),
                core_thesis=str(data.get("core_thesis", "")),
                entry_trigger=str(data.get("entry_trigger", "")),
                invalidation_stop=str(data.get("invalidation_stop", "")),
                take_profit_targets=targets,
                risk_reward_ratio=str(data.get("risk_reward_ratio", "")),
                technical_indicators=indicators,
                structural_confluences=confluences,
                step_by_step_rules=rules
            )
        except Exception as e:
            logger.warning(f"Failed to parse TradeIdeaSpec JSON from Nemotron: {e}\nRaw output: {raw[:300]}")
            return None

    def synthesize_milestone_events(
        self,
        chunks: List[VideoChunk],
        chunk_captions: List[VlmCaptionResponse]
    ) -> Optional[List[TradingEvent]]:
        """
        Synthesizes chronological milestone events (TradingEvent) from video chunks and captions.
        """
        if not self.is_available():
            return None

        timeline_blocks = []
        cap_map = {c.chunk_id: c for c in chunk_captions}

        for chunk in chunks:
            cap = cap_map.get(chunk.chunk_id)
            block = [f"CHUNK #{chunk.chunk_id} [{chunk.start_str} - {chunk.end_str}] ({chunk.start_s}s - {chunk.end_s}s):"]
            if chunk.synchronized_transcript:
                block.append(f'  Transcript: "{chunk.synchronized_transcript}"')
            if cap:
                block.append(f'  Observation: "{cap.content}"')
                if cap.reasoning_description and not cap.reasoning_description.strip().startswith("[Detailed visual"):
                    block.append(f'  Details: "{cap.reasoning_description[:250]}..."')
            timeline_blocks.append("\n".join(block))

        timeline_text = "\n\n".join(timeline_blocks)

        prompt = f"""Review the sequential video chunks and chart observations from this trading video:

{timeline_text}

Task: Group and synthesize the video into chronological milestone trading events.
Extract each key event as a JSON object in an array matching this schema:
[
  {{
    "event_id": 1,
    "timestamp_range": "string (e.g. '00:00 - 00:35')",
    "start_s": float,
    "end_s": float,
    "event_type": "string (e.g. 'Market Structure Mapping', 'Breakout Validation', 'Demand Zone Retest', 'Momentum Divergence', 'Invalidation Stop Placement')",
    "action_or_concept": "string (Concise description of the trading concept or market action)",
    "chart_evidence": "string (Specific visual chart evidence, candle patterns, levels, or indicators)",
    "spoken_rationale": "string (Key quote or explanation spoken by presenter)",
    "trade_significance": "string (Significance to the trade setup)"
  }}
]

Return ONLY the valid JSON array.
"""

        messages = [
            {"role": "system", "content": SYSTEM_REASONING_PROMPT},
            {"role": "user", "content": prompt}
        ]

        raw = self.chat_completion(messages, max_tokens=6000, temperature=0.1)
        if not raw:
            return None

        clean_json = _clean_json_markdown(raw)
        try:
            items = json.loads(clean_json)
            events = []
            for idx, item in enumerate(items):
                if not isinstance(item, dict):
                    continue
                events.append(TradingEvent(
                    event_id=int(item.get("event_id", idx + 1)),
                    timestamp_range=str(item.get("timestamp_range", "")),
                    start_s=float(item.get("start_s", 0.0)),
                    end_s=float(item.get("end_s", 0.0)),
                    event_type=str(item.get("event_type", "Market Structure")),
                    action_or_concept=str(item.get("action_or_concept", "")),
                    chart_evidence=str(item.get("chart_evidence", "")),
                    spoken_rationale=str(item.get("spoken_rationale", "")),
                    trade_significance=str(item.get("trade_significance", ""))
                ))
            return events if events else None
        except Exception as e:
            logger.warning(f"Failed to parse TradingEvent JSON from Nemotron: {e}\nRaw output: {raw[:300]}")
            return None

    def synthesize_narrative_summary(
        self,
        chunks: List[VideoChunk],
        chunk_captions: List[VlmCaptionResponse],
        events: List[TradingEvent],
        trade_spec: Optional[TradeIdeaSpec]
    ) -> Optional[str]:
        """
        Synthesizes an executive narrative summary explaining the trading progression
        and deictic grounding using Nemotron reasoning.
        """
        if not self.is_available():
            return None

        total_dur = chunks[-1].end_s if chunks else 0.0
        n_chunks = len(chunks)

        events_summary = []
        for ev in events:
            events_summary.append(
                f"- [{ev.timestamp_range}] {ev.event_type}: {ev.action_or_concept} | Evidence: {ev.chart_evidence}"
            )
        events_text = "\n".join(events_summary)

        spec_text = ""
        if trade_spec:
            spec_text = (
                f"Asset: {trade_spec.asset} ({trade_spec.timeframe})\n"
                f"Bias: {trade_spec.directional_bias}\n"
                f"Core Thesis: {trade_spec.core_thesis}\n"
                f"Entry Trigger: {trade_spec.entry_trigger}\n"
                f"Stop Loss: {trade_spec.invalidation_stop}"
            )

        prompt = f"""You are the lead technical trading analyst. Synthesize an executive summary of this video clip ({n_chunks} segments, {total_dur:.1f}s total duration).

CHRONOLOGICAL MILESTONE EVENTS:
{events_text}

STRATEGY PARAMETERS:
{spec_text}

Write a comprehensive executive summary in Markdown covering:
1. Executive Summary paragraph introducing the market regime, asset, and educational objective.
2. Key Methodological Progression highlighting the sequential steps demonstrated.
3. Deictic Grounding & Chart Confluences explaining how informal verbal phrases ('here', 'this candle', 'this point', 'buy zone') resolve to specific price levels, candle patterns, and indicators on the chart.

Keep it professional, concise, and grounded in the data. Do NOT include extraneous conversational filler.
"""

        messages = [
            {"role": "system", "content": SYSTEM_REASONING_PROMPT},
            {"role": "user", "content": prompt}
        ]

        raw = self.chat_completion(messages, max_tokens=3000, temperature=0.2)
        return raw.strip() if raw else None
