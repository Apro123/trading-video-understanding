"""
VSS Vision-Language Model (VLM) Reasoner for Trading Video Understanding.
Implements the VLM inference and reasoning layer using NVIDIA Cosmos 3 Nano.
Adheres to the VSS Blueprint VLM reasoning contract:
https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization
"""

import os
import re
import json
import time
import subprocess
import logging
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image

logger = logging.getLogger(__name__)

try:
    from .config import DEFAULT_COSMOS3_PATH, VSSVLMConfig
    from .schemas import VideoChunk, VlmCaptionResponse, ChartGroundingData
    from .chart_grounding import HybridChartGroundingPipeline
except (ImportError, ValueError):
    from config import DEFAULT_COSMOS3_PATH, VSSVLMConfig
    from schemas import VideoChunk, VlmCaptionResponse, ChartGroundingData
    from chart_grounding import HybridChartGroundingPipeline

# VSS Reasoning Prompt Template for Trading Video Understanding
VSS_CHUNK_REASONING_PROMPT = """You are the NVIDIA Video Search and Summarization (VSS) Trading AI Agent analyzing a technical trading video clip.
Analyze this video segment and the synchronized spoken transcript to extract the trader's technical reasoning and trade ideas.

Video Segment Time Window: [{start_str} - {end_str}]
Synchronized Spoken Subtitles:
{transcript}

Task Instructions:
1. Examine the visual charts across the sampled frames thoroughly:
   - Asset/Ticker Symbol: Read symbol from top-left header, chart label, or watchlist.
   - Timeframe & Candle Type: Note chart timeframe (e.g. Weekly, Daily, 4H, 15m) and candlestick type (Heikin Ashi or Japanese candlesticks).
   - Numerical Price Scale (Y-axis): Read exact numerical price coordinates for swing highs, swing lows, breakout levels, horizontal lines, and consolidation ranges.
   - Timeline Scale (X-axis): Read dates, months, or years corresponding to highlighted candles.
   - Market Structure & Price Action: Identify higher highs/lows, lower highs/lows, swing pivots (e.g. 3-candle fractal patterns), breakout candles, rejection wicks, consolidation channels.
   - Technical Indicators & Oscillators: Read sub-window indicators (RSI, Moving Averages, MACD, Volume) and note any momentum divergences (e.g. price printing higher highs while RSI prints lower highs).
2. Deictic Reference Grounding:
   - Ground all informal, ambiguous, or deictic spoken references ("here", "boom boom", "this candle", "this level", "broke above", "buy zone", "lowest point") directly to the specific visual chart coordinates, price values, dates, or drawn shapes on screen.
3. Determine Trade Setup & Concepts:
   - Identify what technical trading concept or action is being taught or executed: Market Structure Shift / CHoCH, Breakout Confirmation, Invalidation Stop Loss Placement, Retest Demand/Supply Zone, Entry Trigger, Profit Targets.

Provide your response strictly with reasoning and answer tags:
<think>
Detailed visual analysis of asset, timeframe, Y-axis price levels, dates, candle patterns, indicators, and grounding of spoken words to visual elements.
</think>
<answer>
Dense visual caption summarizing the market structure, price action, and actionable trading ideas in this segment.
</answer>"""

VSS_TEXT_STRATEGY_PROMPT = """You are the NVIDIA VSS Algorithmic Strategy Synthesizer.
Review the aggregated multimodal timeline and visual observations below from the trading lesson video.
Synthesize the complete, formal trading strategy and actionable trade idea.

AGGREGATED VIDEO TIMELINE & OBSERVATIONS:
{aggregated_observations}

Provide your synthesis structured as:
1. **Strategy Name & Core Methodology**
2. **Directional Bias & Market Context** (Bullish/Bearish/Neutral, Trend, Consolidation)
3. **Key Technical Indicators & Confluences** (RSI divergence, Moving Averages, Fractal Pivots)
4. **Step-by-Step Execution Rules**:
   - Rule 1: Entry Trigger (Exact price or candlestick condition)
   - Rule 2: Invalidation / Stop Loss (Exact level or condition)
   - Rule 3: Take Profit Targets & Risk-to-Reward Ratio
5. **Chart Entities & Grounded Price Levels** (Asset, Timeframe, Key Price Points)
"""

def get_system_vram_usage() -> float:
    """Queries nvidia-smi for total used GPU memory in GB."""
    try:
        cmd = "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits"
        out = subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL).decode("utf-8").strip()
        return round(float(out.split("\n")[0]) / 1024.0, 3)
    except Exception:
        return 0.0

class VSSVLMReasoner:
    """
    Multimodal Video Reasoner wrapping NVIDIA Cosmos 3 Nano.
    Supports both GPU BF16 tensor execution and robust offline fallback.
    """

    def __init__(self, config: Optional[VSSVLMConfig] = None):
        self.config = config or VSSVLMConfig()
        self.model = None
        self.processor = None
        self.grounding_pipeline = HybridChartGroundingPipeline()
        self._is_loaded = False
        self._has_cuda = False

        try:
            import torch
            self._has_cuda = torch.cuda.is_available()
            if self._has_cuda:
                torch.cuda.reset_peak_memory_stats()
        except ImportError:
            self._has_cuda = False

    def get_vram_stats(self) -> Dict[str, float]:
        """Returns peak and current GPU VRAM statistics."""
        try:
            import torch
            if not torch.cuda.is_available():
                return {"allocated_gb": 0.0, "reserved_gb": 0.0, "peak_allocated_gb": 0.0, "peak_reserved_gb": 0.0, "system_used_gb": 0.0}
            return {
                "allocated_gb": round(torch.cuda.memory_allocated() / (1024 ** 3), 3),
                "reserved_gb": round(torch.cuda.memory_reserved() / (1024 ** 3), 3),
                "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3),
                "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / (1024 ** 3), 3),
                "system_used_gb": get_system_vram_usage()
            }
        except Exception:
            return {"allocated_gb": 0.0, "reserved_gb": 0.0, "peak_allocated_gb": 0.0, "peak_reserved_gb": 0.0, "system_used_gb": 0.0}

    def load_model(self):
        """Loads Cosmos 3 Nano model on GPU if available and not skipped."""
        if self._is_loaded or self.config.skip_model:
            return

        if not self._has_cuda:
            logger.info("CUDA not available; running in CPU fallback/mock mode.")
            return

        if not os.path.exists(self.config.model_path):
            logger.info(f"Model path not found at {self.config.model_path}; using fallback reasoning.")
            return

        try:
            import torch
            from transformers import AutoProcessor
            from transformers.models.cosmos3_omni.modeling_cosmos3_omni import Cosmos3OmniForConditionalGeneration

            dtype = torch.bfloat16 if self.config.torch_dtype == "bfloat16" else torch.float16
            logger.info(f"Loading NVIDIA Cosmos 3 Nano from {self.config.model_path} on {self.config.device}...")
            self.processor = AutoProcessor.from_pretrained(self.config.model_path, local_files_only=True)
            self.model = Cosmos3OmniForConditionalGeneration.from_pretrained(
                self.config.model_path,
                torch_dtype=dtype,
                device_map=self.config.device,
                local_files_only=True
            )
            self.model.eval()
            self._is_loaded = True
            vram = self.get_vram_stats()
            logger.info(f"Model loaded successfully. Peak allocated VRAM: {vram['peak_allocated_gb']} GB.")
        except Exception as e:
            logger.warning(f"Failed to load Cosmos 3 Nano on GPU: {e}. Switching to fallback mode.")
            self._is_loaded = False

    def reason_over_chunk(
        self,
        chunk: VideoChunk,
        frames: List[Image.Image]
    ) -> VlmCaptionResponse:
        """
        Processes a single video chunk through Cosmos 3 Nano with synchronized subtitles
        and hybrid visual grounding.
        Returns a VlmCaptionResponse adhering to the VSS schema.
        """
        # 1. Execute hybrid visual grounding over chunk and sampled frames
        grounding_data = self.grounding_pipeline.ground_chunk(chunk, frames)
        grounding_prompt_block = self.grounding_pipeline.format_grounding_prompt_block(grounding_data)

        transcript = chunk.synchronized_transcript or "(No dialogue in this interval)"
        prompt = (
            f"{VSS_CHUNK_REASONING_PROMPT.format(start_str=chunk.start_str, end_str=chunk.end_str, transcript=transcript)}\n\n"
            f"{grounding_prompt_block}"
        )

        raw_output = ""
        reasoning_content = ""
        answer_content = ""

        # GPU Model Inference
        if self._is_loaded and self.model is not None and self.processor is not None and frames:
            try:
                import torch
                # Build chat template structure
                content = []
                for img in frames[:self.config.max_new_tokens]:
                    content.append({"type": "image", "image": img})
                content.append({"type": "text", "text": prompt})

                messages = [{"role": "user", "content": content}]
                text_input = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = self.processor(text=[text_input], images=frames, return_tensors="pt")
                inputs = {k: v.to(self.config.device) for k, v in inputs.items()}

                with torch.inference_mode():
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=self.config.max_new_tokens,
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                        do_sample=True
                    )
                # Decode
                prompt_len = inputs["input_ids"].shape[1]
                output_ids = generated_ids[:, prompt_len:]
                raw_output = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            except Exception as e:
                logger.warning(f"Inference failed on chunk {chunk.chunk_id}: {e}. Using rule-based fallback.")
                raw_output = ""

        # Fallback / parsing logic
        if not raw_output:
            raw_output = self._generate_fallback_chunk_analysis(chunk, frames)

        # Parse <think> and <answer> tags
        think_match = re.search(r'<think>(.*?)</think>', raw_output, re.DOTALL | re.IGNORECASE)
        answer_match = re.search(r'<answer>(.*?)</answer>', raw_output, re.DOTALL | re.IGNORECASE)

        if think_match and answer_match:
            reasoning_content = think_match.group(1).strip()
            answer_content = answer_match.group(1).strip()
        elif think_match:
            reasoning_content = think_match.group(1).strip()
            answer_content = raw_output[think_match.end():].strip()
        else:
            reasoning_content = "Extracted candlestick patterns, chart drawings, and aligned transcript."
            answer_content = raw_output

        # Clean prompt echo if model echoed the template instruction
        if "Detailed visual analysis" in reasoning_content or "Detailed visual observations" in reasoning_content or reasoning_content.strip().startswith("[Detailed"):
            reasoning_content = answer_content

        # Enrich answer content with grounded coordinates if not already present
        if "[Grounded Coordinates:" not in answer_content:
            answer_content = self.grounding_pipeline.enrich_caption_with_grounding(answer_content, grounding_data)

        # Extract detected signals
        signals = self._detect_trade_signals(chunk, reasoning_content + " " + answer_content)

        return VlmCaptionResponse(
            chunk_id=chunk.chunk_id,
            start_time=chunk.start_str,
            end_time=chunk.end_str,
            start_s=chunk.start_s,
            end_s=chunk.end_s,
            content=answer_content,
            reasoning_description=reasoning_content,
            detected_trade_signals=signals,
            grounding=grounding_data,
            visual_observations={
                "frame_count": len(frames),
                "timestamps": chunk.sampled_frame_timestamps,
                "has_transcript": bool(chunk.synchronized_transcript),
                "grounded_coordinates": grounding_data.model_dump()
            }
        )

    def synthesize_strategy(self, chunk_captions: List[VlmCaptionResponse]) -> str:
        """
        Uses text LLM reasoning to synthesize formal trade strategy rules
        from all aggregated chunk captions and observations.
        """
        obs_text = []
        for c in chunk_captions:
            obs_text.append(f"[{c.start_time} - {c.end_time}]")
            obs_text.append(f"Reasoning: {c.reasoning_description}")
            obs_text.append(f"Summary: {c.content}")
            if c.detected_trade_signals:
                obs_text.append(f"Signals: {', '.join(c.detected_trade_signals)}")
            obs_text.append("-" * 40)

        aggregated = "\n".join(obs_text)
        prompt = VSS_TEXT_STRATEGY_PROMPT.format(aggregated_observations=aggregated)

        # If model loaded, generate via text LLM
        if self._is_loaded and self.model is not None and self.processor is not None:
            try:
                import torch
                messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
                text_input = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = self.processor(text=[text_input], return_tensors="pt")
                inputs = {k: v.to(self.config.device) for k, v in inputs.items()}

                with torch.inference_mode():
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=1024,
                        temperature=0.2,
                        top_p=0.8,
                        do_sample=True
                    )
                prompt_len = inputs["input_ids"].shape[1]
                output_ids = generated_ids[:, prompt_len:]
                synth = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                if len(synth) > 100:
                    return synth
            except Exception as e:
                logger.warning(f"Text synthesis failed: {e}. Using deterministic synthesis.")

        return self._generate_fallback_strategy_synthesis(chunk_captions)

    def _detect_trade_signals(self, chunk: VideoChunk, text: str) -> List[str]:
        """Detects trading signal categories from text and chunk context."""
        lower = (chunk.synchronized_transcript + " " + text).lower()
        signals = []
        if any(w in lower for w in ["broke above", "breakout", "crossing above", "breached resistance"]):
            signals.append("BREAKOUT_LEVEL_VALIDATION")
        if any(w in lower for w in ["bullish", "went bullish", "trend reversal", "choch", "change of character"]):
            signals.append("BULLISH_TREND_REVERSAL")
        if any(w in lower for w in ["bearish", "breakdown", "lower low", "broke below"]):
            signals.append("BEARISH_TREND_REVERSAL")
        if any(w in lower for w in ["swing high", "swing low", "fractal", "pivot", "market structure"]):
            signals.append("MARKET_STRUCTURE_SHIFT")
        if any(w in lower for w in ["buy zone", "consolidation", "retest", "pullback", "support", "demand"]):
            signals.append("DEMAND_ZONE_RETEST")
        if any(w in lower for w in ["sell zone", "supply", "resistance", "liquidity pool"]):
            signals.append("SUPPLY_ZONE_RETEST")
        if any(w in lower for w in ["rsi", "divergence", "overbought", "oversold", "oscillator"]):
            signals.append("MOMENTUM_RSI_DIVERGENCE")
        if any(w in lower for w in ["stop loss", "invalidation", "lowest point", "protected swing", "stop"]):
            signals.append("PROTECTED_SWING_STOP")
        return signals

    def _generate_fallback_chunk_analysis(self, chunk: VideoChunk, frames: List[Image.Image]) -> str:
        """Deterministic fallback analysis when model weights or GPU are unavailable."""
        transcript = chunk.synchronized_transcript
        lower = transcript.lower()

        observations = []
        signals = self._detect_trade_signals(chunk, transcript)

        observations.append(f"Visual candlestick chart analysis between {chunk.start_str} and {chunk.end_str}.")
        if "bullish" in lower:
            observations.append("Price action demonstrates bullish momentum and upside structural progression.")
        elif "bearish" in lower:
            observations.append("Price action demonstrates bearish momentum and downside structural progression.")

        if "breakout" in lower or "broke" in lower:
            observations.append("Chart highlights price breaking beyond structural reference pivot levels.")
        if "buy zone" in lower or "demand" in lower or "support" in lower:
            observations.append("Chart highlights consolidation zone acting as support for pullback retest.")
        if "divergence" in lower or "rsi" in lower:
            observations.append("Sub-window oscillator exhibits momentum divergence against price extremes.")
        if "stop" in lower or "lowest" in lower or "invalidation" in lower:
            observations.append("Structural invalidation anchor established at key swing pivot level.")

        if transcript:
            observations.append(f"Speaker notes: '{transcript[:100]}...'.")

        reasoning = " ".join(observations)
        caption = f"Market structure review ({', '.join(signals) if signals else 'Price Action Evaluation'}) during interval {chunk.start_str} - {chunk.end_str}."

        return f"<think>\n{reasoning}\n</think>\n<answer>\n{caption}\n</answer>"

    def _generate_fallback_strategy_synthesis(self, chunk_captions: List[VlmCaptionResponse]) -> str:
        """Generates deterministic strategy specification when text model is unavailable."""
        all_text = " ".join([c.content + " " + c.reasoning_description for c in chunk_captions]).lower()

        if "rsi" in all_text and "divergence" in all_text:
            return """**Strategy Name:** Momentum Divergence Reversal Strategy
**Core Methodology:** Exploit momentum exhaustion where price reaches extreme levels while the momentum oscillator prints divergence at key structural boundaries.

**Key Technical Indicators & Confluences:**
- Momentum Oscillator: Divergence against price swing highs/lows.
- Overhead Structure: Key horizontal support or resistance level test.
- Candlestick Action: Reversal candle confirmation.

**Step-by-Step Execution Rules:**
1. **Entry Trigger:** Enter upon confirmed close of reversal trigger candle following divergence.
2. **Invalidation/Stop Loss:** Place protective stop beyond the most recent extreme pivot.
3. **Take Profit Targets:** Target opposite structural liquidity pool with minimum 2:1 Risk-to-Reward ratio."""

        return """**Strategy Name:** Market Structure Breakout & Zone Retest
**Core Methodology:** Multi-timeframe trend mapping using fractal swing pivots, confirming structural breakouts and entering on pullbacks into institutional demand/supply zones.

**Key Technical Indicators & Confluences:**
- Candlestick Action: Decisive breakout close above previous structural swing pivot.
- Swing Pivots: Valid fractal swing high/low confirmation.
- Zone Retest: Consolidation boundary retested as support/resistance.

**Step-by-Step Execution Rules:**
1. **Entry Trigger:** Enter upon price pulling back to test the previous consolidation level with candlestick confirmation.
2. **Invalidation/Stop Loss:** Protective stop placed beyond the structural pivot trough.
3. **Take Profit Targets:** Target next major structural resistance/liquidity zone with favorable Risk-to-Reward ratio."""
