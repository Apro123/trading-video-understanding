"""
Pydantic Data Models and Schemas for NVIDIA Video Search and Summarization (VSS).
Implements the contract specified in NVIDIA AI Blueprints:
https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization
"""

from typing import List, Dict, Any, Optional, Union, Tuple
from pydantic import BaseModel, Field
import uuid
import time

class SubtitleEntry(BaseModel):
    """Represents a normalized subtitle entry linked to clip timestamps."""
    index: int
    orig_timestamp_str: str
    orig_timestamp_s: float
    clip_timestamp_s: float
    clip_timestamp_str: str
    clip_end_s: float
    duration_s: float
    text: str

class VideoChunk(BaseModel):
    """Represents a discrete temporal slice of a video clip with associated subtitles."""
    chunk_id: int
    start_s: float
    end_s: float
    duration_s: float
    start_str: str
    end_str: str
    sampled_frame_timestamps: List[float] = Field(default_factory=list)
    subtitle_entries: List[SubtitleEntry] = Field(default_factory=list)
    synchronized_transcript: str = ""

class BoundingBox(BaseModel):
    """Normalized bounding box coordinates [0.0 - 1.0]."""
    ymin: float = 0.0
    xmin: float = 0.0
    ymax: float = 1.0
    xmax: float = 1.0

    def to_pixels(self, width: int, height: int) -> Tuple[int, int, int, int]:
        """Converts normalized coords to (left, top, right, bottom) pixel tuple."""
        return (
            int(self.xmin * width),
            int(self.ymin * height),
            int(self.xmax * width),
            int(self.ymax * height),
        )

class GroundedPriceLevel(BaseModel):
    """Specific price level grounded to chart pixel coordinates and visual cues."""
    price: float
    level_type: str = "support"  # support, resistance, breakout, pivot_high, pivot_low, stop_loss, target, demand_zone, supply_zone
    confidence: float = 0.90
    bounding_box: Optional[BoundingBox] = None
    label: str = ""

class DeicticAnchor(BaseModel):
    """Grounding of informal spoken phrase ('here', 'this level') to visual price coordinate."""
    spoken_phrase: str
    grounded_price: Optional[float] = None
    target_type: str = "price_level"  # price_level, zone, indicator, candle
    bounding_box: Optional[BoundingBox] = None
    explanation: str = ""

class ChartRegion(BaseModel):
    """Segmented region of interest in a trading chart."""
    region_type: str  # price_axis, time_axis, header_ticker, main_canvas, indicator_pane
    bounding_box: BoundingBox
    extracted_text: Optional[str] = None

class ChartGroundingData(BaseModel):
    """Structured visual grounding metadata extracted from chart frames."""
    ticker: Optional[str] = None
    timeframe: Optional[str] = None
    price_range: Optional[Tuple[float, float]] = None  # (min_price, max_price)
    detected_price_levels: List[GroundedPriceLevel] = Field(default_factory=list)
    deictic_anchors: List[DeicticAnchor] = Field(default_factory=list)
    chart_regions: List[ChartRegion] = Field(default_factory=list)
    timeline_labels: List[str] = Field(default_factory=list)
    indicator_summary: Dict[str, Any] = Field(default_factory=dict)

class VlmCaptionResponse(BaseModel):
    """Represents a VLM caption response for a single chunk (standard VSS schema)."""
    chunk_id: int
    start_time: str
    end_time: str
    start_s: float
    end_s: float
    content: str
    reasoning_description: str = ""
    visual_observations: Dict[str, Any] = Field(default_factory=dict)
    detected_trade_signals: List[str] = Field(default_factory=list)
    grounding: Optional[ChartGroundingData] = None

class SummarizationQuery(BaseModel):
    """Standard VSS Summarization Query request model."""
    id: Optional[str] = None
    url: Optional[str] = None
    video_path: Optional[str] = None
    subtitles_path: Optional[str] = None
    prompt: Optional[str] = "Provide a comprehensive technical trading analysis and dense captions."
    scenario: Optional[str] = "trading_technical_analysis"
    custom_events: Optional[List[str]] = Field(default_factory=lambda: [
        "trend_reversal", "breakout", "demand_retest", "rsi_divergence", "entry_trigger"
    ])
    chunk_duration: float = 20.0
    chunk_overlap_duration: float = 3.0
    enable_reasoning: bool = True
    max_tokens: Optional[int] = 1024

class TradingEvent(BaseModel):
    """Chronological trading milestone event identified in the video."""
    event_id: int
    timestamp_range: str
    start_s: float
    end_s: float
    event_type: str
    action_or_concept: str
    chart_evidence: str
    spoken_rationale: str
    trade_significance: str

class TradingRule(BaseModel):
    """Algorithmic execution rule synthesized from trader instructions."""
    rule_id: int
    name: str
    description: str
    condition: str
    action: str

class TradeIdeaSpec(BaseModel):
    """Formal Trade Idea Specification synthesized by VSS Strategy Agent."""
    asset: str = "Unknown"
    timeframe: str = "Unknown"
    directional_bias: str = "Neutral"          # Bullish, Bearish, Neutral
    market_context: str = ""
    core_thesis: str = ""
    entry_trigger: str = ""
    invalidation_stop: str = ""
    take_profit_targets: List[str] = Field(default_factory=list)
    risk_reward_ratio: str = ""
    technical_indicators: List[str] = Field(default_factory=list)
    structural_confluences: List[str] = Field(default_factory=list)
    step_by_step_rules: List[TradingRule] = Field(default_factory=list)

class StrategySegment(BaseModel):
    """A distinct trading strategy identified within a specific temporal segment/chunk of a video."""
    strategy_id: int
    name: str                                  # e.g., "Support & Resistance Retest Strategy"
    timestamp_range: str                       # e.g., "16:08 - 25:40"
    start_s: float
    end_s: float
    duration_s: float
    asset: str = "Market Asset"
    timeframe: str = "Chart Timeframe"
    directional_bias: str = "Neutral"          # Bullish, Bearish, Neutral, Both
    core_thesis: str = ""
    entry_trigger: str = ""
    invalidation_stop: str = ""
    take_profit_targets: List[str] = Field(default_factory=list)
    risk_reward_ratio: str = "1:2"
    technical_indicators: List[str] = Field(default_factory=list)
    structural_confluences: List[str] = Field(default_factory=list)
    step_by_step_rules: List[TradingRule] = Field(default_factory=list)
    transcript_summary: str = ""

class MultiStrategyBreakdown(BaseModel):
    """Collection of trading strategies identified across video intervals."""
    video_id: str
    total_strategies: int
    strategies: List[StrategySegment] = Field(default_factory=list)

class CompletionResponse(BaseModel):
    """Standard VSS Completion Response model."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    video_id: str = "default_video"
    model: str = "nvidia/Cosmos3-Nano"
    created: int = Field(default_factory=lambda: int(time.time()))
    summary: str
    key_events: List[TradingEvent] = Field(default_factory=list)
    chunk_captions: List[VlmCaptionResponse] = Field(default_factory=list)
    trade_idea: Optional[TradeIdeaSpec] = None

class SearchQuery(BaseModel):
    """Search request model querying the video archive or clip."""
    query: str
    top_k: int = 3
    min_score: float = 0.0
    video_id: Optional[str] = None

class SearchResultItem(BaseModel):
    """Single matched video segment from VSS semantic search."""
    chunk_id: int
    start_s: float
    end_s: float
    timestamp_range: str
    score: float
    matched_caption: str
    matched_transcript: str
    reasoning: str
    trade_context: str

class SearchResponse(BaseModel):
    """Response containing ranked search results."""
    query: str
    total_chunks: int
    results: List[SearchResultItem]

class VSSPipelineResult(BaseModel):
    """Comprehensive output bundle returned by the VSS Trading Pipeline."""
    video_path: str
    subtitles_path: str
    duration_s: float
    total_chunks: int
    summary_response: CompletionResponse
    trade_spec: TradeIdeaSpec
    search_index_size: int
    hardware_metrics: Dict[str, float]
    generated_files: Dict[str, str] = Field(default_factory=dict)
