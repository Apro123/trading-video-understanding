import pytest
try:
    from trade_strategy_agent import VSSTradeStrategyAgent
    from summarizer import VSSLVSSummarizer
    from schemas import VideoChunk, VlmCaptionResponse, CompletionResponse
except ImportError:
    from video_search_and_summarization.trade_strategy_agent import VSSTradeStrategyAgent
    from video_search_and_summarization.summarizer import VSSLVSSummarizer
    from video_search_and_summarization.schemas import VideoChunk, VlmCaptionResponse, CompletionResponse

def test_trade_strategy_agent_synthesis():
    agent = VSSTradeStrategyAgent()

    chunks = [
        VideoChunk(chunk_id=0, start_s=0.0, end_s=20.0, duration_s=20.0, start_str="00:00", end_str="00:20", synchronized_transcript="NVDA went bullish January 30 2023 broke above 1927"),
        VideoChunk(chunk_id=1, start_s=20.0, end_s=40.0, duration_s=20.0, start_str="00:20", end_str="00:40", synchronized_transcript="pulling back into this nice buy zone consolidation")
    ]
    captions = [
        VlmCaptionResponse(chunk_id=0, start_time="00:00", end_time="00:20", start_s=0.0, end_s=20.0, content="Bullish breakout above 19.27", detected_trade_signals=["BREAKOUT_LEVEL_VALIDATION", "BULLISH_TREND_REVERSAL"]),
        VlmCaptionResponse(chunk_id=1, start_time="00:20", end_time="00:40", start_s=20.0, end_s=40.0, content="Retest of demand buy zone", detected_trade_signals=["DEMAND_ZONE_RETEST"])
    ]

    summarizer = VSSLVSSummarizer()
    completion = summarizer.summarize(chunks, captions)

    spec = agent.synthesize_trade_idea(completion)

    assert "NVDA" in spec.asset
    assert spec.directional_bias == "Bullish"
    assert "19.27" in spec.core_thesis or "1927" in spec.core_thesis or "breakout" in spec.core_thesis.lower()
    assert len(spec.step_by_step_rules) >= 3

    # Check Markdown formatting
    md = agent.export_spec_markdown(spec)
    assert "# 🎯 NVIDIA VSS Algorithmic Trade Strategy Specification" in md
    assert "NVDA" in md
    assert "Rule 1" in md
