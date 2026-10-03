import pytest
try:
    from search_engine import VSSSearchIndex
    from schemas import VideoChunk, VlmCaptionResponse
except ImportError:
    from video_search_and_summarization.search_engine import VSSSearchIndex
    from video_search_and_summarization.schemas import VideoChunk, VlmCaptionResponse

def test_search_engine_indexing_and_retrieval():
    index = VSSSearchIndex(video_id="nvda_test")

    chunks = [
        VideoChunk(
            chunk_id=0, start_s=0.0, end_s=20.0, duration_s=20.0,
            start_str="00:00", end_str="00:20",
            synchronized_transcript="speaker reviews initial downtrend on weekly chart"
        ),
        VideoChunk(
            chunk_id=1, start_s=20.0, end_s=40.0, duration_s=20.0,
            start_str="00:20", end_str="00:40",
            synchronized_transcript="January 30 2023 Nvidia broke above 1927 officially went bullish"
        ),
        VideoChunk(
            chunk_id=2, start_s=40.0, end_s=60.0, duration_s=20.0,
            start_str="00:40", end_str="01:00",
            synchronized_transcript="retesting the previous consolidation demand buy zone"
        )
    ]

    captions = [
        VlmCaptionResponse(
            chunk_id=0, start_time="00:00", end_time="00:20", start_s=0.0, end_s=20.0,
            content="Downtrend analysis on weekly candles",
            detected_trade_signals=["MARKET_STRUCTURE_SHIFT"]
        ),
        VlmCaptionResponse(
            chunk_id=1, start_time="00:20", end_time="00:40", start_s=20.0, end_s=40.0,
            content="Bullish breakout candle above $19.27",
            detected_trade_signals=["BREAKOUT_LEVEL_VALIDATION", "BULLISH_TREND_REVERSAL"]
        ),
        VlmCaptionResponse(
            chunk_id=2, start_time="00:40", end_time="01:00", start_s=40.0, end_s=60.0,
            content="Retest of accumulation buy zone",
            detected_trade_signals=["DEMAND_ZONE_RETEST"]
        )
    ]

    index.index_chunks(chunks, captions)
    assert len(index.chunks_data) == 3

    # Query 1: exact price level breakout
    res1 = index.search("1927 breakout", top_k=1)
    assert len(res1.results) == 1
    assert res1.results[0].chunk_id == 1
    assert "19.27" in res1.results[0].matched_caption or "1927" in res1.results[0].matched_transcript

    # Query 2: buy zone retest
    res2 = index.search("buy zone accumulation", top_k=1)
    assert len(res2.results) == 1
    assert res2.results[0].chunk_id == 2
    assert "buy zone" in res2.results[0].matched_caption.lower() or "buy zone" in res2.results[0].matched_transcript.lower()

    # Export index test
    exported = index.export_index_json()
    assert exported["total_chunks"] == 3
    assert exported["video_id"] == "nvda_test"
