import os
import tempfile
import pytest
from utils import (
    safe_load_json,
    clean_json_markdown,
    extract_ticker_symbol,
    extract_timeframe,
    extract_directional_bias
)

def test_safe_load_json():
    # Non-existent file returns default
    assert safe_load_json("non_existent_file.json", default={"status": "missing"}) == {"status": "missing"}

    # Valid JSON file
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as f:
        f.write('{"key": "value", "count": 42}')
        tmp_valid = f.name

    # Invalid / Corrupted JSON file
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as f:
        f.write('{broken json content')
        tmp_broken = f.name

    try:
        assert safe_load_json(tmp_valid) == {"key": "value", "count": 42}
        assert safe_load_json(tmp_broken, default=None) is None
    finally:
        if os.path.exists(tmp_valid):
            os.remove(tmp_valid)
        if os.path.exists(tmp_broken):
            os.remove(tmp_broken)

def test_clean_json_markdown():
    # Markdown with code fences and trailing commas
    raw = """```json
    {
      "asset": "NVDA",
      "targets": ["TP1", "TP2",],
      "active": true,
    }
    ```"""
    cleaned = clean_json_markdown(raw)
    assert "```" not in cleaned
    assert '["TP1", "TP2"]' in cleaned or '["TP1", "TP2" ]' in cleaned
    assert '"active": true' in cleaned

def test_extract_ticker_symbol():
    assert extract_ticker_symbol("In this video NVDA breaks above key resistance") == "NVDA"
    assert extract_ticker_symbol("Reviewing AUDCAD momentum on daily chart") == "AUDCAD"
    # Should ignore excluded non-ticker acronyms like VSS, RSI, CPU
    assert extract_ticker_symbol("VSS pipeline and RSI oscillator review", default="DefaultAsset") == "DefaultAsset"

def test_extract_timeframe():
    assert extract_timeframe("Looking at the weekly candles") == "Weekly (1W)"
    assert extract_timeframe("On the 1D chart here") == "Daily (1D)"
    assert extract_timeframe("Switching to 4h chart") == "4-Hour (4H)"
    assert extract_timeframe("Weekly chart with heikin ashi candles") == "Weekly (1W) Heikin Ashi"
    assert extract_timeframe("No timeframe mentioned", default="Unknown") == "Unknown"

def test_extract_directional_bias():
    assert extract_directional_bias("Bullish breakout above resistance") == "Bullish"
    assert extract_directional_bias("Bearish breakdown below support") == "Bearish"
    assert extract_directional_bias("Bearish divergence with RSI indicator") == "Bearish Reversal"
    assert extract_directional_bias("No signals", default="Neutral") == "Neutral"
