"""
Shared Utilities for Trading Video Understanding.
Provides centralized helpers for JSON loading, Markdown/LLM output cleaning,
and trading NLP heuristics (asset, timeframe, and directional bias extraction).
"""

import os
import re
import json
import logging
from typing import Any, Optional, Set

logger = logging.getLogger(__name__)

# Common non-ticker tokens and acronyms to exclude during ticker symbol extraction
EXCLUDE_TICKER_TOKENS: Set[str] = {
    "VSS", "VLM", "LVS", "CHoCH", "BOS", "FVG", "RSI", "MACD", "SMA", "EMA",
    "GPU", "CPU", "AI", "TP1", "TP2", "TP", "SL", "TV", "HTF", "LTF", "RR",
    "USD", "CAD", "EUR", "GBP", "JPY", "AUD", "NZD", "CHF"
}


# Common 2-letter futures tickers
KNOWN_2LETTER_TICKERS: Set[str] = {"ES", "NQ", "YM", "CL", "GC", "SI", "NG", "ZB", "ZN", "UB"}


def safe_load_json(file_path: str, default: Any = None) -> Any:
    """Safely loads and parses a JSON file, returning `default` if missing or invalid."""
    if not os.path.exists(file_path):
        return default
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Error loading JSON from {file_path}: {e}")
        return default


def clean_json_markdown(text: str) -> str:
    """
    Strips markdown code fences (e.g. ```json ... ```) and removes
    illegal trailing commas inside JSON objects and arrays.
    """
    text = text.strip()
    match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text, re.IGNORECASE)
    if match:
        text = match.group(1).strip()
    # Strip illegal trailing commas before closing braces/brackets
    text = re.sub(r',\s*([\]}])', r'\1', text)
    return text


def extract_ticker_symbol(text: str, default: str = "Market Asset") -> str:
    """Extracts ticker symbol or currency pair dynamically from text."""
    matches = re.findall(r'\b([A-Z]{2,6}(?:/[A-Z]{2,6})?)\b', text)
    for token in matches:
        if token not in EXCLUDE_TICKER_TOKENS and (len(token) >= 3 or token in KNOWN_2LETTER_TICKERS):
            return token
    return default


def extract_timeframe(text: str, default: str = "Chart Timeframe") -> str:
    """Infers chart timeframe from transcript or visual notes."""
    lower = text.lower()
    timeframe = default
    if "weekly" in lower or "1w" in lower:
        timeframe = "Weekly (1W)"
    elif "daily" in lower or "1d" in lower:
        timeframe = "Daily (1D)"
    elif "4h" in lower:
        timeframe = "4-Hour (4H)"
    elif "1h" in lower:
        timeframe = "1-Hour (1H)"
    elif "15m" in lower:
        timeframe = "15-Minute (15M)"

    if "heikin ashi" in lower and "heikin ashi" not in timeframe.lower():
        timeframe = f"{timeframe} Heikin Ashi" if timeframe != default else "Heikin Ashi"

    return timeframe


def extract_directional_bias(text: str, default: str = "Neutral") -> str:
    """Determines dominant directional bias from technical vocabulary."""
    lower = text.lower()
    bullish_signals = ["bullish", "buy zone", "demand", "breakout", "higher high", "support bounce"]
    bearish_signals = ["bearish", "sell zone", "supply", "breakdown", "lower low", "divergence", "rejection"]

    bullish_score = sum(1 for w in bullish_signals if w in lower)
    bearish_score = sum(1 for w in bearish_signals if w in lower)

    if bearish_score > bullish_score and "divergence" in lower:
        return "Bearish Reversal" if "reversal" not in default.lower() else "Bearish"
    if bullish_score > 0 and bearish_score > 0 and abs(bullish_score - bearish_score) <= 1:
        return "Both"
    if bullish_score > bearish_score:
        return "Bullish"
    if bearish_score > bullish_score:
        return "Bearish"
    return default
