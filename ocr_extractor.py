"""
Generic OCR Extractor for Trading Chart Videos.
Extracts visible chart text, potential tickers, timeframes, indicator labels
(e.g., RSI, MA, MACD), numerical axis levels, and calendar dates without hardcoded biases.
"""

import cv2
import numpy as np
import re
from typing import Dict, List, Any, Optional

class TradingChartOCRExtractor:
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu
        self._reader = None

    @property
    def reader(self):
        if self._reader is None:
            import easyocr
            print("[OCRExtractor] Initializing EasyOCR engine...")
            self._reader = easyocr.Reader(['en'], gpu=self.use_gpu)
        return self._reader

    def extract_chart_metadata(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Extracts key UI text blocks from an arbitrary trading chart frame.
        """
        h, w, _ = frame.shape
        meta = {
            "detected_header_lines": [],
            "ticker_candidates": [],
            "timeframe_candidates": [],
            "indicators_detected": [],
            "visible_price_levels": [],
            "visible_dates": []
        }

        try:
            # 1. Header region: top 18% height, across the width
            header_crop = frame[0:int(h * 0.18), 0:w]
            header_results = self.reader.readtext(header_crop)
            for bbox, text, conf in header_results:
                if conf < 0.25:
                    continue
                clean = text.strip()
                meta["detected_header_lines"].append(clean)

                # Detect potential tickers (uppercase words of 2-6 chars or pairs like EURUSD, CADCHF)
                words = re.findall(r"\b[A-Z]{2,7}(?:/[A-Z]{2,7})?\b", clean)
                for word in words:
                    if word not in ["NASDAQ", "NYSE", "USD", "EUR", "GBP", "JPY", "CAD", "CHF", "AUD", "NEW", "SAVE"]:
                        if word not in meta["ticker_candidates"]:
                            meta["ticker_candidates"].append(word)

                # Detect timeframe indicators (1m, 5m, 15m, 1h, 4h, 1D, 1W, Daily, Weekly)
                tf_matches = re.findall(r"\b(?:1m|5m|15m|30m|1h|2h|4h|1D|1W|1M)\b", clean, re.IGNORECASE)
                for tf in tf_matches:
                    tf_u = tf.upper()
                    if tf_u not in meta["timeframe_candidates"]:
                        meta["timeframe_candidates"].append(tf_u)

            # 2. Main chart & Indicator detection (scan whole frame for technical keywords)
            full_results = self.reader.readtext(frame)
            technical_keywords = ["RSI", "MACD", "EMA", "SMA", "ATR", "STOCH", "BOLL", "HEIKIN", "ASHI", "MA"]
            for bbox, text, conf in full_results:
                if conf < 0.3:
                    continue
                clean = text.strip().upper()
                for kw in technical_keywords:
                    if kw in clean and kw not in meta["indicators_detected"]:
                        meta["indicators_detected"].append(kw)

                # Collect price-like numbers (integers or decimals)
                num_matches = re.findall(r"\b\d{1,6}(?:\.\d{1,5})?\b", text)
                for num_s in num_matches:
                    try:
                        val = float(num_s)
                        if val > 0 and val not in meta["visible_price_levels"]:
                            # Keep top 15 most prominent numbers
                            if len(meta["visible_price_levels"]) < 15:
                                meta["visible_price_levels"].append(val)
                    except ValueError:
                        pass

                # Detect dates
                months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                if any(m in text for m in months) or re.search(r"202\d", text):
                    clean_date = text.strip()
                    if clean_date not in meta["visible_dates"]:
                        meta["visible_dates"].append(clean_date)

        except Exception as e:
            print(f"[OCRExtractor Warning] OCR extraction encountered: {e}")

        return meta

if __name__ == "__main__":
    extractor = TradingChartOCRExtractor(use_gpu=False)
    dummy = np.zeros((1080, 1920, 3), dtype=np.uint8)
    res = extractor.extract_chart_metadata(dummy)
    print("Generic OCR Output:", res)
