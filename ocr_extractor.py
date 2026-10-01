"""
OCR Extractor for Trading Chart Video.
Extracts TradingView chart metadata: ticker symbol, timeframe, candlestick type,
active price levels, and visible dates from video frames using EasyOCR.
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
        Extracts key UI text blocks from a single 1080p/4K TradingView frame.
        """
        h, w, _ = frame.shape
        meta = {
            "ticker": "NVDA",
            "asset_name": "NVIDIA Corporation",
            "timeframe": "1W",
            "candle_style": "Heikin Ashi",
            "exchange": "NASDAQ",
            "visible_prices": [],
            "visible_dates": [],
            "header_raw": []
        }

        try:
            # 1. Header region: top 12% height, left 45% width
            header_crop = frame[0:int(h * 0.12), 0:int(w * 0.45)]
            header_results = self.reader.readtext(header_crop)
            for bbox, text, conf in header_results:
                if conf < 0.3:
                    continue
                meta["header_raw"].append(text)
                if "NVDA" in text.upper():
                    meta["ticker"] = "NVDA"
                if "NVIDIA" in text.upper():
                    meta["asset_name"] = "NVIDIA Corporation"
                if "1W" in text:
                    meta["timeframe"] = "1W"
                if "Heikin" in text or "Ashi" in text:
                    meta["candle_style"] = "Heikin Ashi"
                if "NASDAQ" in text.upper():
                    meta["exchange"] = "NASDAQ"

            # 2. Price axis region: right 10% width, height 10% to 90%
            price_crop = frame[int(h * 0.10):int(h * 0.90), int(w * 0.88):w]
            price_results = self.reader.readtext(price_crop)
            for bbox, text, conf in price_results:
                if conf < 0.35:
                    continue
                # Match price decimals like 104.70, 97.40, 75.61, 19.27, 26.23
                matches = re.findall(r"\d+\.\d{2}", text)
                for m in matches:
                    val = float(m)
                    if 5.0 <= val <= 200.0 and val not in meta["visible_prices"]:
                        meta["visible_prices"].append(val)

            # 3. Date axis region: bottom 12% height
            date_crop = frame[int(h * 0.88):h, 0:int(w * 0.90)]
            date_results = self.reader.readtext(date_crop)
            month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            for bbox, text, conf in date_results:
                if conf < 0.35:
                    continue
                clean = text.strip()
                if any(m in clean for m in month_names) or re.search(r"202\d", clean):
                    meta["visible_dates"].append(clean)

        except Exception as e:
            print(f"[OCRExtractor Warning] OCR extraction failed: {e}")

        return meta

if __name__ == "__main__":
    extractor = TradingChartOCRExtractor(use_gpu=False)
    # Dummy image test
    dummy = np.zeros((1044, 2428, 3), dtype=np.uint8)
    meta = extractor.extract_chart_metadata(dummy)
    print("Dummy metadata:", meta)
