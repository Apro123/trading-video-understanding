"""VLM crop reading (text only, never coordinates) with quantized-crop cache keys."""

import cv2
import numpy as np
from typing import Optional, Dict, Any, List, Tuple

from .layout import Layout, crop
from cache import extract_json, stable_hash, DiskCache

_cache = DiskCache("crops")

R1_HEADER = (
    'Transcribe this TradingView chart header. The top line is the symbol box (ticker). The bottom line '
    'is the chart title "<name> · <interval> · <exchange>[ · <chart type>]" followed by O H L C values. '
    'timeframe = the interval from the title (e.g. 1W, 1D, 4h). chart_type = "Heikin Ashi" if the title '
    'says so, else "Candles". Return ONLY JSON: {"ticker":str,"ticker_name":str,"timeframe":str,'
    '"chart_type":str,"o":num,"h":num,"l":num,"c":num}'
)
R2_AXIS = (
    'This is the right price axis of a TradingView chart. List every label exactly as printed, top to '
    'bottom. Mark labels on a colored background with their background color. One colored label may '
    'contain a candle countdown timer (mm:ss, e.g. 08:53) under its price: output the timer as a string, '
    'never as a number. Return ONLY JSON: [{"value":num|str,"color":"none|green|red|black|gray|blue|other"}]'
)
R3_DATE = (
    'This is the bottom time axis of a TradingView chart. If there is a highlighted date label (dark '
    'background), return it. Return ONLY JSON: {"highlighted":str|null,"labels":[str]}'
)


def _png(img: np.ndarray, scale: float) -> bytes:
    up = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return cv2.imencode(".png", up)[1].tobytes()


def _qkey(prompt: str, img: np.ndarray) -> str:
    """Cache key robust to codec noise: grayscale quantized to 16 levels."""
    if img.size == 0:
        return stable_hash(prompt, b"empty")
    q = (img.mean(axis=2) // 16).astype(np.uint8)
    return stable_hash(prompt, q.tobytes(), q.shape)


def axis_crop(img: np.ndarray, lay: Layout) -> np.ndarray:
    x0, y0, x1, y1 = lay.price_axis
    return crop(img, (max(0, x0 - 54), y0, min(img.shape[1], x1 + 4), y1))


def header_canvas(img: np.ndarray, lay: Layout) -> np.ndarray:
    sym = crop(img, lay.symbol_box)
    hdr = crop(img, lay.header)
    if sym.size == 0 or hdr.size == 0:
        return np.full((30, 200, 3), 255, np.uint8)
    w = max(sym.shape[1], hdr.shape[1])
    canvas = np.full((sym.shape[0] + hdr.shape[0] + 6, w, 3), 255, np.uint8)
    canvas[:sym.shape[0], :sym.shape[1]] = sym
    canvas[sym.shape[0] + 6:, :hdr.shape[1]] = hdr
    return canvas


def read_axis(img: np.ndarray, lay: Layout, vlm_client: Optional[Any] = None) -> Any:
    c = axis_crop(img, lay)
    k = _qkey(R2_AXIS, c)
    cached = _cache.get(k)
    if cached is not None:
        return cached

    if vlm_client and hasattr(vlm_client, "read_image"):
        try:
            text = vlm_client.read_image(R2_AXIS, _png(c, 3), max_tokens=4000)
            res = extract_json(text)
            _cache.set(k, res)
            return res
        except Exception:
            pass

    # Heuristic fallback if VLM endpoint is not reachable
    return []


def read_header(img: np.ndarray, lay: Layout, vlm_client: Optional[Any] = None) -> Dict[str, Any]:
    c = header_canvas(img, lay)
    k = _qkey(R1_HEADER, c)
    cached = _cache.get(k)
    if cached is not None:
        return cached

    if vlm_client and hasattr(vlm_client, "read_image"):
        try:
            text = vlm_client.read_image(R1_HEADER, _png(c, 2), max_tokens=2000)
            res = extract_json(text)
            if isinstance(res, dict):
                _cache.set(k, res)
                return res
        except Exception:
            pass

    return {}


def read_date_label(img: np.ndarray, lay: Layout, xr: Tuple[int, int], vlm_client: Optional[Any] = None) -> Optional[str]:
    x0, y0, x1, y1 = lay.date_axis
    c = crop(img, (max(xr[0] - 30, x0), y0, min(xr[1] + 30, x1), y1))
    k = _qkey(R3_DATE, c)
    cached = _cache.get(k)
    if cached is not None:
        return cached.get("highlighted")

    if vlm_client and hasattr(vlm_client, "read_image"):
        try:
            text = vlm_client.read_image(R3_DATE, _png(c, 3), max_tokens=1500)
            res = extract_json(text)
            if isinstance(res, dict):
                _cache.set(k, res)
                return res.get("highlighted")
        except Exception:
            pass

    return None
