"""Locate TradingView regions (chart pane, price axis, date axis, header) by pixel analysis."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np

Rect = Tuple[int, int, int, int]  # x0, y0, x1, y1 (exclusive)


@dataclass
class Layout:
    width: int
    height: int
    theme: str
    pane: Rect
    price_axis: Rect
    date_axis: Rect
    header: Rect
    symbol_box: Rect
    bg: Tuple[int, int, int]
    skipped_reason: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def _groups(idx: np.ndarray) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for i in idx.tolist():
        if out and i <= out[-1][1] + 1:
            out[-1] = (out[-1][0], i)
        else:
            out.append((i, i))
    return out


def _bg_color(region: np.ndarray) -> Tuple[int, int, int]:
    q = (region.reshape(-1, 3) // 4).astype(np.int32)
    keys = q[:, 0] * 4096 + q[:, 1] * 64 + q[:, 2]
    k = np.bincount(keys).argmax()
    return (int(k // 4096) * 4 + 2, int(k // 64 % 64) * 4 + 2, int(k % 64) * 4 + 2)


def _luma(c: Tuple[int, int, int]) -> float:
    b, g, r = c
    return 0.114 * b + 0.587 * g + 0.299 * r


def detect_layout(img: np.ndarray, video_id: Optional[str] = None, t: float = 0.0) -> Layout:
    """Detects chart boundaries from BGR uint8 full resolution image."""
    H, W = img.shape[:2]
    g = img.mean(axis=2).astype(np.int16)

    dx = np.abs(np.diff(g, axis=1)) > 25
    vsep = _groups(np.where(dx.mean(axis=0) > 0.6)[0])
    left = next((b + 1 for a, b in vsep if 5 < b < W * 0.1), 0)
    axis_pair = None
    for (a1, b1), (a2, b2) in zip(vsep, vsep[1:]):
        if a1 > W * 0.5 and 35 <= a2 - b1 <= 140:
            axis_pair = (a1, b1, a2)
            break

    skipped = None
    if axis_pair is None:
        skipped = "no_chart_layout"
        axis_pair = (int(W * 0.808), int(W * 0.809), int(W * 0.836))
    pane_x1 = axis_pair[0]
    ax0, ax1 = axis_pair[1] + 1, axis_pair[2]

    dy = np.abs(np.diff(g[:, left:pane_x1], axis=0)) > 25
    hsep = _groups(np.where(dy.mean(axis=1) > 0.6)[0])
    full = _groups(np.where(dy.mean(axis=1) > 0.9)[0])
    top_seps = [b for a, b in full if b < H * 0.15]
    pane_top = (top_seps[-1] + 1) if top_seps else int(H * 0.1)

    date_axis = None
    low = [(a, b) for a, b in hsep if a > H * 0.5]
    for (a1, b1), (a2, b2) in zip(low, low[1:]):
        if 15 <= a2 - b1 <= 32 and g[b1 + 1:a2, left:pane_x1].mean() > 120:
            date_axis = (left, b1 + 1, pane_x1, a2)
            break
    if date_axis is None:
        date_axis = (left, int(H * 0.893), pane_x1, int(H * 0.919))
        skipped = skipped or "no_date_axis"
    pane_bottom = date_axis[1] - 1

    pane = (left, pane_top, pane_x1, pane_bottom)
    inner = img[pane_top + 40:pane_bottom - 10, left + 10:pane_x1 - 10]
    if inner.size > 0:
        bg = _bg_color(inner)
    else:
        bg = (245, 245, 245)
    lum = _luma(bg)
    theme = "dark" if lum < 90 else "light"
    if skipped is None and 90 <= lum < 200:
        skipped = "modal"

    lay = Layout(
        width=W, height=H, theme=theme, pane=pane,
        price_axis=(ax0, pane_top, ax1, pane_bottom),
        date_axis=date_axis,
        header=(left, pane_top, min(left + 1000, pane_x1), pane_top + 24),
        symbol_box=(max(left - 2, 0), 0, left + 300, 34),
        bg=bg, skipped_reason=skipped,
    )
    return lay


def crop(img: np.ndarray, r: Rect) -> np.ndarray:
    x0, y0, x1, y1 = r
    return img[max(y0, 0):y1, max(x0, 0):x1]


def layout_signature(img: np.ndarray) -> np.ndarray:
    small = img[::8, ::8].mean(axis=2)
    return small.astype(np.float32)
