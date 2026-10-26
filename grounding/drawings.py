"""Chart-pane drawings in pixel space: horizontal levels, zones, position tool, crosshair."""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np

from .layout import Layout


def color_name(bgr: Tuple[int, int, int]) -> str:
    b, g, r = (int(v) for v in bgr)
    mx, mn = max(b, g, r), min(b, g, r)
    if mx < 70:
        return "black"
    if mx - mn < 25:
        return "gray" if mx < 200 else "white"
    if g > r + 30 and g > b + 10:
        return "green"
    if r > g + 40 and r > b + 40:
        return "red" if max(g, b) < 50 else "salmon"
    if b > r + 30 and b > g - 10:
        return "blue"
    if r > 150 and g > 100 and b < 100:
        return "orange"
    if r > b + 30 and b > g:
        return "purple"
    return "other"


@dataclass
class HLine:
    y: float
    x0: int
    x1: int
    thickness: int
    color: str


@dataclass
class Zone:
    color: str          # green | red
    x0: int
    y0: int
    x1: int
    y1: int
    fill: float         # fraction of bbox covered by the mask

    @property
    def area(self) -> int:
        return (self.x1 - self.x0) * (self.y1 - self.y0)


@dataclass
class PositionBox:
    direction: str      # long | short
    x0: int
    x1: int
    entry_y: float
    stop_y: float
    target_y: float
    edges_aligned: bool   # both left and right edges of the two boxes line up


@dataclass
class Crosshair:
    x: Optional[int]
    y: Optional[int]


def _longest_runs(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per row: (length, start, end) of the longest run of True."""
    H, W = mask.shape
    lengths = np.zeros(H, int)
    starts = np.zeros(H, int)
    row_has_any = np.where(mask.any(axis=1))[0]
    if len(row_has_any) == 0:
        return lengths, starts, starts

    padded = np.zeros((len(row_has_any), W + 2), dtype=np.int8)
    padded[:, 1:-1] = mask[row_has_any]
    d = np.diff(padded, axis=1)

    for idx, y in enumerate(row_has_any):
        s = np.where(d[idx] == 1)[0]
        e = np.where(d[idx] == -1)[0]
        if len(s):
            L = e - s
            i = int(L.argmax())
            lengths[y], starts[y] = L[i], s[i]
    return lengths, starts, starts + lengths


def _line_masks(pane: np.ndarray) -> Dict[str, np.ndarray]:
    b, g, r = (pane[..., i] for i in range(3))
    return {
        "black": pane.max(axis=2) < 90,
        "blue": (b > r + 60) & (b > g + 30),
    }


def find_hlines(img: np.ndarray, lay: Layout, min_cover: float = 0.12, min_run: float = 0.06) -> List[HLine]:
    """Rows where one line color covers much of the pane width (candles only break the run)."""
    x0, y0, x1, y1 = lay.pane
    pane = img[y0:y1, x0:x1].astype(np.int16)
    W = x1 - x0
    if W <= 0 or pane.shape[0] <= 0:
        return []
    lines: List[HLine] = []
    for color, mask in _line_masks(pane).items():
        lengths, starts, ends = _longest_runs(mask)
        cover = mask.sum(axis=1)
        rows = np.where((cover > min_cover * W) & (lengths > min_run * W))[0]
        group: List[int] = []

        def flush():
            if group and len(group) <= 4:
                ys = np.array(group)
                cols = np.where(mask[ys].any(axis=0))[0]
                if len(cols) > 0:
                    lines.append(HLine(y=float(ys.mean()) + y0, x0=int(cols.min()) + x0, x1=int(cols.max()) + x0,
                                       thickness=len(group), color=color))
            group.clear()

        for y in rows.tolist():
            if group and y != group[-1] + 1:
                flush()
            group.append(y)
        flush()
    lines.sort(key=lambda l: l.y)
    return lines


def _masks(pane: np.ndarray) -> Dict[str, np.ndarray]:
    b, g, r = (pane[..., i].astype(np.int16) for i in range(3))
    green = (r > 90) & (b > 90) & (g > r + 12) & (g > b + 8) & (g > 180)
    red = (r > 220) & (g > 120) & (g < 235) & (b > 120) & (b < 240) & (r > g + 15) & (abs(g - b) < 30)
    return {"green": green, "red": red}


def find_zones(img: np.ndarray, lay: Layout, min_area_frac: float = 0.002) -> List[Zone]:
    x0, y0, x1, y1 = lay.pane
    pane = img[y0:y1, x0:x1]
    if pane.size == 0:
        return []
    total = pane.shape[0] * pane.shape[1]
    zones: List[Zone] = []
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    for color, mask in _masks(pane).items():
        m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3)))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < min_area_frac * total or w < 15 or h < 4:
                continue
            fill = area / float(w * h)
            if fill < 0.55:
                continue
            zones.append(Zone(color=color, x0=int(x + x0), y0=int(y + y0), x1=int(x + w + x0),
                              y1=int(y + h + y0), fill=round(fill, 3)))
    zones.sort(key=lambda z: -z.area)
    return zones[:24]


def find_position_tools(zones: List[Zone], tol: int = 5, edge_tol: int = 8) -> List[PositionBox]:
    """Green/red box pairs sharing a horizontal edge. Matches long and short TradingView position tools."""
    out = []
    greens = [z for z in zones if z.color == "green"]
    reds = [z for z in zones if z.color == "red"]
    for gz in greens:
        for rz in reds:
            if abs(gz.x1 - rz.x1) > edge_tol:
                continue
            gw, rw = gz.x1 - gz.x0, rz.x1 - rz.x0
            if min(gw, rw) < 0.6 * max(gw, rw):
                continue
            aligned = abs(gz.x0 - rz.x0) <= edge_tol
            x0, x1 = max(gz.x0, rz.x0), min(gz.x1, rz.x1)
            if abs(gz.y1 - rz.y0) <= tol:      # green above red -> long
                out.append(PositionBox("long", x0, x1, entry_y=(gz.y1 + rz.y0) / 2, stop_y=rz.y1,
                                       target_y=gz.y0, edges_aligned=aligned))
            elif abs(rz.y1 - gz.y0) <= tol:    # red above green -> short
                out.append(PositionBox("short", x0, x1, entry_y=(rz.y1 + gz.y0) / 2, stop_y=rz.y0,
                                       target_y=gz.y1, edges_aligned=aligned))
    return out


def _dashed(line: np.ndarray, bg: np.ndarray) -> bool:
    """A 1-px strip (N,3) that alternates between background and a darker neutral dash."""
    line = line.astype(np.int16)
    on = (np.abs(line - bg).max(axis=1) > 20)
    frac = on.mean()
    trans = np.count_nonzero(np.diff(on.astype(np.int8)))
    return 0.2 < frac < 0.85 and trans > 0.05 * len(line)


def find_crosshair(img: np.ndarray, lay: Layout, candidate_ys: Optional[List[float]] = None) -> Crosshair:
    """Dashed horizontal + vertical lines; y is confirmed against candidate axis-tag rows when given."""
    x0, y0, x1, y1 = lay.pane
    bg = np.array(lay.bg, dtype=np.int16)
    y = None
    for cy in candidate_ys or []:
        for dy in (0, -1, 1, -2, 2):
            yy = int(round(cy)) + dy
            if y0 <= yy < y1 and _dashed(img[yy, x0:x1], bg):
                y = yy
                break
        if y is not None:
            break
    x = None
    xr = find_date_label(img, lay)
    if xr:
        cx = (xr[0] + xr[1]) // 2
        for dx in (0, -1, 1, -2, 2, -3, 3):
            if y0 < y1 and 0 <= cx + dx < img.shape[1] and _dashed(img[y0:y1, cx + dx], bg):
                x = cx + dx
                break
        x = x if x is not None else cx
    return Crosshair(x=x, y=y)


def find_date_label(img: np.ndarray, lay: Layout) -> Optional[Tuple[int, int]]:
    """x-range of the dark highlighted label in the date axis (crosshair date)."""
    x0, y0, x1, y1 = lay.date_axis
    if y0 >= y1 or x0 >= x1:
        return None
    strip = img[y0:y1, x0:x1].astype(np.int16)
    if strip.size == 0:
        return None
    dark = (strip.max(axis=2) < 80).mean(axis=0) > 0.5
    idx = np.where(dark)[0]
    if len(idx) < 30:
        return None
    return int(idx.min() + x0), int(idx.max() + x0)
