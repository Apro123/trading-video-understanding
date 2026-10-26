"""Price-axis calibration: pixel text bands + OCR values -> price = f(y), plus tag cross-check."""

import math
import re
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Any

import numpy as np

from .layout import Layout

AXIS_TOP_SKIP = 20          # currency button at the top of the price scale
_TIME = re.compile(r"^\s*\d{1,2}:\d{2}(:\d{2})?\s*$")


@dataclass
class TextLine:
    y: float
    y0: int
    y1: int


@dataclass
class TagSegment:
    y0: int
    y1: int
    color: str
    bgr: Tuple[int, int, int]
    lines: List[TextLine] = field(default_factory=list)
    crosshair: bool = False
    countdown_line: Optional[int] = None   # index into lines of a countdown timer row

    @property
    def yc(self) -> float:
        return (self.y0 + self.y1) / 2


@dataclass
class AxisPixels:
    grid_bands: List[TextLine]
    tags: List[TagSegment]


@dataclass
class AxisCalibration:
    frame_t: float
    scale: str  # "linear" | "log"
    a: float
    b: float
    r2: float
    tick_step: float
    n_ticks: int
    y_range_px: Tuple[int, int]
    ok: bool

    def price(self, y: float) -> float:
        v = self.a * y + self.b
        return math.exp(v) if self.scale == "log" else v


@dataclass
class TagReading:
    seg: TagSegment
    line: TextLine
    price: float
    source: str        # both_agree | pixel_fit
    confidence: float
    read_value: Optional[float]


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


def _runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    idx = np.where(mask)[0]
    out: List[Tuple[int, int]] = []
    for i in idx.tolist():
        if out and i == out[-1][1] + 1:
            out[-1] = (out[-1][0], i)
        else:
            out.append((i, i))
    return out


def analyze_axis_pixels(img: np.ndarray, lay: Layout) -> AxisPixels:
    x0, y0, x1, y1 = lay.price_axis
    top = y0 + AXIS_TOP_SKIP
    if top >= y1 or x0 >= x1:
        return AxisPixels(grid_bands=[], tags=[])
    strip = img[top:y1, x0:x1].astype(np.int16)
    if strip.size == 0 or strip.shape[1] < 10:
        return AxisPixels(grid_bands=[], tags=[])

    # per-row background from the right margin
    bg_rows = np.median(strip[:, -9:-2], axis=1)
    dist_bg = np.abs(strip - bg_rows[:, None, :]).max(axis=2)

    edge = strip[:, 1:4].mean(axis=1)
    edge_lum = edge @ np.array([0.114, 0.587, 0.299])
    edge_off = (np.abs(edge - bg_rows).max(axis=1) > 50) & (edge_lum < 165)

    tags: List[TagSegment] = []
    for a, b in _runs(edge_off):
        start = a
        for y in range(a + 1, b + 2):
            if y > b or np.abs(edge[y] - edge[y - 1]).max() > 45:
                if y - start >= 6:
                    fill = np.median(edge[start:y], axis=0)
                    tags.append(TagSegment(y0=start + top, y1=y - 1 + top, color=color_name(fill),
                                           bgr=tuple(int(v) for v in fill)))
                start = y

    for seg in tags:
        fill = np.array(seg.bgr, dtype=np.int16)
        rows = strip[seg.y0 - top:seg.y1 - top + 1]
        if rows.size == 0:
            continue
        near_fill = (np.abs(rows - fill).max(axis=2) < 40).mean(axis=0) >= 0.4
        cols = np.where(near_fill)[0]
        if len(cols) < 6:
            continue
        rows = rows[:, cols.min() + 2:cols.max() - 1]
        if rows.size == 0:
            continue
        contrast = (np.abs(rows - fill).max(axis=2) > 70).sum(axis=1) >= 2
        for a, b in _runs(contrast):
            if 3 <= b - a + 1 <= 13:
                seg.lines.append(TextLine(y=(a + b) / 2 + seg.y0, y0=a + seg.y0, y1=b + seg.y0))
        if len(seg.lines) == 2 and seg.lines[1].y - seg.lines[0].y <= 13:
            seg.countdown_line = 1
        px0, px1 = lay.pane[2] - 20, lay.pane[2] - 2
        if px0 < px1 and 0 <= px0 < img.shape[1] and 0 <= px1 <= img.shape[1]:
            btn = img[seg.y0:seg.y1 + 1, px0:px1].astype(np.int16)
            if btn.size and (btn.max(axis=2) < 80).mean() > 0.3:
                seg.crosshair = True

    lum = strip[:, 3:-10] @ np.array([0.114, 0.587, 0.299])
    neutral_rows = (~edge_off) & (((dist_bg[:, 3:-10] > 50) & (lum < 200)).sum(axis=1) > 0)
    grid = [TextLine(y=(a + b) / 2 + top, y0=a + top, y1=b + top)
            for a, b in _runs(neutral_rows) if 4 <= b - a + 1 <= 12]
    return AxisPixels(grid_bands=grid, tags=tags)


def parse_axis_read(raw: Any) -> Tuple[List[float], bool]:
    """-> (numeric values top to bottom, whether a countdown timer was reported)."""
    vals: List[float] = []
    has_timer = False
    items = raw if isinstance(raw, list) else (raw or {}).get("labels", [])
    for it in items:
        v = it.get("value") if isinstance(it, dict) else it
        if isinstance(v, str):
            if _TIME.match(v):
                has_timer = True
                continue
            try:
                v = float(v.replace(",", ""))
            except ValueError:
                continue
        if isinstance(v, (int, float)) and math.isfinite(v):
            vals.append(float(v))
    return vals, has_timer


def find_value_lattice(values: List[float]) -> Optional[Tuple[float, float, List[float]]]:
    """Most-populated arithmetic lattice among the read values -> (step, anchor, members)."""
    uniq = sorted(set(round(v, 6) for v in values), reverse=True)
    if len(uniq) < 3:
        return None
    cands = sorted({round(abs(a - b), 6) for i, a in enumerate(uniq) for b in uniq[i + 1:i + 4] if abs(a - b) > 0})
    best = None
    for dv in cands:
        for v0 in uniq:
            members = [v for v in uniq if abs((v0 - v) / dv - round((v0 - v) / dv)) < 0.01]
            holes = (max(members) - min(members)) / dv + 1 - len(members)
            key = (len(members) - 0.5 * holes, dv)
            if best is None or key > best[0]:
                best = (key, dv, v0, members)
    if best is None or len(best[3]) < 3:
        return None
    _, dv, _, members = best
    return dv, max(members), members


def _fit(ys: np.ndarray, ps: np.ndarray, log: bool) -> Tuple[float, float, float, float]:
    target = np.log(ps) if log else ps
    A = np.vstack([ys, np.ones_like(ys)]).T
    (a, b), *_ = np.linalg.lstsq(A, target, rcond=None)
    pred = a * ys + b
    pred_p = np.exp(pred) if log else pred
    ss_res = float(((target - pred) ** 2).sum())
    ss_tot = float(((target - target.mean()) ** 2).sum()) or 1e-12
    return float(a), float(b), 1 - ss_res / ss_tot, float(np.abs(pred_p - ps).max())


def calibrate(
    t: float,
    pix: AxisPixels,
    values: List[float],
    lay: Layout
) -> Tuple[Optional[AxisCalibration], List[str]]:
    """Calibrates pixel y positions to monetary prices."""
    issues: List[str] = []
    lat = find_value_lattice(values)
    ys = [b.y for b in pix.grid_bands]
    if lat is None or len(ys) < 3:
        issues.append(f"axis: insufficient grid (lattice={lat is not None}, bands={len(ys)})")
        return None, issues
    dv, vtop, members = lat

    diffs = np.diff(ys)
    d0 = min(d for d in diffs if d >= 12) if any(d >= 12 for d in diffs) else None
    if d0 is None:
        issues.append("axis: grid bands too dense")
        return None, issues
    dy = float(np.median([d / max(round(d / d0), 1) for d in diffs if d >= 12]))
    k = [round((y - ys[0]) / dy) for y in ys]

    best = None
    for mult in (1, 2, 0.5):
        j = [round((vtop - v) / dv * mult) for v in members]
        if any(abs((vtop - v) / dv * mult - jj) > 0.05 for v, jj in zip(members, j)):
            continue
        for s in range(-len(j) - len(k), len(j) + len(k)):
            kj = {kk: y for kk, y in zip(k, ys)}
            pairs = [(kj[jj + s], v) for jj, v in zip(j, members) if jj + s in kj]
            if len(pairs) < 3:
                continue
            py, pv = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
            a, b, r2, maxerr = _fit(py, pv, log=False)
            if a >= 0:
                continue
            key = (len(pairs), r2)
            if best is None or key > best[0]:
                best = (key, py, pv, dv / mult)
    if best is None:
        issues.append("axis: could not align read values to grid bands")
        return None, issues
    _, py, pv, step = best

    a, b, r2, maxerr = _fit(py, pv, log=False)
    scale = "linear"
    if maxerr > 0.25 * step and (pv > 0).all():
        la, lb, lr2, lerr = _fit(py, pv, log=True)
        if lerr < maxerr:
            a, b, r2, maxerr, scale = la, lb, lr2, lerr, "log"
    ok = r2 > 0.999 and len(py) >= 4 and maxerr <= 0.25 * step
    if not ok:
        issues.append(f"axis: weak fit r2={r2:.5f} n={len(py)} maxerr={maxerr:.3f}")
    cal = AxisCalibration(frame_t=t, scale=scale, a=a, b=b, r2=r2, tick_step=step, n_ticks=len(py),
                          y_range_px=(lay.price_axis[1], lay.price_axis[3]), ok=ok)
    return cal, issues


def inverse(cal: AxisCalibration, price: float) -> float:
    v = math.log(price) if cal.scale == "log" else price
    return (v - cal.b) / cal.a


def cross_check_tags(
    cal: AxisCalibration,
    pix: AxisPixels,
    values: List[float],
    has_timer: bool = False
) -> Tuple[List[TagReading], List[str]]:
    """Match OCR tag values to tag text rows."""
    issues: List[str] = []
    lat = find_value_lattice(values)
    members = set(lat[2]) if lat else set()
    tag_vals = [v for v in values if round(v, 6) not in members]

    lines = [(seg, i, ln) for seg in pix.tags for i, ln in enumerate(seg.lines)
             if not (has_timer and seg.countdown_line == i)]
    cand = []
    for vi, v in enumerate(tag_vals):
        y_true = inverse(cal, v)
        for li, (seg, _, ln) in enumerate(lines):
            d = abs(ln.y - y_true)
            if d <= 16:
                cand.append((d, vi, li))
    cand.sort()
    used_v, used_l, out = set(), set(), []
    for d, vi, li in cand:
        if vi in used_v or li in used_l:
            continue
        used_v.add(vi)
        used_l.add(li)
        seg, _, ln = lines[li]
        v = tag_vals[vi]
        conf = 0.95 if d <= 4 else 0.8
        out.append(TagReading(seg=seg, line=ln, price=v, source="both_agree", confidence=conf, read_value=v))
    for vi, v in enumerate(tag_vals):
        if vi not in used_v:
            issues.append(f"axis tag read {v} has no matching tag row (dropped)")
    for li, (seg, _, ln) in enumerate(lines):
        if li not in used_l:
            p = cal.price(ln.y)
            stacked = any(o is not seg and (abs(o.y0 - seg.y1) <= 2 or abs(seg.y0 - o.y1) <= 2) for o in pix.tags)
            out.append(TagReading(seg=seg, line=ln, price=round(p, 4), source="pixel_fit",
                                  confidence=0.3 if stacked else 0.5, read_value=None))
            issues.append(f"axis tag at y={ln.y:.0f} ({seg.color}) unread; pixel fit {p:.2f}")
    out.sort(key=lambda r: r.line.y)
    return out, issues
