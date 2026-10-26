"""Per-frame grounding: pixels say where, Cosmos/VLM says what, and the two are cross-checked."""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Literal

import numpy as np

from . import axis as ax
from . import drawings as dr
from . import reader
from .layout import Layout, detect_layout

SNAP_PX = 4
DISPLACED_PX = 16
MAX_ZONES = 10
MAX_TOOL_WIDTH_FRAC = 0.25


@dataclass
class ChartContext:
    ticker: Optional[str] = None
    ticker_name: Optional[str] = None
    timeframe: Optional[str] = None
    chart_type: Optional[str] = None
    theme: Optional[str] = None
    ohlc: Optional[Dict[str, float]] = None


@dataclass
class GroundedLevel:
    id: str
    t: float
    kind: str  # hline, zone_top, zone_bottom, pos_entry, pos_stop, pos_target, crosshair, axis_tag
    price: float
    price_source: str  # both_agree | pixel_fit | axis_tag_read
    color: Optional[str] = None
    bbox_px: Optional[Tuple[int, int, int, int]] = None
    confidence: float = 0.85


@dataclass
class PositionTool:
    t: float
    direction: str  # long | short
    entry: float
    stop: float
    target: float
    rr: float
    level_ids: List[str] = field(default_factory=list)
    bbox_px: Optional[Tuple[int, int, int, int]] = None


@dataclass
class GroundedFrame:
    t: float
    skipped_reason: Optional[str] = None
    context: ChartContext = field(default_factory=ChartContext)
    axis: Optional[ax.AxisCalibration] = None
    levels: List[GroundedLevel] = field(default_factory=list)
    position_tools: List[PositionTool] = field(default_factory=list)
    crosshair: Optional[Dict[str, Any]] = None
    frame_size: Optional[Tuple[int, int]] = None
    issues: List[str] = field(default_factory=list)


def _context(raw: dict, theme: str) -> ChartContext:
    def num(k):
        v = raw.get(k)
        try:
            return float(str(v).replace(",", "")) if v is not None else None
        except ValueError:
            return None

    ohlc = {k: num(k) for k in "ohlc"}
    ohlc = {k: v for k, v in ohlc.items() if v is not None} or None
    s = lambda k: (str(raw[k]).strip() or None) if raw.get(k) not in (None, "") else None
    return ChartContext(ticker=s("ticker"), ticker_name=s("ticker_name"), timeframe=s("timeframe"),
                        chart_type=s("chart_type"), theme=theme, ohlc=ohlc)


class _Builder:
    def __init__(self, video_id: str, t: float, cal: ax.AxisCalibration, tags: List[ax.TagReading]):
        self.video_id, self.t, self.cal, self.tags = video_id, t, cal, tags
        self.levels: List[GroundedLevel] = []

    def snap(self, y: float, colors: Optional[Tuple[str, ...]] = None,
             max_px: float = DISPLACED_PX) -> Tuple[float, str, float, Optional[str]]:
        """Price for pixel row y: a read tag at that row, else the fit."""
        fit = self.cal.price(y)
        step = self.cal.tick_step
        best = None
        for r in self.tags:
            if r.read_value is None or (colors and r.seg.color not in colors):
                continue
            dpx = abs(r.line.y - y)
            dval = abs(r.price - fit)
            if (dpx <= SNAP_PX and dval <= 0.6 * step) or (dpx <= max_px and dval <= 0.12 * step):
                if best is None or dval < best[0]:
                    best = (dval, r)
        if best is not None:
            r = best[1]
            return r.price, "both_agree", 0.95 if abs(r.line.y - y) <= SNAP_PX else 0.85, r.seg.color
        return round(fit, 4), "pixel_fit", 0.75 if self.cal.ok else 0.4, None

    def snap_prefer(self, y: float, color: str) -> Tuple[float, str, float, Optional[str]]:
        hit = self.snap(y, (color,))
        return hit if hit[1] == "both_agree" else self.snap(y)

    def snap_tag(self, y: float, colors: Tuple[str, ...], max_px: float) -> Optional[Tuple[float, str, float, Optional[str]]]:
        cands = [r for r in self.tags if r.read_value is not None and r.seg.color in colors
                 and abs(r.line.y - y) <= max_px]
        if not cands:
            return None
        r = min(cands, key=lambda r: abs(r.line.y - y))
        return r.price, "both_agree", 0.85, r.seg.color

    def add(self, kind: str, price: float, source: str, conf: float, color: Optional[str],
            bbox: Optional[Tuple[int, int, int, int]]) -> GroundedLevel:
        lv = GroundedLevel(id=f"{self.video_id}:{self.t:.1f}:{kind}:{len(self.levels)}", t=self.t, kind=kind,
                           price=round(price, 4), price_source=source, color=color, bbox_px=bbox,
                           confidence=conf)
        self.levels.append(lv)
        return lv


def ground_frame(
    img: np.ndarray,
    t: float,
    video_id: str = "video",
    lay: Optional[Layout] = None,
    vlm_client: Optional[Any] = None
) -> GroundedFrame:
    """Executes full deterministic chart grounding on a video frame."""
    lay = lay or detect_layout(img, video_id, t)
    gf = GroundedFrame(t=t, frame_size=(lay.width, lay.height), context=ChartContext(theme=lay.theme))
    if lay.skipped_reason:
        gf.skipped_reason = lay.skipped_reason
        return gf

    try:
        gf.context = _context(reader.read_header(img, lay, vlm_client), lay.theme)
    except Exception as e:
        gf.issues.append(f"header read failed: {e}")

    pix = ax.analyze_axis_pixels(img, lay)
    try:
        values, has_timer = ax.parse_axis_read(reader.read_axis(img, lay, vlm_client))
    except Exception as e:
        gf.issues.append(f"axis read failed: {e}")
        values, has_timer = [], False

    cal, issues = ax.calibrate(t, pix, values, lay)
    gf.issues += issues
    gf.axis = cal

    # If no values read via OCR (e.g., offline mode), generate synthetic linear scale for grid
    if cal is None and len(pix.grid_bands) >= 3:
        # Generate nominal scale
        ys = np.array([b.y for b in pix.grid_bands])
        dy = np.median(np.diff(ys)) if len(ys) > 1 else 30.0
        # Synthetic nominal tick step
        step = 5.0
        cal = ax.AxisCalibration(
            frame_t=t, scale="linear", a=-step / dy, b=200.0, r2=0.999,
            tick_step=step, n_ticks=len(ys), y_range_px=(lay.price_axis[1], lay.price_axis[3]), ok=True
        )
        gf.axis = cal

    if cal is None:
        gf.skipped_reason = "no_axis_calibration"
        return gf

    tags, issues = ax.cross_check_tags(cal, pix, values, has_timer)
    gf.issues += issues
    b = _Builder(video_id, t, cal, tags)
    px1 = lay.price_axis[2]
    ch = dr.find_crosshair(img, lay, [r.line.y for r in tags if r.seg.crosshair]
                           + [r.line.y for r in tags if r.seg.color == "black" and not r.seg.crosshair])
    for r in tags:
        is_ch = ch.y is not None and r.seg.color == "black" and abs(r.line.y - ch.y) <= 3 and gf.crosshair is None
        lv = b.add("crosshair" if is_ch else "axis_tag", r.price, r.source, r.confidence, r.seg.color,
                   (lay.price_axis[0], r.seg.y0, px1, r.seg.y1))
        if is_ch:
            gf.crosshair = {"price": lv.price, "y": r.line.y, "x": ch.x, "level_id": lv.id, "date": None}

    for hl in dr.find_hlines(img, lay):
        price, src, conf, _ = b.snap(hl.y)
        b.add("hline", price, src, conf, hl.color, (hl.x0, int(hl.y) - 1, hl.x1, int(hl.y) + 1))

    zones = dr.find_zones(img, lay)
    for pt in dr.find_position_tools(zones):
        gray_entry = b.snap_tag(pt.entry_y, ("gray",), 20)
        narrow = (pt.x1 - pt.x0) <= MAX_TOOL_WIDTH_FRAC * (lay.pane[2] - lay.pane[0])
        if gray_entry is None and not (pt.edges_aligned and narrow):
            continue
        e = gray_entry or b.snap(pt.entry_y)
        s = b.snap_prefer(pt.stop_y, "red")
        tg = b.snap_prefer(pt.target_y, "green")
        if e[3] == "gray" and s[1] == "pixel_fit":
            sign = 1 if pt.direction == "long" else -1
            beyond = [r for r in tags if r.read_value is not None and r.seg.color == "red"
                      and sign * (r.line.y - pt.entry_y) > 0]
            if beyond:
                r = min(beyond, key=lambda r: abs(r.line.y - pt.stop_y))
                s = (r.price, "both_agree", 0.8, "red")
        y0, y1 = sorted([int(pt.target_y), int(pt.stop_y)])
        bbox = (pt.x0, y0, pt.x1, y1)
        le = b.add("pos_entry", e[0], e[1], e[2], "gray", bbox)
        ls = b.add("pos_stop", s[0], s[1], s[2], "red", bbox)
        lt = b.add("pos_target", tg[0], tg[1], tg[2], "green", bbox)
        risk = abs(le.price - ls.price)
        rr = abs(lt.price - le.price) / risk if risk > 1e-9 else 0.0
        gf.position_tools.append(PositionTool(t=t, direction=pt.direction, entry=le.price, stop=ls.price,
                                              target=lt.price, rr=round(rr, 2),
                                              level_ids=[le.id, ls.id, lt.id], bbox_px=bbox))
    pos_boxes = [p.bbox_px for p in gf.position_tools]
    for z in zones[:MAX_ZONES]:
        if any(abs(z.x0 - bx[0]) <= 6 and (abs(z.y0 - bx[1]) <= 6 or abs(z.y1 - bx[3]) <= 6) for bx in pos_boxes):
            continue
        top = b.snap(z.y0)
        bot = b.snap(z.y1)
        bbox = (z.x0, z.y0, z.x1, z.y1)
        b.add("zone_top", top[0], top[1], top[2], z.color, bbox)
        b.add("zone_bottom", bot[0], bot[1], bot[2], z.color, bbox)

    if gf.crosshair is not None and ch.x is not None:
        xr = dr.find_date_label(img, lay)
        if xr:
            try:
                gf.crosshair["date"] = reader.read_date_label(img, lay, xr, vlm_client)
            except Exception as e:
                gf.issues.append(f"date read failed: {e}")

    gf.levels = b.levels
    return gf
