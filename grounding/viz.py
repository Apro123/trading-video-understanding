"""Debug and UI overlay rendering of a GroundedFrame on its source image."""

from typing import Tuple
import cv2
import numpy as np

from .ground import GroundedFrame

_COLORS = {
    "hline": (0, 0, 0),
    "zone_top": (0, 160, 0),
    "zone_bottom": (0, 160, 0),
    "pos_entry": (90, 90, 90),
    "pos_stop": (0, 0, 255),
    "pos_target": (0, 170, 0),
    "crosshair": (255, 0, 255),
    "axis_tag": (255, 120, 0)
}


def draw_overlay(img: np.ndarray, gf: GroundedFrame) -> np.ndarray:
    """Draws bounding boxes, level tags, and position tools on top of an image."""
    out = img.copy()
    if gf.skipped_reason:
        cv2.putText(out, f"SKIPPED: {gf.skipped_reason}", (60, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        return out

    for lv in gf.levels:
        if not lv.bbox_px:
            continue
        x0, y0, x1, y1 = lv.bbox_px
        c = _COLORS.get(lv.kind, (255, 0, 0))
        if lv.kind == "axis_tag" or (lv.kind == "crosshair" and x0 > img.shape[1] * 0.7):
            cv2.rectangle(out, (max(0, x0 - 60), y0), (max(0, x0 - 2), y1), c, 1)
            cv2.putText(out, f"{lv.price:g}", (max(0, x0 - 58), y1 - 2), cv2.FONT_HERSHEY_PLAIN, 0.8, c, 1)
            continue
        y = {"zone_top": y0, "zone_bottom": y1}.get(lv.kind, (y0 + y1) // 2)
        if lv.kind.startswith("pos_"):
            y = None
        if y is not None:
            cv2.line(out, (x0, y), (x1, y), c, 1)
            cv2.putText(out, f"{lv.kind} {lv.price:g} [{lv.price_source}]", (max(x0, 40), y - 3),
                        cv2.FONT_HERSHEY_PLAIN, 0.9, c, 1)

    for pt in gf.position_tools:
        if not pt.bbox_px:
            continue
        x0, y0, x1, y1 = pt.bbox_px
        cv2.rectangle(out, (x0, y0), (x1, y1), (255, 0, 255), 2)
        cv2.putText(out, f"{pt.direction.upper()} E:{pt.entry:g} S:{pt.stop:g} T:{pt.target:g} (R:R {pt.rr})",
                    (max(x0 - 200, 40), max(y0 - 6, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2)

    if gf.crosshair and gf.crosshair.get("x") and gf.crosshair.get("y"):
        cv2.circle(out, (int(gf.crosshair["x"]), int(gf.crosshair["y"])), 8, (255, 0, 255), 2)

    ctx = gf.context
    if ctx.ticker:
        label = f"{ctx.ticker} {ctx.timeframe or ''} {ctx.chart_type or ''}".strip()
        cv2.putText(out, f"{label} | Axis: {'OK' if gf.axis and gf.axis.ok else 'Pending'}",
                    (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 220), 2)

    return out
