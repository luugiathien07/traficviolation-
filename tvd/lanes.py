"""Lane-line detection for a camera on a moving vehicle (dashcam): painted lane lines found in every frame and
tracked over time, so the lanes of an R.412 sign can be matched to the road ahead.

Per frame: white paint in the road part of the frame (vehicles masked out) -> Canny -> Hough segments; segments
that are steep enough to be lane lines are grouped by where they cross a reference row near the bottom
(x_ref) and fitted as one line each. Lines are tracked by x_ref with smoothing and kept for `keep_seconds`
after the last sighting, so the gaps of a dashed line do not drop it.

Lanes are counted from the RIGHT road edge (the rightmost line): in right-hand traffic the edge line is the
one a dashcam sees most reliably, while the median on the left is often off-screen. The last panel of a lane
sign is the lane along the right edge, the panel before it the next lane to the left, and so on.
"""
from __future__ import annotations

import cv2
import numpy as np


class LaneLine:
    def __init__(self, x_ref: float, slope: float, t: float):
        self.x_ref, self.slope = x_ref, slope   # x at the reference row, dx/dy
        self.first = self.last = t
        self.hits = 1

    def x_at(self, y: float, y_ref: float) -> float:
        return self.x_ref + (y - y_ref) * self.slope


class LaneTracker:
    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        self.top = float(cfg.get("roi_top", 0.55))       # road part of the frame, as a share of the height
        self.bottom = float(cfg.get("roi_bottom", 0.92))  # above the dashcam's date/time overlay
        self.ref = float(cfg.get("ref_row", 0.85))        # reference row for x_ref
        self.min_slope = float(cfg.get("min_abs_dxdy", 0.15))
        self.max_slope = float(cfg.get("max_abs_dxdy", 4.0))
        self.merge_px = float(cfg.get("merge_px", 0.04))  # share of the width: segments closer belong to one line
        self.keep = float(cfg.get("keep_seconds", 1.0))
        self.min_hits = int(cfg.get("min_hits", 6))       # a line counts once seen in this many frames
        self.alpha = float(cfg.get("smoothing", 0.3))
        # lane lines run towards the vanishing point: at the horizon row their x is within this band
        self.horizon = float(cfg.get("horizon_row", 0.5))
        self.vp_band = tuple(cfg.get("vanishing_band", (0.3, 0.7)))
        self.top_half_width = float(cfg.get("roi_top_half_width", 0.2))  # road trapezoid: half width at roi_top
        self.min_lane = float(cfg.get("min_lane_width", 0.18))  # share of the width at ref_row
        self.lines: list[LaneLine] = []
        self.size = (0, 0)

    def _segments(self, frame: np.ndarray, vehicle_boxes) -> list[tuple[float, float, float]]:
        """(x_ref, dx/dy, length) of the paint segments of this frame."""
        h, w = frame.shape[:2]
        y0, y1 = int(self.top * h), int(self.bottom * h)
        roi = frame[y0:y1]
        light = cv2.cvtColor(roi, cv2.COLOR_BGR2HLS)
        # paint = thin and brighter than the asphalt around it (top-hat), with little colour
        k = max(9, int(0.025 * w)) | 1
        tophat = cv2.morphologyEx(light[..., 1], cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1)))
        paint = ((tophat > 35) & (light[..., 1] > 150) & (light[..., 2] < 110)).astype(np.uint8) * 255
        # only the road ahead: a trapezoid from the bottom corners to a narrower top
        road = np.zeros_like(paint)
        rh = y1 - y0
        cx = w / 2
        cv2.fillPoly(road, [np.array([[0, rh], [w, rh], [cx + self.top_half_width * w, 0],
                                      [cx - self.top_half_width * w, 0]], np.int32)], 255)
        paint &= road
        for x1, y1b, x2, y2b in vehicle_boxes:  # vehicles and riders are not paint
            cv2.rectangle(paint, (int(x1), int(y1b) - y0), (int(x2), int(y2b) - y0), 0, -1)
        segs = cv2.HoughLinesP(paint, 1, np.pi / 180, threshold=15, minLineLength=int(0.02 * h), maxLineGap=int(0.03 * h))
        out = []
        y_ref, y_hor = self.ref * h, self.horizon * h
        for x1, ya, x2, yb in (segs.reshape(-1, 4).tolist() if segs is not None else []):
            ya, yb = ya + y0, yb + y0
            if yb == ya:
                continue
            k = (x2 - x1) / (yb - ya)
            if not self.min_slope <= abs(k) <= self.max_slope:
                continue
            x_hor = x1 + (y_hor - ya) * k
            if not self.vp_band[0] * w <= x_hor <= self.vp_band[1] * w:
                continue  # does not run towards the vanishing point: a kerb corner, shadow, building edge
            # the side towards the middle of the road must be asphalt (grey), not sidewalk, grass or a wall
            mx, my = (x1 + x2) / 2, (ya + yb) / 2 - y0
            inner = int(mx + (1 if mx < cx else -1) * 0.02 * w)
            if not 0 <= inner < w or not self._asphalt(light, inner, int(my)):
                continue
            out.append((x1 + (y_ref - ya) * k, k, float(np.hypot(x2 - x1, yb - ya))))
        return out

    @staticmethod
    def _asphalt(hls: np.ndarray, x: int, y: int) -> bool:
        patch = hls[max(0, y - 4):y + 5, max(0, x - 4):x + 5].reshape(-1, 3)
        l, s = float(np.median(patch[:, 1])), float(np.median(patch[:, 2]))
        return 40 <= l <= 185 and s <= 70

    def update(self, frame: np.ndarray, t: float, vehicle_boxes=()) -> None:
        h, w = frame.shape[:2]
        self.size = (w, h)
        segs = sorted(self._segments(frame, vehicle_boxes))
        groups, merge = [], self.merge_px * w
        for s in segs:
            if groups and s[0] - groups[-1][-1][0] <= merge and np.sign(s[1]) == np.sign(groups[-1][-1][1]):
                groups[-1].append(s)
            else:
                groups.append([s])
        for g in groups:
            wsum = sum(l for _, _, l in g)
            x = sum(x * l for x, _, l in g) / wsum
            k = sum(k * l for _, k, l in g) / wsum
            near = min(self.lines, key=lambda ln: abs(ln.x_ref - x), default=None)
            if near is not None and abs(near.x_ref - x) <= 2 * merge:
                near.x_ref += self.alpha * (x - near.x_ref)
                near.slope += self.alpha * (k - near.slope)
                near.last, near.hits = t, near.hits + 1
            else:
                self.lines.append(LaneLine(x, k, t))
        self.lines = sorted((ln for ln in self.lines if t - ln.last <= self.keep), key=lambda ln: ln.x_ref)

    def confirmed(self) -> list[LaneLine]:
        """Lines seen often enough, at least one lane width apart (the more often seen line wins)."""
        w = self.size[0]
        kept: list[LaneLine] = []
        for ln in sorted((ln for ln in self.lines if ln.hits >= self.min_hits), key=lambda ln: -ln.hits):
            if all(abs(ln.x_ref - k.x_ref) >= self.min_lane * w for k in kept):
                kept.append(ln)
        return sorted(kept, key=lambda ln: ln.x_ref)

    def lanes_from_right(self) -> list[np.ndarray]:
        """Polygons of the lanes between confirmed lines, the first being the lane along the right edge."""
        w, h = self.size
        lines = self.confirmed()
        y0, y1, y_ref = self.top * h, self.bottom * h, self.ref * h
        polys = []
        for left, right in zip(lines[-2::-1], lines[::-1]):  # (second last, last), (third last, second last), ...
            polys.append(np.array([[left.x_at(y0, y_ref), y0], [right.x_at(y0, y_ref), y0],
                                   [right.x_at(y1, y_ref), y1], [left.x_at(y1, y_ref), y1]], np.float32))
        return polys

    def draw(self, img: np.ndarray, colors=None) -> None:
        w, h = self.size
        y0, y1, y_ref = self.top * h, self.bottom * h, self.ref * h
        for ln in self.confirmed():
            cv2.line(img, (int(ln.x_at(y0, y_ref)), int(y0)), (int(ln.x_at(y1, y_ref)), int(y1)), (0, 255, 255), 3, cv2.LINE_AA)
