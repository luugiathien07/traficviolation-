"""Read an R.412 lane sign ("vehicle permission lane"): which vehicles each lane is for, left to right.

The white pictograms on the blue sign are found as connected white blobs (tools/prepare_lane_signs.py `icons`),
each is classified (car, moto, truck, bus, arrow, other) by a small YOLO classifier trained on labelled
pictograms, and the vehicle pictograms are grouped into lanes by their x position: pictograms of one lane are
stacked in a column, lanes sit side by side.
"""
from __future__ import annotations

import cv2
import numpy as np

# pictogram class -> detector (COCO) vehicle classes allowed in the lane
VEHICLE_OF = {"car": "car", "moto": "motorcycle", "truck": "truck", "bus": "bus"}


def to_height(sign: np.ndarray, h: int = 120) -> np.ndarray:
    return cv2.resize(sign, (max(40, sign.shape[1] * h // max(1, sign.shape[0])), h), interpolation=cv2.INTER_CUBIC)


def icon_boxes(sign: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Boxes of the white pictograms of a lane sign (at height 120): connected white blobs that are neither
    specks, nor the sign border, nor the thin lane separators."""
    hsv = cv2.cvtColor(sign, cv2.COLOR_BGR2HSV)
    white = ((hsv[..., 1] < 90) & (hsv[..., 2] > 130)).astype(np.uint8)
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    h, w = white.shape
    n, _, st, _ = cv2.connectedComponentsWithStats(white, 8)
    out = []
    for i in range(1, n):
        x, y, bw, bh, area = map(int, st[i])
        if area < 0.004 * h * w or bw > 0.6 * w or bh > 0.7 * h or bh > 4 * bw:
            continue  # speck, sign border, lane separator
        if x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1:
            continue  # cut by the crop edge
        out.append((x, y, x + bw, y + bh))
    return out


class LaneSignReader:
    def __init__(self, model: str = "models/lane_icons.pt", min_conf: float = 0.6):
        from ultralytics import YOLO

        self.model = YOLO(model)
        self.min_conf = min_conf

    def read(self, sign_bgr: np.ndarray) -> list[set[str]] | None:
        """Allowed vehicle classes per lane, left to right; None when no vehicle pictogram is readable."""
        sign = to_height(sign_bgr)
        icons = []  # (x centre, width, vehicle)
        for x1, y1, x2, y2 in icon_boxes(sign):
            crop = sign[max(0, y1 - 3):y2 + 3, max(0, x1 - 3):x2 + 3]
            p = self.model.predict(crop, imgsz=64, verbose=False)[0].probs
            name = self.model.names[p.top1]
            if float(p.top1conf) >= self.min_conf and name in VEHICLE_OF:
                icons.append(((x1 + x2) / 2, x2 - x1, VEHICLE_OF[name]))
        if not icons:
            return None
        icons.sort()
        gap = 0.6 * float(np.median([w for _, w, _ in icons]))
        lanes, last_x = [], None
        for x, _, v in icons:
            if last_x is None or x - last_x > gap:
                lanes.append(set())
            lanes[-1].add(v)
            last_x = x
        return lanes
