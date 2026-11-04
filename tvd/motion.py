"""Moving camera check (camera di động).

Lines and zones (stop line, sidewalk, one-way road, junction box...) are drawn once for a camera that stands still.
On a dashcam or a camera carried by hand they land on the wrong part of the road, and every parked car seems to
move because the camera does: a stopped car beside the dashcam "drives along the sidewalk", a car the dashcam
overtakes "drives the wrong way". This module measures how far the background moves between frames (sparse optical
flow on corners outside the vehicle boxes) and says when the camera itself is moving, so the pipeline can switch off
the rules that need a fixed camera and keep the others (helmet, riders, plates, signs).

Modes: "off", "auto" (default). Auto smooths the median shift so a swaying pole or a passing bus does not flip it.
Measured on the sample videos: fixed cameras stay under 0.4 px per frame (leaves moving in the wind), dashcams run
at 13-27 px. A fixed camera never starts moving, so with `latch` (default) the first sustained movement keeps the
camera "moving" for the rest of the video: a dashcam stopped at a red light still has its zones in the wrong place.
"""
from __future__ import annotations

import cv2
import numpy as np

MODES = ("off", "auto")


class CameraMotion:
    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        self.mode = cfg.get("mode", "auto")
        if self.mode not in MODES:
            raise ValueError(f"camera_motion.mode must be one of {MODES}")
        self.width = int(cfg.get("width", 480))          # frames are measured at this width
        self.on_px = float(cfg.get("on_px", 2.0))        # median background shift per frame, full-size px
        self.off_px = float(cfg.get("off_px", 0.6))
        self.latch = bool(cfg.get("latch", True))
        self.smooth = float(cfg.get("smooth", 0.1))      # EMA weight of a new frame
        self.prev = None
        self.shift = 0.0       # smoothed median background shift, px per frame at full size
        self.moving = False

    def update(self, frame: np.ndarray, boxes) -> bool:
        """Feed every frame with this frame's vehicle boxes; returns whether the camera is moving."""
        if self.mode == "off":
            return False
        h, w = frame.shape[:2]
        scale = self.width / w
        gray = cv2.cvtColor(cv2.resize(frame, (self.width, int(h * scale))), cv2.COLOR_BGR2GRAY)
        prev, self.prev = self.prev, gray
        if prev is None:
            return self.moving
        mask = np.full(prev.shape, 255, np.uint8)
        for x1, y1, x2, y2 in boxes:  # vehicles move on their own: measure only the background
            mask[max(0, int(y1 * scale)):int(y2 * scale) + 1, max(0, int(x1 * scale)):int(x2 * scale) + 1] = 0
        pts = cv2.goodFeaturesToTrack(prev, maxCorners=200, qualityLevel=0.01, minDistance=8, mask=mask)
        if pts is None or len(pts) < 10:
            return self.moving
        nxt, ok, _ = cv2.calcOpticalFlowPyrLK(prev, gray, pts, None, winSize=(15, 15), maxLevel=2)
        ok = ok.ravel() == 1
        if ok.sum() < 10:
            return self.moving
        d = np.linalg.norm((nxt - pts).reshape(-1, 2)[ok], axis=1)
        self.shift = (1 - self.smooth) * self.shift + self.smooth * float(np.median(d)) / scale
        if not self.moving and self.shift > self.on_px:
            self.moving = True
        elif self.moving and not self.latch and self.shift < self.off_px:
            self.moving = False
        return self.moving
