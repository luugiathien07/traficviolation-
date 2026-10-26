"""Traffic-light state sources.

- roi:    read the lamp colour from a region of the frame (HSV thresholds)
- timer:  fixed signal plan (green/yellow/red seconds + offset)
- manual: state set from the dashboard / API (useful when the light is off-camera)

Any source can be overridden at runtime with `override(state)`; `override(None)`
returns to the configured source.
"""
from __future__ import annotations

from collections import Counter, deque

import cv2
import numpy as np

STATES = ("red", "yellow", "green")


class LightSource:
    def __init__(self, cfg: dict):
        self.mode = cfg.get("mode", "manual")
        self.roi = cfg.get("roi")  # [x1, y1, x2, y2]
        self.min_ratio = float(cfg.get("min_ratio", 0.04))
        self.history: deque[str] = deque(maxlen=int(cfg.get("smooth_frames", 5)))
        self.plan = cfg.get("timer", {"green": 20, "yellow": 3, "red": 20, "offset": 0})
        self.manual_state = cfg.get("initial", "green")
        self._override: str | None = None
        self.state = "unknown"
        self.ratios: dict[str, float] = {}

    def override(self, state: str | None) -> None:
        if state is not None and state not in STATES:
            raise ValueError(f"state must be one of {STATES} or None")
        self._override = state

    @property
    def overridden(self) -> bool:
        return self._override is not None

    def update(self, frame: np.ndarray, t: float) -> str:
        if self._override:
            self.state = self._override
        elif self.mode == "roi" and self.roi:
            self.history.append(self._classify(frame))
            known = [s for s in self.history if s != "unknown"]
            self.state = Counter(known).most_common(1)[0][0] if known else "unknown"
        elif self.mode == "timer":
            g, y, r = self.plan["green"], self.plan["yellow"], self.plan["red"]
            tt = (t + self.plan.get("offset", 0)) % (g + y + r)
            self.state = "green" if tt < g else "yellow" if tt < g + y else "red"
        else:
            self.state = self.manual_state
        return self.state

    def _classify(self, frame: np.ndarray) -> str:
        x1, y1, x2, y2 = map(int, self.roi)
        crop = frame[max(0, y1):y2, max(0, x1):x2]
        if crop.size == 0:
            return "unknown"
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        lit = (hsv[..., 1] > 90) & (hsv[..., 2] > 140)
        h = hsv[..., 0]
        masks = {
            "red": lit & ((h < 10) | (h > 165)),
            "yellow": lit & (h >= 12) & (h <= 35),
            "green": lit & (h >= 40) & (h <= 95),
        }
        n = crop.shape[0] * crop.shape[1]
        self.ratios = {k: float(m.sum()) / n for k, m in masks.items()}
        best = max(self.ratios, key=self.ratios.get)
        return best if self.ratios[best] >= self.min_ratio else "unknown"
