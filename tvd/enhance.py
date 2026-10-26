"""Backlight compensation (bù ngược sáng).

When the camera faces the sun, a bright sky or oncoming headlights, the exposure is
set by the bright area and vehicles end up as dark silhouettes, which hurts both
detection and evidence readability. This module:

1. measures how backlit the frame is (share of clipped highlights vs. deep shadows),
2. lifts shadows with a tone curve and compresses highlights,
3. applies CLAHE on the luminance channel to restore local contrast.

Only luminance (L in LAB) is changed, so colours stay usable. The traffic-light
classifier still reads the raw frame, because enhancement shifts lamp colours.

Modes: "off", "on" (always, fixed strength), "auto" (strength follows the measured
backlight score, with hysteresis so it doesn't flicker).
"""
from __future__ import annotations

import cv2
import numpy as np

MODES = ("off", "on", "auto")


class BacklightCompensator:
    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        self.mode = cfg.get("mode", "auto")
        if self.mode not in MODES:
            raise ValueError(f"backlight.mode must be one of {MODES}")
        self.strength = float(cfg.get("strength", 0.8))      # 0..1, used in "on" mode
        self.gamma = float(cfg.get("shadow_gamma", 0.55))     # <1 lifts shadows
        self.knee = int(cfg.get("highlight_knee", 210))       # highlights above are compressed
        self.on_at = float(cfg.get("auto_on", 0.40))          # score to switch on in auto
        self.off_at = float(cfg.get("auto_off", 0.25))        # score to switch off in auto
        self.clahe = cv2.createCLAHE(clipLimit=float(cfg.get("clahe_clip", 2.5)),
                                     tileGridSize=(int(cfg.get("clahe_tiles", 8)),) * 2)
        self.score = 0.0      # smoothed backlight score 0..1
        self.active = False   # whether compensation is currently applied
        self.applied_strength = 0.0
        self._lut_cache: dict[int, np.ndarray] = {}

    def set_mode(self, mode: str) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.mode = mode

    @staticmethod
    def measure(gray: np.ndarray) -> float:
        """0 = evenly lit, 1 = strongly backlit (lots of clipped highlights AND deep shadows)."""
        small = cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)
        n = small.size
        bright = np.count_nonzero(small > 230) / n
        dark = np.count_nonzero(small < 60) / n
        # Need both: a bright scene alone is just daylight, a dark one alone is night.
        return float(np.clip(min(bright / 0.08, 1.0) * min(dark / 0.30, 1.0), 0.0, 1.0))

    def _lut(self, strength: float) -> np.ndarray:
        key = int(round(strength * 20))
        lut = self._lut_cache.get(key)
        if lut is None:
            s = key / 20
            x = np.arange(256, dtype=np.float32)
            lifted = 255.0 * (x / 255.0) ** self.gamma
            w = s * (1.0 - x / 255.0)  # lift dark tones most, leave bright tones alone
            y = x + (lifted - x) * w
            hi = y > self.knee
            y[hi] = self.knee + (y[hi] - self.knee) * (1.0 - 0.5 * s)
            lut = self._lut_cache[key] = np.clip(y, 0, 255).astype(np.uint8)
        return lut

    def process(self, frame: np.ndarray) -> np.ndarray:
        if self.mode == "off":
            self.active, self.applied_strength = False, 0.0
            return frame
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        L = lab[..., 0]
        self.score = 0.85 * self.score + 0.15 * self.measure(L)
        if self.mode == "on":
            self.active, strength = True, self.strength
        else:
            if self.active and self.score < self.off_at:
                self.active = False
            elif not self.active and self.score > self.on_at:
                self.active = True
            strength = float(np.clip(self.score * 1.4, 0.35, 1.0)) if self.active else 0.0
        self.applied_strength = strength
        if not self.active:
            return frame
        L = cv2.LUT(L, self._lut(strength))
        lab[..., 0] = self.clahe.apply(L)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
