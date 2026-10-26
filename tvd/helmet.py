"""Helmet check for tracked motorcycles.

A detector trained on the HELMET dataset (tools/train_helmet.py) finds motorcycles with
their riders and says whether everyone wears a helmet (class 0) or not (class 1).
Each detection is matched to the tracked motorcycle it covers and added as a vote,
weighted by confidence; NoHelmetRule decides once enough frames agree.
"""
from __future__ import annotations

import numpy as np

from .rules import Track


def _cover(moto, det) -> float:
    """Share of the tracked motorcycle box that lies inside the helmet-model box."""
    ix1, iy1 = max(moto[0], det[0]), max(moto[1], det[1])
    ix2, iy2 = min(moto[2], det[2]), min(moto[3], det[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    return inter / max(1.0, (moto[2] - moto[0]) * (moto[3] - moto[1]))


class HelmetChecker:
    def __init__(self, cfg: dict):
        from ultralytics import YOLO

        self.model = YOLO(cfg.get("model", "models/helmet.pt"))
        self.conf = float(cfg.get("conf", 0.35))
        self.imgsz = int(cfg.get("imgsz", 1280))
        self.every = int(cfg.get("every_n_frames", 3))
        self.min_cover = float(cfg.get("min_cover", 0.5))
        self.device = cfg.get("device")
        self._n = 0
        self.last: list[tuple[np.ndarray, int, float]] = []  # boxes of the latest run, for drawing

    def update(self, frame: np.ndarray, tracks: list[Track]) -> None:
        self._n += 1
        motos = [tr for tr in tracks if tr.cls == "motorcycle"]
        if not motos or self._n % self.every:
            return
        res = self.model(frame, conf=self.conf, imgsz=self.imgsz, device=self.device, verbose=False)[0]
        dets = list(zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.int().cpu().tolist(), res.boxes.conf.cpu().tolist()))
        self.last = dets
        for tr in motos:
            best = max(dets, key=lambda d: _cover(tr.bbox, d[0]), default=None)
            if best is None or _cover(tr.bbox, best[0]) < self.min_cover:
                continue
            _, cls, conf = best
            tr.helmet_votes[cls] += conf
            tr.helmet_frames += 1
