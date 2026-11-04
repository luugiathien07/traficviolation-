"""Traffic-sign detection (biển báo) with a YOLO model trained on Vietnamese signs (tools/train_signs.py).

Runs every `every_n_frames` frames on the whole frame and keeps a short memory of what is visible,
so the dashboard and the video overlay can show the signs at this camera, and the rules that need a sign
(`require_sign`) know it is there. The model has one "speed limit" class for every limit; a small classifier
(`speed_model`) reads the number, listed as an extra visible sign "speed limit 50".
"""
from __future__ import annotations

from collections import Counter

import cv2
import numpy as np

# Class names of the Vietnamese traffic-sign dataset -> Vietnamese label shown to users.
LABELS_VI = {
    "one way prohibition": "Cấm đi ngược chiều",
    "speed limit": "Giới hạn tốc độ",
    "no parking": "Cấm đỗ xe",
    "intersection danger": "Nguy hiểm: giao nhau",
    "indication": "Biển hiệu lệnh",
    "no stopping and parking": "Cấm dừng và đỗ xe",
    "other prohibition": "Biển cấm khác",
    "vehicle permission lane": "Làn theo loại xe",
    "road danger": "Nguy hiểm trên đường",
    "direction": "Chỉ hướng đi",
    "slow warning": "Đi chậm",
    "no turn left": "Cấm rẽ trái",
    "pedestrian danger": "Nguy hiểm: người đi bộ",
    "no truck entry/turning": "Cấm xe tải",
    "no car entry/turning": "Cấm ô tô",
    "pedestrian crossing": "Đường người đi bộ sang ngang",
    "no motorbike entry/turning": "Cấm xe máy",
    "no turn right": "Cấm rẽ phải",
    "vehicle and speed permission lane": "Làn theo loại xe và tốc độ",
    "other warning": "Biển cảnh báo khác",
    "other": "Biển khác",
    "height limit": "Hạn chế chiều cao",
    "no u turn": "Cấm quay đầu",
    "no more prohibition": "Hết lệnh cấm",
    "construction danger": "Công trường",
    "no u and left turn": "Cấm quay đầu và rẽ trái",
    "weight limit": "Hạn chế tải trọng",
    "no u and right turn": "Cấm quay đầu và rẽ phải",
    "overpass route": "Lối đi cầu vượt",
}
SIGN_BGR = (0, 200, 255)


def label_vi(name: str) -> str:
    n = name.lower().replace("_", " ")
    if n.startswith("speed limit ") and n[12:].isdigit():
        return f"Giới hạn tốc độ {n[12:]} km/h"
    return LABELS_VI.get(n, name)


class SignDetector:
    def __init__(self, cfg: dict):
        from ultralytics import YOLO

        self.model = YOLO(cfg.get("model", "models/vn_signs.pt"))
        self.conf = float(cfg.get("conf", 0.4))
        self.imgsz = int(cfg.get("imgsz", 1600))
        self.every = int(cfg.get("every_n_frames", 5))
        self.hold = float(cfg.get("hold_seconds", 2.0))  # keep a sign listed this long after last seeing it
        self.device = cfg.get("device")
        self._n = 0
        self.visible: dict[str, dict] = {}   # class name -> {label, conf, box, last}
        self.boxes: list[tuple] = []         # (name, conf, xyxy) of the latest run, for drawing
        # reads the number on speed-limit signs (tools/prepare_speed_signs.py); None when not configured
        from pathlib import Path
        sm = cfg.get("speed_model", "models/speed_value.pt")
        self.speed = YOLO(sm) if Path(sm).exists() else None
        self.speed_min_conf = float(cfg.get("speed_min_conf", 0.8))  # below this a reading is left out
        self.speed_votes: list[tuple[float, int]] = []  # (t, km/h); the majority within hold_seconds is used

    def update(self, frame: np.ndarray, t: float) -> None:
        self._n += 1
        if self._n % self.every == 1 or self.every == 1:
            res = self.model(frame, conf=self.conf, imgsz=self.imgsz, device=self.device, verbose=False)[0]
            self.boxes = [(res.names[int(c)], float(cf), tuple(map(float, b)))
                          for b, c, cf in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist(), res.boxes.conf.tolist())]
            for name, cf, box in self.boxes:
                v = self.visible.get(name)
                if v is None or t - v["last"] > self.hold or cf >= v["conf"]:
                    self.visible[name] = {"label": label_vi(name), "conf": cf, "box": box, "last": t}
                else:
                    v["last"], v["box"] = t, box
            self._read_speeds(frame, t)
        # "speed limit 50": the number read from the sign, listed like a sign of its own for the rules
        for k in [k for k in self.visible if k.startswith("speed limit ")]:
            del self.visible[k]
        self.speed_votes = [v for v in self.speed_votes if t - v[0] <= self.hold]
        if self.speed_votes and "speed limit" in self.visible:
            kmh = Counter(v[1] for v in self.speed_votes).most_common(1)[0][0]
            base = self.visible["speed limit"]
            self.visible[f"speed limit {kmh}"] = {**base, "label": label_vi(f"speed limit {kmh}")}
        for name in [k for k, v in self.visible.items() if t - v["last"] > self.hold]:
            del self.visible[name]
            self.boxes = [b for b in self.boxes if b[0] != name]

    def _read_speeds(self, frame: np.ndarray, t: float) -> None:
        """Read the number of each speed-limit sign of this run and vote; the drawn label shows it."""
        if self.speed is None:
            return
        for i, (name, cf, (x1, y1, x2, y2)) in enumerate(self.boxes):
            if name != "speed limit" or x2 - x1 < 16:
                continue
            crop = frame[max(0, int(y1)):int(y2), max(0, int(x1)):int(x2)]
            p = self.speed.predict(crop, imgsz=64, verbose=False)[0].probs
            if float(p.top1conf) >= self.speed_min_conf:
                kmh = int(self.speed.names[p.top1])
                self.speed_votes.append((t, kmh))
                self.boxes[i] = (f"speed limit {kmh}", cf, (x1, y1, x2, y2))

    def visible_names(self) -> set[str]:
        """Model class names of the signs currently visible (within hold_seconds)."""
        return set(self.visible)

    def summary(self) -> list[dict]:
        return [{"name": k, "label": v["label"], "conf": round(v["conf"], 2)}
                for k, v in sorted(self.visible.items(), key=lambda kv: -kv[1]["conf"])]

    def draw(self, img: np.ndarray, ascii_fn) -> None:
        for name, cf, (x1, y1, x2, y2) in self.boxes:
            p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
            cv2.rectangle(img, p1, p2, SIGN_BGR, 2)
            text = ascii_fn(f"{label_vi(name)} {cf:.2f}")
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            ty = p2[1] + th + 6 if p1[1] < th + 8 else p1[1]
            cv2.rectangle(img, (p1[0], ty - th - 6), (p1[0] + tw + 6, ty), SIGN_BGR, -1)
            cv2.putText(img, text, (p1[0] + 3, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
