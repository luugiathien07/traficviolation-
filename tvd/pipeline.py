"""Video -> detect & track -> rules -> evidence. Runs in its own thread when served."""
from __future__ import annotations

import logging
import threading
import time
from collections import Counter

import cv2
import numpy as np

from .enhance import BacklightCompensator
from .light import LightSource
from .rules import VEHICLES, Track, Violation, build_rules
from .storage import Store

log = logging.getLogger("tvd")

COCO_IDS = {"bicycle": 1, "car": 2, "motorcycle": 3, "bus": 5, "truck": 7}
LIGHT_BGR = {"red": (60, 60, 255), "yellow": (40, 190, 250), "green": (100, 210, 60), "unknown": (160, 160, 160)}
TYPE_BGR = {"red_light": (60, 60, 255), "speeding": (40, 150, 255), "wrong_lane": (255, 120, 180)}


class Pipeline:
    def __init__(self, cfg: dict, store: Store):
        self.cfg = cfg
        self.store = store
        self.camera_id = cfg.get("camera_id", "cam01")
        self.source = cfg["source"]
        det = cfg.get("detector", {})
        self.model_path = det.get("model", "yolo11n.pt")
        self.conf = float(det.get("conf", 0.3))
        self.imgsz = int(det.get("imgsz", 960))
        self.device = det.get("device")  # None = auto
        self.tracker = det.get("tracker", "bytetrack.yaml")
        self.class_ids = [COCO_IDS[c] for c in det.get("classes", VEHICLES)]
        self.loop = bool(cfg.get("loop", False))
        self.realtime = bool(cfg.get("realtime", True))
        self.light = LightSource(cfg.get("light", {}))
        self.backlight = BacklightCompensator(cfg.get("backlight", {}))
        self.rules = build_rules(cfg)
        self.plate_cfg = cfg.get("plate", {})
        self.plate_reader = None  # loaded in run() when plate.enabled
        # track_id -> {"rows": [violation ids], "until": t, "next": t}; retried on later frames
        self.plate_jobs: dict[int, dict] = {}

        self.tracks: dict[int, Track] = {}
        self.frame_lock = threading.Lock()
        self.latest_jpeg: bytes | None = None
        self.frame_size: tuple[int, int] | None = None
        self.stats = {"frames": 0, "fps": 0.0, "active_tracks": 0, "vehicles_seen": 0,
                      "by_class": Counter(), "light": "unknown", "running": False, "source": str(self.source)}
        self._seen_ids: set[int] = set()
        self._plate_img: dict[int, np.ndarray] = {}
        self.stop_event = threading.Event()
        self.on_violation = None  # optional callback(row_id, Violation)

    # ---------- main loop ----------
    def run(self, max_frames: int | None = None, writer_path: str | None = None) -> None:
        from ultralytics import YOLO  # heavy import, keep it lazy

        model = YOLO(self.model_path)
        if self.plate_cfg.get("enabled") and self.plate_reader is None:
            from .plate import PlateReader
            self.plate_reader = PlateReader(self.plate_cfg)
        src = int(self.source) if str(self.source).isdigit() else self.source
        is_file = isinstance(src, str) and not src.startswith(("rtsp://", "http://", "https://"))
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise RuntimeError(f"Không mở được nguồn video: {self.source}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        writer = None
        frame_idx, t_start, t_last = 0, time.time(), time.time()
        self.stats["running"] = True
        log.info("Đang chạy %s (fps=%.1f, model=%s)", self.source, fps, self.model_path)
        try:
            while not self.stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    if is_file and self.loop:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        model.predictor = None  # reset tracker ids between loops
                        self.tracks.clear()
                        continue
                    break
                # Video time: frame clock for files (exact), wall clock for live streams.
                t = frame_idx / fps if is_file else time.time() - t_start
                annotated, _ = self.process(model, frame, t)
                if writer_path:
                    if writer is None:
                        h, w = annotated.shape[:2]
                        writer = cv2.VideoWriter(writer_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    writer.write(annotated)
                frame_idx += 1
                now = time.time()
                self.stats["fps"] = 0.9 * self.stats["fps"] + 0.1 / max(1e-6, now - t_last)
                t_last = now
                if self.realtime and is_file:
                    lag = (frame_idx / fps) - (now - t_start)
                    if lag > 0:
                        time.sleep(lag)
                if max_frames and frame_idx >= max_frames:
                    break
        finally:
            cap.release()
            if writer:
                writer.release()
            self.stats["running"] = False

    def process(self, model, frame: np.ndarray, t: float) -> tuple[np.ndarray, list[Violation]]:
        self.frame_size = (frame.shape[1], frame.shape[0])
        light = self.light.update(frame, t)  # raw frame: enhancement would shift lamp colours
        frame = self.backlight.process(frame)
        res = model.track(frame, persist=True, classes=self.class_ids, conf=self.conf, imgsz=self.imgsz,
                          tracker=self.tracker, device=self.device, verbose=False)[0]
        names = res.names
        violations: list[Violation] = []
        boxes = res.boxes
        if boxes is not None and boxes.id is not None:
            for xyxy, tid, cls, conf in zip(boxes.xyxy.cpu().numpy(), boxes.id.int().cpu().tolist(),
                                             boxes.cls.int().cpu().tolist(), boxes.conf.cpu().tolist()):
                cname = names[cls]
                tr = self.tracks.get(tid)
                if tr is None:
                    tr = self.tracks[tid] = Track(tid, cname, conf, tuple(xyxy), t)
                    if tid not in self._seen_ids:
                        self._seen_ids.add(tid)
                        self.stats["vehicles_seen"] += 1
                        self.stats["by_class"][cname] += 1
                tr.bbox, tr.last_seen, tr.conf = tuple(map(float, xyxy)), t, max(tr.conf, conf)
                tr.history.append((t, tr.anchor))
                for rule in self.rules:
                    violations.extend(rule.update(tr, light))
        # forget tracks not seen for 3 s
        for tid in [k for k, v in self.tracks.items() if t - v.last_seen > 3.0]:
            del self.tracks[tid]
            self._plate_img.pop(tid, None)

        annotated = self.annotate(frame, t, light)
        for v in violations:
            self._save(frame, annotated, v, light)
        if self.plate_reader:
            self._read_plates(frame, t)

        self.stats["frames"] += 1
        self.stats["active_tracks"] = sum(1 for v in self.tracks.values() if v.last_seen == t)
        self.stats["light"] = light
        self.stats["backlight"] = {"mode": self.backlight.mode, "active": self.backlight.active,
                                   "score": round(self.backlight.score, 3),
                                   "strength": round(self.backlight.applied_strength, 2)}
        ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with self.frame_lock:
                self.latest_jpeg = buf.tobytes()
        return annotated, violations

    def _save(self, raw: np.ndarray, annotated: np.ndarray, v: Violation, light: str) -> None:
        tr = v.track
        x1, y1, x2, y2 = map(int, tr.bbox)
        h, w = raw.shape[:2]
        pad = int(0.15 * max(x2 - x1, y2 - y1)) + 8
        crop = raw[max(0, y1 - pad):min(h, y2 + pad), max(0, x1 - pad):min(w, x2 + pad)].copy()
        evidence = annotated.copy()
        color = TYPE_BGR.get(v.type, (0, 0, 255))
        cv2.rectangle(evidence, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), color, 4)
        _banner(evidence, f"{v.type.upper()}  #{tr.id}  {v.detail}", color)
        row_id = self.store.add(camera_id=self.camera_id, video_t=v.t, type=v.type, track_id=tr.id,
                                vclass=tr.cls, conf=tr.conf, speed_kmh=v.speed_kmh, light=light,
                                detail=v.detail, frame=evidence, crop=crop)
        log.info("VI PHẠM #%d %s track=%d %s (%s)", row_id, v.type, tr.id, tr.cls, v.detail)
        if self.plate_reader:
            if tr.plate:
                self.store.set_plate_auto(row_id, tr.plate, tr.plate_conf, self._plate_img.get(tr.id, crop))
            else:
                job = self.plate_jobs.setdefault(tr.id, {"rows": [], "next": v.t})
                job["rows"].append(row_id)
                job["until"] = v.t + float(self.plate_cfg.get("retry_seconds", 2.0))
        if self.on_violation:
            self.on_violation(row_id, v)

    def _read_plates(self, frame: np.ndarray, t: float) -> None:
        """Try to read the plate of every vehicle with a pending violation; keep the
        most confident reading and write it to all of that vehicle's violations."""
        step = float(self.plate_cfg.get("retry_every", 0.1))
        for tid in list(self.plate_jobs):
            job, tr = self.plate_jobs[tid], self.tracks.get(tid)
            if tr is None or t > job["until"]:
                if tr is None or not tr.plate:
                    log.info("Không đọc được biển số xe #%d", tid)
                del self.plate_jobs[tid]
                continue
            if tr.last_seen != t or t < job["next"]:
                continue
            job["next"] = t + step
            x1, y1, x2, y2 = map(int, tr.bbox)
            h, w = frame.shape[:2]
            res = self.plate_reader.read(frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)],
                                        motorbike=tr.cls == "motorcycle")
            if res and res.text and res.conf > tr.plate_conf:
                tr.plate, tr.plate_conf = res.text, res.conf
                self._plate_img[tid] = res.image
                for row_id in job["rows"]:
                    self.store.set_plate_auto(row_id, res.text, res.conf, res.image)
                log.info("Biển số xe #%d: %s (%.2f)", tid, res.text, res.conf)
                if res.conf >= 0.95:
                    del self.plate_jobs[tid]

    # ---------- drawing ----------
    def annotate(self, frame: np.ndarray, t: float, light: str) -> np.ndarray:
        img = frame.copy()
        r = self.cfg.get("rules", {})
        lc = LIGHT_BGR.get(light, LIGHT_BGR["unknown"])
        if r.get("wrong_lane", {}).get("enabled"):
            overlay = img.copy()
            for ln in r["wrong_lane"]["lanes"]:
                poly = np.array(ln["polygon"], np.int32)
                cv2.fillPoly(overlay, [poly], ln.get("color", (200, 160, 60)))
                cv2.putText(img, ln["name"], tuple(poly[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            img = cv2.addWeighted(overlay, 0.18, img, 0.82, 0)
        if r.get("speeding", {}).get("enabled"):
            for key in ("line_a", "line_b"):
                a, b = r["speeding"][key]
                cv2.line(img, tuple(map(int, a)), tuple(map(int, b)), (40, 190, 250), 1, cv2.LINE_AA)
        if r.get("red_light", {}).get("enabled"):
            a, b = r["red_light"]["stop_line"]
            cv2.line(img, tuple(map(int, a)), tuple(map(int, b)), lc, 3, cv2.LINE_AA)
        if self.light.mode == "roi" and self.light.roi:
            x1, y1, x2, y2 = map(int, self.light.roi)
            cv2.rectangle(img, (x1, y1), (x2, y2), lc, 2)

        for tr in self.tracks.values():
            if tr.last_seen != t:
                continue
            x1, y1, x2, y2 = map(int, tr.bbox)
            color = TYPE_BGR[sorted(tr.flagged)[0]] if tr.flagged else (200, 210, 80)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2 if tr.flagged else 1)
            label = f"#{tr.id} {tr.cls} {tr.conf:.2f}"
            if tr.speed_kmh is not None:
                label += f" {tr.speed_kmh:.0f}km/h"
            if tr.plate:
                label += f" [{tr.plate}]"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
            cv2.putText(img, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1, cv2.LINE_AA)
            if len(tr.history) > 1:
                pts = np.array([p for _, p in tr.history], np.int32)
                cv2.polylines(img, [pts], False, color, 1, cv2.LINE_AA)

        hud = f"{self.camera_id}  t={t:6.1f}s  den: {light.upper()}{' (tay)' if self.light.overridden else ''}  xe: {self.stats['vehicles_seen']}"
        if self.backlight.active:
            hud += f"  BLC {self.backlight.applied_strength:.1f}"
        cv2.rectangle(img, (0, 0), (img.shape[1], 26), (15, 15, 15), -1)
        cv2.circle(img, (14, 13), 7, lc, -1)
        cv2.putText(img, hud, (28, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
        return img


def _banner(img: np.ndarray, text: str, color) -> None:
    # OpenCV's Hershey fonts have no Vietnamese glyphs; keep evidence banners ASCII.
    import unicodedata
    ascii_text = unicodedata.normalize("NFKD", text).replace("đ", "d").replace("Đ", "D")
    ascii_text = ascii_text.encode("ascii", "ignore").decode()
    h = img.shape[0]
    cv2.rectangle(img, (0, h - 30), (img.shape[1], h), color, -1)
    cv2.putText(img, ascii_text, (10, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
