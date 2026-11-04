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
from .motion import CameraMotion
from .rules import VEHICLES, Track, Violation, ZoneRule, _cover, build_rules
from .storage import Store

log = logging.getLogger("tvd")

COCO_IDS = {"person": 0, "bicycle": 1, "car": 2, "motorcycle": 3, "bus": 5, "truck": 7}
LIGHT_BGR = {"red": (60, 60, 255), "yellow": (40, 190, 250), "green": (100, 210, 60), "unknown": (160, 160, 160)}
TYPE_BGR = {"red_light": (60, 60, 255), "speeding": (40, 150, 255), "wrong_lane": (255, 120, 180),
            "no_helmet": (255, 190, 60), "no_turn": (200, 80, 255), "wrong_way": (0, 140, 255),
            "no_entry": (80, 80, 200), "sidewalk": (60, 200, 160), "no_parking": (200, 200, 0),
            "overload": (0, 220, 255), "solid_line": (0, 230, 230), "no_yield": (255, 255, 120),
            "block_junction": (120, 60, 255)}
# rules that need the people in the frame (riders, pedestrians): the detector then also finds COCO person
PEOPLE_RULES = ("overload", "no_yield")


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
        if any(cfg.get("rules", {}).get(k, {}).get("enabled") for k in PEOPLE_RULES) and 0 not in self.class_ids:
            self.class_ids.append(0)
            # bicycles too, only to tell cyclists from pedestrians (not tracked unless detector.classes has them)
            self.bikes_only = 1 not in self.class_ids
            if self.bikes_only:
                self.class_ids.append(1)
        self.loop = bool(cfg.get("loop", False))
        self.realtime = bool(cfg.get("realtime", True))
        self.light = LightSource(cfg.get("light", {}))
        self.backlight = BacklightCompensator(cfg.get("backlight", {}))
        self.motion = CameraMotion(cfg.get("camera_motion", {}))
        self.rules = build_rules(cfg)
        self.plate_cfg = cfg.get("plate", {})
        self.plate_reader = None  # loaded in run() when plate.enabled
        self.helmet_cfg = cfg.get("helmet", {})
        self.helmet = None        # loaded in run() when helmet.enabled
        self.signs_cfg = cfg.get("signs", {})
        self.signs = None         # loaded in run() when signs.enabled
        # track_id -> {"rows": [violation ids], "until": t, "next": t}; retried on later frames
        self.plate_jobs: dict[int, dict] = {}
        # type + plate -> video time of the last saved violation, to drop repeats when a vehicle gets a new track id
        self._last_by_plate: dict[str, float] = {}
        # violation rows of each live track, so a plate read later (read_all) is filled into all of them
        self._rows_by_track: dict[int, list[int]] = {}

        self.tracks: dict[int, Track] = {}
        self.frame_lock = threading.Lock()
        self.latest_jpeg: bytes | None = None
        self.frame_size: tuple[int, int] | None = None
        self.stats = {"frames": 0, "fps": 0.0, "active_tracks": 0, "vehicles_seen": 0,
                      "by_class": Counter(), "light": "unknown", "running": False, "source": str(self.source)}
        self._seen_ids: set[int] = set()
        self._plate_img: dict[int, np.ndarray] = {}
        self.stop_event = threading.Event()
        self._people: list[tuple[float, float, float, float]] = []
        self._bikes: list[tuple[float, float, float, float]] = []
        self.on_violation = None  # optional callback(row_id, Violation)

    # ---------- main loop ----------
    def run(self, max_frames: int | None = None, writer_path: str | None = None) -> None:
        from ultralytics import YOLO  # heavy import, keep it lazy

        model = YOLO(self.model_path)
        if 0 in self.class_ids:
            # registered before model.track() adds the tracker's callback, so it runs first: people go to the
            # rules but not to ByteTrack, which ignores classes and would hand a motorbike's id to its rider
            model.add_callback("on_predict_postprocess_end", self._take_people)
        if self.helmet_cfg.get("enabled") and self.helmet is None:
            from pathlib import Path
            if Path(self.helmet_cfg.get("model", "models/helmet.pt")).exists():
                from .helmet import HelmetChecker
                self.helmet = HelmetChecker(self.helmet_cfg)
            else:
                log.warning("Chưa có model mũ bảo hiểm (%s); bỏ qua luật mũ bảo hiểm. Huấn luyện bằng tools/train_helmet.py",
                            self.helmet_cfg.get("model", "models/helmet.pt"))
        if self.signs_cfg.get("enabled") and self.signs is None:
            from .signs import SignDetector
            self.signs = SignDetector(self.signs_cfg)
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

    def _take_people(self, predictor) -> None:
        self._people, self._bikes = [], []
        for i, result in enumerate(predictor.results):
            cls = result.boxes.cls
            is_person = cls == 0
            is_bike = (cls == 1) & getattr(self, "bikes_only", False)
            self._people += [tuple(map(float, b)) for b in result.boxes.xyxy[is_person].cpu().numpy()]
            self._bikes += [tuple(map(float, b)) for b in result.boxes.xyxy[is_bike].cpu().numpy()]
            predictor.results[i] = result[~(is_person | is_bike)]

    def process(self, model, frame: np.ndarray, t: float) -> tuple[np.ndarray, list[Violation]]:
        self.frame_size = (frame.shape[1], frame.shape[0])
        light = self.light.update(frame, t)  # raw frame: enhancement would shift lamp colours
        frame = self.backlight.process(frame)
        res = model.track(frame, persist=True, classes=self.class_ids, conf=self.conf, imgsz=self.imgsz,
                          tracker=self.tracker, device=self.device, verbose=False)[0]
        names = res.names
        violations: list[Violation] = []
        boxes = res.boxes
        seen: list[Track] = []
        # riders and pedestrians of this frame, taken out before tracking by _take_people
        people, self._people = self._people, []
        # a person over a bicycle is a cyclist, neither a pedestrian nor a motorbike rider
        people = [p for p in people if not any(_cover(p, b) >= 0.25 for b in self._bikes)]
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
                tr.cls_votes[cname] += 1
                seen.append(tr)
        # helmet votes need this frame's boxes of every track, so rules run after all tracks moved
        if self.helmet:
            self.helmet.update(frame, seen)
        if self.signs:
            self.signs.update(frame, t)
            visible = self.signs.visible_names()
            for rule in self.rules:
                if hasattr(rule, "signs_seen"):
                    rule.signs_seen(visible, t)
        for rule in self.rules:
            if hasattr(rule, "people_seen"):
                rule.people_seen(people, seen, t)
        # dashcam / handheld: lines and zones drawn for a fixed camera no longer fit, and parked cars seem to move
        moving = self.motion.update(frame, [tr.bbox for tr in seen])
        for tr in seen:
            for rule in self.rules:
                if moving and getattr(rule, "fixed_camera", False):
                    continue
                violations.extend(rule.update(tr, light))
        # forget tracks not seen for 3 s
        for tid in [k for k, v in self.tracks.items() if t - v.last_seen > 3.0]:
            del self.tracks[tid]
            self._plate_img.pop(tid, None)
            self._rows_by_track.pop(tid, None)

        annotated = self.annotate(frame, t, light)
        for v in violations:
            self._save(frame, annotated, v, light)
        if self.plate_reader:
            self._read_plates(frame, t)
            if self.plate_cfg.get("read_all"):
                self._read_all_plates(frame, seen, t)

        self.stats["frames"] += 1
        self.stats["active_tracks"] = sum(1 for v in self.tracks.values() if v.last_seen == t)
        self.stats["light"] = light
        self.stats["camera_moving"] = self.motion.moving
        self.stats["backlight"] = {"mode": self.backlight.mode, "active": self.backlight.active,
                                   "score": round(self.backlight.score, 3),
                                   "strength": round(self.backlight.applied_strength, 2)}
        if self.signs:
            self.stats["signs"] = self.signs.summary()
        ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with self.frame_lock:
                self.latest_jpeg = buf.tobytes()
        return annotated, violations

    def _save(self, raw: np.ndarray, annotated: np.ndarray, v: Violation, light: str) -> None:
        tr = v.track
        if tr.plate:
            key = f"{v.type}|{tr.plate}"
            last = self._last_by_plate.get(key)
            self._last_by_plate[key] = v.t
            if last is not None and abs(v.t - last) < float(self.cfg.get("dedupe_seconds", 15)):
                log.info("Bỏ lỗi trùng %s của xe %s (track mới #%d)", v.type, tr.plate, tr.id)
                return
        x1, y1, x2, y2 = map(int, tr.bbox)
        if v.type == "no_helmet" and tr.rider_box:
            # the motorcycle box leaves out the riders' heads: crop the helmet-model box (bike + riders) as well
            rx1, ry1, rx2, ry2 = map(int, tr.rider_box)
            x1, y1, x2, y2 = min(x1, rx1), min(y1, ry1), max(x2, rx2), max(y2, ry2)
        if v.type == "overload" and tr.overload_box:
            x1, y1, x2, y2 = map(int, tr.overload_box)
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
        self._rows_by_track.setdefault(tr.id, []).append(row_id)
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

    def _read_all_plates(self, frame: np.ndarray, seen: list[Track], t: float) -> None:
        """plate.read_all: read the plate of every vehicle on screen (not only violators) so it can be shown
        live, at most every read_all_every seconds per vehicle and only while no confident reading exists yet."""
        every = float(self.plate_cfg.get("read_all_every", 0.5))
        min_w = float(self.plate_cfg.get("read_all_min_width", 80))  # vehicle box width; smaller plates can't be read
        h, w = frame.shape[:2]
        for tr in seen:
            x1, y1, x2, y2 = map(int, tr.bbox)
            if tr.plate_conf >= 0.95 or t < tr.plate_next or x2 - x1 < min_w:
                continue
            tr.plate_next = t + every
            res = self.plate_reader.read(frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)],
                                         motorbike=tr.cls == "motorcycle")
            if res and res.text and res.conf > tr.plate_conf:
                tr.plate, tr.plate_conf = res.text, res.conf
                self._plate_img[tr.id] = res.image
                # a better reading after the violation was saved: fill it into the vehicle's violations too
                for row_id in self._rows_by_track.get(tr.id, []):
                    self.store.set_plate_auto(row_id, res.text, res.conf, res.image)

    # ---------- drawing ----------
    def annotate(self, frame: np.ndarray, t: float, light: str) -> np.ndarray:
        img = frame.copy()
        # a moving camera has its lines and zones in the wrong place: don't draw them (their rules are off)
        r = {} if self.motion.moving else self.cfg.get("rules", {})
        lc = LIGHT_BGR.get(light, LIGHT_BGR["unknown"])
        if r.get("wrong_lane", {}).get("enabled"):
            overlay = img.copy()
            for ln in r["wrong_lane"]["lanes"]:
                poly = np.array(ln["polygon"], np.int32)
                cv2.fillPoly(overlay, [poly], ln.get("color", (200, 160, 60)))
                cv2.putText(img, ln["name"], tuple(poly[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            img = cv2.addWeighted(overlay, 0.18, img, 0.82, 0)
        for kind in ZoneRule.KINDS:
            if not r.get(kind, {}).get("enabled"):
                continue
            zc = TYPE_BGR[kind]
            for z in r[kind].get("zones", []):
                poly = np.array(z["polygon"], np.int32)
                cv2.polylines(img, [poly], True, zc, 2, cv2.LINE_AA)
                x0, y0 = map(int, poly[0])
                cv2.putText(img, _ascii(z.get("name", kind)), (x0 + 6, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, zc, 2, cv2.LINE_AA)
                if "direction" in z:  # allowed direction of travel
                    dx, dy = z["direction"]
                    n = float(np.hypot(dx, dy))
                    c = (x0 + 60, y0 + 45)
                    cv2.arrowedLine(img, c, (int(c[0] + 80 * dx / n), int(c[1] + 80 * dy / n)), zc, 3, cv2.LINE_AA, 0, 0.3)
        if r.get("speeding", {}).get("enabled"):
            for key in ("line_a", "line_b"):
                a, b = r["speeding"][key]
                cv2.line(img, tuple(map(int, a)), tuple(map(int, b)), (40, 190, 250), 1, cv2.LINE_AA)
        if r.get("red_light", {}).get("enabled"):
            a, b = r["red_light"]["stop_line"]
            cv2.line(img, tuple(map(int, a)), tuple(map(int, b)), lc, 3, cv2.LINE_AA)
        if r.get("no_turn", {}).get("enabled"):
            a, b = r["no_turn"]["entry_line"]
            cv2.line(img, tuple(map(int, a)), tuple(map(int, b)), (230, 230, 230), 2, cv2.LINE_AA)
            for ex in r["no_turn"]["banned_exits"]:
                a, b = ex["line"]
                cv2.line(img, tuple(map(int, a)), tuple(map(int, b)), TYPE_BGR["no_turn"], 2, cv2.LINE_AA)
                cv2.putText(img, "cam " + ex["name"], tuple(map(int, a)), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            TYPE_BGR["no_turn"], 1, cv2.LINE_AA)
        if r.get("solid_line", {}).get("enabled"):
            for ln in r["solid_line"].get("lines", []):
                pts = np.array(ln["points"], np.int32)
                cv2.polylines(img, [pts], False, TYPE_BGR["solid_line"], 3, cv2.LINE_AA)
                cv2.putText(img, _ascii(ln.get("name", "vach lien")), tuple(map(int, pts[0])), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, TYPE_BGR["solid_line"], 1, cv2.LINE_AA)
        if r.get("no_yield", {}).get("enabled"):
            for c in r["no_yield"].get("crosswalks", []):
                poly = np.array(c["polygon"], np.int32)
                cv2.polylines(img, [poly], True, TYPE_BGR["no_yield"], 2, cv2.LINE_AA)
                cv2.putText(img, _ascii(c.get("name", "vach nguoi di bo")), (int(poly[0][0]) + 6, int(poly[0][1]) + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, TYPE_BGR["no_yield"], 2, cv2.LINE_AA)
        if self.signs:
            self.signs.draw(img, _ascii)
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
        if self.motion.moving:
            hud += "  CAMERA DI DONG: tat luat theo vach/vung"
        cv2.rectangle(img, (0, 0), (img.shape[1], 26), (15, 15, 15), -1)
        cv2.circle(img, (14, 13), 7, lc, -1)
        cv2.putText(img, hud, (28, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
        return img


def _ascii(text: str) -> str:
    # OpenCV's Hershey fonts have no Vietnamese glyphs; keep drawn text ASCII.
    import unicodedata
    text = text.replace("đ", "d").replace("Đ", "D")
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


def _banner(img: np.ndarray, text: str, color) -> None:
    ascii_text = _ascii(text)
    h = img.shape[0]
    cv2.rectangle(img, (0, h - 30), (img.shape[1], h), color, -1)
    cv2.putText(img, ascii_text, (10, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
