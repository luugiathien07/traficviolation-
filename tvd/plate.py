"""Licence-plate reading: find the plate inside a vehicle crop, then OCR it.

Vietnamese plates come in two shapes:
- long, one row  (cars):            51A-123.45
- square, two rows (motorbikes, some cars): 29-B1 / 234.56

The global OCR model reads most square plates whole; for plates whose
height/width ratio is above `two_row_ratio` the rows are also read separately
and the better reading is kept (`two_row_mode: best`). The raw OCR text is then fitted to the
Vietnamese layout, fixing look-alike characters by position (a letter where
a digit belongs, and the other way round).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import cv2
import numpy as np

log = logging.getLogger("tvd")

TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "U": "0", "I": "1", "L": "1", "T": "1", "Z": "2",
            "J": "3", "A": "4", "S": "5", "G": "6", "B": "8"}
TO_LETTER = {"0": "D", "1": "T", "2": "Z", "4": "A", "5": "S", "6": "G", "8": "B"}
# Letters used in Vietnamese plate series (no I, J, O, Q, R, W)
SERIES_LETTERS = set("ABCDEFGHKLMNPSTUVXYZ")


@dataclass
class PlateResult:
    text: str | None        # formatted, e.g. "51A-123.45"; None if it doesn't fit the VN layout
    raw: str                # OCR output before formatting
    conf: float             # mean character confidence (0..1)
    det_conf: float         # plate detector confidence
    box: tuple[int, int, int, int]  # plate box inside the vehicle crop
    two_rows: bool          # read as two separate rows
    image: np.ndarray       # the plate crop


def normalize_vn(raw: str, motorbike: bool = False) -> str | None:
    """Fit OCR text to the Vietnamese plate layout:
    2 digits (province) + series (1 letter + optional letter/digit) + 4 or 5 digits.
    The same characters can fit both layouts (30L90860 is car "30L-908.60" or
    motorbike "30-L9 0860"); `motorbike` - from the detected vehicle class, since
    some cars also carry square plates - decides which one is tried first."""
    s = re.sub(r"[^A-Z0-9]", "", raw.upper())
    if not 7 <= len(s) <= 10:
        return None
    best = None
    # Try series of length 1 (51A-12345) and 2 (29B1-23456 / 51LD-12345)
    for series_len in ((2, 1) if motorbike else (1, 2)):
        num = s[2 + series_len:]
        if len(num) not in (4, 5):
            continue
        prov = "".join(TO_DIGIT.get(c, c) for c in s[:2])
        first = TO_LETTER.get(s[2], s[2])
        series = first + (s[3] if series_len == 2 else "")
        digits = "".join(TO_DIGIT.get(c, c) for c in num)
        if not (prov.isdigit() and first in SERIES_LETTERS and digits.isdigit()):
            continue
        if series_len == 2 and not (s[3].isdigit() or s[3] in SERIES_LETTERS):
            continue
        tail = f"{digits[:3]}.{digits[3:]}" if len(digits) == 5 else digits
        # Motorbike series end in a digit and are written "29-B1 256.62"; cars "51A-123.45" / "50LD-004.54"
        cand = f"{prov}-{series} {tail}" if series[-1].isdigit() else f"{prov}{series}-{tail}"
        # Prefer the reading that needed fewer substitutions
        changes = sum(a != b for a, b in zip(s, re.sub(r"[^A-Z0-9]", "", cand)))
        if best is None or changes < best[0]:
            best = (changes, cand)
    return best[1] if best else None


class PlateReader:
    def __init__(self, cfg: dict):
        from fast_plate_ocr import LicensePlateRecognizer
        from ultralytics import YOLO

        self.detector = YOLO(cfg.get("detector", "models/plate-v1s.pt"))
        self.det_conf = float(cfg.get("det_conf", 0.25))
        self.device = cfg.get("device")
        self.ocr = LicensePlateRecognizer(cfg.get("ocr_model", "cct-s-v2-global-model"),
                                          device=cfg.get("ocr_device", "auto"))
        self.two_row_mode = cfg.get("two_row_mode", "best")  # best | whole | split
        self.two_row_ratio = float(cfg.get("two_row_ratio", 0.45))
        self.min_width = int(cfg.get("min_plate_width", 40))
        self.min_conf = float(cfg.get("min_conf", 0.6))  # below this the reading is discarded

    def find(self, vehicle: np.ndarray) -> tuple[tuple[int, int, int, int], float] | None:
        res = self.detector(vehicle, conf=self.det_conf, device=self.device, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return None
        i = int(res.boxes.conf.argmax())
        x1, y1, x2, y2 = map(int, res.boxes.xyxy[i].tolist())
        return (x1, y1, x2, y2), float(res.boxes.conf[i])

    def _ocr(self, img: np.ndarray) -> tuple[str, float]:
        mode = self.ocr.config.image_color_mode
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY if mode == "grayscale" else cv2.COLOR_BGR2RGB)
        pred = self.ocr.run(img, return_confidence=True)[0]
        text = pred.plate
        probs = getattr(pred, "char_probs", None)
        conf = float(np.mean(np.asarray(probs)[: max(1, len(text))])) if probs is not None else 0.0
        return text, conf

    def read_plate(self, plate: np.ndarray) -> tuple[str, float, bool]:
        """Return (raw text, confidence, read_as_two_rows).
        Square plates are read both whole and split into rows; the reading that
        fits the Vietnamese layout (then the more confident one) wins."""
        h, w = plate.shape[:2]
        whole, cw = self._ocr(plate)
        if h / max(w, 1) <= self.two_row_ratio or self.two_row_mode == "whole":
            return whole, cw, False
        cut, pad = h // 2, max(1, h // 20)
        top, c1 = self._ocr(plate[: cut + pad])
        bot, c2 = self._ocr(plate[cut - pad:])
        split, cs = top + bot, (c1 + c2) / 2
        if self.two_row_mode == "split":
            return split, cs, True
        cands = [(normalize_vn(whole, True) is not None, cw, whole, False),
                 (normalize_vn(split, True) is not None, cs, split, True)]
        ok, conf, text, two = max(cands)
        return text, conf, two

    def read(self, vehicle: np.ndarray, motorbike: bool = False) -> PlateResult | None:
        if vehicle.size == 0:
            return None
        found = self.find(vehicle)
        if not found:
            return None
        (x1, y1, x2, y2), det_conf = found
        if x2 - x1 < self.min_width:
            return None
        # small margin so edge characters are not clipped
        mx, my = int(0.04 * (x2 - x1)), int(0.08 * (y2 - y1))
        H, W = vehicle.shape[:2]
        plate = vehicle[max(0, y1 - my):min(H, y2 + my), max(0, x1 - mx):min(W, x2 + mx)].copy()
        raw, conf, two = self.read_plate(plate)
        text = normalize_vn(raw, motorbike) if conf >= self.min_conf else None
        return PlateResult(text, raw, conf, det_conf, (x1, y1, x2, y2), two, plate)
