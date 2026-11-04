"""Measure how much backlight compensation helps detection.

Uses a clean video and its synthetic backlit copy (tools/make_backlit.py keeps frames aligned):
detections on the clean frame are the reference, then we count how many of them are found again
on the backlit frame, raw and after each compensation method (recall), and how many extra boxes
appear (precision).

  python eval/eval_backlight.py samples/raw_hanoi_slowtv.mp4 samples/hanoi_backlit.mp4
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tvd.enhance import BacklightCompensator  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("clean")
ap.add_argument("backlit")
ap.add_argument("--start", type=float, default=70, help="first second to sample (effect fully on)")
ap.add_argument("--n", type=int, default=40)
ap.add_argument("--model", default="models/yolo11s.pt")
ap.add_argument("--save", default="", help="folder to write side-by-side samples")
args = ap.parse_args()

CLASSES = [0, 2, 3, 5, 7]  # person, car, motorcycle, bus, truck
model = YOLO(args.model)


def detect(img):
    r = model.predict(img, imgsz=1280, conf=0.3, classes=CLASSES, verbose=False)[0]
    return r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)


def iou(a, b):
    x1, y1 = np.maximum(a[:, None, 0], b[None, :, 0]), np.maximum(a[:, None, 1], b[None, :, 1])
    x2, y2 = np.minimum(a[:, None, 2], b[None, :, 2]), np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def match(ref, got):
    """Greedy same-class matching at IoU 0.5 -> (true positives, n_ref, n_got)."""
    (rb, rc), (gb, gc) = ref, got
    if len(rb) == 0 or len(gb) == 0:
        return 0, len(rb), len(gb)
    m = iou(rb, gb) * (rc[:, None] == gc[None, :])
    tp, used = 0, set()
    for i in np.argsort(-m.max(1)):
        j = int(np.argmax(np.where([k in used for k in range(len(gb))], -1, m[i])))
        if m[i, j] >= 0.5 and j not in used:
            used.add(j)
            tp += 1
    return tp, len(rb), len(gb)


def methods():
    """name -> function(frame) -> frame. Add new compensation variants here."""
    out = {"khong_bu": lambda f: f}
    old = BacklightCompensator({"mode": "on", "strength": 1.0})
    out["bu"] = old.process
    out["retinex"] = BacklightCompensator({"mode": "on", "strength": 0.4, "method": "retinex"}).process
    return out


cap_c, cap_b = cv2.VideoCapture(args.clean), cv2.VideoCapture(args.backlit)
fps = cap_b.get(cv2.CAP_PROP_FPS)
end = cap_b.get(cv2.CAP_PROP_FRAME_COUNT) / fps
times = np.linspace(args.start, end - 1, args.n)
ms = methods()
tot = {k: [0, 0, 0] for k in ms}
if args.save:
    Path(args.save).mkdir(parents=True, exist_ok=True)
for k, t in enumerate(times):
    idx = int(t * fps)
    cap_c.set(cv2.CAP_PROP_POS_FRAMES, idx)
    cap_b.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok1, clean = cap_c.read()
    ok2, back = cap_b.read()
    if not (ok1 and ok2):
        break
    ref = detect(clean)
    shown = [clean]
    for name, fn in ms.items():
        img = fn(back)
        tp, nr, ng = match(ref, detect(img))
        tot[name][0] += tp
        tot[name][1] += nr
        tot[name][2] += ng
        shown.append(img)
    if args.save and k % 8 == 0:
        cv2.imwrite(f"{args.save}/t{int(t):03d}.jpg", cv2.resize(np.hstack(shown), None, fx=0.25, fy=0.25))

print(f"{'cách':10s} {'recall':>7s} {'precision':>9s}   (đáp án: {tot['khong_bu'][1]} vật trên hình gốc)")
for name, (tp, nr, ng) in tot.items():
    print(f"{name:10s} {tp / max(nr, 1):7.1%} {tp / max(ng, 1):9.1%}")
