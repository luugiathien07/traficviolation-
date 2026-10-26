"""Run plate detection + OCR on a folder of images and save plate crops for review.

  python eval/run_plates.py eval/vn [--labels eval/vn/labels.json]
labels.json maps image file name -> list of expected plates, e.g. {"1.jpg": ["51F-972.21"]}.
"""
import argparse, json, re, sys
from pathlib import Path

import cv2
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tvd.plate import PlateReader, normalize_vn  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("folder")
ap.add_argument("--labels")
ap.add_argument("--ocr", default="cct-s-v2-global-model")
ap.add_argument("--mode", default="best")
args = ap.parse_args()

reader = PlateReader({"ocr_model": args.ocr, "two_row_mode": args.mode, "det_conf": 0.25})
out = Path(args.folder) / "crops"; out.mkdir(exist_ok=True)
labels = json.load(open(args.labels)) if args.labels else {}
# Compare characters only: the car/motorbike layout comes from the vehicle class in the
# pipeline, which still images don't have.
key = lambda s: re.sub(r"[^A-Z0-9]", "", (s or "").upper())
n = hit = chars = chars_ok = 0
for img_path in sorted(p for p in Path(args.folder).iterdir() if p.suffix.lower() in (".jpg", ".png", ".jpeg")):
    img = cv2.imread(str(img_path))
    res = reader.detector(img, conf=0.25, verbose=False)[0]
    got = []
    for j, b in enumerate(res.boxes.xyxy.tolist()):
        x1, y1, x2, y2 = map(int, b)
        mx, my = int(0.04 * (x2 - x1)), int(0.08 * (y2 - y1))
        plate = img[max(0, y1 - my):y2 + my, max(0, x1 - mx):x2 + mx]
        raw, conf, two = reader.read_plate(plate)
        txt = normalize_vn(raw) if conf >= reader.min_conf else None
        cv2.imwrite(str(out / f"{img_path.stem}_{j}.jpg"), plate)
        got.append(txt or "")
        print(f"{img_path.name:14s} #{j} raw={raw:12s} -> {txt!s:12s} conf={conf:.2f} {'2 dong' if two else '1 dong'} {x2-x1}x{y2-y1}")
    for exp in labels.get(img_path.name, []):
        n += 1
        best = max(got, key=lambda g: sum(a == b for a, b in zip(key(g), key(exp))), default="")
        hit += key(best) == key(exp)
        chars += len(key(exp)); chars_ok += sum(a == b for a, b in zip(key(best), key(exp)))
if n:
    print(f"\nDung ca bien: {hit}/{n} = {hit/n:.0%} | dung ky tu: {chars_ok}/{chars} = {chars_ok/chars:.0%}")
