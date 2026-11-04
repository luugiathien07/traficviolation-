"""Build a small Vietnamese helmet set from a video, in the HELMET dataset format (one box per motorcycle
including its riders; 0 helmet = everyone wears one, 1 no_helmet = at least one rider doesn't).

Step 1 (propose): sample frames, find motorcycles + people with COCO YOLO, merge each motorcycle with the
people riding it into one box, and write review grids with a number on every box.
  python tools/prepare_vn_helmet.py propose samples/raw_hanoi_slowtv.mp4 --out datasets/vn_helmet \
      --split train --start 0 --end 360 --every 2

Step 2 (write): after looking at the grids, write a labels file: one line per box number,
"<n> h" (helmet), "<n> n" (no helmet) or "<n> x" (unclear: the box is greyed out in the image so it is
neither a positive nor a background example). Boxes not listed default to helmet.
  python tools/prepare_vn_helmet.py write --out datasets/vn_helmet --split train --labels review_train.txt
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def propose(a):
    from ultralytics import YOLO

    out = Path(a.out)
    (out / "images" / a.split).mkdir(parents=True, exist_ok=True)
    (out / "review").mkdir(parents=True, exist_ok=True)
    model = YOLO(a.model)
    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    boxes, tiles = [], []
    for t in np.arange(a.start, a.end, a.every):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, frame = cap.read()
        if not ok:
            break
        r = model.predict(frame, imgsz=1280, conf=0.35, classes=[0, 3], verbose=False)[0]
        xyxy, cls = r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)
        motos, people = xyxy[cls == 3], xyxy[cls == 0]
        name = f"{Path(a.video).stem}_{int(t * 10):05d}"
        kept = []
        for m in motos:
            if m[2] - m[0] < a.min_width:
                continue
            x1, y1, x2, y2 = m
            for p in people:  # a rider: person whose lower half overlaps the motorcycle horizontally
                ov = min(x2, p[2]) - max(x1, p[0])
                if ov > 0.5 * (p[2] - p[0]) and p[3] > y1 and p[3] < y2 + 0.1 * (y2 - y1):
                    x1, y1, x2, y2 = min(x1, p[0]), min(y1, p[1]), max(x2, p[2]), max(y2, p[3])
            if y1 < m[1] - 5:  # someone rides it (parked bikes have no rider and no helmet question)
                kept.append([float(x1), float(y1), float(x2), float(y2)])
        if not kept:
            continue
        cv2.imwrite(str(out / "images" / a.split / f"{name}.jpg"), frame)
        for b in kept:
            n = len(boxes)
            boxes.append({"n": n, "image": f"{name}.jpg", "box": b})
            x1, y1, x2, y2 = map(int, b)
            head = frame[max(0, y1 - 10):y1 + max((y2 - y1) // 2, 20), max(0, x1 - 10):x2 + 10]
            tile = cv2.resize(head, (150, int(150 * head.shape[0] / max(head.shape[1], 1))))[:200]
            tile = cv2.copyMakeBorder(tile, 18, 200 - tile.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            cv2.putText(tile, str(n), (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1, cv2.LINE_AA)
            tiles.append(tile)
    json.dump(boxes, open(out / "review" / f"{a.split}_boxes.json", "w"))
    per = 40
    for g in range(0, len(tiles), per):
        t = tiles[g:g + per] + [np.full_like(tiles[0], 255)] * (per - len(tiles[g:g + per]))
        grid = np.vstack([np.hstack(t[r * 10:(r + 1) * 10]) for r in range(4)])
        cv2.imwrite(str(out / "review" / f"{a.split}_{g // per:02d}.jpg"), grid)
    print(f"{len(boxes)} boxes in {len(set(b['image'] for b in boxes))} frames; grids in {out / 'review'}")


def write(a):
    out = Path(a.out)
    boxes = json.load(open(out / "review" / f"{a.split}_boxes.json"))
    marks = {}
    for line in open(a.labels):
        p = line.split()
        if len(p) == 2:
            marks[int(p[0])] = p[1]
    (out / "labels" / a.split).mkdir(parents=True, exist_ok=True)
    by_img = {}
    for b in boxes:
        by_img.setdefault(b["image"], []).append(b)
    count = {"h": 0, "n": 0, "x": 0}
    for img_name, bs in by_img.items():
        img_path = out / "images" / a.split / img_name
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        lines, grey = [], False
        for b in bs:
            m = marks.get(b["n"], "h")
            count[m] += 1
            x1, y1, x2, y2 = b["box"]
            if m == "x":
                img[int(y1):int(y2), int(x1):int(x2)] = 114
                grey = True
                continue
            lines.append(f"{0 if m == 'h' else 1} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                         f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
        if grey:
            cv2.imwrite(str(img_path), img)
        (out / "labels" / a.split / img_name.replace(".jpg", ".txt")).write_text("\n".join(lines) + "\n")
    print(f"{a.split}: helmet {count['h']}, no_helmet {count['n']}, greyed {count['x']}")


ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest="cmd", required=True)
p1 = sub.add_parser("propose")
p1.add_argument("video")
p1.add_argument("--out", required=True)
p1.add_argument("--split", required=True)
p1.add_argument("--start", type=float, default=0)
p1.add_argument("--end", type=float, default=1e9)
p1.add_argument("--every", type=float, default=2)
p1.add_argument("--min-width", type=int, default=60)
p1.add_argument("--model", default="models/yolo11s.pt")
p2 = sub.add_parser("write")
p2.add_argument("--out", required=True)
p2.add_argument("--split", required=True)
p2.add_argument("--labels", required=True)
args = ap.parse_args()
propose(args) if args.cmd == "propose" else write(args)
