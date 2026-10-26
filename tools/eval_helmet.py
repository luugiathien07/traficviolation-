"""Evaluate the helmet model per motorcycle on a split of the converted dataset.

For every labelled motorcycle, take the predicted box with the highest IoU (>= 0.5):
  - caught      no_helmet labelled and predicted no_helmet  -> violation found
  - missed      no_helmet labelled, predicted helmet or not found
  - false flag  helmet labelled, predicted no_helmet        -> would fine an innocent rider

  python tools/eval_helmet.py --model runs/helmet/helmet_v1/weights/best.pt --data datasets/helmet/yolo --split test
"""
import argparse
from pathlib import Path

import numpy as np
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--split", default="test")
ap.add_argument("--conf", type=float, default=0.35)
ap.add_argument("--imgsz", type=int, default=1280)
args = ap.parse_args()


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


model = YOLO(args.model)
imgs = sorted((Path(args.data) / "images" / args.split).glob("*.jpg"))
cm = np.zeros((2, 3), int)  # rows: true helmet/no_helmet; cols: pred helmet/no_helmet/not found
for img in imgs:  # one at a time: a list would be inferred as a single batch
    r = model.predict(str(img), conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
    H, W = r.orig_shape
    lab = Path(args.data) / "labels" / args.split / (Path(r.path).stem + ".txt")
    preds = list(zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.int().cpu().tolist()))
    for line in lab.read_text().splitlines():
        c, cx, cy, w, h = map(float, line.split())
        g = ((cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H)
        best = max(preds, key=lambda p: iou(g, p[0]), default=None)
        col = best[1] if best is not None and iou(g, best[0]) >= 0.5 else 2
        cm[int(c), col] += 1

caught, missed = cm[1, 1], cm[1, 0] + cm[1, 2]
false_flag, ok_helmet = cm[0, 1], cm[0, 0]
print(f"Ảnh: {len(imgs)}  |  xe có nhãn: {cm.sum()}")
print(f"Xe có người KHÔNG đội mũ: {cm[1].sum():5d}  -> bắt được {caught} ({caught / max(cm[1].sum(), 1):.0%}), bỏ sót {missed}")
print(f"Xe mọi người ĐỘI mũ:      {cm[0].sum():5d}  -> phạt nhầm {false_flag} ({false_flag / max(cm[0].sum(), 1):.0%}), không phát hiện {cm[0, 2]}")
print(f"Độ chính xác khi báo lỗi (precision): {caught / max(caught + false_flag, 1):.0%}")
