"""Train the Vietnamese traffic-sign detector.

Data: "Vietnamese Traffic Signs Detection and Recognition" (Dat Nguyen, Kaggle
jaydenguyenx/vietnamese-traffic-signs-detection-and-recognition; licence not stated, use for demos only):
1170 dashcam frames (1622x626), 29 sign groups, already in YOLO format.

  curl -L -o datasets/vn_signs/archive.zip \
      https://www.kaggle.com/api/v1/datasets/download/jaydenguyenx/vietnamese-traffic-signs-detection-and-recognition
  (unzip into datasets/vn_signs/, write datasets/vn_signs/data.yaml with the local path and the names list)
  python tools/train_signs.py --data datasets/vn_signs/data.yaml
  cp runs/detect/runs/signs/vn_signs/weights/best.pt models/vn_signs.pt

Signs are small (median ~26 px wide), so training runs at imgsz 1600.
"""
import argparse

from ultralytics import YOLO

from gpu_wait import wait_for_free_gpu

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--model", default="models/yolo11s.pt")
ap.add_argument("--name", default="vn_signs")
ap.add_argument("--epochs", type=int, default=120)
ap.add_argument("--imgsz", type=int, default=1600)
ap.add_argument("--batch", type=int, default=4)
args = ap.parse_args()
wait_for_free_gpu()

YOLO(args.model).train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                       patience=25, project="runs/signs", name=args.name, exist_ok=True,
                       fliplr=0.0,  # mirrored signs mean something else (no left turn <-> no right turn)
                       workers=4, verbose=False)
