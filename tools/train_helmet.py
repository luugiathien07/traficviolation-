"""Train the helmet / no_helmet detector on the converted HELMET dataset.

  python tools/train_helmet.py --data datasets/helmet/yolo/data.yaml --name helmet_v1
"""
import argparse

from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--base", default="models/yolo11s.pt")
ap.add_argument("--name", default="helmet")
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--imgsz", type=int, default=1280)
ap.add_argument("--batch", type=int, default=8)
args = ap.parse_args()

YOLO(args.base).train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                      patience=15, workers=4, project="runs/helmet", name=args.name, exist_ok=True,
                      fliplr=0.5, mosaic=1.0, close_mosaic=10, plots=True, verbose=False)
