"""Train the helmet / no_helmet detector on the converted HELMET dataset.

  python tools/train_helmet.py --data datasets/helmet/yolo/data.yaml --name helmet_v1
"""
import argparse

from ultralytics import YOLO

from gpu_wait import wait_for_free_gpu

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--base", default="models/yolo11s.pt")
ap.add_argument("--name", default="helmet")
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--imgsz", type=int, default=1280)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--optimizer", default="auto", help="auto ignores --lr0; pass SGD or AdamW to use it")
ap.add_argument("--lr0", type=float, default=0.01)
ap.add_argument("--warmup", type=float, default=3.0, help="warmup epochs (0 when fine-tuning a trained model)")
ap.add_argument("--patience", type=int, default=15)
ap.add_argument("--no-gpu-wait", action="store_true", help="train even if another process (e.g. the demo) holds the GPU")
args = ap.parse_args()
if not args.no_gpu_wait:
    wait_for_free_gpu()

YOLO(args.base).train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                      patience=args.patience, workers=4, optimizer=args.optimizer, lr0=args.lr0, warmup_epochs=args.warmup, project="runs/helmet", name=args.name, exist_ok=True,
                      fliplr=0.5, mosaic=1.0, close_mosaic=10, plots=True, verbose=False)
