"""Export YOLO .pt models to ONNX for the Java version (java/).

  python tools/export_onnx.py models/yolo11n.pt models/yolo11s.pt models/plate-v1s.pt models/helmet.pt

Writes models/onnx/<name>.onnx with a dynamic input size, so Java can letterbox each frame to the
smallest multiple of 32 (e.g. 960x544 for 16:9) exactly like Ultralytics' predict() does.
"""
import shutil
import sys
from pathlib import Path

from ultralytics import YOLO

for pt in sys.argv[1:]:
    pt = Path(pt)
    out = YOLO(str(pt)).export(format="onnx", dynamic=True, simplify=True, opset=17)
    dst = pt.parent / "onnx" / (pt.stem + ".onnx")
    dst.parent.mkdir(exist_ok=True)
    shutil.move(out, dst)
    print(dst)
