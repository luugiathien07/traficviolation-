#!/usr/bin/env bash
# Tải weights: YOLO11n (nhận diện xe) và YOLO11s tinh chỉnh cho biển số (AGPL-3.0).
set -euo pipefail
cd "$(dirname "$0")"
curl -fL -o yolo11n.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt
curl -fL -o plate-v1s.pt https://huggingface.co/morsetechlab/yolov11-license-plate-detection/resolve/main/license-plate-finetune-v1s.pt
