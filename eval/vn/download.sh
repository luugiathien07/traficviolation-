#!/usr/bin/env bash
# Tải 8 ảnh biển số Việt Nam dùng để đánh giá OCR (nguồn: github.com/trungdinh22/License-Plate-Recognition).
set -euo pipefail
cd "$(dirname "$0")"
for f in 1.jpg 101.jpg 117.jpg 119.jpg 3.jpg 4.jpg 557.png bien_so.jpg; do
  curl -fsSL -o "$f" "https://raw.githubusercontent.com/trungdinh22/License-Plate-Recognition/HEAD/test_image/$f"
done
echo "Đã tải xong. Chạy: python eval/run_plates.py eval/vn --labels eval/vn/labels.json"
