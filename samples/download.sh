#!/usr/bin/env bash
# Video mẫu: camera đường đô thị 640x360 (nguồn: github.com/kraten/vehicle-speed-check).
set -euo pipefail
cd "$(dirname "$0")"
curl -fsSL -o cars.mp4 https://raw.githubusercontent.com/kraten/vehicle-speed-check/master/cars.mp4
