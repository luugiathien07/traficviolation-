#!/usr/bin/env bash
# Chạy bản Java: ./java/tvd.sh --config configs/sample.yaml --serve
# Tự thêm thư viện CUDA 13 / cuDNN 9 (từ gói pip nvidia-* trong một venv) vào LD_LIBRARY_PATH nếu tìm thấy.
# Đặt TVD_CUDA_LIBS=/đường/dẫn/lib để chỉ định thư mục khác, TVD_DEVICE=cpu để ép chạy CPU.
here="$(cd "$(dirname "$0")" && pwd)"
libs="${TVD_CUDA_LIBS:-}"
for venv in "$HOME/venv" "$here/../.venv"; do
  for d in "$venv"/lib/python3*/site-packages/nvidia/cu13/lib "$venv"/lib/python3*/site-packages/nvidia/cudnn/lib; do
    [ -d "$d" ] && libs="${libs:+$libs:}$d"
  done
done
export LD_LIBRARY_PATH="${libs}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
JAVA="${JAVA_HOME:+$JAVA_HOME/bin/}java"
exec "$JAVA" -jar "$here/target/tvd.jar" "$@"
