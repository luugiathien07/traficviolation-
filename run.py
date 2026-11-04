"""Entry point.

  python run.py --config configs/sample.yaml --serve          # dashboard at http://localhost:8000
  python run.py --config configs/sample.yaml --out out.mp4    # headless, write annotated video
  python run.py --demos configs/demos.yaml --serve            # one button per demo + video upload
"""
from __future__ import annotations

import argparse
import logging

import yaml

from tvd.pipeline import Pipeline
from tvd.storage import Store


def main() -> None:
    ap = argparse.ArgumentParser(description="Phát hiện vi phạm giao thông từ video")
    ap.add_argument("--config", help="file cấu hình YAML (bắt buộc, trừ khi dùng --demos)")
    ap.add_argument("--demos", help="danh sách demo (mỗi demo một nút trên dashboard) và bộ luật cho video tải lên")
    ap.add_argument("--source", help="ghi đè nguồn video (file, rtsp://..., hoặc số webcam)")
    ap.add_argument("--data", default="data", help="thư mục lưu CSDL và ảnh bằng chứng")
    ap.add_argument("--serve", action="store_true", help="chạy dashboard web")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--out", help="ghi video đã chú thích ra file .mp4")
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--fast", action="store_true", help="không chờ theo FPS gốc khi đọc file")
    ap.add_argument("--preview", help="vẽ vạch/vùng cấu hình lên một khung hình rồi lưu ra file ảnh")
    ap.add_argument("--preview-frame", type=int, default=75, help="khung hình dùng cho --preview")
    args = ap.parse_args()

    if not args.config and not (args.demos and args.serve):
        ap.error("cần --config (hoặc --demos cùng --serve)")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    store = Store(args.data)
    if args.demos:
        import uvicorn
        from tvd.runner import Runner, load_cfg
        from tvd.server import create_app
        runner = Runner(store, load_cfg(args.demos), fast=args.fast)
        if not runner.demos or not runner.start_demo(runner.demos[0]["key"]):
            raise SystemExit(f"Không chạy được demo đầu tiên trong {args.demos}")
        uvicorn.run(create_app(runner, store), host=args.host, port=args.port, log_level="warning")
        return
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.source:
        cfg["source"] = args.source
    if args.fast or not args.serve:
        cfg["realtime"] = False

    pipe = Pipeline(cfg, store)

    if args.preview:
        import cv2
        cap = cv2.VideoCapture(int(cfg["source"]) if str(cfg["source"]).isdigit() else cfg["source"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.preview_frame)
        ok, frame = cap.read()
        if not ok:
            raise SystemExit(f"Không đọc được khung hình từ {cfg['source']}")
        img = pipe.annotate(pipe.backlight.process(frame), 0.0, pipe.light.update(frame, 0.0))
        h, w = img.shape[:2]
        for x in range(0, w, 50):  # coordinate grid to help place lines
            cv2.line(img, (x, 0), (x, h), (90, 90, 90), 1)
            cv2.putText(img, str(x), (x + 2, h - 4), cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 255, 255), 1)
        for y in range(0, h, 50):
            cv2.line(img, (0, y), (w, y), (90, 90, 90), 1)
            cv2.putText(img, str(y), (2, y - 2), cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 255, 255), 1)
        cv2.imwrite(args.preview, img)
        print("Đã lưu", args.preview)
        return

    if not args.serve:
        pipe.run(max_frames=args.max_frames, writer_path=args.out)
        print("Tổng xe:", pipe.stats["vehicles_seen"], "| Vi phạm:", store.counts())
        return

    import uvicorn
    from tvd.runner import Runner
    from tvd.server import create_app

    runner = Runner(store)
    runner.start("main", cfg)
    uvicorn.run(create_app(runner, store), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
