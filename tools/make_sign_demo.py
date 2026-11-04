"""Paste real Vietnamese traffic-sign photos (crops from the VN sign dataset) onto a fixed-camera video, so the
sign model finds them like real signs on a pole and the sign-gated rules (wrong_way, no_entry, no_parking)
switch on. The vehicles are real; the signs are staged, so the violations are illustrations only.

  python tools/make_sign_demo.py samples/raw_hanoi_slowtv.mp4 samples/hanoi_signs.mp4 --seconds 300 \
      --sign samples/signs/p102_cam_nguoc_chieu.png 575 600 90 --sign samples/signs/p103a_cam_oto.png 575 700 90
"""
import argparse
import subprocess

import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("dst")
ap.add_argument("--seconds", type=float, default=300)
ap.add_argument("--sign", nargs=4, action="append", default=[], metavar=("PNG", "X", "Y", "SIZE"),
                help="round sign image and its top-left corner and width on the frame")
ap.add_argument("--rect-sign", nargs=5, action="append", default=[], metavar=("PNG", "X", "Y", "W", "H"),
                help="rectangular sign (e.g. R.412 lane sign) and its top-left corner, width and height")
args = ap.parse_args()

signs = []
for path, x, y, size in args.sign:
    x, y, size = int(x), int(y), int(size)
    img = cv2.resize(cv2.imread(path), (size, size), interpolation=cv2.INTER_AREA)
    # round sign: keep the disc, feather the edge so no rectangle of the original background shows
    mask = np.zeros((size, size), np.float32)
    cv2.circle(mask, (size // 2, size // 2), size // 2 - 2, 1.0, -1, cv2.LINE_AA)
    mask = cv2.GaussianBlur(mask, (0, 0), 1.2)[..., None]
    signs.append((img.astype(np.float32), mask, x, y, size, size))
for path, x, y, sw, sh in args.rect_sign:
    x, y, sw, sh = int(x), int(y), int(sw), int(sh)
    img = cv2.resize(cv2.imread(path), (sw, sh), interpolation=cv2.INTER_AREA)
    signs.append((img.astype(np.float32), np.ones((sh, sw, 1), np.float32), x, y, sw, sh))

cap = cv2.VideoCapture(args.src)
fps = cap.get(cv2.CAP_PROP_FPS) or 25
w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                       "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                       "-pix_fmt", "yuv420p", args.dst], stdin=subprocess.PIPE)
i = 0
for i in range(int(args.seconds * fps)):
    ok, frame = cap.read()
    if not ok:
        break
    for img, mask, x, y, sw, sh in signs:
        roi = frame[y:y + sh, x:x + sw].astype(np.float32)
        frame[y:y + sh, x:x + sw] = (img * mask + roi * (1 - mask)).astype(np.uint8)
    ff.stdin.write(frame.tobytes())
ff.stdin.close()
ff.wait()
print(f"{i + 1} frames -> {args.dst}")
