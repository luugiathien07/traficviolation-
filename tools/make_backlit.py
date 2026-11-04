"""Make a synthetic backlit (ngược sáng) copy of a daytime video, to try backlight compensation.

Real backlight: the camera faces the sun, exposure is set by the bright sky, so vehicles and riders
turn into dark silhouettes and a veil of glare washes out the frame. This imitates that: underexposes
the scene, adds a clipped sun glare from the top and a light haze. The effect ramps in after
--clean seconds over --ramp seconds, so the demo shows the auto mode switching on.

  python tools/make_backlit.py samples/raw_hanoi_slowtv.mp4 samples/hanoi_backlit.mp4 --seconds 300
"""
import argparse
import subprocess

import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("dst")
ap.add_argument("--seconds", type=float, default=300)
ap.add_argument("--clean", type=float, default=45, help="seconds left untouched at the start")
ap.add_argument("--ramp", type=float, default=15, help="seconds to fade the effect in")
ap.add_argument("--sun-x", type=float, default=0.62, help="sun position, fraction of width")
args = ap.parse_args()

cap = cv2.VideoCapture(args.src)
fps = cap.get(cv2.CAP_PROP_FPS) or 25
w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Glare map: a bright sun just above the frame plus a falloff that covers the upper part.
yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
sx, sy = args.sun_x * w, -0.08 * h
d = np.sqrt((xx - sx) ** 2 + (yy - sy) ** 2) / h
glare = (1.6 * np.exp(-(d / 0.45) ** 2) + 0.35 * np.exp(-(d / 1.1) ** 2))[..., None]
warm = np.array([0.80, 0.95, 1.0], np.float32)  # BGR: glare is slightly yellow

ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                       "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                       "-pix_fmt", "yuv420p", args.dst], stdin=subprocess.PIPE)
n_max = int(args.seconds * fps)
for i in range(n_max):
    ok, frame = cap.read()
    if not ok:
        break
    t = i / fps
    a = float(np.clip((t - args.clean) / args.ramp, 0.0, 1.0))
    if a > 0:
        f = frame.astype(np.float32) / 255.0
        under = 0.30 * f ** 1.8                      # subjects underexposed into silhouettes
        out = under + glare * warm + 0.03            # sun glare + haze veil
        f = (1 - a) * f + a * out
        frame = (np.clip(f, 0, 1) * 255).astype(np.uint8)
    ff.stdin.write(frame.tobytes())
ff.stdin.close()
ff.wait()
print(f"{i + 1} frames -> {args.dst}")
