"""Make a red-light demo video from real footage that has no traffic light: draws a stop line on the road
and a traffic light that cycles green -> yellow -> red (and with --lane-y a dashed lane line + "XE MAY"). The real vehicles don't see the drawn light, so
those crossing the line while it is red become red-light violations. The system reads the lamp colour
from the frame (light.mode: roi), like with a real camera. For demos only: the violations are staged.

  python tools/make_violation_demo.py samples/raw_hanoi_slowtv.mp4 samples/hanoi_redlight.mp4 --seconds 300
"""
import argparse
import subprocess

import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("dst")
ap.add_argument("--seconds", type=float, default=300)
ap.add_argument("--green", type=float, default=12)
ap.add_argument("--yellow", type=float, default=3)
ap.add_argument("--red", type=float, default=10)
ap.add_argument("--stop-line", type=int, nargs=4, default=[1010, 860, 990, 1080], metavar=("X1", "Y1", "X2", "Y2"))
ap.add_argument("--lane-y", type=int, default=0, help="paint a lane line at this y (0 = none); below it: XE MAY lane")
ap.add_argument("--no-light", action="store_true", help="only the lane (no traffic light, no stop line)")
ap.add_argument("--light-box", type=int, nargs=4, default=[1700, 120, 1790, 380], metavar=("X1", "Y1", "X2", "Y2"))
args = ap.parse_args()

cap = cv2.VideoCapture(args.src)
fps = cap.get(cv2.CAP_PROP_FPS) or 25
w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cycle = args.green + args.yellow + args.red
X1, Y1, X2, Y2 = args.light_box
lamps = {"red": (0, 0, 255), "yellow": (0, 210, 255), "green": (60, 220, 0)}
cy = {k: Y1 + (Y2 - Y1) * (2 * i + 1) // 6 for i, k in enumerate(lamps)}
r = min(X2 - X1, (Y2 - Y1) // 3) * 2 // 5


def state(t):
    tt = t % cycle
    return "green" if tt < args.green else "yellow" if tt < args.green + args.yellow else "red"


ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                       "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                       "-pix_fmt", "yuv420p", args.dst], stdin=subprocess.PIPE)
for i in range(int(args.seconds * fps)):
    ok, frame = cap.read()
    if not ok:
        break
    # stop line painted on the road, blended a little so the asphalt texture shows through
    paint = frame.copy()
    sx1, sy1, sx2, sy2 = args.stop_line
    if not args.no_light:
        cv2.line(paint, (sx1, sy1), (sx2, sy2), (235, 235, 235), 14, cv2.LINE_AA)
    if args.lane_y:
        for x in range(0, w, 120):  # dashed lane line
            cv2.line(paint, (x, args.lane_y), (x + 70, args.lane_y), (235, 235, 235), 6, cv2.LINE_AA)
        cv2.putText(paint, "XE MAY", (300, args.lane_y + 75), cv2.FONT_HERSHEY_DUPLEX, 1.6, (235, 235, 235), 4, cv2.LINE_AA)
    frame = cv2.addWeighted(paint, 0.8, frame, 0.2, 0)
    if args.no_light:
        ff.stdin.write(frame.tobytes())
        continue
    # pole + housing + lamps; only the active lamp is lit
    cv2.rectangle(frame, ((X1 + X2) // 2 - 6, Y2), ((X1 + X2) // 2 + 6, h), (60, 60, 60), -1)
    cv2.rectangle(frame, (X1, Y1), (X2, Y2), (25, 25, 25), -1)
    on = state(i / fps)
    for k, col in lamps.items():
        c = col if k == on else tuple(v // 6 for v in col)
        cv2.circle(frame, ((X1 + X2) // 2, cy[k]), r, c, -1, cv2.LINE_AA)
    ff.stdin.write(frame.tobytes())
ff.stdin.close()
ff.wait()
print(f"{i + 1} frames -> {args.dst}")
