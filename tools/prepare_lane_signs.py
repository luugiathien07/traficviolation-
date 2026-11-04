"""Build a small dataset of R.412 lane-sign panels (one picture per lane: car, motorbike, ...) for a classifier
that reads which vehicles each lane of a "vehicle permission lane" sign is for.

Step 1 (crops): cut every lane sign out of the VN sign dataset (classes "vehicle permission lane" and
"vehicle and speed permission lane") and, optionally, out of videos with the sign model; split each sign into
vertical panels at its white separator lines and write review grids with a number on every panel.
  python tools/prepare_lane_signs.py crops --out datasets/lane_panels --video samples/vo_van_kiet.mp4

Step 2 (write): after looking at the grids, write a labels file with one line per panel number,
"<n> <class>" with class one of car, moto, truck, bus, x (x = not a vehicle panel / unreadable, left out),
then copy the panels into an ImageFolder layout for `yolo classify`.
  python tools/prepare_lane_signs.py write --out datasets/lane_panels --labels review.txt
"""
import argparse
import glob
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

LANE_CLASSES = {24, 25}  # vehicle permission lane, vehicle and speed permission lane (models/vn_signs.pt)
PANEL_CLASSES = ("car", "moto", "truck", "bus")


def to_height(sign: np.ndarray, h: int = 120) -> np.ndarray:
    """Signs come at any size; split and classify them at one height."""
    return cv2.resize(sign, (max(40, sign.shape[1] * h // max(1, sign.shape[0])), h), interpolation=cv2.INTER_CUBIC)


def split_panels(sign: np.ndarray) -> list[tuple[int, int]]:
    """x ranges of the panels of a lane sign. Lanes are separated by thin white lines, solid or dashed: a
    column counts as separator where it is white with blue a few pixels left and right, over a good part of
    the height (pictograms are wider than the line, so their columns fail the blue-on-both-sides test)."""
    hsv = cv2.cvtColor(sign, cv2.COLOR_BGR2HSV)
    white = (hsv[..., 1] < 90) & (hsv[..., 2] > 130)
    blue = (hsv[..., 0] > 95) & (hsv[..., 0] < 130) & (hsv[..., 1] > 80)
    h, w = white.shape
    d = max(3, w // 40)
    thin = np.zeros_like(white)
    thin[:, d:w - d] = white[:, d:w - d] & blue[:, :w - 2 * d] & blue[:, 2 * d:]
    col = thin[int(0.1 * h):int(0.9 * h)].mean(0)
    col = np.convolve(col, np.ones(3) / 3, mode="same")
    cuts, run = [], []
    for x in range(int(0.12 * w), int(0.88 * w)):
        if col[x] > 0.22:
            run.append(x)
        elif run:
            cuts.append(sum(run) // len(run))
            run = []
    if run:
        cuts.append(sum(run) // len(run))
    edges = [0] + cuts + [w]
    return [(a, b) for a, b in zip(edges, edges[1:]) if b - a > 0.12 * w]


def sign_crops(a) -> list[tuple[str, np.ndarray]]:
    out = []
    root = Path(a.dataset)
    for lf in sorted(root.glob("labels/*/*.txt")):
        imgs = glob.glob(str(lf.parent.parent.parent / "images" / lf.parent.name / (lf.stem + ".*")))
        if not imgs:
            continue
        img = None
        for i, line in enumerate(open(lf)):
            p = line.split()
            if len(p) < 5 or int(p[0]) not in LANE_CLASSES:
                continue
            img = cv2.imread(imgs[0]) if img is None else img
            H, W = img.shape[:2]
            x, y, bw, bh = map(float, p[1:5])
            x1, y1, x2, y2 = int((x - bw / 2) * W), int((y - bh / 2) * H), int((x + bw / 2) * W), int((y + bh / 2) * H)
            if x2 - x1 >= a.min_width:
                out.append((f"ds_{lf.parent.name}_{lf.stem}_{i}", img[max(0, y1):y2, max(0, x1):x2]))
    for v in a.video:
        from ultralytics import YOLO
        model = YOLO(a.sign_model)
        cap = cv2.VideoCapture(v)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        for t in np.arange(0, cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps, a.every):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
            ok, f = cap.read()
            if not ok:
                break
            r = model.predict(f, imgsz=1600, conf=0.4, verbose=False)[0]
            for j, (b, c) in enumerate(zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist())):
                x1, y1, x2, y2 = map(int, b)
                if int(c) in LANE_CLASSES and x2 - x1 >= a.min_width:
                    out.append((f"vid_{Path(v).stem}_{int(t * 10):05d}_{j}", f[y1:y2, x1:x2]))
    return out


ICON_CLASSES = ("car", "moto", "truck", "bus", "arrow", "other")


def icons(sign: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Boxes (x1, y1, x2, y2) of the white pictograms of a lane sign (at height 120): connected white blobs that
    are neither specks, nor the sign border, nor the thin lane separators."""
    hsv = cv2.cvtColor(sign, cv2.COLOR_BGR2HSV)
    white = ((hsv[..., 1] < 90) & (hsv[..., 2] > 130)).astype(np.uint8)
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    h, w = white.shape
    n, _, st, _ = cv2.connectedComponentsWithStats(white, 8)
    out = []
    for i in range(1, n):
        x, y, bw, bh, area = st[i]
        if area < 0.004 * h * w or bw > 0.6 * w or bh > 0.7 * h:
            continue  # speck, or the white border of the sign
        if bh > 4 * bw:
            continue  # a lane separator line
        if x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1:
            continue  # cut by the crop edge
        out.append((int(x), int(y), int(x + bw), int(y + bh)))
    return out


def icon_crops(a):
    """Step 1b: one review tile per pictogram (in its sign, framed) for labelling car / moto / truck / ..."""
    out = Path(a.out)
    shutil.rmtree(out / "icons", ignore_errors=True)
    (out / "icons").mkdir(parents=True)
    (out / "review_icons").mkdir(parents=True, exist_ok=True)
    items, tiles = [], []
    for name, sign in sign_crops(a):
        sign = to_height(sign)
        for k, (x1, y1, x2, y2) in enumerate(icons(sign)):
            n = len(items)
            pad = 3
            crop = sign[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad]
            fn = f"{n:04d}_{name}_{k}.png"
            cv2.imwrite(str(out / "icons" / fn), crop)
            items.append({"n": n, "file": fn, "sign": name, "box": [x1, y1, x2, y2]})
            t = cv2.resize(crop, (72, 72), interpolation=cv2.INTER_CUBIC)
            ctx = sign.copy()
            cv2.rectangle(ctx, (x1, y1), (x2, y2), (0, 0, 255), 2)
            ctx = cv2.resize(ctx, (int(72 * ctx.shape[1] / ctx.shape[0]), 72))[:, :144]
            ctx = cv2.copyMakeBorder(ctx, 0, 0, 0, 144 - ctx.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))
            t = cv2.copyMakeBorder(np.hstack([t, ctx]), 16, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            cv2.putText(t, str(n), (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 200), 1)
            tiles.append(t)
    json.dump(items, open(out / "review_icons" / "icons.json", "w"), indent=0)
    per, cols = 40, 5
    for g in range(0, len(tiles), per):
        t = tiles[g:g + per] + [np.full_like(tiles[0], 255)] * (per - len(tiles[g:g + per]))
        cv2.imwrite(str(out / "review_icons" / f"grid_{g // per:02d}.jpg"),
                    np.vstack([np.hstack(t[r * cols:(r + 1) * cols]) for r in range(per // cols)]))
    print(f"{len(items)} pictograms; grids in {out / 'review_icons'}")


def write_icons(a):
    """Step 2b: copy the labelled pictograms into an ImageFolder layout (split by sign, not by pictogram)."""
    out = Path(a.out)
    items = {i["n"]: i for i in json.load(open(out / "review_icons" / "icons.json"))}
    marks = {int(p[0]): p[1] for p in (l.split() for l in open(a.labels)) if len(p) == 2 and p[0].isdigit()}
    signs = sorted({items[n]["sign"] for n, c in marks.items() if c in ICON_CLASSES})
    rng = np.random.default_rng(0)
    val_signs = set(rng.choice(signs, size=max(1, len(signs) // 5), replace=False))
    shutil.rmtree(out / "icons_cls", ignore_errors=True)
    counts = {}
    for n, c in marks.items():
        if c not in ICON_CLASSES:
            continue
        split = "val" if items[n]["sign"] in val_signs else "train"
        (out / "icons_cls" / split / c).mkdir(parents=True, exist_ok=True)
        shutil.copy(out / "icons" / items[n]["file"], out / "icons_cls" / split / c / items[n]["file"])
        counts[(split, c)] = counts.get((split, c), 0) + 1
    for k in sorted(counts):
        print(*k, counts[k])


def crops(a):
    out = Path(a.out)
    (out / "panels").mkdir(parents=True, exist_ok=True)
    (out / "review").mkdir(parents=True, exist_ok=True)
    panels, tiles = [], []
    for name, sign in sign_crops(a):
        sign = to_height(sign)
        parts = split_panels(sign)
        for k, (x1, x2) in enumerate(parts):
            n = len(panels)
            fn = f"{n:04d}_{name}_{k}.png"
            cv2.imwrite(str(out / "panels" / fn), sign[:, x1:x2])
            panels.append({"n": n, "file": fn, "sign": name, "index": k, "of": len(parts)})
            # review tile: the whole sign with this panel framed, so the panel is read in context
            t = sign.copy()
            cv2.rectangle(t, (x1, 0), (x2 - 1, t.shape[0] - 1), (0, 0, 255), 2)
            t = cv2.resize(t, (160, max(1, int(160 * t.shape[0] / t.shape[1]))))[:110]
            t = cv2.copyMakeBorder(t, 16, 110 - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            cv2.putText(t, f"{n} ({k + 1}/{len(parts)})", (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 200), 1)
            tiles.append(t)
    json.dump(panels, open(out / "review" / "panels.json", "w"), indent=0)
    per = 50
    for g in range(0, len(tiles), per):
        t = tiles[g:g + per] + [np.full_like(tiles[0], 255)] * (per - len(tiles[g:g + per]))
        cv2.imwrite(str(out / "review" / f"grid_{g // per:02d}.jpg"),
                    np.vstack([np.hstack(t[r * 10:(r + 1) * 10]) for r in range(5)]))
    signs = len({p["sign"] for p in panels})
    print(f"{signs} signs -> {len(panels)} panels; grids in {out / 'review'}")


def write(a):
    out = Path(a.out)
    panels = {p["n"]: p for p in json.load(open(out / "review" / "panels.json"))}
    marks = {}
    for line in open(a.labels):
        p = line.split()
        if len(p) == 2 and p[0].isdigit():
            marks[int(p[0])] = p[1]
    counts = {}
    rng = np.random.default_rng(0)
    signs = sorted({p["sign"] for p in panels.values()})
    val_signs = set(rng.choice(signs, size=max(1, len(signs) // 5), replace=False))  # split by sign, not panel
    for split in ("train", "val"):
        shutil.rmtree(out / "cls" / split, ignore_errors=True)
    for n, p in panels.items():
        c = marks.get(n)
        if c not in PANEL_CLASSES:
            continue
        split = "val" if p["sign"] in val_signs else "train"
        (out / "cls" / split / c).mkdir(parents=True, exist_ok=True)
        shutil.copy(out / "panels" / p["file"], out / "cls" / split / c / p["file"])
        counts[(split, c)] = counts.get((split, c), 0) + 1
    for k in sorted(counts):
        print(*k, counts[k])


ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest="cmd", required=True)
p1 = sub.add_parser("crops")
p1.add_argument("--out", required=True)
p1.add_argument("--dataset", default="datasets/vn_signs/train_data")
p1.add_argument("--video", action="append", default=[])
p1.add_argument("--every", type=float, default=1.0)
p1.add_argument("--min-width", type=int, default=30)
p1.add_argument("--sign-model", default="models/vn_signs.pt")
p1b = sub.add_parser("icons")
p1b.add_argument("--out", required=True)
p1b.add_argument("--dataset", default="datasets/vn_signs/train_data")
p1b.add_argument("--video", action="append", default=[])
p1b.add_argument("--every", type=float, default=1.0)
p1b.add_argument("--min-width", type=int, default=30)
p1b.add_argument("--sign-model", default="models/vn_signs.pt")
p2 = sub.add_parser("write")
p2.add_argument("--out", required=True)
p2.add_argument("--labels", required=True)
p2b = sub.add_parser("write-icons")
p2b.add_argument("--out", required=True)
p2b.add_argument("--labels", required=True)
args = ap.parse_args()
{"crops": crops, "icons": icon_crops, "write": write, "write-icons": write_icons}[args.cmd](args)
