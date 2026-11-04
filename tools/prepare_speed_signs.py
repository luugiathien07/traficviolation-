"""Build a dataset for reading the number on speed-limit signs (P.127): the sign model has one "speed limit"
class for every limit, so a small classifier tells 40 from 50 from 60...

Step 1 (crops): cut every "speed limit" box out of the VN sign dataset (train/val/test kept apart) and write
review grids with a number on every sign.
  python tools/prepare_speed_signs.py crops --out datasets/speed_signs

Step 2 (write): after looking at the grids, write a labels file "<n> <value>" (e.g. "12 50"; "x" = unreadable,
left out) and copy the crops into an ImageFolder layout for `yolo classify`.
  python tools/prepare_speed_signs.py write --out datasets/speed_signs --labels review.txt
"""
import argparse
import glob
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

SPEED_CLASS = 14  # "speed limit" in models/vn_signs.pt


def crops(a):
    out = Path(a.out)
    shutil.rmtree(out / "crops", ignore_errors=True)
    (out / "crops").mkdir(parents=True)
    (out / "review").mkdir(parents=True, exist_ok=True)
    items, tiles = [], []
    for split in ("train", "val", "test"):
        for lf in sorted(Path(a.dataset, "labels", split).glob("*.txt")):
            img = None
            for i, line in enumerate(open(lf)):
                p = line.split()
                if len(p) < 5 or int(p[0]) != SPEED_CLASS:
                    continue
                if img is None:
                    img = cv2.imread(glob.glob(str(Path(a.dataset, "images", split, lf.stem + ".*")))[0])
                H, W = img.shape[:2]
                x, y, bw, bh = map(float, p[1:5])
                x1, y1 = max(0, int((x - bw / 2) * W)), max(0, int((y - bh / 2) * H))
                x2, y2 = int((x + bw / 2) * W), int((y + bh / 2) * H)
                crop = img[y1:y2, x1:x2]
                if min(crop.shape[:2]) < a.min_size:
                    continue
                n = len(items)
                fn = f"{n:04d}_{split}_{lf.stem}_{i}.png"
                cv2.imwrite(str(out / "crops" / fn), crop)
                items.append({"n": n, "file": fn, "split": split, "w": x2 - x1})
                t = cv2.resize(crop, (96, 96), interpolation=cv2.INTER_CUBIC)
                t = cv2.copyMakeBorder(t, 16, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
                cv2.putText(t, str(n), (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 200), 1)
                tiles.append(t)
    json.dump(items, open(out / "review" / "items.json", "w"), indent=0)
    per, cols = 60, 12
    for g in range(0, len(tiles), per):
        t = tiles[g:g + per] + [np.full_like(tiles[0], 255)] * (per - len(tiles[g:g + per]))
        cv2.imwrite(str(out / "review" / f"grid_{g // per:02d}.jpg"),
                    np.vstack([np.hstack(t[r * cols:(r + 1) * cols]) for r in range(per // cols)]))
    print(f"{len(items)} speed-limit signs; grids in {out / 'review'}")


def write(a):
    out = Path(a.out)
    items = {i["n"]: i for i in json.load(open(out / "review" / "items.json"))}
    counts = {}
    shutil.rmtree(out / "cls", ignore_errors=True)
    for line in open(a.labels):
        p = line.split()
        if len(p) != 2 or not p[0].isdigit() or not p[1].isdigit():
            continue
        it = items[int(p[0])]
        split = "val" if it["split"] != "train" else "train"  # dataset val + test -> our val
        (out / "cls" / split / p[1]).mkdir(parents=True, exist_ok=True)
        shutil.copy(out / "crops" / it["file"], out / "cls" / split / p[1] / it["file"])
        counts[(split, p[1])] = counts.get((split, p[1]), 0) + 1
    for k in sorted(counts, key=lambda k: (k[0], int(k[1]))):
        print(*k, counts[k])


ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest="cmd", required=True)
p1 = sub.add_parser("crops")
p1.add_argument("--out", required=True)
p1.add_argument("--dataset", default="datasets/vn_signs/train_data")
p1.add_argument("--min-size", type=int, default=18)
p2 = sub.add_parser("write")
p2.add_argument("--out", required=True)
p2.add_argument("--labels", required=True)
args = ap.parse_args()
crops(args) if args.cmd == "crops" else write(args)
