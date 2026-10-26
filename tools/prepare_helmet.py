"""Convert the HELMET dataset (Lin et al. 2020, CC BY 4.0, https://osf.io/4pwj8/) to YOLO format.

Each motorcycle box is labelled with who wears a helmet, e.g. DHelmetP1NoHelmet
(driver yes, passenger 1 no). We collapse that to the question the rule needs:
  0 helmet     everyone on the motorcycle wears a helmet
  1 no_helmet  at least one rider does not

  python tools/prepare_helmet.py --raw datasets/helmet/raw --ann datasets/helmet/ann/annotation \
      --split datasets/helmet/data_split.csv --out datasets/helmet/yolo --every 4

Images are symlinked, not copied. Clips keep the official training/validation/test split,
so frames of one clip never end up in two splits.
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import Counter, defaultdict
from pathlib import Path

W, H = 1920, 1080
SPLITS = {"training": "train", "validation": "val", "test": "test"}


def to_class(label: str) -> int:
    return 1 if "NoHelmet" in label else 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="folder(s) containing part_N/<clip>/<frame>.jpg", nargs="+")
    ap.add_argument("--ann", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=4, help="keep every Nth frame (clips are 10 fps)")
    ap.add_argument("--balance", action="store_true",
                    help="repeat training images that contain helmeted riders until both classes are about even")
    args = ap.parse_args()

    split_of = {r["video_id"]: SPLITS[r["Set"]] for r in csv.DictReader(open(args.split))}
    clips = {}
    for raw in args.raw:
        for part in Path(raw).iterdir():
            for clip in part.iterdir():
                if clip.is_dir():
                    clips[clip.name] = clip
    out = Path(args.out)
    stats = defaultdict(Counter)
    for name, clip_dir in sorted(clips.items()):
        split = split_of.get(name)
        ann_file = Path(args.ann) / f"{name}.csv"
        if split is None or not ann_file.exists():
            continue
        boxes = defaultdict(list)
        for r in csv.DictReader(open(ann_file)):
            x, y, w, h = (float(r[k]) for k in ("x", "y", "w", "h"))
            # clip to the frame; annotations can run past the edge
            x1, y1, x2, y2 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            boxes[int(r["frame_id"])].append((to_class(r["label"]), (x1 + x2) / 2 / W, (y1 + y2) / 2 / H,
                                              (x2 - x1) / W, (y2 - y1) / H))
        for d in ("images", "labels"):
            (out / d / split).mkdir(parents=True, exist_ok=True)
        for frame in range(1, 101, args.every):
            img = clip_dir / f"{frame:02d}.jpg"
            if not img.exists():
                img = clip_dir / f"{frame}.jpg"
            if not img.exists():
                continue
            stem = f"{name}_{frame:03d}"
            link = out / "images" / split / f"{stem}.jpg"
            if not link.exists():
                os.symlink(img.resolve(), link)
            with open(out / "labels" / split / f"{stem}.txt", "w") as f:
                for c, cx, cy, bw, bh in boxes.get(frame, []):
                    f.write(f"{c} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
                    stats[split][c] += 1
            stats[split]["images"] += 1
        stats[split]["clips"] += 1

    if args.balance:
        balance_train(out, stats)

    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        "names:\n  0: helmet\n  1: no_helmet\n")
    for split, c in stats.items():
        print(f"{split:5s} clips={c['clips']:3d} images={c['images']:5d} helmet={c[0]:6d} no_helmet={c[1]:6d}")


def balance_train(out: Path, stats) -> None:
    """Oversample training images that contain `helmet` boxes (extra symlinks + label copies).
    Only the training split is touched; val/test keep their real class mix."""
    lab_dir, img_dir = out / "labels" / "train", out / "images" / "train"
    helmet_imgs = []
    for lab in lab_dir.glob("*.txt"):
        if "_rep" in lab.stem:
            continue
        cls = [int(l.split()[0]) for l in lab.read_text().splitlines()]
        if cls.count(0) > cls.count(1):
            helmet_imgs.append((lab, cls.count(0), cls.count(1)))
    n_help, n_no = stats["train"][0], stats["train"][1]
    gain = sum(h - n for _, h, n in helmet_imgs)  # net helmet boxes added per repetition
    reps = 0 if gain <= 0 else min(8, max(0, round((n_no - n_help) / gain)))
    for lab, h, n in helmet_imgs:
        src = img_dir / f"{lab.stem}.jpg"
        for k in range(reps):
            stem = f"{lab.stem}_rep{k}"
            if not (img_dir / f"{stem}.jpg").exists():
                os.symlink(os.readlink(src), img_dir / f"{stem}.jpg")
            (lab_dir / f"{stem}.txt").write_text(lab.read_text())
            stats["train"][0] += h
            stats["train"][1] += n
            stats["train"]["images"] += 1
    print(f"balance: {len(helmet_imgs)} helmet-majority images repeated {reps}x")


if __name__ == "__main__":
    main()
