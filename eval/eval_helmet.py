"""Compare helmet models on labelled test sets: mAP50 per class (Ultralytics val) and, the number that matters
for the no_helmet rule, how many boxes each model calls no_helmet on riders who wear one (false alarms).

  python eval/eval_helmet.py models/helmet_vn.pt runs/.../best.pt
"""
import sys
from pathlib import Path

from ultralytics import YOLO

SETS = {
    "HELMET test (Myanmar, camera cao)": "datasets/helmet/yolo/data.yaml",
    "VN Ha Noi (nhin ngang)": "datasets/vn_helmet/data.yaml",
    "VN dashcam TP.HCM (nhin tu sau)": "datasets/vn_helmet_rear/data.yaml",
}


def false_alarms(model, images: Path) -> tuple[int, int, int]:
    """(predicted no_helmet on a box labelled helmet, labelled helmet boxes, labelled no_helmet boxes)."""
    fa = n_helmet = n_no = 0
    for img in sorted(images.glob("*.jpg")):
        lab = images.parent.parent / "labels" / images.name / (img.stem + ".txt")
        gt = [l.split() for l in open(lab) if l.strip()] if lab.exists() else []
        n_helmet += sum(g[0] == "0" for g in gt)
        n_no += sum(g[0] == "1" for g in gt)
        r = model.predict(str(img), imgsz=1280, conf=0.35, verbose=False)[0]
        h, w = r.orig_shape
        for b, c in zip(r.boxes.xyxy.tolist(), r.boxes.cls.int().tolist()):
            if c != 1:
                continue
            for g in gt:  # matched to a helmet box by IoU >= 0.5
                cx, cy, bw, bh = map(float, g[1:5])
                gx1, gy1, gx2, gy2 = (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h
                iw = max(0, min(b[2], gx2) - max(b[0], gx1))
                ih = max(0, min(b[3], gy2) - max(b[1], gy1))
                inter = iw * ih
                union = (b[2] - b[0]) * (b[3] - b[1]) + (gx2 - gx1) * (gy2 - gy1) - inter
                if g[0] == "0" and inter / max(union, 1e-6) >= 0.5:
                    fa += 1
                    break
    return fa, n_helmet, n_no


for path in sys.argv[1:]:
    m = YOLO(path)
    print(f"\n=== {path}")
    for name, data in SETS.items():
        if not Path(data).exists():
            continue
        import yaml
        d = yaml.safe_load(open(data))
        res = m.val(data=data, split="test", imgsz=1280, batch=4, plots=False, verbose=False)
        maps = dict(zip([m.names[i] for i in res.box.ap_class_index], res.box.ap50))
        test = Path(d["test"] if Path(d["test"]).is_absolute() else Path(d.get("path", ".")) / d["test"])
        fa, nh, nn = false_alarms(m, test)
        print(f"{name}: mAP50 helmet {maps.get('helmet', float('nan')):.3f} no_helmet {maps.get('no_helmet', float('nan')):.3f}"
              f" | bao nham 'khong doi mu' {fa}/{nh} nguoi co doi mu ({100 * fa / max(nh, 1):.0f}%), so nguoi khong doi mu {nn}")
