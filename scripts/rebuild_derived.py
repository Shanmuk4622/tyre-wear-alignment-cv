"""Regenerate boxes and COCO from the authoritative mask PNGs.

The masks are the source of truth. Boxes and COCO are derived, so if the
derivation logic changes they can be rebuilt without touching the annotations
or re-parsing labelme.

Usage (Windows, one line):
    conda activate cv_conda
    python scripts\\rebuild_derived.py --annotations "D:\\Dataset Download\\Tire Dataset Prepared\\annotations"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from annotation_regions import CLASSES, CLASS_IDX, REGIONS, tyre_region


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", required=True)
    args = ap.parse_args()

    out = Path(args.annotations)
    masks = sorted((out / "clean" / "masks").glob("*.png"))
    (out / "clean" / "boxes").mkdir(parents=True, exist_ok=True)
    (out / "coco").mkdir(parents=True, exist_ok=True)
    print(f"masks: {len(masks)}")

    coco = {"images": [], "annotations": [],
            "categories": [{"id": i + 1, "name": c} for i, c in enumerate(CLASSES)]}
    ann_id = 1
    n_box = n_empty = 0
    counts = {c: 0 for c in CLASSES}

    for mp in masks:
        m = np.asarray(Image.open(mp))
        h, w = m.shape
        iid = mp.stem

        ys, xs = np.where(tyre_region(m))
        if len(xs):
            x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
            (out / "clean" / "boxes" / f"{iid}.txt").write_text(
                f"0 {((x0+x1)/2)/w:.6f} {((y0+y1)/2)/h:.6f} "
                f"{(x1-x0)/w:.6f} {(y1-y0)/h:.6f}")
            n_box += 1
        else:
            (out / "clean" / "boxes" / f"{iid}.txt").write_text("")
            n_empty += 1

        coco["images"].append({"id": len(coco["images"]) + 1, "file_name": f"{iid}.jpg",
                               "width": int(w), "height": int(h)})
        for name in CLASSES:
            reg = REGIONS[name](m)
            ys, xs = np.where(reg)
            if not len(xs):
                continue
            counts[name] += 1
            x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
            coco["annotations"].append({
                "id": ann_id, "image_id": len(coco["images"]), "category_id": CLASS_IDX[name],
                "iscrowd": 0, "bbox": [x0, y0, x1 - x0, y1 - y0], "area": int(reg.sum())})
            ann_id += 1

    (out / "coco" / "instances_clean.json").write_text(json.dumps(coco))
    print(f"\nboxes written : {n_box}   empty: {n_empty}")
    print(f"coco          : {len(coco['images'])} images, {len(coco['annotations'])} annotations")
    print("\nper-class annotation counts:")
    for c, n in counts.items():
        print(f"  {c:<9} {n:>4}")
    if n_empty:
        print(f"\n  WARNING: {n_empty} image(s) produced no box -- no tyre region at all.")


if __name__ == "__main__":
    main()
