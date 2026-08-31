"""Turn annotation exports into the annotations/ tree.

Primary format is labelme (one .json per image). CVAT 'Segmentation mask 1.1'
zips are also accepted with --format cvat.

Usage (Windows, one line):
    conda activate cv_conda
    python scripts\\import_annotations.py --exports "D:\\...\\all_annotations" --map "D:\\...\\annotation_upload\\filename_map.csv" --out "D:\\Dataset Download\\Tire Dataset Prepared\\annotations" --format labelme

--exports may contain one sub-folder per annotator; all are merged.

Writes:
    clean/masks/<image_id>.png     indexed PNG, 0=bg 1=tyre 2=tread 3=marking 4=damage
    clean/polygons/<image_id>.json the source labelme file, kept editable
    clean/boxes/<image_id>.txt     YOLO format, derived from the tyre mask
    coco/instances_clean.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

import sys

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
# Region accessors, NOT raw class indices. The mask is a single indexed layer,
# so `m == 1` means "tyre minus whatever was painted on top of it" -- and on a
# head-on tyre photo, tread covers essentially all of it, leaving `m == 1`
# empty. Deriving the box from that produced 160 empty box files.
from annotation_regions import CLASSES, CLASS_IDX, REGIONS, tyre_region
CVAT_COLOURS = {"tyre": (255, 0, 0), "tread": (0, 255, 0),
                "marking": (0, 0, 255), "damage": (255, 255, 0)}


def mask_from_labelme(js_path: Path) -> tuple[np.ndarray, dict]:
    d = json.loads(js_path.read_text())
    w, h = int(d["imageWidth"]), int(d["imageHeight"])
    img = Image.new("L", (w, h), 0)
    dr = ImageDraw.Draw(img)
    # Draw in class order so tread sits on top of tyre, and marking/damage on
    # top of both -- the taxonomy is nested by construction.
    for name in CLASSES:
        for s in d.get("shapes", []):
            if s.get("label", "").strip().lower() != name:
                continue
            pts = [tuple(map(float, p)) for p in s.get("points", [])]
            st = s.get("shape_type", "polygon")
            if st == "rectangle" and len(pts) == 2:
                (x0, y0), (x1, y1) = pts
                dr.rectangle([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)],
                             fill=CLASS_IDX[name])
            elif st == "circle" and len(pts) == 2:
                (cx, cy), (ex, ey) = pts
                r = ((ex - cx) ** 2 + (ey - cy) ** 2) ** 0.5
                dr.ellipse([cx - r, cy - r, cx + r, cy + r], fill=CLASS_IDX[name])
            elif len(pts) >= 3:
                dr.polygon(pts, fill=CLASS_IDX[name])
    return np.asarray(img), d


def mask_from_cvat_png(png: Path) -> tuple[np.ndarray, dict]:
    rgb = np.asarray(Image.open(png).convert("RGB"))
    out = np.zeros(rgb.shape[:2], np.uint8)
    for name, col in CVAT_COLOURS.items():
        out[(np.abs(rgb.astype(int) - np.array(col)).sum(2) < 40)] = CLASS_IDX[name]
    return out, {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exports", required=True)
    ap.add_argument("--map", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--format", default="labelme", choices=["labelme", "cvat"])
    args = ap.parse_args()

    out = Path(args.out)
    for sub in ("clean/masks", "clean/boxes", "clean/polygons", "coco", "audit"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    fmap = pd.read_csv(args.map)
    # labelme keeps the image filename, so match on stem; also allow a direct
    # image_id match in case someone renamed things.
    by_stem = {Path(r.filename).stem: r for r in fmap.itertuples()}
    by_id = {r.image_id: r for r in fmap.itertuples()}

    src = Path(args.exports)
    if args.format == "cvat":
        work = out / "_tmp_extract"
        work.mkdir(exist_ok=True)
        for z in sorted(src.rglob("*.zip")):
            with zipfile.ZipFile(z) as f:
                f.extractall(work / z.stem)
            print(f"  extracted {z.name}")
        files = sorted(work.rglob("*.png"))
    else:
        files = sorted(src.rglob("*.json"))
        files = [f for f in files if f.name not in ("filename_map.json",)]
    print(f"format={args.format}   found {len(files)} annotation file(s)")

    coco = {"images": [], "annotations": [],
            "categories": [{"id": i + 1, "name": c} for i, c in enumerate(CLASSES)]}
    n_ok = n_skip = n_dupe = 0
    ann_id = 1
    seen: set[str] = set()

    for p in files:
        stem = p.stem
        rec = by_stem.get(stem) or by_id.get(stem)
        if rec is None:
            n_skip += 1
            continue
        if rec.image_id in seen:
            n_dupe += 1                 # calibration images annotated by several people
            continue
        seen.add(rec.image_id)

        m, raw = (mask_from_labelme(p) if args.format == "labelme" else mask_from_cvat_png(p))
        Image.fromarray(m.astype(np.uint8), mode="L").save(
            out / "clean/masks" / f"{rec.image_id}.png")
        if args.format == "labelme":
            shutil.copy2(p, out / "clean/polygons" / f"{rec.image_id}.json")

        h, w = m.shape
        ys, xs = np.where(tyre_region(m))          # region, not raw index
        lines = []
        if len(xs):
            x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
            lines.append(f"0 {((x0+x1)/2)/w:.6f} {((y0+y1)/2)/h:.6f} "
                         f"{(x1-x0)/w:.6f} {(y1-y0)/h:.6f}")
        (out / "clean/boxes" / f"{rec.image_id}.txt").write_text("\n".join(lines))

        coco["images"].append({"id": len(coco["images"]) + 1,
                               "file_name": f"{rec.image_id}.jpg",
                               "width": int(w), "height": int(h)})
        for name in CLASSES:
            idx = CLASS_IDX[name]
            ys, xs = np.where(REGIONS[name](m))     # region, not raw index
            if not len(xs):
                continue
            x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
            coco["annotations"].append({"id": ann_id, "image_id": len(coco["images"]),
                                        "category_id": idx, "iscrowd": 0,
                                        "bbox": [x0, y0, x1 - x0, y1 - y0],
                                        "area": int(REGIONS[name](m).sum())})
            ann_id += 1
        n_ok += 1

    (out / "coco" / "instances_clean.json").write_text(json.dumps(coco))
    print(f"\nimported {n_ok}   duplicates skipped {n_dupe}   unmatched filename {n_skip}")

    present = {c: 0 for c in CLASSES}
    for f in (out / "clean/masks").glob("*.png"):
        m = np.asarray(Image.open(f))
        for c in CLASSES:
            present[c] += int(REGIONS[c](m).any())
    print("\nclass coverage:")
    for c, n in present.items():
        req = c in ("tyre", "tread")
        print(f"  {c:<9} {n:>4} / {n_ok}" + ("   (required on every image)" if req
                                             else "   (only when visible)"))
    if present["tyre"] < n_ok or present["tread"] < n_ok:
        print("\n  WARNING: some images lack a tyre or tread mask.")
        print("  Run validate_annotations.py to list exactly which.")
    print("\nNext: python scripts\\validate_annotations.py ...")


if __name__ == "__main__":
    main()
