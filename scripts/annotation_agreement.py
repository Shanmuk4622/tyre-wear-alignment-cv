"""Annotation consistency check.

TWO MODES
---------
--mode self  (default, for a one-person job)
    Compares your FIRST pass at the self-check images against a SECOND pass
    made after finishing everything. This is the solo equivalent of
    inter-annotator agreement, and it measures the thing that actually
    threatens a one-person job: your tread-boundary judgement drifting over
    3-4 hours as your eye gets better.

--mode team
    Pairwise agreement between several annotators on a shared calibration set.

Either way, if tread IoU is below 0.90 the boundary rule is not being applied
consistently, and every attention metric in the study inherits it.

Usage (Windows, one line):
    conda activate cv_conda
    python scripts\annotation_agreement.py --dir "D:\...\calibration" --format labelme

Expects one sub-folder per annotator:
    calibration\shanmukesh\*.json    nehru\*.json    manu\*.json    harish\*.json

--format labelme (default) reads labelme JSON directly -- no conversion needed.
--format mask reads pre-rendered indexed PNG masks instead.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# Canonical region definitions. NEVER compare raw class indices: the mask is a
# single indexed layer, so `m == 1` means "tyre minus whatever was painted on
# top", and two annotators with identical tyre outlines but a 4 px tread
# difference score 0.93 instead of 1.00.
sys.path.insert(0, str(Path(__file__).parent))
from annotation_regions import CLASSES, CLASS_IDX, REGIONS

TARGETS = {"tyre": 0.95, "tread": 0.90, "marking": 0.80, "damage": 0.70}


def render_labelme(js_path: Path) -> np.ndarray:
    """labelme JSON -> indexed mask, same class order as import_annotations."""
    d = json.loads(js_path.read_text())
    img = Image.new("L", (int(d["imageWidth"]), int(d["imageHeight"])), 0)
    dr = ImageDraw.Draw(img)
    for name in CLASSES:
        for s in d.get("shapes", []):
            if s.get("label", "").strip().lower() != name:
                continue
            pts = [tuple(map(float, p)) for p in s.get("points", [])]
            if len(pts) >= 3:
                dr.polygon(pts, fill=CLASS_IDX[name])
    return np.asarray(img)


def load_person(folder: Path, fmt: str) -> dict[str, np.ndarray]:
    if fmt == "labelme":
        return {f.stem: render_labelme(f) for f in folder.glob("*.json")}
    return {f.stem: np.asarray(Image.open(f)) for f in folder.glob("*.png")}


def iou(a: np.ndarray, b: np.ndarray) -> float | None:
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else None       # both empty -> undefined, not 1.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="self", choices=["self", "team"])
    ap.add_argument("--pass1", help="mode=self: first-pass annotation folder")
    ap.add_argument("--pass2", help="mode=self: second-pass annotation folder")
    ap.add_argument("--dir", help="mode=team: folder containing one sub-folder per annotator")
    ap.add_argument("--out", default=None)
    ap.add_argument("--format", default="labelme", choices=["labelme", "mask"])
    args = ap.parse_args()

    if args.mode == "self":
        if not (args.pass1 and args.pass2):
            raise SystemExit("--mode self needs --pass1 and --pass2")
        people = ["pass1", "pass2"]
        loaded = {"pass1": load_person(Path(args.pass1), args.format),
                  "pass2": load_person(Path(args.pass2), args.format)}
        root = Path(args.pass2).parent
        print("mode: SELF-CONSISTENCY (same person, two passes)")
    else:
        if not args.dir:
            raise SystemExit("--mode team needs --dir")
        root = Path(args.dir)
        people = sorted(p.name for p in root.iterdir() if p.is_dir())
        if len(people) < 2:
            raise SystemExit(f"need >= 2 annotator folders in {root}, found {people}")
        loaded = {p: load_person(root / p, args.format) for p in people}
        print(f"mode: TEAM  annotators: {people}")
    for p in people:
        print(f"  {p:<14} {len(loaded[p]):>4} annotated images")
    common = None
    for p in people:
        ids = set(loaded[p])
        common = ids if common is None else (common & ids)
    common = sorted(common)
    label = "in both passes" if args.mode == "self" else "annotated by everyone"
    print(f"images {label}: {len(common)}\n")
    if not common:
        raise SystemExit("no overlapping image_ids -- the two folders share no filenames. "
                         "In self mode, pass 2 must re-annotate the SAME images as batch_SELFCHECK.")

    scores = {c: [] for c in CLASSES}
    damage_agree = []
    for img in common:
        masks = {p: loaded[p][img] for p in people}
        shapes = {m.shape for m in masks.values()}
        if len(shapes) > 1:
            print(f"  skipping {img}: annotators disagree on image size {shapes}")
            continue
        for a, b in itertools.combinations(people, 2):
            for c in CLASSES:
                region = REGIONS[c]
                v = iou(region(masks[a]), region(masks[b]))
                if v is not None:
                    scores[c].append(v)
            damage_agree.append(int(REGIONS["damage"](masks[a]).any()
                                    == REGIONS["damage"](masks[b]).any()))

    print(f"{'class':<10}{'mean IoU':>10}{'min':>8}{'n pairs':>9}{'target':>9}{'':>8}")
    print("-" * 54)
    report, all_ok = {}, True
    for c in CLASSES:
        v = scores[c]
        if not v:
            print(f"{c:<10}{'n/a':>10}{'':>8}{0:>9}{TARGETS[c]:>9.2f}   (never drawn)")
            report[c] = None
            continue
        m = float(np.mean(v))
        ok = m >= TARGETS[c]
        all_ok = all_ok and ok
        report[c] = {"mean_iou": round(m, 4), "min": round(float(np.min(v)), 4), "n": len(v)}
        print(f"{c:<10}{m:>10.3f}{np.min(v):>8.3f}{len(v):>9}{TARGETS[c]:>9.2f}"
              + ("   PASS" if ok else "   FAIL"))

    kappa = float(np.mean(damage_agree)) if damage_agree else None
    if kappa is not None:
        print(f"\ndamage presence agreement: {kappa:.3f}")
        report["damage_presence_agreement"] = round(kappa, 4)

    print()
    if args.mode == "self":
        if all_ok:
            print("SELF-CONSISTENCY ACCEPTABLE.")
            print("Your tread-boundary judgement stayed stable across the whole job.")
            print("Report these numbers in the paper -- most projects never measure this.")
        else:
            print("SELF-CONSISTENCY TOO LOW -- your boundary judgement drifted.")
            print("Your later work is probably the better work (your eye improved).")
            print("Re-read guide Part C Step 2, then RE-ANNOTATE THE FIRST ~40 IMAGES")
            print("of batch_ALL -- those were done before you had the rule internalised.")
    else:
        if all_ok:
            print("AGREEMENT ACCEPTABLE -- start the main batch.")
        else:
            print("AGREEMENT TOO LOW -- do NOT start the main batch.")
            print("Get on a call, put the same image on screen, and agree where the")
            print("tread boundary is (guide Part C, Step 2). Then redo the 50.")

    dest = Path(args.out) if args.out else root / "agreement_report.json"
    dest.write_text(json.dumps({"mode": args.mode, "annotators": people,
                                "n_images": len(common), "per_class": report,
                                "passed": all_ok}, indent=2))
    print(f"\nwritten: {dest}")


if __name__ == "__main__":
    main()
