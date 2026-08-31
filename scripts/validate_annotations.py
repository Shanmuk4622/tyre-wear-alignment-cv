"""Structural checks on annotations/ -- run BEFORE anything depends on the masks.

A mask/image size mismatch is silent and poisons a whole training run. A missing
tread mask makes that image's attention metric undefined. Catch both here.

Usage:
    conda activate cv_conda
    python scripts/validate_annotations.py \
        --annotations "D:/.../annotations" \
        --final       "D:/Dataset Download/Tire Dataset Prepared/FINAL"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from annotation_regions import CLASSES, CLASS_IDX, REGIONS, tyre_region, tread_region


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--final", required=True)
    args = ap.parse_args()

    ann, root = Path(args.annotations), Path(args.final)
    df = pd.read_csv(root / "manifests" / "clean_manifest.csv")
    df.columns = [c.lstrip("\ufeff") for c in df.columns]

    problems: list[str] = []
    stats = {c: 0 for c in CLASSES}
    ratios: list[float] = []
    n_found = 0

    for r in df.itertuples():
        mp = ann / "clean" / "masks" / f"{r.image_id}.png"
        if not mp.exists():
            problems.append(f"MISSING MASK        {r.image_id}")
            continue
        n_found += 1
        m = np.asarray(Image.open(mp))

        with Image.open(root / r.relative_path) as im:
            iw, ih = im.size
        if m.shape != (ih, iw):
            problems.append(f"SIZE MISMATCH       {r.image_id}  mask {m.shape} vs image {(ih, iw)}")

        # region accessors, not raw indices -- see annotation_regions.py
        tyre = tyre_region(m)
        tread = tread_region(m)
        mark = REGIONS["marking"](m)
        dmg = REGIONS["damage"](m)
        for c, mask in zip(CLASSES, (tyre, tread, mark, dmg)):
            stats[c] += int(mask.any())

        if not tyre.any():
            problems.append(f"NO TYRE MASK        {r.image_id}")
        if not tread.any():
            problems.append(f"NO TREAD MASK       {r.image_id}")

        # The taxonomy is nested by construction: tread, marking and damage all
        # live on the tyre. A stray pixel outside it means a mis-click.
        for name, sub in (("tread", tread), ("marking", mark), ("damage", dmg)):
            if sub.any():
                outside = int((sub & ~tyre).sum())
                if outside > 0.02 * sub.sum():
                    problems.append(f"{name.upper()} OUTSIDE TYRE  {r.image_id}  "
                                    f"{outside} px ({outside/sub.sum():.0%})")

        if tyre.any() and tread.any():
            frac = tread.sum() / tyre.sum()
            ratios.append(frac)
            # NOT an error on this dataset. The camera faces the tread crown
            # head-on, so the shoulders curve out of frame and tread genuinely
            # is ~99% of the visible tyre. Verified by eye on the extremes.
            # Recorded as an observation so the XAI stage words its claim
            # correctly (TER = tyre vs background here, not tread vs shoulder).
            if frac < 0.15:
                problems.append(f"TREAD VERY SMALL    {r.image_id}  {frac:.0%} of the tyre")
        if tyre.any() and tyre.sum() < 0.05 * m.size:
            problems.append(f"TYRE VERY SMALL     {r.image_id}  {tyre.sum()/m.size:.1%} of frame")

    print(f"clean images in manifest : {len(df)}")
    print(f"masks found              : {n_found}")
    print(f"\nclass coverage:")
    for c in CLASSES:
        print(f"  {c:<9} {stats[c]:>4} / {n_found}"
              + ("   (required on every image)" if c in ("tyre", "tread") else "   (only when visible)"))

    if ratios:
        a = np.array(ratios)
        print(f"\ntread / tyre area ratio:  median {np.median(a):.3f}   "
              f"min {a.min():.3f}   max {a.max():.3f}")
        n_full = int((a > 0.999).sum())
        print(f"  {n_full}/{len(a)} images have no visible shoulder at all")
        if np.median(a) > 0.95:
            print("  -> head-on viewpoint: tread IS essentially the whole visible tyre.")
            print("     Expected for this dataset. TER will therefore measure")
            print("     'tyre vs background', not 'tread vs shoulder' (14_XAI_PROTOCOL).")

    print(f"\nproblems: {len(problems)}")
    for p in problems[:60]:
        print("  " + p)
    if len(problems) > 60:
        print(f"  ... and {len(problems)-60} more")

    out = ann / "audit" / "validation_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "n_manifest": len(df), "n_masks": n_found, "coverage": stats,
        "tread_tyre_ratio": {"median": float(np.median(ratios)), "min": float(np.min(ratios)),
                             "max": float(np.max(ratios)),
                             "n_no_shoulder": int((np.array(ratios) > 0.999).sum())} if ratios else None,
        "n_problems": len(problems), "problems": problems}, indent=2))
    print(f"\nwritten: {out}")
    print("\nVALIDATION PASSED" if not problems and n_found == len(df)
          else "\nVALIDATION FAILED -- fix the above before propagating")


if __name__ == "__main__":
    main()
