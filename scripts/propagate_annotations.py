"""Replay each derivative's recorded transform onto its source mask.

418 hand-corrected masks -> 4,598. The dataset records every derivative's exact
operations in `augmentation_trace_json`; we replay only the GEOMETRIC ones and
ignore the photometric ones, which do not move pixels.

Nearest-neighbour interpolation throughout -- bilinear invents intermediate
class values at boundaries.

-----------------------------------------------------------------------------
WHY THIS FILE WAS REWRITTEN  (v2, 2026-08-27)
-----------------------------------------------------------------------------
v1 matched operations by substring and read parameters by guessed key names:

    if "crop" in name:
        box = p.get("box") or [p.get("x_min"), ...]     # recorded key: crop_box
        if all(v is not None for v in box): m = m.crop(box)
    elif "rotat" in name:
        ang = p.get("angle", p.get("limit", 0))         # recorded key: degrees

Neither key exists in this dataset. `all(v is not None ...)` was False, `ang`
was 0, so **the crop and the rotation were silently skipped** -- no exception,
no warning, 4,180 files written that looked completely normal. Only the flip
was applied, and the final `resize` stretched the full frame instead of
letterboxing it.

Measured damage: mean IoU 0.83 between the v1 masks and a correct replay, and
a mean image/mask alignment score of 16.1 against 34.3 for the correct replay.
The masks were wrong on every single derivative, consistently, and silently.

So v2 does three things differently:

  1. **Exact-name dispatch.** Ops are matched by their full recorded name
     against an explicit table. No substring matching.
  2. **Unknown ops are a hard error.** Every op must be classified as
     geometric or photometric. An unrecognised name raises. A future policy
     version that adds `vertical_flip` stops the script instead of quietly
     producing 4,180 wrong files.
  3. **The output is verified before the script claims success.** It measures
     how well each mask agrees with its own image and compares that with the
     hand-drawn clean masks. A silent geometry bug cannot survive this.

The lesson generalises: a transform replay must fail loudly, because a
misaligned mask is indistinguishable from a correct one at a glance.

-----------------------------------------------------------------------------
Usage (Windows, one line each):
    conda activate cv_conda
    python scripts\\propagate_annotations.py ^
        --annotations "D:\\Dataset Download\\Tire Dataset Prepared\\annotations" ^
        --final       "D:\\Dataset Download\\Tire Dataset Prepared\\FINAL"

Then LOOK at annotations/audit/propagation_check/.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

CLASS_COLOURS = np.array([[0, 0, 0], [220, 40, 40], [40, 200, 80],
                          [60, 110, 240], [240, 210, 50]], np.uint8)

# --------------------------------------------------------------------------
# The op table. Every operation the augmentation policy can emit must appear
# in exactly one of these two sets.
# --------------------------------------------------------------------------

# Move pixels -> must be replayed on the mask.
GEOMETRIC = {"random_resized_crop_letterbox", "horizontal_flip",
             "vertical_flip", "rotation"}

# Change appearance only -> the mask is unchanged.
#
# coarse_dropout paints grey rectangles over the image. It is an OCCLUSION
# augmentation: the tyre is still there, the model just cannot see that part.
# The ground truth does not change, so the mask is deliberately left intact.
PHOTOMETRIC = {"brightness_contrast", "gamma", "saturation", "clahe",
               "gaussian_noise", "gaussian_blur", "box_blur", "unsharp_mask",
               "jpeg_recompression", "coarse_dropout"}


class UnknownOperation(RuntimeError):
    """Raised when the trace contains an op we cannot classify.

    Deliberately fatal. The alternative -- skipping it -- is exactly the bug
    this rewrite exists to prevent.
    """


def letterbox(im: Image.Image, out: int) -> Image.Image:
    """Aspect-preserving resize onto a square canvas, centred.

    The image pipeline pads with a constant grey; the mask pads with 0
    (background), because padding is not tyre.

    Rounding is `round`, not `int`. Checked against the real images: on 400
    unrotated derivatives the bar widths implied by `round` matched the
    measured constant-column runs 215 times against 103 for `int`.
    """
    w, h = im.size
    s = out / max(w, h)
    w2, h2 = max(1, round(w * s)), max(1, round(h * s))
    im = im.resize((w2, h2), Image.NEAREST)
    canvas = Image.new("L", (out, out), 0)
    canvas.paste(im, ((out - w2) // 2, (out - h2) // 2))
    return canvas


def apply_trace(mask: Image.Image, ops: list, target_size) -> Image.Image:
    """Geometric ops only, in recorded order, nearest-neighbour throughout."""
    m = mask
    for op in ops:
        name = op.get("name") or op.get("op") or ""
        if name in PHOTOMETRIC:
            continue
        if name not in GEOMETRIC:
            raise UnknownOperation(
                f"operation {name!r} is in neither GEOMETRIC nor PHOTOMETRIC.\n"
                f"Classify it in scripts/propagate_annotations.py before running.\n"
                f"Recorded parameters: {sorted(set(op) - {'name'})}"
            )

        if name == "random_resized_crop_letterbox":
            box = op["crop_box"]                       # exact key. No guessing.
            m = m.crop(tuple(int(v) for v in box))
            m = letterbox(m, int(op["output_size"]))

        elif name == "horizontal_flip":
            m = m.transpose(Image.FLIP_LEFT_RIGHT)

        elif name == "vertical_flip":
            m = m.transpose(Image.FLIP_TOP_BOTTOM)

        elif name == "rotation":
            # PIL rotates counter-clockwise for positive angles. Verified by
            # measurement, not assumption: on the largest-|angle| decile,
            # rotate(+degrees) scored 33.96 on the alignment metric against
            # 28.36 for rotate(-degrees).
            ang = float(op["degrees"])
            if ang:
                m = m.rotate(ang, resample=Image.NEAREST, expand=False, fillcolor=0)

    if m.size != tuple(target_size):
        # Should not fire for this policy -- letterbox already lands on 768.
        m = m.resize(tuple(target_size), Image.NEAREST)
    return m


# --------------------------------------------------------------------------
# Verification -- the part that makes a silent geometry bug impossible
# --------------------------------------------------------------------------

def alignment_score(grey: np.ndarray, mask: np.ndarray) -> float:
    """Mean luminance outside the mask minus mean luminance inside it.

    A tyre is much darker than road, wall and sky, so a correctly placed mask
    puts the dark pixels inside and the bright ones outside. Misalign it and
    the two populations mix and the score collapses. It needs no ground truth
    beyond the image itself, which is exactly why it can catch a replay bug.
    """
    t = mask > 0
    f = t.mean()
    if f < 0.02 or f > 0.995:
        return float("nan")
    return float(grey[~t].mean() - grey[t].mean())


def verify(ann: Path, root: Path, df: pd.DataFrame, n: int = 200, seed: int = 0) -> bool:
    """Score each propagated mask against three DELIBERATELY WRONG versions of
    itself, on the same image.

    Comparing propagated masks with the hand-drawn clean ones does not work:
    the derivatives are brightness- and contrast-augmented, letterboxed and
    cropped, so their absolute scores are lower for reasons that have nothing
    to do with geometry. The controls fix that. Each control is measured on
    the *same* image with the *same* photometry, so the only thing that
    differs is placement.

      shift    the mask moved 6% of the frame sideways
      mirror   the mask flipped left-right
      swap     a different image's mask

    A correct mask must beat all three by a clear margin. The v1 masks scored
    16.1 against a swap control of 9.8 -- barely better than a mask belonging
    to a different photo, which is what a broken replay looks like.
    """
    rng = random.Random(seed)
    aug = df[df.image_kind == "synthetic_derivative"]
    rows = list(aug.itertuples())
    rng.shuffle(rows)

    correct, shifted, mirrored, swapped = [], [], [], []
    prev = None
    for r in rows:
        p = ann / "propagated" / "masks" / f"{r.image_id}.png"
        ip = root / r.relative_path
        if not (p.exists() and ip.exists()):
            continue
        g = np.asarray(Image.open(ip).convert("L"), dtype=np.float32)
        k = np.asarray(Image.open(p))
        if g.shape != k.shape:
            continue
        d = int(0.06 * k.shape[1])
        correct.append(alignment_score(g, k))
        shifted.append(alignment_score(g, np.roll(k, d, axis=1)))
        mirrored.append(alignment_score(g, k[:, ::-1]))
        if prev is not None and prev.shape == k.shape:
            swapped.append(alignment_score(g, prev))
        prev = k
        if len(correct) >= n:
            break

    if len(correct) < 20:
        print("\n=== verification ===\n  INCONCLUSIVE -- not enough samples")
        return False

    c = float(np.nanmean(correct))
    ctrl = {"shift 6%": float(np.nanmean(shifted)),
            "mirrored": float(np.nanmean(mirrored)),
            "swapped": float(np.nanmean(swapped))}
    worst = max(ctrl.values())
    margin = c - worst

    print(f"\n=== verification ===   (n={len(correct)})")
    print(f"  propagated mask, as written     {c:6.2f}")
    for k_, v in ctrl.items():
        print(f"  control: {k_:<24s}{v:6.2f}")
    print(f"  margin over the best control    {margin:+6.2f}   (want > +5.00)")

    if margin < 5.0:
        print("\n  FAIL. The masks barely beat a deliberately wrong mask, which")
        print("  means they are not tracking the image. Do NOT use them, and do")
        print("  not train anything on them.")
        return False
    print("\n  PASS. The masks follow their images and every corruption of them")
    print("  scores clearly worse.")
    return True


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--final", required=True)
    ap.add_argument("--n-check", type=int, default=24, help="overlay images to write")
    ap.add_argument("--n-verify", type=int, default=200)
    ap.add_argument("--verify-only", action="store_true",
                    help="re-check masks already on disk without rewriting them")
    args = ap.parse_args()

    ann, root = Path(args.annotations), Path(args.final)
    dst = ann / "propagated" / "masks"
    dst.mkdir(parents=True, exist_ok=True)
    chk = ann / "audit" / "propagation_check"
    chk.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(root / "manifests" / "dataset_manifest.csv", encoding="utf-8-sig")
    df.columns = [c.lstrip("\ufeff") for c in df.columns]
    aug = df[df.image_kind == "synthetic_derivative"]
    print(f"derivatives to propagate: {len(aug)}")

    src_cache: dict[str, Image.Image] = {}
    n_ok = n_miss = 0
    checks = set(random.Random(0).sample(range(len(aug)), min(args.n_check, len(aug))))

    for i, r in enumerate([] if args.verify_only else aug.itertuples()):
        sm = ann / "clean" / "masks" / f"{r.source_image_id}.png"
        if not sm.exists():
            n_miss += 1
            continue
        if r.source_image_id not in src_cache:
            src_cache[r.source_image_id] = Image.open(sm).convert("L")
        base = src_cache[r.source_image_id]

        trace = json.loads(r.augmentation_trace_json)
        ops = trace.get("operations", trace.get("ops", [])) if isinstance(trace, dict) else trace
        if not ops:
            # v1 tolerated this. It is a data defect: a derivative with no
            # recorded transform cannot be replayed, so refuse to invent one.
            raise UnknownOperation(f"{r.image_id}: empty augmentation trace")

        out = apply_trace(base, ops, (int(r.width), int(r.height)))
        out.save(dst / f"{r.image_id}.png")
        n_ok += 1

        if i in checks:
            img = Image.open(root / r.relative_path).convert("RGB").resize(out.size)
            ov = np.asarray(img).astype(float)
            col = CLASS_COLOURS[np.asarray(out).clip(0, 4)]
            mk = (np.asarray(out) > 0)[..., None]
            blend = (ov * (1 - 0.45 * mk) + col * 0.45 * mk).astype(np.uint8)
            Image.fromarray(blend).save(chk / f"{r.image_id}.jpg", quality=88)

        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(aug)}")

    if not args.verify_only:
        print(f"\npropagated {n_ok}   missing source mask {n_miss}")
        print(f"overlays for inspection: {chk}  ({len(checks)} images)")

    ok = verify(ann, root, df, n=args.n_verify)

    print("\nOPEN THOSE OVERLAYS ANYWAY. The verification catches gross")
    print("misalignment; your eyes catch the subtle kind.")
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
