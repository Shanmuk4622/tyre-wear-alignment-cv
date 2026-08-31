"""Audit whether the dataset's `session_group` values are really distinct tyres.

WHY THIS EXISTS
---------------
`final_v1` grouped captures into 12 "sessions" using a 12-second timestamp gap.
Timestamps are a proxy for tyre identity, not a measurement of it: photograph
one tyre, walk away for 20 seconds, come back and photograph it again, and the
rule creates two "sessions" for one tyre. If those land in different folds, a
model can memorise the tyre in training and be graded on it in validation --
and the leak is completely silent.

This compares tread PATTERN between sessions. The tread pattern is what
identifies a physical tyre; brightness and dirt are not.

    ratio = between-session similarity / sqrt(within(A) x within(B))

Within-session similarity is the "same tyre, different frame" reference, so
ratio ~ 1.0 means two sessions are indistinguishable from one tyre.

Sessions with fewer than --min-n images are excluded: their within-session
baseline is too noisy to normalise by, and including them produces false
merges (a first version of this script "found" 6 tyres because of exactly that).

Usage (Windows, one line):
    conda activate cv_conda
    python scripts\\tyre_identity_audit.py --final "D:\\Dataset Download\\Tire Dataset Prepared\\FINAL" --annotations "D:\\Dataset Download\\Tire Dataset Prepared\\annotations"
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
from annotation_regions import tyre_region

SAME_TYRE = 0.93        # ratio at or above this: treat as one tyre
SUSPECT = 0.85          # between this and SAME_TYRE: flag, do not merge


def descriptor(img_path: Path, mask_path: Path | None) -> np.ndarray:
    """Contrast-normalised tread crop. Captures pattern, not lighting."""
    im = Image.open(img_path).convert("L")
    if mask_path and mask_path.exists():
        m = np.asarray(Image.open(mask_path))
        ys, xs = np.where(tyre_region(m))
        if len(xs):
            im = im.crop((int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())))
    a = np.asarray(im.resize((48, 64)), np.float32)
    v = ((a - a.mean()) / (a.std() + 1e-8)).ravel()
    return v / (np.linalg.norm(v) + 1e-9)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", required=True)
    ap.add_argument("--annotations", default=None)
    ap.add_argument("--min-n", type=int, default=15)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.final)
    ann = Path(args.annotations) if args.annotations else None
    man = pd.read_csv(root / "manifests" / "clean_manifest.csv")
    man.columns = [c.lstrip("\ufeff") for c in man.columns]

    D, meta = [], []
    for r in man.itertuples():
        mp = (ann / "clean" / "masks" / f"{r.image_id}.png") if ann else None
        D.append(descriptor(root / r.relative_path, mp))
        meta.append((r.session_group, r.proxy_label, r.fold_id))
    D = np.array(D)
    meta = pd.DataFrame(meta, columns=["sess", "cls", "fold"])

    sessions = sorted(meta.sess.unique())
    n_of = {s: int((meta.sess == s).sum()) for s in sessions}
    reliable = [s for s in sessions if n_of[s] >= args.min_n]
    tiny = [s for s in sessions if n_of[s] < args.min_n]

    print(f"images {len(D)}   sessions {len(sessions)}   "
          f"reliable (>= {args.min_n} imgs) {len(reliable)}")
    if tiny:
        print("  too small to adjudicate:",
              ", ".join(f"{s.split('__')[-1]}({n_of[s]})" for s in tiny))

    def within(s):
        X = D[(meta.sess == s).values]
        return float((X @ X.T)[np.triu_indices(len(X), 1)].mean()) if len(X) > 1 else np.nan

    def between(a, b):
        return float((D[(meta.sess == a).values] @ D[(meta.sess == b).values].T).mean())

    print(f"\n{'session A':<22}{'session B':<22}{'ratio':>7}   verdict")
    print("-" * 66)
    pairs = []
    for i, a in enumerate(reliable):
        for b in reliable[i + 1:]:
            if meta[meta.sess == a].cls.iloc[0] != meta[meta.sess == b].cls.iloc[0]:
                continue
            ratio = between(a, b) / np.sqrt(within(a) * within(b))
            fa = int(meta[meta.sess == a].fold.iloc[0])
            fb = int(meta[meta.sess == b].fold.iloc[0])
            v = "SAME TYRE" if ratio >= SAME_TYRE else ("suspect" if ratio >= SUSPECT else "different")
            pairs.append({"a": a, "b": b, "ratio": round(float(ratio), 3),
                          "verdict": v, "fold_a": fa, "fold_b": fb,
                          "cross_fold": fa != fb})
            if v != "different":
                sa, sb = a.split("__")[-1], b.split("__")[-1]
                flag = "  *** CROSS-FOLD ***" if fa != fb else ""
                print(f"{a[:21]:<22}{b[:21]:<22}{ratio:>7.2f}   {v}{flag}")

    bad = [p for p in pairs if p["verdict"] != "different" and p["cross_fold"]]
    print("\n" + "=" * 66)
    if bad:
        print(f"{len(bad)} same-or-suspect tyre pair(s) span DIFFERENT folds:")
        for p in sorted(bad, key=lambda x: -x["ratio"]):
            print(f"  {p['a'].split('__')[-1]} (fold {p['fold_a']})  ==  "
                  f"{p['b'].split('__')[-1]} (fold {p['fold_b']})   ratio {p['ratio']}  [{p['verdict']}]")
        print("\nA model can memorise that tyre in training and be graded on it in")
        print("validation. Report every affected fold's result with this caveat.")
    else:
        print("No cross-fold same-tyre pairs among reliable sessions.")
    print("=" * 66)

    print("\nDISTINCT TYRES IN EACH FOLD'S VALIDATION SET")
    print("(this number, not the image count, is the real sample size)\n")
    for k in sorted(meta.fold.unique()):
        v = meta[meta.fold == k]
        per = v.groupby("cls").sess.nunique()
        print(f"  fold {k}: {len(v):>4} images, {v.sess.nunique()} sessions  "
              + "  ".join(f"{c.replace('_mileage_proxy','')}={n}" for c, n in per.items()))
    print("\n  With roughly one tyre per class in validation, a model only has to")
    print("  tell three specific tyres apart. Near-perfect scores are EXPECTED and")
    print("  are not evidence of learning wear.")

    out = Path(args.out) if args.out else (ann or root).parent / "tyre_identity_audit.json"
    out.write_text(json.dumps({
        "min_n": args.min_n, "thresholds": {"same": SAME_TYRE, "suspect": SUSPECT},
        "n_sessions": len(sessions), "n_reliable": len(reliable),
        "too_small": tiny, "pairs": pairs, "cross_fold_flags": bad,
        "val_sessions_per_fold": {int(k): int(meta[meta.fold == k].sess.nunique())
                                  for k in sorted(meta.fold.unique())}}, indent=2))
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
