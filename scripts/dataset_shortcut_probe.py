"""Shortcut / difficulty probe for the FINAL v1 tyre dataset.

Establishes the floor that any deep model must beat, using only trivial features
and the dataset's own supplied group folds.

Probe A - COLOUR ONLY   : 10 global colour statistics from a 64x64 thumbnail.
Probe B - STRUCTURE ONLY: 9 contrast-normalised texture/groove statistics from
                          the centre tread band (background and global brightness
                          removed).

If a deep model does not clearly beat these, it has learned nothing that the
trivial baselines had not already captured.

Usage:
    conda activate cv_conda
    python scripts/dataset_shortcut_probe.py --root "D:/Dataset Download/Tire Dataset Prepared/FINAL"

Reference results, FINAL v1, clean images only (recorded 2026-08-26):

    Probe A (colour)    fold0 0.952 | fold1 0.399 | fold2 0.123 | mean 0.491
    Probe B (structure) fold0 0.354 | fold1 0.119 | fold2 0.976 | mean 0.483
    majority-class      mean accuracy 0.423

The enormous fold-to-fold swing is the finding: with one to two capture sessions
per class per fold, "class" is very nearly "which tyre", so cross-fold
generalisation is not reliably measurable on this package.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

LABELS = {"low_mileage_proxy": 0, "mid_mileage_proxy": 1, "high_mileage_proxy": 2}
COLOUR_COLS = ["R", "G", "B", "bright", "contrast", "sat", "sat_p99",
               "blueish", "bright_frac", "dark_frac"]
STRUCT_COLS = ["d10", "d15", "d20", "gmean", "gp95", "lapvar",
               "colstd", "colmin", "rowstd"]


# ---------------------------------------------------------------- features

def colour_features(path: Path) -> dict:
    a = np.asarray(Image.open(path).convert("RGB").resize((64, 64)), np.float32) / 255.0
    mx, mn = a.max(2), a.min(2)
    sat = np.where(mx > 0, (mx - mn) / (mx + 1e-6), 0)
    g = a.mean(2)
    return dict(
        R=a[..., 0].mean(), G=a[..., 1].mean(), B=a[..., 2].mean(),
        bright=g.mean(), contrast=g.std(),
        sat=sat.mean(), sat_p99=float(np.percentile(sat, 99)),
        blueish=float(((a[..., 2] - a[..., 0]) > 0.10).mean()),   # new-tyre paint stripe
        bright_frac=float((g > 0.62).mean()),                     # pale dust / background
        dark_frac=float((g < 0.18).mean()),                       # deep groove shadow
    )


def _grad(g: np.ndarray) -> np.ndarray:
    gx, gy = np.zeros_like(g), np.zeros_like(g)
    gx[:, 1:-1] = g[:, 2:] - g[:, :-2]
    gy[1:-1, :] = g[2:, :] - g[:-2, :]
    return np.hypot(gx, gy)


def structure_features(path: Path) -> dict:
    im = Image.open(path).convert("L")
    w, h = im.size
    # centre tread band only: removes background and framing differences
    im = im.crop((int(w * 0.28), int(h * 0.28), int(w * 0.72), int(h * 0.72))).resize((256, 256))
    g = np.asarray(im, np.float32) / 255.0
    gn = (g - g.mean()) / (g.std() + 1e-8)          # kills global brightness/exposure
    gm = _grad(gn)
    lap = gn[1:-1, 1:-1] * 4 - gn[:-2, 1:-1] - gn[2:, 1:-1] - gn[1:-1, :-2] - gn[1:-1, 2:]
    return dict(
        d10=float((gn < -1.0).mean()),               # groove-shadow fractions
        d15=float((gn < -1.5).mean()),
        d20=float((gn < -2.0).mean()),
        gmean=float(gm.mean()), gp95=float(np.percentile(gm, 95)),
        lapvar=float(lap.var()),
        colstd=float(gn.mean(0).std()),              # vertical groove banding strength
        colmin=float(gn.mean(0).min()),
        rowstd=float(gn.mean(1).std()),
    )


# ---------------------------------------------------------------- model

def _softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def _fit(X, y, k=3, iters=4000, lr=0.5, l2=1e-2):
    n, d = X.shape
    W, b, Y = np.zeros((d, k)), np.zeros(k), np.eye(k)[y]
    for _ in range(iters):
        G = (_softmax(X @ W + b) - Y) / n
        W -= lr * (X.T @ G + l2 * W)
        b -= lr * G.sum(0)
    return W, b


def macro_f1(y, p, k=3):
    fs = []
    for c in range(k):
        tp = ((p == c) & (y == c)).sum()
        fp = ((p == c) & (y != c)).sum()
        fn = ((p != c) & (y == c)).sum()
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        fs.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return float(np.mean(fs)), [round(f, 3) for f in fs]


def run_probe(df: pd.DataFrame, cols: list[str], name: str) -> float:
    print(f"\n=== {name} ===")
    scores = []
    for k in sorted(df.fold_id.unique()):
        tr, va = df[df.fold_id != k], df[df.fold_id == k]
        mu, sd = tr[cols].values.mean(0), tr[cols].values.std(0) + 1e-8
        W, b = _fit((tr[cols].values - mu) / sd, tr.y.values)
        pv = _softmax(((va[cols].values - mu) / sd) @ W + b).argmax(1)
        f1, per = macro_f1(va.y.values, pv)
        maj = np.bincount(va.y.values, minlength=3).argmax()
        scores.append(f1)
        print(f"  fold {k}: n={len(va):4d}  macroF1={f1:.3f}  acc={(pv == va.y.values).mean():.3f}"
              f"  (majority acc={(va.y.values == maj).mean():.3f})  per-class F1={per}")
        cm = np.zeros((3, 3), int)
        for t, p in zip(va.y.values, pv):
            cm[t, p] += 1
        print(f"     confusion rows=true[low,mid,high]:\n{cm}")
    print(f"  MEAN macroF1 = {np.mean(scores):.3f}   (fold spread "
          f"{min(scores):.3f} .. {max(scores):.3f})")
    return float(np.mean(scores))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="path to the FINAL package")
    ap.add_argument("--out", default=None, help="optional CSV output directory")
    args = ap.parse_args()

    root = Path(args.root)
    df = pd.read_csv(root / "manifests" / "clean_manifest.csv")
    df.columns = [c.lstrip("\ufeff") for c in df.columns]
    print(f"clean images: {len(df)}   sessions: {df.session_group.nunique()}")

    recs = []
    for _, r in df.iterrows():
        p = root / r["relative_path"]
        recs.append({
            "image_id": r["image_id"], "proxy": r["proxy_label"],
            "sess": r["session_group"], "fold_id": r["fold_id"],
            **colour_features(p), **structure_features(p),
        })
    F = pd.DataFrame(recs)
    F["y"] = F["proxy"].map(LABELS)

    print("\nper-class structure means (physically meaningful wear cues):")
    print(F.groupby("proxy")[["d20", "colstd", "gmean"]].mean().round(4).to_string())
    print("  d20    = deep-groove shadow fraction   -> expect low > mid > high")
    print("  colstd = vertical groove banding       -> expect low > mid > high")

    a = run_probe(F, COLOUR_COLS, "PROBE A - COLOUR ONLY (10 global colour stats)")
    b = run_probe(F, STRUCT_COLS, "PROBE B - STRUCTURE ONLY (contrast-normalised tread band)")

    print("\n" + "=" * 72)
    print(f"FLOOR TO BEAT:  colour {a:.3f}   structure {b:.3f}   -> any model must beat {max(a, b):.3f}")
    print("Report ALL THREE FOLDS. A single-fold number from this package is not evidence.")
    print("=" * 72)

    if args.out:
        outd = Path(args.out)
        outd.mkdir(parents=True, exist_ok=True)
        F.to_csv(outd / "probe_features.csv", index=False)
        print(f"features written to {outd / 'probe_features.csv'}")


if __name__ == "__main__":
    main()
