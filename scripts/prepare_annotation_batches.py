"""Build the annotation package.

SOLO MODE (default): one folder with all 418 clean images, ordered so whole
sessions stay together -- you interpret one tyre consistently before moving to
the next -- plus a 30-image self-check subset for the consistency pass.

TEAM MODE (--team): four session-aligned batches plus a shared 50-image
calibration batch, for when more than one person annotates.

Usage (Windows, one line):
    conda activate cv_conda
    python scripts\prepare_annotation_batches.py --final "D:\Dataset Download\Tire Dataset Prepared\FINAL" --out "D:\Dataset Download\Tire Dataset Prepared\annotation_upload" --no-zip

Produces batch_A/B/C/D folders, batch_CAL, labels.txt and filename_map.csv.
Use --no-zip for labelme (which reads folders). Omit it to also produce .zip
files for a browser tool.

Keep filename_map.csv -- the import script needs it to turn annotation
filenames back into image_ids.
"""
from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

import pandas as pd

# Whole sessions to one annotator. Never split a session across people --
# the tread-boundary interpretation must stay consistent within a tyre.
BATCHES = {
    "A": ["new_tire__session_001", "mileage_100000_plus__session_002"],
    "B": ["mileage_000100_005000__session_001", "mileage_000100_005000__session_002"],
    "C": ["mileage_040000__session_001", "mileage_070000__session_001",
          "mileage_090000__session_001"],
    "D": ["mileage_100000_plus__session_001", "mileage_100000_plus__session_003",
          "mileage_100000_plus__session_004", "mileage_100000_plus__session_005",
          "mileage_100000_plus__session_006"],
}
N_CALIBRATION = 50
N_SELFCHECK = 30


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=4622)
    ap.add_argument("--team", action="store_true",
                    help="four-person split instead of one solo batch")
    ap.add_argument("--zip", action="store_true",
                    help="also write .zip files (labelme reads folders; zips are for browser tools)")
    ap.add_argument("--selfcheck-n", type=int, default=N_SELFCHECK)
    args = ap.parse_args()

    root, out = Path(args.final), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(root / "manifests" / "clean_manifest.csv")
    df.columns = [c.lstrip("\ufeff") for c in df.columns]
    print(f"clean images: {len(df)}   sessions: {df.session_group.nunique()}")

    rows = []

    if args.team:
        covered = {x for ss in BATCHES.values() for x in ss}
        missing = set(df.session_group) - covered
        if missing:
            raise SystemExit(f"sessions not assigned to any batch: {sorted(missing)}")
        for name, sessions in BATCHES.items():
            sub = df[df.session_group.isin(sessions)].sort_values(
                ["session_group", "image_id"]).reset_index(drop=True)
            rows += _stage(sub, out / f"batch_{name}", root, name, args.zip)
            print(f"  batch_{name:<3} {len(sub):>4} images   "
                  f"{len(sessions)} session(s)")
        cal = (df.groupby("proxy_label", group_keys=False)[df.columns.tolist()]
                 .apply(lambda g: g.sample(min(len(g), N_CALIBRATION // 3 + 1),
                                           random_state=args.seed))
                 .head(N_CALIBRATION).sort_values("image_id").reset_index(drop=True))
        rows += _stage(cal, out / "batch_CAL", root, "CAL", args.zip)
        print(f"  batch_CAL {len(cal):>4} images   (everyone annotates these independently)")

    else:
        # SOLO. One folder, ordered by session so you finish one tyre before
        # starting the next -- the tread-boundary judgement stays consistent
        # within a tyre, which is where consistency matters most.
        allimg = df.sort_values(["session_group", "image_id"]).reset_index(drop=True)
        rows += _stage(allimg, out / "batch_ALL", root, "ALL", args.zip)
        print(f"  batch_ALL       {len(allimg):>4} images, session-ordered")
        by_sess = allimg.groupby("session_group").size()
        print(f"\n  session order and sizes (annotate top to bottom):")
        for sg, n in by_sess.items():
            cls = allimg[allimg.session_group == sg].proxy_label.iloc[0].replace("_mileage_proxy", "")
            print(f"    {sg:<38} {n:>4}  ({cls})")

        # Self-consistency subset. You re-annotate these at the END, without
        # looking at the first pass. Comparing the two measures whether your
        # tread-boundary rule stayed stable across ~3.5 hours of work -- which
        # is the solo equivalent of inter-annotator agreement, and it is the
        # drift that actually threatens a one-person job.
        n_per = max(1, args.selfcheck_n // 3)
        sc = (df.groupby("proxy_label", group_keys=False)[df.columns.tolist()]
                .apply(lambda g: g.sample(min(len(g), n_per), random_state=args.seed))
                .head(args.selfcheck_n).sort_values("image_id").reset_index(drop=True))
        _stage(sc, out / "batch_SELFCHECK", root, "SELFCHECK", args.zip)   # copies, not new rows
        (out / "selfcheck_ids.txt").write_text("\n".join(sc.image_id) + "\n")
        print(f"\n  batch_SELFCHECK {len(sc):>4} images   (re-annotate these LAST, "
              f"in a separate folder)")
        print(f"    class balance: {sc.proxy_label.value_counts().to_dict()}")

    m = pd.DataFrame(rows)
    m.to_csv(out / "filename_map.csv", index=False)
    print(f"\nfilename_map.csv written ({len(m)} rows) -- KEEP THIS, the import needs it")

    # A fixed label list means annotators PICK from a list instead of typing.
    # "Tread" or "tyre " (trailing space) can then never happen, and that one
    # detail saves an afternoon of cleanup.
    (out / "labels.txt").write_text(
        "__ignore__\n_background_\ntyre\ntread\nmarking\ndamage\n")
    print("labels.txt written -- everyone must pass this to labelme with --labels")

    if args.team:
        print("\nNext: send each person their batch_X folder, batch_CAL, and labels.txt.")
    else:
        print("\nNext -- start annotating (all on one line):")
        print(f'  labelme "{out}\\batch_ALL" --labels "{out}\\labels.txt" '
              f'--output "{out}\\ann_pass1" --nodata --autosave')
        print("\nWhen all 418 are done, the consistency pass:")
        print(f'  labelme "{out}\\batch_SELFCHECK" --labels "{out}\\labels.txt" '
              f'--output "{out}\\ann_pass2" --nodata --autosave')
        print("\nThen:  python scripts\\annotation_agreement.py --mode self "
              f'--pass1 "{out}\\ann_pass1" --pass2 "{out}\\ann_pass2"')


def _short(session: str) -> str:
    """'mileage_100000_plus__session_004' -> '100000_plus/004'.

    Taking only the part after '__' gave three identical 'session_001' entries
    for batch C, which told you nothing about which tyres were in it.
    """
    if "__" not in session:
        return session
    head, tail = session.split("__", 1)
    return f"{head.replace('mileage_', '')}/{tail.replace('session_', '')}"


def _stage(sub: pd.DataFrame, stage: Path, root: Path, batch: str, do_zip: bool) -> list[dict]:
    """Copy images into a flat folder with a zero-padded index prefix.

    The prefix keeps labelme's file order stable, so "image 137" means the same
    thing every time you reopen the folder.
    """
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    rows = []
    for i, r in sub.iterrows():
        fn = f"{i:04d}_{r.image_id}.jpg"
        shutil.copy2(root / r.relative_path, stage / fn)
        rows.append({"batch": batch, "filename": fn, "image_id": r.image_id,
                     "session_group": r.session_group, "proxy_label": r.proxy_label,
                     "fold_id": r.fold_id, "relative_path": r.relative_path})
    if do_zip:
        _zip(stage, stage.with_suffix(".zip"))
    return rows


def _zip(folder: Path, dest: Path) -> None:
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(folder.glob("*.jpg")):
            z.write(f, f.name)


if __name__ == "__main__":
    main()
