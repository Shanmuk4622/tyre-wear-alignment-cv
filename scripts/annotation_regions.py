"""Canonical region definitions for the annotation masks.

Imported by annotation_agreement.py, validate_annotations.py and the XAI stage
so that "the tread region" means exactly one thing everywhere.

WHY THIS FILE EXISTS
--------------------
Masks are stored as a SINGLE indexed PNG (0=bg 1=tyre 2=tread 3=marking
4=damage). One pixel holds one value, so a later class painted over an earlier
one ERASES it. Drawing order is tyre -> tread -> marking -> damage, therefore:

    m == 1  is NOT "the tyre".
            It is "tyre, excluding whatever was painted on top of it".

Comparing raw `m == 1` between two annotators penalises them for disagreeing
about the TREAD boundary -- the leftover tyre ring changes shape. In testing
this produced a spurious 0.944 tyre IoU for two annotators whose tyre outlines
were pixel-identical.

Always go through these accessors. Never compare raw class indices.
"""
from __future__ import annotations

import numpy as np

CLASSES = ["tyre", "tread", "marking", "damage"]
CLASS_IDX = {c: i + 1 for i, c in enumerate(CLASSES)}

BG, TYRE, TREAD, MARKING, DAMAGE = 0, 1, 2, 3, 4


def tyre_region(m: np.ndarray) -> np.ndarray:
    """Everything rubber. All four classes are nested inside the tyre."""
    return m > BG


def tread_region(m: np.ndarray) -> np.ndarray:
    """The grooved face.

    Includes `marking`, because factory paint stripes and moulded lettering sit
    ON the tread and would otherwise punch holes in it.

    EXCLUDES `damage`, because damage can be on the sidewall or shoulder and we
    cannot tell which from a single-layer mask. That slightly under-counts tread
    on damaged tyres -- the safe direction, since it makes attention metrics
    pessimistic rather than optimistic.
    """
    return (m == TREAD) | (m == MARKING)


def marking_region(m: np.ndarray) -> np.ndarray:
    """Factory paint stripes and lettering -- the known `low`-class shortcut."""
    return m == MARKING


def damage_region(m: np.ndarray) -> np.ndarray:
    return m == DAMAGE


def shoulder_region(m: np.ndarray) -> np.ndarray:
    """Tyre minus tread. Derived, never annotated directly.

    MEASURED ON final_v1 (418 images, annotation v1):
        tread / tyre area ratio   median 0.990   min 0.943   max 1.000
        27% of images have NO shoulder pixels at all

    That is not an annotation error -- it is the viewpoint. The camera faces
    the tread crown head-on, so the shoulders curve away and are barely in
    frame. Annotating a shoulder here would mean inventing a region the image
    does not contain.

    CONSEQUENCE, and it matters for the XAI stage: on this dataset
    `tread_region` and `tyre_region` are nearly the same set, so TER measures
    "attention on the TYRE vs the BACKGROUND", not "tread vs shoulder".
    That is still a real and useful question -- backgrounds vary per session
    (concrete, brick wall, parked car, vegetation) and are a documented
    shortcut -- but the claim must be worded accordingly. See 14_XAI_PROTOCOL.
    """
    return tyre_region(m) & ~tread_region(m)


def background_region(m: np.ndarray) -> np.ndarray:
    """Outside the tyre entirely -- the BAR denominator."""
    return m == BG


REGIONS = {
    "tyre": tyre_region,
    "tread": tread_region,
    "marking": marking_region,
    "damage": damage_region,
    "shoulder": shoulder_region,
    "background": background_region,
}


def area_fractions(m: np.ndarray) -> dict[str, float]:
    n = m.size
    return {k: float(f(m).sum()) / n for k, f in REGIONS.items()}


def _selftest() -> bool:
    m = np.zeros((100, 100), np.uint8)
    m[10:90, 10:90] = TYRE          # tyre block
    m[20:80, 25:75] = TREAD         # tread painted over part of it
    m[40:60, 45:55] = MARKING       # marking painted over part of the tread
    ok = True

    def t(name, cond):
        nonlocal ok
        print(("  PASS  " if cond else "  FAIL  ") + name)
        ok = ok and bool(cond)

    print("=== annotation_regions selftest ===")
    t("tyre_region covers all painted pixels", tyre_region(m).sum() == 80 * 80)
    t("raw m==1 is SMALLER than tyre_region (the trap)",
      (m == TYRE).sum() < tyre_region(m).sum())
    t("tread_region includes marking", tread_region(m).sum() == 60 * 50)
    t("marking inside tread", (marking_region(m) & ~tread_region(m)).sum() == 0)
    t("tread inside tyre", (tread_region(m) & ~tyre_region(m)).sum() == 0)
    t("shoulder = tyre - tread",
      shoulder_region(m).sum() == tyre_region(m).sum() - tread_region(m).sum())
    t("background is the complement", background_region(m).sum() == 100 * 100 - 80 * 80)

    # the exact false-FAIL this file prevents: identical tyre outlines, tread
    # boundary differing by 4 px
    a = np.zeros((100, 100), np.uint8); a[10:90, 10:90] = TYRE; a[20:80, 25:75] = TREAD
    b = np.zeros((100, 100), np.uint8); b[10:90, 10:90] = TYRE; b[20:80, 25:79] = TREAD
    raw = (a == TYRE) & (b == TYRE); rawu = (a == TYRE) | (b == TYRE)
    reg = tyre_region(a) & tyre_region(b); regu = tyre_region(a) | tyre_region(b)
    raw_iou, reg_iou = raw.sum() / rawu.sum(), reg.sum() / regu.sum()
    print(f"  raw m==1 tyre IoU  {raw_iou:.3f}   <- spuriously low")
    print(f"  tyre_region IoU    {reg_iou:.3f}   <- correct")
    t("region accessor gives IoU 1.0 for identical tyre outlines", reg_iou == 1.0)
    print("=== selftest", "PASSED" if ok else "FAILED", "===")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if _selftest() else 1)
