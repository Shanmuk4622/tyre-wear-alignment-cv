# 02_RIG_BUILD.md — SUPERSEDED

**This document is obsolete. See `02_CAPTURE_AND_PREPROCESSING.md`.**

---

It described building a camera rig: sensor and lens selection, mounting geometry, a bill of materials, a photometric-stereo illumination array, cross-polarisation, ChArUco calibration, laser triangulation, and an alignment jig.

**None of that is being built.** The project now works from the existing `final_v1` dataset plus separately-captured video, and the goal is a broad comparative study of models and techniques rather than a hardware system (`13_EXPERIMENT_PLAN.md`).

What survived the change moved to **`02_CAPTURE_AND_PREPROCESSING.md`**:

- How the existing data was captured, and what follows from having no calibration
- Guidance for capturing `final_v2` — more tyres, a scale reference in frame, gauge measurements, consistent framing, locked camera settings
- The image-processing filter table, with a verdict and a reason for each
- Augmentation policy, including the horizontal-flip rule

The removed content was correct for its purpose. If the project later returns to metric depth or alignment, recover it from git history:

```bash
git log --oneline -- docs/02_RIG_BUILD.md
git show <commit>:docs/02_RIG_BUILD.md
```

*Safe to delete this file.*
