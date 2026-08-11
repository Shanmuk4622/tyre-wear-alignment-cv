# Project Logbook

> One entry per week, 30 minutes every Friday. This is not bureaucracy — in week 22 you will need to remember why you chose `λ = 0.3` in week 14, and you will not.

**Template:**

```
## Week N — <dates>

**Planned:**  (from 07_ROADMAP.md)

**Done:**

**Broke / didn't work:**

**Decisions made (and why):**

**Numbers:**  (any measurement taken this week, with units)

**Next week:**

**Roadmap changes:**
```

---

## Week 0 — 2026-08-09

**Done:**

- Project concept designed and documented. Reframed from "two models glued together" to a single coupled wear↔alignment system with a ground-embedded FTIR rig.
- Physics sanity checks run: rolling-constraint blur, tread circumference vs plate length, laser triangulation resolution, FOV.
- Repo scaffold + docs 01–08 written.
- `environment.yml` and `verify_env.py` created for `cv_conda`.

**Decisions made (and why):**

| Decision | Why |
|---|---|
| Ground-embedded upward camera | Zero blur at contact patch; rig defines vehicle reference frame; enables FTIR |
| FTIR contact imaging | Highest-SNR signal available; direct camber/toe signature |
| Laser as *training-time teacher* only | Cheap deployment, better paper |
| Ordinal ranking + isotonic anchoring for depth | Better conditioned than mm regression on small data |
| Analytic prior + learned residual for geometry | Interpretable, data-efficient, defensible in viva |
| Alignment jig instead of workshop rack | Exact labels, uniform coverage, 400 passes/day vs 20 vehicles/day |
| Toe reframed as binary screening | ±0.1° spec is not achievable from a single wheel; screening is honest and more useful |

**Numbers:**

- 195/65R15 → 634 mm dia, 1.99 m circumference. 1.2 m plate → ~60% coverage/pass.
- Blur at contact patch: ~0 mm at any speed. At top of tyre, 5.6 mm @ 10 km/h, 1 ms.
- Laser triangulation target: f=1400 px, b=200 mm, z₀=150 mm → 12.4 px/mm → 0.081 mm/px.
- 120 fps @ 10 km/h → 23 mm travel/frame → ~6× overlap on a 150 mm contact patch.

---

## Week 0b — 2026-08-09 (literature audit)

**Done:** Full research pass. `09_RELATED_WORK.md` written with annotated bibliography and novelty audit.

**Findings that changed the design:**

| Finding | Consequence |
|---|---|
| FTIR tyre footprint is **prior art** (Chodera; Cabrera et al., *Sensors* 2017, 17(4) 707) | Novelty claim #1 downgraded. Reframed as *inverting* a validated forward model, in motion. Net effect: **lower risk**, physics is peer-validated. |
| Prior art interposes a **clear plastic lamina**; black rubber absorbs coupled light rather than scattering it | **Design change: interface film (PPF/PET) is mandatory.** Week-1 test revised from thumb-only to a 4-step rubber-on-film protocol. |
| Cabrera et al. published *contact area vs camber angle* fits | The camber signal provably exists. Strong justification to cite. |
| Image-based tread depth SOTA = **±1.5 mm on 90%** (QBurst) | Our 0.35 mm target is a ~4× improvement. New headline baseline. |
| Drive-over laser tread depth is heavily patented (US 11820178, 10352688, 8621919, 11421982) | Our laser is train-time only; RGB at inference. That is the differentiator — state it explicitly. |
| Hunter holds rolling/no-stop alignment patents using drive-direction calculation (US 11698250, 12467747) | Cite; distinguish (they use clamp-on targets, we are target-free from below). Don't claim novelty. |
| Cross-task consistency is established (Zamir CVPR'20; Nishi CVPR'24) | Reposition: novelty is the *brush-model-derived link function*, not the mechanism. |
| **Toe = permanent static slip angle**; brush/Archard models give wear rate as an explicit function of slip angle | `h_τ` is now **derivable**. Also suggests inferring toe from wear may be *more* sensitive than geometry. Test early. |
| Nothing in the literature distinguishes recent vs chronic misalignment | **Disagreement diagnostic is the strongest claim. Protect it.** |

**Decisions:**

- Add clear PPF interface film to BOM; revise the go/no-go test
- Reframe FTIR claim as inversion of validated prior art
- Derive `h_τ` from the brush model rather than fitting blind
- Abstract headline order: disagreement diagnostic → 4× SOTA → dataset → in-motion inverse FTIR

**Next week (Week 1):**

- [ ] Glass + LED strip + **PPF film + scrap rubber** → **4-step go/no-go test**  ← GO / NO-GO
- [ ] Bicycle wheel rolled over v0 rig, video captured
- [ ] Buy digital tread depth gauge; start longitudinal study on 6 vehicles
- [ ] Order v1 parts (camera, polished glass — longest lead times)
- [ ] Read Tier-1 papers (`09_RELATED_WORK.md §9`), especially Cabrera 2017 + brush-model theses

---
