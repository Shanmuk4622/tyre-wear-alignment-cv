# Project Logbook

> One entry per week, 30 minutes every Friday, as a group. This is not bureaucracy — at Review-3 you will need to remember why a loss weight was set the way it was, and you will not.

**Template**

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

## Entry 2 — 2026-08-25 · Documentation rebuilt on the Review-1 specification

**Context.** The repository docs had been written against a wrong assumption about the capture setup (a ground-embedded glass-plate rig). The Review-1 report and `VISION_MODELS_AND_FILTERS_README.md` define the actual setup: **a low-mounted camera ahead of the wheel, facing the front of one tyre.** All documents rewritten accordingly.

**Rewritten:** `README.md`, `docs/01`–`04`, `06`–`09`, `ENVIRONMENT.md`, `environment.yml`, `CITATION.cff`, `LICENSE`, `GITHUB_SETUP.md`.
**New:** `docs/10_VISION_TECHNIQUES.md`, `docs/11_APP.md`.
**Removed entirely:** every reference to FTIR, glass plates, contact-patch imaging, drive-over rigs and the rolling-constraint blur argument. None of it applies.

**Research pass — findings that changed the design:**

| Finding | Source | Consequence |
|---|---|---|
| **Photometric stereo** is proven for defect detection on specular/low-contrast industrial surfaces; a static ring light provably cannot distinguish a stain from a shadowed cavity | [Sensors 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC8838491/), [MVA 2021](https://link.springer.com/article/10.1007/s00138-021-01244-z) | **Largest addition to the stack.** 4 LEDs (~₹2,000) turn the camera into a surface-geometry sensor. Now Core + Ablation #2 |
| **clDice / Skeleton Recall** topology-preserving losses, explicitly proposed for industrial crack detection | [arXiv 2003.07311](https://arxiv.org/pdf/2003.07311), [2404.03010](https://arxiv.org/html/2404.03010v1) | Added to `L_seg` for sipes and cracks. Connectivity is now a reported metric, not just IoU |
| **SAM2 memory propagation**: annotation throughput 37.8 s/frame → 4.5 s/frame; FS-SAM2 gains from ~50 imgs/class | [arXiv 2509.12105](https://arxiv.org/html/2509.12105) | Adopted as the core annotation workflow. This is what makes 300 tyres feasible |
| **Frozen SSL features give no clear advantage** on RGB industrial tasks; fully fine-tuned SSL init is strongest | [arXiv 2605.23472](https://arxiv.org/html/2605.23472) | Changed the training recipe: initialise from SSL, **fine-tune the whole backbone**, do not linear-probe |
| **Depth Anything V2** documented weak at fine detail and close range | [arXiv 2406.09414](https://arxiv.org/html/2406.09414v2) | Rejected for metrology. Kept as a **negative-result ablation** |
| **Ko et al.**: stacking depth + equalised depth + height map improved mIoU by >7 points | [doi:10.3390/app112110376](https://doi.org/10.3390/app112110376) | Supports the `[RGB \| normals \| albedo \| CLAHE]` input stack |
| **Huber TireEye**: 0.57 mm using TWI bars as an in-frame scale reference | [doi:10.36001/phmconf.2022.v14i1.3242](https://doi.org/10.36001/phmconf.2022.v14i1.3242) | TWI anchor formalised as a training loss term. Benchmark to beat |
| **Vivekanandan & Rajeswari**: unseen-brand accuracy 88.2% → 92.4% with domain adaptation | [doi:10.1016/j.measurement.2026.121509](https://doi.org/10.1016/j.measurement.2026.121509) | Brand shift is a **measured** gap. Unseen-brand split mandatory from day one |

**Decisions:**

- Illumination promoted from acquisition detail to the project's most distinctive design choice
- `L_seg` = focal + 0.7·dice + 0.3·boundary + **0.2·clDice**
- SAM2-assisted annotation before scaling collection, not after
- Backbone: SSL init, **fully fine-tuned**
- Toe positioned as **binary screening (AUROC)**, continuous MAE secondary
- Resolution budget stated explicitly: **0.3 mm sipe at 3 px requires ≤0.1 mm/px** → ≥8 MP sensor or a cropped-region claim
- Team split by subsystem with four written interface contracts

**Numbers established:**

- Resolution: 1080p across a 250 mm tread = 0.130 mm/px — **insufficient for sipes**; 12 MP = 0.062 mm/px
- Reference benchmarks: on-board optical depth **0.57 mm**; structured light **<0.2 mm**; marker stereo alignment **~0.025°**; RGB-D alignment **<0.1°**; front-view tread segmentation **mAP 0.608**

**Novelty audit (see `09_RELATED_WORK.md §6`):** strongest claims are the wear↔geometry cross-check (recent vs chronic misalignment — nothing comparable found), the dataset, the four-modality controlled comparison, and coverage-reported unrolling. TWI anchoring is prior art (cite Huber). The app is not a research contribution.

**Next week (P0):**

- [ ] Photograph a tyre from the intended viewpoint. **Can you see a sipe? A TWI bar?** Measure mm/px against a ruler in frame ← GO/NO-GO
- [ ] **Hand-torch photometric test** — 4 torch positions vs 1 flat photo. Decides Ablation #2
- [ ] Buy the digital tread depth gauge; measure 4 tyres; longitudinal study starts
- [ ] Assign the four roles and write the four interface contracts
- [ ] Order long-lead parts: camera, lens, polarising film, LEDs
- [ ] All four members read Tier-1 papers (`09_RELATED_WORK.md §7`)

---

## Entry 1 — 2026-08-09 · Repository initialised

Initial scaffold created. *(Superseded by Entry 2 — the capture setup assumed here was incorrect.)*

---
