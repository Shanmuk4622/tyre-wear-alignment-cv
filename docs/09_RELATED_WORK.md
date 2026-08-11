# 09 — Related Work and Novelty Audit

> Research pass completed 2026-08-09. Two findings **change the design**; read §2 and §3 before building anything.
> Novelty audit in §7 replaces the optimistic table in `README.md`.

---

## 1. Executive summary of the research pass

| Question | Answer | Consequence |
|---|---|---|
| Is FTIR tyre-footprint imaging novel? | **No.** Established since Chodera; validated lab test bench published in *Sensors* 2017. | **Good news, not bad.** The physics is peer-validated. Novelty moves to in-motion deployment + inverting the model. |
| Does the footprint actually encode camber? | **Yes — measured and fitted in prior work.** | The signal provably exists. De-risks the core premise. |
| Will black tyre rubber work directly on glass? | **Probably not.** Prior art uses a clear plastic interface film for a reason. | **Design change required.** See §3. |
| Is drive-over laser tread depth novel? | **No.** Multiple granted patents. | Our laser is a *training-time teacher only*; RGB at inference. That is the differentiator. |
| What's SOTA for image-based tread depth? | **±1.5 mm on 90% of images** (QBurst, monocular CNN). | Our 0.35 mm MAE target would be a **4× improvement**. Strong positioning. |
| Is "no-stop" alignment novel? | **No.** Hunter holds patents on rolling alignment with drive-direction calculation. | But those use clamp-on wheel targets. Target-free, ground-embedded is open. |
| Is cross-task consistency loss novel? | **No.** Established (Zamir et al. CVPR'20; Joint-Task Regularization CVPR'24). | Novelty is the *physically-derived link function*, not the mechanism. Reposition. |
| Can toe be grounded in physics? | **Yes — better than expected.** Toe = permanent static slip angle; brush/Archard models give wear rate as an explicit function of it. | Upgrades the consistency loss from heuristic to derived. **Big win.** |

---

## 2. Finding A — FTIR is prior art, and that is good for you

**Cabrera Carrillo et al., "Optimization of an Optical Test Bench for Tire Properties Measurement and Tread Defects Characterization," *Sensors* 2017, 17(4), 707.**

What they built:
- 12 mm glass plate, illuminated from the sides by fluorescent tubes
- A **plastic lamina placed between the tyre and the glass**
- Camera below, grey level calibrated to normal contact pressure
- Tests on a Nankang 165/65R13 at 1000 N vertical load
- Swept **camber angle** and **inflation pressure**; fitted contact area and pressure-centre as functions of both
- Also used it to detect belt defects and abrasion

They cite Chodera as the originator of FTIR pressure sensing, and note that optical techniques give **better spatial resolution than piezoresistive sensor mats**, which is the competing technology.

### Why this is good news, not bad

1. **The physics is peer-validated.** You are not gambling on an untested idea; you are deploying a proven measurement principle in a new setting. That is a much safer capstone.
2. **They proved the signal you need exists.** Figures 13–17 of that paper are literally *contact area vs camber angle* and *pressure centre vs camber angle*, with fitted models. Somebody has already demonstrated that camber is recoverable from an FTIR footprint. Cite it as your justification.
3. **They built the forward model. You build the inverse.** They set camber and measured the footprint. You observe the footprint and estimate camber. Forward → inverse is a legitimate, well-understood form of contribution.
4. **They were static, indoors, on a lab bench, with a hand-placed film.** You are in the road plane, under a moving vehicle, at 8 km/h. That gap is the contribution.

### Revised claim wording

> ~~"FTIR contact-footprint imaging repurposed for wheel-alignment estimation."~~
>
> **"Prior work has established FTIR footprint imaging as a laboratory instrument for characterising tyres with *known* alignment [Cabrera 2017, Chodera]. We invert this relationship and deploy it in the road plane: estimating *unknown* alignment state from the footprint of a moving vehicle, without markers, clamps, or driver intervention."**

That is a smaller claim than the original, and a much more defensible one. It also gives you a strong related-work paragraph instead of a nervous one.

---

## 3. Finding B — the design change: you need an interface film

**This is the most important practical result of the research pass.**

The prior art does not press rubber directly onto glass. It interposes a **clear plastic lamina**, and the reason is optical:

> *"the plastic interface absorbs one fraction, lets another pass through, and if it is clear, it reflects a large amount of it."* — Cabrera et al.

FTIR only produces a **bright** footprint if the contacting material **scatters light back down** toward the camera. Tyre tread rubber does the opposite:

- Carbon-black-filled rubber has a **high refractive index** (couples light in efficiently — good) and is **strongly absorbing** (destroys it — bad).
- Light that enters black rubber is absorbed, not scattered back.
- Meanwhile, in non-contact regions TIR is intact, so no light exits downward there either.

**Net result: with bare rubber on bare glass, both contact and non-contact regions look dark. You get almost no contrast.** The elegant fingerprint-scanner picture in `01_CONCEPT.md` does not transfer to a black tyre.

### The fix

Laminate a **thin, clear, sacrificial interface film** onto the top surface of the glass:

| Property | Requirement | Candidate |
|---|---|---|
| Refractive index | > glass (1.52) | Polyurethane PPF (n ≈ 1.50–1.56), PET (n ≈ 1.57) |
| Thickness | 75–150 µm | thicker = more hysteresis/creep |
| Optical | clear, scattering back-face | slight matte on the underside helps |
| Durability | sacrificial, replaceable | car paint-protection film is designed for exactly this abuse |
| Cost | low | PPF ≈ ₹400/m² |

**Automotive paint-protection film (PPF) is close to ideal**: optically clear polyurethane, self-healing, abrasion-resistant, designed for road grit, sold by the roll, and trivially replaceable when scratched. It is a genuinely good fit for a drive-over plate.

Caveat the prior art already flagged: plastic laminae exhibit **creep and hysteresis** (grey level drifts under sustained pressure). For a *rolling* contact the load is applied for only ~50 ms per point, which is far better than a static bench test — arguably your dynamic use case suffers *less* from creep than theirs did. Worth measuring and reporting; it is a nice small result.

### Week-1 test, revised

The original "press your thumb on the glass" test is **not sufficient** — a fingertip is soft, pale, moist, and scatters beautifully. It will pass even if the concept fails on rubber.

**Revised go/no-go protocol:**

1. Edge-light the glass. Press your thumb. Expect a bright fingerprint. *(Confirms the rig works.)*
2. Press a **piece of black tread rubber cut from a scrap tyre**. Photograph. *(This is the real test.)*
3. Lay a piece of clear PPF/PET film on the glass, press the same rubber. Photograph.
4. Compare contrast between 2 and 3.

**Decision rule:**

| Outcome | Action |
|---|---|
| (3) shows a clear bright footprint | Proceed with film. Expected outcome. |
| (2) alone shows usable contrast | Even better — no film needed. Report it, it contradicts prior practice. |
| Neither works | Fall back to flood-lit ground view + laser. Still novel, still viable. Rewrite `01_CONCEPT.md §3`. |

Total cost: one afternoon and about ₹600 of film. **Do this before ordering the v1 camera.**

---

## 4. Competitive positioning — the numbers that matter

### Tread depth

| System | Method | Reported accuracy |
|---|---|---|
| Digital tread gauge (reference) | Mechanical | ~0.10–0.15 mm test–retest |
| **QBurst tire inspection** | Monocular CNN (U-Net/DenseNet) | **±1.5 mm on 90% of images** |
| Industrial 2D laser profilers | Laser triangulation, up to 4000 fps | ~0.01–0.05 mm |
| Drive-over laser ramps (patented) | Structured-light triangulation | sub-mm |
| **GRIP target** | RGB, laser-distilled, TWI-anchored | **0.35 mm MAE, 85% within ±0.5 mm** |

**This is your headline comparison.** The best published *image-based* result is ±1.5 mm. Laser systems reach 0.05 mm but require shipping a laser. If GRIP hits 0.35 mm with RGB-only inference, you have closed most of the gap between cheap cameras and expensive lasers — and that is a clean, quotable one-sentence contribution.

### Alignment

| System | Method | Accuracy | Requires |
|---|---|---|---|
| Hunter HawkEye Elite | Camera + clamp-on targets | ~0.02–0.05° | Targets, rack, 70 s, technician |
| Furferi et al. 2013 | Stereovision + NIR sidewall markers | "compatible with commercial systems" | Markers on tyre |
| IEEE Access 2022 (RVM-DBSCAN) | Calibration-free vision | camber & toe, high accuracy claimed | Static, controlled |
| **GRIP target** | Target-free, ground-embedded, in-motion | camber 0.30°, toe **screening** (AUC > 0.90) | Nothing — driver just drives |

**Do not claim to compete with Hunter on accuracy. You will lose.** Claim a different axis entirely: **zero-effort, zero-hardware-on-vehicle, universal screening.** Hunter measures 20 cars a day that a technician already suspected. GRIP screens every car that enters the forecourt, for free, and tells the technician which ones to put on the Hunter. Those are complementary products, and framing it that way makes your modest accuracy a design choice rather than a shortfall.

---

## 5. Finding C — a proper physical basis for the toe consistency loss

This upgrades `01_CONCEPT.md §6` from a plausible heuristic to a derived relationship, and it is the strongest theoretical result of the research pass.

**The key realisation: a wheel with static toe angle τ, driving in a straight line, is permanently operating at slip angle α = τ.**

Toe is not merely *correlated* with wear. Toe **is** a continuously-applied slip angle, and the tyre-wear literature gives explicit models for wear rate as a function of slip angle:

- **Brush tyre model** (Salminen, KTH; Analysis of tyre wear using the expanded brush tyre model, DiVA) — models tread elements as elastic bristles, gives sliding velocity and frictional work in the contact patch as a function of slip angle.
- **Archard-type abrasion:** wear volume ∝ frictional work = ∫ (shear stress × sliding distance) over the contact patch.
- Empirical wear-rate form: `W = f(A_b, F_y, α, S, F_N/F_N0)` — abrasion coefficient, lateral force, slip angle, distance, normalised normal load.
- Reported sensitivity is extreme: side-slip angle and longitudinal slip are the **most influential** parameters, with wear rates varying by orders of magnitude across the range.

### What this buys you

1. **`h_τ` is derivable, not fitted from thin air.** You can write down the brush-model prediction of lateral wear-rate distribution across the tread as a function of τ, and use *that* as the link function. A physics-informed loss with actual physics in it.
2. **Strong signal.** If wear rate depends steeply on slip angle, then small toe errors produce large, detectable wear-pattern differences — precisely the regime where inferring toe *from wear* beats measuring toe *geometrically*. This is an argument that your indirect route may be **more sensitive** than the direct one.
3. **A real theoretical section for the paper.** Derive the wear-rate-vs-slip-angle relation from the brush model, show your learned link function recovers it from data, and you have a genuine physics-informed-ML contribution rather than a regulariser with a nice name.

Camber gets similar treatment: camber thrust produces lateral force and asymmetric normal pressure, both of which feed the same wear model.

**Action:** read the two DiVA theses on brush-model tyre wear in Phase 0. They are free, thorough, and directly usable. This is the highest-value reading in your literature review.

---

## 6. Positioning against cross-task consistency literature

Cross-task consistency is an established family:

- Zamir et al., *Robust Learning Through Cross-Task Consistency*, CVPR 2020 — the canonical reference.
- Nishi et al., *Joint-Task Regularization for Partially Labeled Multi-Task Learning*, CVPR 2024 — directly relevant, handles exactly your "abundant labels for task A, scarce for task B" setting.
- Broad semi-supervised consistency-regularisation literature.

**Reposition accordingly.** Your contribution is not "we used a consistency loss." It is:

> **"Where cross-task consistency is usually enforced through learned cross-task mappings, we derive the link function from tyre contact mechanics. The constraint is therefore falsifiable and interpretable: the residual has physical meaning, and we show it identifies *temporal* discrepancies between a vehicle's current geometry and its accumulated wear history."**

That last clause — the disagreement diagnostic — remains, after this entire research pass, **the most novel thing in the project.** Nothing found in the search does it. Protect it.

---

## 7. Revised novelty audit

Replaces the table in `README.md`. Honest grades.

| # | Claim | Verdict | Grade |
|---|---|---|---|
| 1 | FTIR footprint imaging for tyres | **Prior art** (Chodera; Cabrera 2017) | ✗ Not novel |
| 1b | FTIR **in the road plane, in motion, inverted to estimate unknown alignment** | No prior art found | ✓ **Novel** |
| 2 | Zero-blur imaging via the rolling constraint | Physically obvious once stated; no paper found that states it | ~ Novel framing, weak claim. Use as *justification*, not contribution. |
| 3 | Laser as train-time teacher, RGB at inference | Cross-modal distillation is established; this application is not | ✓ **Novel application** |
| 4 | Consistency loss coupling wear to alignment | Mechanism is prior art; **brush-model-derived link function is not** | ✓ **Novel as reframed (§5)** |
| 4b | **Disagreement diagnostic** (recent vs chronic misalignment) | Nothing comparable found | ✓✓ **Strongest claim** |
| 5 | TWI bars as in-image metric ruler | No paper found doing this. Simple, but genuinely useful. | ✓ **Novel, small** |
| 6 | GRIP-Roll dataset | No comparable public dataset exists | ✓✓ **Solid, durable** |
| 7 | Conformal intervals on tyre safety decisions | Standard method, new application | ~ Good practice, not a contribution |
| 8 | **Beating the ±1.5 mm image-based SOTA by 4×** | Verifiable, quotable | ✓ **Strong if achieved** |

**Headline claims for the abstract, in order:** 4b (disagreement diagnostic) → 8 (accuracy vs image-based SOTA) → 6 (dataset) → 1b (in-motion inverse FTIR).

**Drop from the abstract:** 2 and 7. Keep them in Methods where they belong.

---

## 8. Patent landscape — what to avoid claiming

You are doing academic research, so freedom-to-operate is not a blocker. But do not claim novelty for these:

| Patent | Claims |
|---|---|
| US 11820178, US 10352688, US 8621919, US 11421982 | Drive-over ramp, laser pattern projected onto tyre, deformation → tread depth, velocity correction |
| US 11698250 / US 12467747 (Hunter) | Rolling wheel aligner using drive-direction calculation from calibrated cameras + gravity from inclinometer, no-stop positioning |
| US 10670392 / US 11408732 / US 10508907 | Wheel aligner, advanced diagnostics, no-stop positioning |
| US 6219134 | Rolling runout compensation |
| DE 19705047A1 | Laser-illuminated tread depth measurement |

**Note especially the Hunter drive-direction patents.** Your per-pass travel-direction estimation (`02_RIG_BUILD.md §3.4`) is conceptually adjacent. Cite them, describe the difference (they use clamp-on targets and side-mounted cameras; you use the footprint centroid track from below, target-free), and do not claim it as novel.

---

## 9. Reading list — Phase 0 priority order

**Tier 1 — read fully, week 1–2**

1. Cabrera Carrillo et al., *Optimization of an Optical Test Bench for Tire Properties Measurement and Tread Defects Characterization*, Sensors 2017, 17(4), 707. **Your single most important reference.** — https://doi.org/10.3390/s17040707
2. Salminen, *Parametrizing tyre wear using a brush tyre model*, KTH — https://kth.diva-portal.org/smash/get/diva2:802101/FULLTEXT01.pdf
3. *Analysis of tyre wear using the expanded brush tyre model*, DiVA — https://www.diva-portal.org/smash/get/diva2:854657/FULLTEXT01.pdf
4. Furferi, Governi, Volpe, Carfagni, *Design and Assessment of a Machine Vision System for Automatic Vehicle Wheel Alignment*, 2013 — https://journals.sagepub.com/doi/10.5772/55928
5. *Automatic and Accurate Vision-Based Measurement of Camber and Toe-In Alignment of Vehicle Wheel*, IEEE Access 2022 — https://ieeexplore.ieee.org/document/9926077/

**Tier 2 — read for method grounding**

6. Zamir et al., *Robust Learning Through Cross-Task Consistency*, CVPR 2020
7. Nishi et al., *Joint-Task Regularization for Partially Labeled Multi-Task Learning*, CVPR 2024 — https://openaccess.thecvf.com/content/CVPR2024/papers/Nishi_Joint-Task_Regularization_for_Partially_Labeled_Multi-Task_Learning_CVPR_2024_paper.pdf
8. Cao et al., *Rank Consistent Ordinal Regression (CORAL)* — https://arxiv.org/html/2111.08851v5
9. Angelopoulos & Bates, *A Gentle Introduction to Conformal Prediction*
10. *Hybrid Synthetic Data Generation with Domain Randomization for Part Inspection* — https://arxiv.org/html/2512.00125v1
11. *Deep learning-based instance segmentation for detection of tire tread area*, 2025 — https://www.sciencedirect.com/science/article/abs/pii/S2950425225000325
12. *A Flexible Wheel Alignment Measurement Method via APCS-SwinUnet and Point Cloud Registration* — https://www.mdpi.com/2673-8244/6/1/4
13. *Camber Angle Inspection for Vehicle Wheel Alignments*, Sensors — https://ncbi.nlm.nih.gov/pmc/articles/PMC5336001
14. Modelling wear of truck tyres under high slip, *Vehicle System Dynamics* 2025 — https://www.tandfonline.com/doi/full/10.1080/00423114.2025.2520489
15. wisetrue95/Tire — *Efficient Tire Wear and Defect Detection* (code) — https://github.com/wisetrue95/Tire

**Tier 3 — skim for context:** the patents in §8, UVeye product documentation, Michelin/Les Schwab wear-pattern guides, Kaggle dataset cards.

---

## 10. Changes to make elsewhere in this repo

- [ ] `01_CONCEPT.md §3` — add the interface-film requirement; correct the "rubber is index-matched to glass" claim
- [ ] `02_RIG_BUILD.md` — add PPF/PET film to the BOM; revise the week-1 go/no-go test to use tyre rubber, not a thumb
- [ ] `02_RIG_BUILD.md §5` — add "footprint has no contrast" failure mode → interface film
- [ ] `README.md` — replace the novelty table with §7 of this document
- [ ] `01_CONCEPT.md §6` — replace the heuristic `h_τ` with the brush-model derivation
- [ ] `06_EVALUATION.md` — add ±1.5 mm (QBurst) as an explicit baseline row
- [ ] `07_ROADMAP.md` — add "read Tier-1 papers" to weeks 1–2; add film procurement to week 1
