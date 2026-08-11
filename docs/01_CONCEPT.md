# 01 — Concept, Physics, and Mathematics

> Read `README.md` first. This document is the technical justification for every design choice.

---

## 1. Restating the problem properly

Your original brief: *"detect whether a car tyre is worn out, and whether the wheel alignment is correct, while the car is in motion."*

Written as an estimation problem, that is:

Given a short video sequence `V = {I_1 … I_T}` of one wheel captured during a rolling pass, estimate:

| Symbol | Quantity | Units | Difficulty |
|---|---|---|---|
| `d(θ, w)` | tread depth as a function of circumferential angle θ and lateral position w | mm | **hard** — sub-mm regression |
| `c ∈ C` | wear-pattern class | 8-way | medium |
| `γ` | camber angle | degrees | **hard** — needs ~0.2° |
| `τ` | toe angle | degrees | **very hard** — spec is ~±0.1° |
| `p` | inflation state proxy | normalised | easy |
| `a` | damage / anomaly flags | binary set | medium |

Note the honesty in that table. Toe tolerance on a passenger car is often **±6 arcminutes (0.1°)**. A vision system that claims to replace a ₹25-lakh Hunter alignment rack is not credible. GRIP is positioned as a **screening instrument**: it flags "this vehicle needs an alignment check" with high recall, and it does so on every car that drives in, for free, with zero driver effort. That is a genuinely useful product *and* an honest scientific claim. See `08_RISKS_AND_MY_OPINION.md`.

---

## 2. The rolling constraint and why blur disappears

For a wheel of radius `R` rolling without slip at forward speed `v`, the velocity of a point on the tyre surface at angle `φ` from the contact point is:

```
|v_surface(φ)| = 2·v·sin(φ/2)
```

At `φ = 0` (contact point): `0`.
At `φ = π` (top of tyre): `2v`.

Blur length for exposure `t_e` at the contact patch is therefore `≈ 0` regardless of vehicle speed. Verified numerically:

```
10 km/h, 1000 µs exposure → 5.56 mm blur at top of tyre,  ~0 mm at contact patch
30 km/h, 1000 µs exposure → 16.7 mm blur at top of tyre,  ~0 mm at contact patch
```

**Consequence:** you do not need a global-shutter camera, a xenon strobe, or a 2000 fps sensor. A ₹4,500 Raspberry Pi Global Shutter camera at 120 fps is sufficient. Every rupee saved on the sensor goes into optics and lighting, which matter far more.

**Caveat to be rigorous about:** the zero-blur zone is a *band* around the contact point, not a point. Within ±20 mm of contact, surface speed is under 6% of `v` — still effectively sharp. Outside that band blur grows fast. So the model must be trained on, and only trust, the near-contact band. Enforce this with a mask derived from the FTIR channel.

---

## 3. FTIR contact imaging

### The optics

A glass plate of refractive index `n₁ = 1.52` in air (`n₂ = 1.0`) has critical angle:

```
θ_c = arcsin(n₂/n₁) = arcsin(1/1.52) = 41.1°
```

Light injected into the plate edge at angles beyond `θ_c` propagates by total internal reflection and never leaves the top face. Where a higher-index material physically contacts the top face, the TIR condition collapses, light couples out of the glass, scatters, and a fraction returns downward into the camera.

> ### ⚠ Correction — you need an interface film. Do not skip this.
>
> An earlier draft of this document assumed bare tread rubber on bare glass would work, because rubber is roughly index-matched to glass (`n ≈ 1.52`). **That is only half the requirement**, and the half it satisfies is not the important one.
>
> FTIR produces a *bright* footprint only if the contacting material **scatters light back down** toward the camera. Carbon-black tread rubber couples light in efficiently (high index — good) and then **absorbs it** (bad). Light goes in and does not come back. Meanwhile non-contact regions keep TIR intact, so nothing exits there either. **Both regions read dark: almost no contrast.**
>
> This is exactly why the validated prior art ([Cabrera et al., *Sensors* 2017](https://doi.org/10.3390/s17040707)) interposes a **clear plastic lamina** between tyre and glass rather than using the rubber directly.
>
> **Fix:** laminate a thin (75–150 µm) optically clear, sacrificial film onto the top of the plate. Automotive **paint-protection film (PPF)** is close to ideal — clear polyurethane, `n ≈ 1.50–1.56`, abrasion-resistant, self-healing, designed for road grit, replaceable, ~₹400/m².
>
> Full analysis and the revised go/no-go test: `09_RELATED_WORK.md §3`.

### What you get

| Region | Appearance in FTIR channel |
|---|---|
| Tread rib carrying load | Bright, intensity ∝ contact pressure |
| Groove void | Black (air gap, TIR intact) |
| Sipe | Black hairline |
| Contact patch boundary | Razor-sharp edge |
| Everything not touching the glass | Black |

This is an **enormously** higher-SNR signal than trying to segment tread from a normally-lit photo. You get near-perfect groove masks for free, which then anchor everything downstream.

### Alignment signatures in the footprint

| Condition | FTIR signature |
|---|---|
| Negative camber | Pressure centroid shifted toward inner shoulder; inner edge brighter |
| Positive camber | Centroid shifted outward |
| Toe-in / toe-out | Footprint sheared into a trapezoid; leading edge asymmetry; rib edges show directional intensity gradient |
| Under-inflation | Footprint longer, shoulders bright, centre dim |
| Over-inflation | Footprint short and narrow, centre bright |
| Static load imbalance | Footprint area asymmetry front/rear |

Formally: define the footprint intensity field `F(x, y)`. Extract
- lateral first moment `μ_w = Σ w·F / Σ F` → camber proxy
- principal-axis angle of `F` relative to travel direction → toe proxy
- longitudinal/lateral second-moment ratio → inflation proxy

These become **analytic features fed to the network as a prior**, not replaced by it. See §6.

### Practical warnings

- **Water and mud also frustrate TIR.** A wet plate produces false bright regions. Mitigations: air knife across the plate, hydrophobic coating, and a "surface-clean" classifier that aborts the pass. For the capstone, collect indoors/dry and list wet operation as future work — this is defensible.
- Use **low-iron toughened glass**, ≥10 mm, with **polished edges** for LED injection.
- Surround the plate underside with matte black flock to kill stray light.
- Use **850 nm IR LEDs** + IR-pass filter on the camera. Ambient sunlight has far less 850 nm energy than visible, and drivers won't see a glowing plate.

---

## 4. Recovering toe and camber geometrically

### Setup

Camera below, optical axis vertical, intrinsics `K` known from calibration. The glass plane is the world `z = 0` plane. The rig's long axis defines world `+x` = direction of travel. This is the crucial advantage: **the rig defines the vehicle reference frame**, so we never need to estimate it.

### Step 1 — wheel plane from the bead/rim ellipse

The tyre bead seat is a circle of radius `r_b` lying in the wheel mid-plane. Under perspective projection a 3D circle images as an ellipse. Fitting the conic `C` and back-projecting gives the classic two-fold-ambiguous solution for the supporting plane normal `n`:

```
Q = Kᵀ C K              (conic in normalised camera coords)
eigendecompose Q → λ₁ ≥ λ₂ > 0 > λ₃
n ∝ [ ±√((λ₁-λ₂)/(λ₁-λ₃)), 0, √((λ₂-λ₃)/(λ₁-λ₃)) ]  (in eigenbasis)
```

The two solutions are mirror images about the optical axis. **Disambiguate using motion**: track the ellipse across frames as the wheel translates; only one hypothesis produces a consistent rigid translation along `+x`. This is a clean, publishable use of the rolling constraint.

### Step 2 — angles

With unit normal `n = (n_x, n_y, n_z)` in the rig frame (x = travel, y = lateral, z = up):

```
toe    τ = arctan( n_x / n_y )      (deviation of wheel plane from travel direction, in ground plane)
camber γ = arcsin( n_z )             (tilt of wheel plane from vertical)
```

Sign conventions: define toe-in positive, negative camber = top of wheel leaning inward. **Write these down once and never change them.** Half the bugs in alignment work are sign errors.

### Step 3 — the practical problem, and the fix

From directly below, the rim is largely occluded by the tyre itself. You will often see only the **sidewall silhouette** near the contact patch. Three complementary estimators, fused:

| Estimator | Signal | Strength | Weakness |
|---|---|---|---|
| **E1 — sidewall silhouette** | Left/right sidewall bulge outlines; a cambered wheel makes them asymmetric | Always visible | Deformable, load-dependent |
| **E2 — FTIR footprint moments** | Pressure asymmetry and shear | Highest SNR, sub-pixel | Confounded by load & pressure |
| **E3 — tread-groove direction flow** | Circumferential grooves are straight lines in the tyre frame; their projected direction reveals the wheel plane | Geometric, principled | Needs clean groove segmentation (which FTIR gives you) |

Fuse with a small learned head that takes all three plus their covariances. **Do not throw all of this away and regress angles end-to-end from pixels.** Analytic-prior-plus-learned-residual will beat end-to-end here by a wide margin, and it is far easier to defend in a viva.

### Step 4 — separating the confounds

Camber, load, and inflation all shift the footprint. You cannot disentangle them from one wheel alone. Two fixes:

1. **Capture both wheels of an axle** (two cameras, or two plates). Load transfer is symmetric; camber is not. Differencing cancels the confound.
2. **Multi-pass consistency.** The same car, same load, driven over the plate twice — geometry repeats, noise doesn't.

Also: capturing both wheels of an axle unlocks **thrust angle**, which is arguably the single most useful alignment number and which no single-wheel system can produce. Strongly recommended for the final rig.

---

## 5. Tread depth: the metrology problem

### Why this is the hard part

New passenger tyre: 8 mm. Legal minimum: 1.6 mm. Useful resolution: **0.3 mm**. That is a 4% change in a quantity you are inferring from shading in a photo. Naïve RGB→mm regression will not get there. Four independent measures:

#### (a) Laser-line triangulation as the ground-truth teacher

A 650 nm line laser mounted at baseline `b` from the camera, projecting a fan across the tyre just *ahead* of the contact patch. Groove depth `Δz` displaces the imaged laser line by

```
Δu = f · b · Δz / (z₀ · (z₀ + Δz))  ≈  f · b · Δz / z₀²
```

With `f ≈ 1400 px`, `b = 80 mm`, `z₀ = 250 mm`: `Δu ≈ 1.79 px per mm`. That is too coarse. Increase baseline to `b = 200 mm` and shorten `z₀` to 150 mm → `Δu ≈ 12.4 px/mm`, i.e. **0.08 mm per pixel**, and with sub-pixel peak fitting (Gaussian centroid on the laser profile, typically 0.1 px) you reach **~0.01 mm**. Comfortably good enough to *supervise* the model.

> **Design rule: maximise baseline, minimise standoff.** Do the arithmetic before you buy anything. A spreadsheet cell is cheaper than a rebuild.

**The trick:** the laser is a *training-time* sensor only. You train an RGB→depth network with the laser profile as dense supervision, then at deployment you ship RGB-only. This is cross-modal distillation, it is cheap, and it is exactly the kind of thing reviewers like.

#### (b) TWI bars as an in-frame ruler

Every road-legal tyre has **Tread Wear Indicator** bars moulded into the main grooves at exactly **1.6 mm** above the groove floor, marked on the sidewall by a small triangle / "TWI" / brand logo. Find them, and you have an absolute metric reference *inside the image*, immune to scale drift, camera height error, and tyre-size variation.

Pipeline: detect TWI bar → measure its apparent height against the adjacent rib surface → that difference *is* 1.6 mm minus remaining depth... no, more precisely: `remaining_depth = 1.6 + height_of_rib_above_TWI_top`. Calibrate every prediction against this. Free, universal, self-calibrating. Novelty claim #5.

#### (c) Relative / ordinal learning instead of absolute regression

Absolute mm regression from a single image is badly conditioned. Pairwise ranking is not. Train a Siamese head:

```
L_rank = BCE( σ(f(x_a) − f(x_b)), 1[d_a > d_b] )
```

Labelling "A is more worn than B" is trivial and near-noise-free, whereas labelling "A is 4.3 mm" requires a gauge and careful technique. Train the ranker on abundant pairs, then **anchor the learned scale to millimetres with a small calibrated set** (isotonic regression from ranking score → mm). This is how you get precision out of a modest dataset. This is probably the single highest-leverage modelling decision in the project.

#### (d) Shape-from-shading in the groove

Under controlled, known illumination, groove-floor irradiance falls off predictably with depth (a deeper groove is more occluded from the light source). With the FTIR channel giving you an exact groove mask, and a fixed illuminator geometry, the *mean intensity inside the groove mask* is a strong monotone depth cue. Cheap to compute, add it as an explicit input channel.

### The unrolled tread map

As the wheel rolls across the plate, successive frames sample successive circumferential sections. Stitch the near-contact band from each frame into a single **unrolled tread map**: a `(circumference × width)` image at ~80–170 µm/px.

Numbers (verified):

```
195/65R15 → 634 mm diameter → 1.99 m circumference
glass plate 1.0 m  → 51% of circumference per pass, 2 passes for full coverage
glass plate 1.5 m  → 77% per pass
at 10 km/h, 120 fps → 23 mm of travel per frame → ~6× overlap on a 150 mm contact patch
```

**Recommendation: 1.2 m glass plate, 120 fps, target 6–10 km/h.** Two passes gives full 360° coverage with margin. Report per-pass coverage as a dataset field.

This unrolled map is the representation everything downstream operates on. It is metric, blur-free, illumination-normalised, and directly comparable across tyres and across time. It is also a very good figure for the paper.

---

## 6. The physics-consistency loss

The heart of the research contribution.

Let the network produce a depth map `D̂(θ, w)` over the unrolled tread and geometry estimates `(τ̂, γ̂)`.

Define the **lateral wear gradient** as the slope of depth across tread width, averaged circumferentially:

```
g_w = (1/Θ) Σ_θ  ∂D̂(θ,w)/∂w     evaluated as a least-squares fit across w
```

Tyre-contact mechanics says: sustained camber produces a monotone lateral wear gradient of the same sign, with magnitude increasing in `|γ|`. Model it as a monotone link `h_γ` (a small monotone MLP, or a fitted affine function — start affine):

```
L_camber_consistency = || g_w − h_γ(γ̂) ||²
```

### Toe: a derived link function, not a heuristic

Toe deserves better than "feathering correlates with toe," and the tyre-mechanics literature provides it.

> **A wheel with static toe angle τ, driving in a straight line, is permanently operating at slip angle α = τ.**

Toe is not merely *associated* with wear — toe **is** a continuously applied slip angle. That means the standard tyre-wear models apply directly:

- The **brush tyre model** treats tread elements as elastic bristles and gives sliding velocity and frictional work across the contact patch as an explicit function of slip angle.
- **Archard-type abrasion**: wear volume ∝ frictional work = ∫ (shear stress × sliding distance) over the patch.
- Empirically, `W = f(A_b, F_y, α, S, F_N/F_N0)`, and side-slip angle is among the *most* influential parameters — wear rate varies by orders of magnitude across the range.

So `h_τ` can be **derived from the brush model** rather than fitted blind. Two consequences:

1. The consistency loss becomes a genuine physics-informed constraint with a falsifiable functional form — you can check whether the learned link recovers the predicted one.
2. Because wear rate depends *steeply* on slip angle, small toe errors produce large wear-pattern differences. **Inferring toe from accumulated wear may be more sensitive than measuring it geometrically.** That is a real argument for the indirect route, and worth testing explicitly.

Reading: `09_RELATED_WORK.md §5` and the two brush-model theses listed there.

Empirically, sustained toe produces **feathering**: rib edges become sharp on one side and rounded on the other. Quantify with a directional edge-asymmetry statistic on the unrolled map:

```
A = Σ_ribs [ |∇⁺ D̂| − |∇⁻ D̂| ]  /  Σ_ribs [ |∇⁺ D̂| + |∇⁻ D̂| ]
L_toe_consistency = || A − h_τ(τ̂) ||²
```

Total:

```
L = L_depth + α·L_pattern + β·L_geometry + λ·(L_camber_consistency + L_toe_consistency)
```

### Why this is a real contribution, not decoration

1. **It is a semi-supervised mechanism.** `L_consistency` needs *no alignment label at all* — it only needs the two heads to agree. So you can train on thousands of unlabelled passes and still improve the alignment head.
2. **It is falsifiable.** Ablate `λ = 0` and report the degradation. If it doesn't help, you report that honestly and it's still a finding.
3. **It gives explainability for free.** When the two heads *disagree*, that is itself informative: geometry says aligned, wear says misaligned → *the car was recently realigned but the old damage remains.* Geometry says misaligned, wear says fine → *the misalignment is recent.* No other system can make that distinction. **This is a genuinely novel diagnostic capability and I would build a whole section of the paper around it.**

That third point is the best idea in this document. Make it a headline.

---

## 7. Wear-pattern taxonomy (your label set)

| Class | Appearance on unrolled map | Usual cause |
|---|---|---|
| `uniform` | flat depth profile | healthy |
| `centre_wear` | centre ribs shallow, shoulders deep | chronic over-inflation |
| `shoulder_wear` | both shoulders shallow, centre deep | chronic under-inflation |
| `one_side_wear` | monotone lateral gradient | camber misalignment |
| `feathering` | rib edges asymmetric, sawtooth cross-section | toe misalignment |
| `cupping` | periodic scalloping around circumference | worn damper / imbalance |
| `flat_spot` | one localised low-depth patch | hard braking / skid |
| `patch_wear` | irregular localised patches | imbalance, bent rim |

Plus independent binary damage flags: `cut`, `bulge`, `embedded_object`, `crack`, `exposed_cord`.

Note that the classes are **not mutually exclusive** — use multi-label sigmoid outputs, not softmax. Real tyres carry two or three at once. Every public tyre dataset gets this wrong by using binary good/defective; being the first to model it properly is a small but real contribution.

---

## 8. Uncertainty — non-negotiable for a safety claim

A number without an interval is not a measurement. Use **split conformal prediction**:

1. Train the model on the training split.
2. On a held-out calibration split, compute residuals `|d_true − d̂|`.
3. Take the `⌈(n+1)(1−α)⌉`-th smallest residual as `q̂`.
4. At test time, output `[d̂ − q̂, d̂ + q̂]`.

This gives a **distribution-free finite-sample coverage guarantee** of `1−α` under exchangeability, with no assumptions about the network. It is five lines of code, it is rigorous, and it makes the safety claim defensible. Do the same for `τ̂` and `γ̂`.

Then define the decision rule in terms of the interval, not the point estimate:

```
if upper_bound(depth) < 3.0 mm      → REPLACE
elif lower_bound(depth) < 3.0 mm    → INSPECT MANUALLY (model is unsure)
else                                → OK
```

That "I don't know" branch is what separates a research demo from something a workshop would actually install.

---

## 9. Design decisions log

Keep this table updated. Examiners love it and it saves you from re-litigating settled questions at 2 a.m.

| Decision | Chosen | Rejected | Why |
|---|---|---|---|
| Viewpoint | Below, ground-embedded | Wheel-arch, dashcam | Zero blur at contact; rig defines reference frame; FTIR possible |
| Contact medium | Toughened low-iron glass, FTIR-lit | Open pit, no glass | FTIR footprint is the highest-SNR signal available |
| Illumination | 850 nm IR edge injection + IR-pass filter | Visible flood | Sunlight immunity; invisible to driver |
| Camera | Global shutter, 120 fps, 1.6 MP | High-speed 1000 fps | Rolling constraint removes the need |
| Depth GT | Laser triangulation at capture | Manual gauge only | Dense, automatic, sub-0.05 mm |
| Depth model | RGB → depth, laser-distilled | Ship the laser | Cheaper deployment, better paper |
| Depth objective | Ordinal ranking + isotonic anchoring | Direct mm regression | Far better conditioned on small data |
| Alignment model | Analytic prior + learned residual | End-to-end regression | Interpretable, data-efficient, defensible |
| Pattern head | Multi-label sigmoid | Softmax multi-class | Real tyres show multiple patterns |
| Uncertainty | Split conformal | Softmax confidence | Distribution-free guarantee |
