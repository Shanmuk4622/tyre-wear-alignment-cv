# 02 — Camera Rig, Illumination and Calibration

> Build this before writing model code. **The rig is the sensor.** An excellent model on badly-lit, uncalibrated data will lose to a mediocre model on good data, every time.

---

## 0. Three tiers — build v0 this week

| Tier | Cost (₹) | Time | Proves |
|---|---|---|---|
| **v0 — Phone on a brick** | ~1,500 | 2 days | Viewpoint works. Can you see grooves, shoulders and a TWI bar at all? |
| **v1 — Bench rig** | ~22,000 | 2–3 weeks | Photometric stereo + calibration + repeatable capture |
| **v2 — Vehicle-mounted** | ~35,000 | 3–4 weeks | Real cars, real conditions, real dataset |

**v0 exists to fail cheaply.** Two days. If you cannot see a TWI bar in a phone photo of a tyre lit from the side, you need to know in week 1.

---

## 1. Geometry of the mount

```
                              ╭───────────╮
                         ╭────╯           ╰────╮
                        ╱      TYRE (front)     ╲
                       │   ← tread crown →       │
                       │  shoulder     shoulder  │
        ───────────────┴─────────────────────────┴──────── ground
                    ╲    ↖ field of view ↗    ╱
                     ╲                       ╱
                      ╲    [CAMERA]         ╱
                       ╲   +4 LEDs         ╱
                    low, ahead of the wheel,
                    aimed backward + slightly upward
```

**Requirements:**

| Parameter | Target | Why |
|---|---|---|
| Both shoulders in frame | Mandatory | Alignment is refused without them |
| Full tread width in frame | Mandatory | Lateral wear profile needs edge-to-edge |
| Elevation angle | 10–25° upward | Too flat → severe foreshortening; too steep → shoulders leave frame |
| Standoff | 300–600 mm | Closer = better mm/px, but FOV and depth-of-field shrink |
| Mount rigidity | **Absolute** | Any flex changes the calibration and biases toe |

> **The image border is not a reference.** Vertical and travel direction come from calibration only. Write this on the rig if you have to.

---

## 2. Resolution budget — do this arithmetic before buying

From `01_CONCEPT.md §5`: resolving a 0.3 mm sipe at 3 px requires **≤ 0.1 mm/px**.

| Sensor | Horizontal px | FOV 250 mm | FOV 150 mm | FOV 100 mm |
|---|---|---|---|---|
| 1080p (1920) | 0.130 mm/px | 0.078 mm/px | 0.052 mm/px |
| 8 MP (3264) | 0.077 mm/px | 0.046 mm/px | 0.031 mm/px |
| 12 MP (4056) | 0.062 mm/px | 0.037 mm/px | 0.025 mm/px |

**Conclusions:**

- 1080p across a full 250 mm tread is **not enough** for sipes. It is fine for grooves, shoulders and wear patterns.
- **Recommended: ≥8 MP**, or accept that fine structures are resolved only in a cropped centre region.
- **State the achieved mm/px in every results table.** A detail claim without a spatial resolution is unfalsifiable.

Also check **depth of field**: a doubly-curved tyre spans 50–100 mm in depth. At f/2.8 and 400 mm standoff DoF is tight — stop down to f/5.6–f/8 and add light rather than opening the aperture.

---

## 3. Bill of materials

### v1 — bench rig

| Item | Spec | Qty | ₹ | Note |
|---|---|---|---|---|
| Camera | Raspberry Pi HQ (IMX477, 12 MP) **or** global-shutter 8 MP USB3 | 1 | 6,000 | **Global shutter required if the wheel rotates** |
| Lens | 12–16 mm C-mount, low distortion | 1 | 1,800 | Match to standoff; verify FOV before buying |
| Host | Raspberry Pi 5 8 GB + 128 GB A2 card | 1 | 9,000 | Or a laptop with USB3 |
| **LEDs for photometric stereo** | 4 × 10 W white COB, individually GPIO-switchable | 4 | 1,600 | 90° apart around the lens, 30–45° elevation |
| **Polarising film** | Linear, A4 sheets | 2 | 500 | One on the lights, one on the lens, crossed |
| LED driver | 4-channel constant current + MOSFET gates | 1 | 900 | Must switch fast enough to strobe |
| Diffusers | Frosted acrylic squares | 4 | 300 | Softens hard shadows without killing directionality |
| Calibration target | ChArUco A3 on rigid foam board | 1 | 300 | **Must be flat** — verify with a straightedge |
| Tread depth gauge | Digital, 0.01 mm | 1 | 900 | Your reference instrument. Buy a good one |
| Grey card | 18% reflectance | 1 | 200 | Flat-field correction |
| Extrusion + brackets | 20×20 V-slot, 2 m | 1 set | 2,000 | Rigid frame |
| Matte black flock | 1 m² | 1 | 400 | Kill stray reflections |
| **Total** | | | **~23,900** | |

### v2 additions — vehicle-mounted

| Item | Spec | ₹ | Note |
|---|---|---|---|
| Enclosure | IP65, impact-resistant | 3,000 | Road debris, water |
| Mounting bracket | Custom, bolted to a jack point or bumper | 4,000 | Local fabricator |
| Power | 12 V automotive → 5 V buck, fused | 1,500 | Never tap the ECU harness |
| Line laser (optional) | 650 nm, 5 mW, focusable | 400 | **Training-time teacher only** — see §7 |
| Edge compute (optional) | Jetson Orin Nano 8 GB | 35,000 | A laptop is fine for the capstone |

> **Budget reality:** a strong capstone is achievable at v1 (~₹24k) plus a phone-grade v2 for real-vehicle captures. Don't let the Jetson block progress.

---

## 4. Illumination — the highest-value subsystem

Read `10_VISION_TECHNIQUES.md §2–3` first. Summary of what to build:

### Photometric stereo array

```
        LED_N (0°)
            │
LED_W ──[CAMERA]── LED_E        4 LEDs, 90° apart
   (270°)   │      (90°)         elevation 30–45°
        LED_S (180°)             all at equal radius from the lens
```

Capture sequence per inspection, GPIO-strobed:

```
frame 1: LED_N only      frame 4: LED_W only
frame 2: LED_E only      frame 5: all LEDs (flat reference)
frame 3: LED_S only      frame 6: all LEDs, cross-polarised
```

Six frames in ~120 ms at 50 fps. Solve `I_i = ρ(n·l_i)` for the normal map `n` and albedo `ρ`.

**Critical constraint: the tyre must be static during the burst.** Either capture stationary, or strobe fast enough that inter-frame motion is sub-pixel, or register the burst before solving. Verify this in v0.

**Calibrate the light directions.** Photograph a matte white sphere (a ping-pong ball works) under each LED; the specular highlight position gives the light direction. Do this once and store it with the calibration.

### Cross-polarisation

Polariser sheet over the LEDs, second polariser on the lens rotated 90°. Verify by pointing at a wet tyre: highlights should vanish. Capture both polarised and unpolarised — the **difference image isolates the specular component**, which is a useful gloss/wetness feature in its own right.

### Flat-field correction

Once per session, photograph an 18% grey card filling the frame under each illumination condition. Store as `flatfield_<condition>.npy`. Divide every subsequent frame by it. This removes vignetting and lighting non-uniformity that otherwise **looks exactly like shoulder wear**.

---

## 5. Calibration — four steps, in order

### 5.1 Intrinsics

ChArUco, OpenCV `calibrateCamera`, **≥30 views** covering all frame corners and a range of tilts.

```bash
conda activate cv_conda
python scripts/calibrate_intrinsics.py --images data/calib/intrinsics/ --board charuco_5x7_35mm
```

**Acceptance: reprojection RMS < 0.3 px.** If not, the target isn't flat or the images are blurred. Foam board, not paper.

### 5.2 Extrinsics — camera to world

Place the ChArUco flat on the ground in a known orientation relative to the vehicle travel axis. Solve PnP for `[R|t]`.

**Acceptance:** a known 100 mm ground distance reconstructs within 0.5 mm.

### 5.3 Light-direction calibration

Sphere method above. **Acceptance:** re-solving a known convex object's normals gives < 5° angular error.

### 5.4 Travel-axis verification — the one everyone gets wrong

**Toe is measured relative to the vehicle travel axis. Any error in that axis biases every toe reading by the same amount.**

Two defences:

1. **Static:** establish the axis from calibration against a marked straight line on the floor.
2. **Dynamic (preferred):** estimate travel direction per-clip from the **tracked motion of the tyre across frames**, not from the mount. Fit a robust line to the motion track. This makes the measurement self-referencing and removes mount bias entirely.

**Acceptance test:** set the jig to exactly 0° toe. Measured toe must be **0 ± 0.2°** across 20 clips. A constant offset means systematic bias — find it *before* collecting 400 tyres, not after.

### Calibration hygiene

- Store a **calibration version ID with every clip.** Non-negotiable.
- Recalibrate after any lens, focus, or mount change.
- Re-run the zero-toe test weekly. If it drifts, the mount is not rigid enough.

---

## 6. Capture software

```
capture/
├── record.py        # burst capture, LED sequencing, metadata
├── illum.py         # GPIO strobe control synced to frame sync
├── flatfield.py     # grey-card capture and correction
└── qc.py            # immediate quality check — reject on the spot
```

### `qc.py` is the most important file here

After every capture it must check and print a large PASS or FAIL:

- [ ] Tread mask coverage above threshold
- [ ] **Both shoulders visible** (alignment refused otherwise)
- [ ] Focus: variance of Laplacian above threshold
- [ ] Exposure: < 0.1% clipped pixels, either end
- [ ] Specular area below threshold (or wetness flagged)
- [ ] Photometric-stereo burst: inter-frame motion sub-pixel
- [ ] ≥ N frames accepted for registration
- [ ] TWI bar detected (flag if not — scale anchor unavailable)

**Nothing wastes a collection day like discovering at home that 60 tyres were out of focus.**

### Metadata for every clip

```json
{
  "clip_id": "2026-09-03_1412_t047_c02",
  "tyre_id": "t047",
  "vehicle": {"make":"Maruti","model":"Swift","year":2019,"odometer_km":48210},
  "wheel": "front_left",
  "tyre": {"brand":"MRF","model":"ZLX","size":"185/65R15","dot":"3419"},
  "inflation_kpa": 220,
  "load_state": "empty",
  "gauge_depth_mm": {"stations": 6, "grooves": 4, "values": [[...]]},
  "alignment_gt": {"source":"jig","toe_deg":0.35,"camber_deg":-1.20,"unc_deg":0.05},
  "capture": {"fps":50,"frames":180,"mm_per_px":0.062,
              "illum":["N","E","S","W","flat","xpol"],
              "surface":"dry","ambient_lux":320,"temp_c":31},
  "calibration_id": "calib_2026-09-01_v3",
  "qc": {"passed":true,"focus":412.7,"shoulders_visible":true,"twi_found":true}
}
```

---

## 7. Optional metric-depth extension

If the project needs defensible millimetre depth, add **one** of:

| Option | Precedent | Cost |
|---|---|---|
| Line laser + triangulation | [Wang et al. 2019](https://doi.org/10.1177/1687814019837828), **<0.2 mm** | ₹400 + calibration |
| Second camera (stereo) | — | ₹6,000 |
| RGB-D sensor | [Shi et al. 2026](https://doi.org/10.3390/metrology6010004), **<0.1°** alignment | ₹15,000+ |

**Use it as a training-time teacher, not a deployment sensor.** Supervise the RGB model with metric depth, then ship RGB-only. That is the cheap-deployment story *and* a clean ablation (`06_EVALUATION.md`).

**Laser safety:** 5 mW 650 nm is Class 3R. Never at eye level. Interlock it off unless a tyre is present. Wear 650 nm eyewear during alignment. Label the enclosure.

---

## 8. Failure modes

| Symptom | Cause | Fix |
|---|---|---|
| Grooves invisible, image flat and grey | Flat/ring illumination | **Photometric stereo.** This is the whole point of §4 |
| Bright blown-out patches | Specular reflection | Cross-polarise; check wetness flag |
| One shoulder consistently darker | Uneven lighting | Flat-field correction |
| Sipes not resolvable | Insufficient mm/px | Crop tighter or use a higher-res sensor (§2) |
| Photometric normals noisy/wrong | Non-Lambertian rubber, shadows in grooves | ≥4 lights, robust/RANSAC normal fit, cross-polarise |
| Normals smeared | Tyre moved during burst | Strobe faster, capture stationary, or register the burst |
| Toe has a constant offset | Travel-axis bias | Per-clip motion-derived axis (§5.4) |
| Toe drifts week to week | Mount flexing | Stiffen the mount; recalibrate; re-run zero-toe test |
| Model great on val, bad on new tyres | Frame-level split leak, or brand shift | **Tyre-level grouped splits**; unseen-brand test set |
| Tread skewed when wheel rotates | Rolling shutter | Global-shutter sensor |

---

## 9. Build checklist

```
Week 1  — v0
  [ ] Phone + side lighting + one tyre, photographed
  [ ] Can you see: grooves? shoulders? a sipe? a TWI bar?     ← GO / NO-GO
  [ ] Measure achieved mm/px against a ruler in frame
  [ ] 4-position hand-torch photometric test: do grooves pop?
  [ ] Start the gauge study: measure 4 tyres, record with date

Week 2-3 — v1
  [ ] Frame built, camera rigidly mounted
  [ ] 4-LED array + drivers, GPIO strobing verified
  [ ] Cross-polarisation verified on a wet tyre
  [ ] Intrinsics calibrated, RMS < 0.3 px
  [ ] Extrinsics + ground plane, 100 mm within 0.5 mm
  [ ] Light directions calibrated, < 5° error
  [ ] Flat-field capture routine working
  [ ] qc.py written and actually used

Week 4-5 — jig + pilot
  [ ] Alignment jig built, verniers verified to 0.05°
  [ ] Zero-toe bias test: 0 ± 0.2° over 20 clips
  [ ] Pilot: 30-50 unique tyres captured with full metadata
  [ ] Photometric normal maps look sane on real tyres

Week 6+ — v2
  [ ] Enclosure + vehicle mount
  [ ] Real-vehicle clips
  [ ] Weekly calibration re-verification
```

Everything runs under `conda activate cv_conda`.
