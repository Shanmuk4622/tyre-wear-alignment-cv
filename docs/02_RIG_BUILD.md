# 02 — The Rig: Hardware, Optics, Calibration

> Build this before you write a single line of model code. **The rig is the project.** A mediocre model on excellent data beats an excellent model on mediocre data, every time.

---

## 0. Build in three tiers

Do **not** try to build the final rig first. Build v0 this week.

| Tier | Cost (₹) | Time | What it proves |
|---|---|---|---|
| **v0 — Cardboard** | ~1,500 | 2 days | Geometry works. Phone under a glass sheet on bricks, roll a *bicycle wheel* over it by hand. |
| **v1 — Bench** | ~12,000 | 2 weeks | FTIR works, laser works, stitching works. Single spare wheel on a jig, pushed by hand. |
| **v2 — Drive-over** | ~45,000 | 4 weeks | Real car, real speed, real data. |

**v0 exists to kill the project cheaply if the idea is wrong.** Spend two days on it. If you can't see grooves clearly through glass with a phone, you need to know that in week 1, not week 12.

---

## 1. Bill of materials

### Tier v1 — Bench rig (build this properly, it generates most of your data)

| Item | Spec | Qty | ₹ (approx) | Notes |
|---|---|---|---|---|
| Camera | Raspberry Pi Global Shutter Camera (IMX296, 1456×1088, C/CS mount) | 1 | 4,500 | Global shutter is still worth it for the laser channel |
| Lens | 6 mm C-mount, f/1.4, low distortion | 1 | 1,200 | Wider = more FOV but more distortion; calibrate it |
| Host | Raspberry Pi 5 8 GB + 128 GB A2 microSD | 1 | 9,000 | Or a laptop with a USB3 camera |
| Glass | Low-iron toughened, 10 mm, 400×300 mm, **polished edges** | 1 | 1,800 | Polished edges are essential for LED injection — specify this |
| **Interface film** | Clear polyurethane PPF or PET, 75–150 µm, 1 m² | 1 | 400 | **Mandatory — see `09_RELATED_WORK.md §3`.** Bare rubber on bare glass gives no FTIR contrast. Sacrificial, replace when scratched. |
| IR LED strip | 850 nm, 5 m, 12 V, high density | 1 | 900 | Two strips along the long polished edges |
| IR-pass filter | 850 nm bandpass, screw-in for lens | 1 | 600 | Blocks ambient visible light |
| Line laser | 650 nm, 5 mW, 90° fan, focusable | 1 | 400 | **Class 3R — see safety §6** |
| Aluminium extrusion | 20×20 mm V-slot, 2 m + brackets | 1 set | 2,500 | Frame, camera mount, laser mount |
| Matte black flock | Self-adhesive sheet, 1 m² | 1 | 400 | Line the enclosure. Stray light is your enemy |
| Calibration target | ChArUco board, A3, printed on rigid foam-board | 1 | 300 | Print flat and check with a ruler |
| Tread depth gauge | Digital, 0.01 mm resolution | 1 | 900 | Your ground-truth instrument. Buy a good one |
| Misc | Wiring, 12 V PSU, fasteners | — | 1,500 | |
| | | **Total** | **~24,000** | |

### Tier v2 additions — Drive-over

| Item | Spec | Qty | ₹ | Notes |
|---|---|---|---|---|
| Glass | Low-iron toughened laminated, **19 mm**, 1200×400 mm | 1 | 9,000 | Must take vehicle load. Do not economise here |
| Steel frame | Welded, flush-mount ramp, load-rated 1.5 t/wheel | 1 | 12,000 | Local fabricator |
| Second camera | Same as above, second wheel of axle | 1 | 5,700 | Unlocks thrust angle |
| Edge compute | Jetson Orin Nano 8 GB dev kit | 1 | 35,000 | Optional — a laptop works for the capstone |
| Trigger | IR break-beam pair | 2 | 1,200 | Starts capture, measures speed |
| Air knife / blower | 12 V | 1 | 1,500 | Clears water/grit from the plate |

> **Budget reality check.** You can produce a strong capstone at v1 only (~₹24k) using a spare wheel on a jig, plus a *few* real-car passes over a v0-grade plate laid on level ground. Do not let v2 cost block progress. The jig data (see `03_DATA.md`) is better labelled than real-car data anyway.

---

## 2. Optical layout

```
              tyre rolling  ──────►  +x (travel)
     ═══════════════════════════════════════════   glass plate (z = 0)
      ▲IR         ▲IR         ▲IR         ▲IR       ← 850 nm injected at edges
     ═══════════════════════════════════════════
                        │
        ╱ laser fan     │  camera optical axis (vertical, +z)
       ╱                │
   [LASER]           [CAMERA]
   ◄────── b = 200 mm ──────►
                        │
                        │  z₀ = 150 mm standoff
                        ▼
```

### The triangulation arithmetic — do this before buying

Disparity per millimetre of groove depth:

```
Δu / Δz  ≈  f · b / z₀²          [pixels per mm]
```

| f (px) | b (mm) | z₀ (mm) | px per mm | mm per px |
|---|---|---|---|---|
| 1400 | 80 | 250 | 1.8 | 0.56 ← **useless** |
| 1400 | 150 | 200 | 5.3 | 0.19 |
| 1400 | 200 | 150 | 12.4 | **0.081** ← target |
| 1400 | 250 | 120 | 24.3 | 0.041 ← better, but FOV shrinks |

With Gaussian sub-pixel peak fitting on the laser line (routinely 0.1 px), the last two rows give **~0.008 mm** effective precision. Absurdly better than you need — which is exactly right for a *teacher* signal.

**Trade-off to be aware of:** shrinking `z₀` shrinks the field of view. At `z₀ = 150 mm` with a 6 mm lens on a 1/2.9" sensor you cover roughly 140 mm laterally — enough for one tyre width on a small car, tight on an SUV. Either accept the crop, or use two cameras side by side, or step back to `z₀ = 200 mm` and accept 0.19 mm/px. **Measure your actual FOV in v0 before committing.**

### FTIR light injection

- Sand and polish both long edges of the plate to optical clarity (a glass shop will do this — ask for "polished edge, C-grind").
- Mount the LED strip flush against the edge with index-matching optical gel or clear silicone. Air gaps kill coupling efficiency.
- Wrap the whole underside enclosure in matte black flock.
- **Laminate the clear interface film onto the top face.** Without it, black tread rubber absorbs the coupled light instead of scattering it back, and you get no contrast. This is not optional — see `09_RELATED_WORK.md §3`.

#### The revised go/no-go test (v0, week 1)

The thumb test alone is **not sufficient**. A fingertip is soft, pale and moist; it scatters beautifully and will pass even if the concept fails on rubber. Run all four steps:

| Step | Do | Expect |
|---|---|---|
| 1 | Edge-light the glass, press your thumb | Bright fingerprint → *the rig works* |
| 2 | Press **black tread rubber cut from a scrap tyre**, bare glass | Probably little/no contrast |
| 3 | Lay clear PPF/PET film on the glass, press the same rubber | Bright footprint → *the concept works* |
| 4 | Compare 2 vs 3 quantitatively (histogram, Michelson contrast) | Document the difference |

| Outcome | Action |
|---|---|
| Step 3 gives clear contrast | Proceed with film. Expected. |
| Step 2 alone gives usable contrast | Better — no film needed. **Report it; it contradicts prior practice.** |
| Neither works | Fall back to flood-lit ground view + laser. Still novel, still viable. Rewrite `01_CONCEPT.md §3`. |

Cost: one afternoon, ~₹600 of film. **Do this before ordering the v1 camera.**

### Channel separation

You need three signals from one camera. Options, in order of preference:

1. **Temporal multiplexing (recommended).** Cycle the illumination frame-by-frame at 120 fps: `[FTIR] → [laser] → [flood IR] → repeat`. Gives you 40 fps per channel, perfectly registered, one camera. Drive the LEDs and laser from Pi GPIO synced to the camera's frame-sync (XVS) pin.
2. **Spectral separation.** 850 nm FTIR + 650 nm laser + dichroic splitter and two cameras. More expensive, no temporal offset.
3. **Two cameras, independent.** Simplest to build, hardest to register.

Go with (1). It is elegant, cheap, and the temporal offset at 8 km/h is 23 mm of travel between channels — recoverable exactly, because you know the speed.

---

## 3. Calibration — do all four, in this order

### 3.1 Intrinsics

Standard ChArUco / checkerboard, OpenCV `calibrateCamera`. **Capture ≥30 views** covering all corners of the frame and a range of tilts.

```bash
conda activate cv_conda
python scripts/calibrate_intrinsics.py --images data/calib/intrinsics/ --board charuco_5x7_35mm
```

Acceptance: reprojection RMS **< 0.3 px**. If not, your target isn't flat or your images are blurry. Print on foam-board, not paper.

### 3.2 Plate-plane pose

Lay the ChArUco flat *on the glass*. Solve PnP. This defines world `z = 0` and gives you a metric px→mm homography for the plate surface.

Acceptance: measure a known 100 mm distance on the plate through the homography, error **< 0.5 mm**.

### 3.3 Laser-plane calibration

The laser fan is a plane in camera coordinates. To find it:

1. Place the ChArUco at ≥6 different known heights above the plate (use gauge blocks or precision-cut spacers).
2. At each height, the laser line intersects the known board plane; extract line points, back-project to 3D.
3. Fit a plane to the union of all 3D points → laser plane `π_L: n·X + d = 0`.

Then any laser pixel back-projects to a ray, which you intersect with `π_L` for a metric 3D point. Standard structured-light calibration.

Acceptance: measure a machined step of known height (a stack of feeler gauges works) — error **< 0.05 mm**.

### 3.4 Travel-axis alignment — the one everyone forgets

Your toe measurement is *relative to the rig's `+x` axis*. If the rig's axis is not parallel to the vehicle's actual direction of travel, every toe reading is biased by that error.

Fix it:
- Machine/mark the plate frame with a precise reference edge.
- Determine actual travel direction per-pass from the **tracked motion of the tyre footprint centroid** across frames, not from the frame geometry. Fit a line to the centroid track; that line *is* the travel direction for that pass.
- This makes the measurement self-referencing and removes rig-alignment bias entirely. **Do this — it is nearly free and it removes an entire error source.**

Acceptance: roll the jig wheel set to exactly 0° toe. Measured toe should be **0 ± 0.1°** across 20 passes. If there's a constant offset, you have a systematic bias — find it before collecting data, not after.

---

## 4. Capture software

```
capture/
  ├─ trigger.py         # break-beam → arm → record → speed estimate
  ├─ illum_sync.py      # GPIO channel cycling, synced to camera XVS
  ├─ record.py          # ring buffer, saves raw frames + metadata JSON
  └─ qc.py              # immediate quality check; reject bad passes on the spot
```

**`qc.py` is the most important file here.** After every pass it must check and report:

- [ ] Footprint detected in ≥ 15 frames
- [ ] Sharpness (variance of Laplacian) above threshold in the contact band
- [ ] Laser line detected and unbroken across the tyre width
- [ ] Speed within 4–15 km/h
- [ ] Plate surface clean (no false FTIR blobs before the tyre arrives)
- [ ] Exposure not clipped (< 0.1% saturated pixels)

Print a big green PASS or red FAIL. **Nothing wastes a data-collection day like discovering at home that 200 passes were out of focus.**

### Metadata to record for every single pass

Store as JSON alongside the frames. This is your dataset's real value.

```json
{
  "pass_id": "2026-08-14_1432_p017",
  "vehicle": {"make":"Maruti","model":"Swift","year":2019,"odometer_km":48210},
  "wheel": "front_left",
  "tyre": {"brand":"MRF","model":"ZLX","size":"185/65R15",
           "dot_code":"3419","load_index":88,"speed_rating":"H"},
  "inflation_kpa": 220,
  "measured_depth_mm": {"positions": "12 clock x 5 lateral", "values": [[...]]},
  "alignment_gt": {"source":"jig","toe_deg":0.35,"camber_deg":-1.20,
                   "uncertainty_deg":0.05},
  "capture": {"speed_kmh":8.4,"fps":120,"frames":58,"coverage_pct":52,
              "surface":"dry","ambient_lux":320,"temp_c":31},
  "operator": "shanmukesh",
  "qc": {"passed": true, "sharpness": 412.7, "laser_ok": true}
}
```

---

## 5. Common failure modes and their fixes

| Symptom | Likely cause | Fix |
|---|---|---|
| **Footprint has no contrast at all — contact and non-contact both dark** | **No interface film.** Black rubber absorbs the coupled light instead of scattering it | **Laminate clear PPF/PET on the plate.** `09_RELATED_WORK.md §3` |
| Footprint brightness drifts during a long static test | Plastic film creep/hysteresis (documented in Cabrera 2017) | Rolling contact loads each point for only ~50 ms, so this should be minor — measure it and report |
| Film scratches quickly | Grit on tyres | It's sacrificial. Buy a roll, replace weekly. Budget for it. |
| FTIR image is grey mush, no contrast | Ambient light leaking in | Flock the enclosure; add IR-pass filter; test at night first |
| Fingerprint test fails | Poor edge coupling, or plate too thick/thin | Polish edges properly; use optical gel; try 8–10 mm glass |
| Grooves invisible in flood channel | Glass surface dirty or scratched | Clean with IPA; replace plate; it *will* get scratched by grit |
| Laser line washed out | Ambient IR / sunlight | 650 nm narrow bandpass filter on that frame; capture indoors or at dusk |
| Laser line broken across black rubber | Rubber absorbs 650 nm strongly | Increase laser power to 5 mW (not more); lengthen exposure on laser frames only |
| Stitched map has seams / duplicated features | Speed estimate wrong | Estimate rolling speed from tread feature tracking, **not** from vehicle speed — slip is real |
| Toe reading has constant offset | Rig axis bias | Use per-pass travel direction from footprint track (§3.4) |
| Depth predictions collapse to the dataset mean | Class-imbalanced tread depths | Stratify sampling by depth bin; use the ranking loss |
| Model works on jig, fails on real cars | Domain gap — load, suspension compliance, dirt | Collect real-car data early and continuously, not at the end |

---

## 6. Safety — read this before you switch anything on

**Laser.** A 5 mW 650 nm line laser is Class 3R. It will not blind you instantly but it can damage vision on direct beam entry. Rules:
- Mount so the beam is **always** pointing upward into an enclosed cavity, never at eye level.
- Interlock: laser off unless the plate is covered or a wheel is present.
- Wear 650 nm laser safety glasses during alignment and calibration work.
- Put a warning label on the enclosure.

**Glass under a vehicle.** A 19 mm laminated toughened plate on a properly supported steel frame handles a passenger-car wheel load (~400 kg). But:
- **Never** let a vehicle onto v0/v1 glass (10 mm, unlaminated). It will shatter.
- Support the plate on its full perimeter, on a compliant gasket, never on point loads.
- Use **laminated** glass in v2 so a failure holds together rather than dropping a wheel into your camera.
- Test with sandbags to 1.5× rated load before any car goes near it.

**Vehicle movement.** Data collection means a moving car near a person operating a camera. Have a second person. Agree hand signals. Never stand in the path. Chock the wheels between passes.

**Electrical.** 12 V is safe but outdoors + water is not. Use an IP-rated enclosure and an RCD if mains-powered.

---

## 7. Build checklist

```
Week 1  — v0
  [ ] Glass sheet + phone + bricks assembled
  [ ] Fingerprint FTIR test passes           ← GO / NO-GO for the whole concept
  [ ] Bicycle wheel rolled over, grooves visible in video
  [ ] Frames extracted, one crude stitch attempted

Week 2-3 — v1
  [ ] Frame built, camera mounted rigidly
  [ ] Edge polish + LED injection working
  [ ] Intrinsics calibrated, RMS < 0.3 px
  [ ] Plate homography verified < 0.5 mm
  [ ] Laser plane calibrated, step test < 0.05 mm
  [ ] Illumination channel cycling synced
  [ ] qc.py written and actually used

Week 4-6 — jig + first data
  [ ] Alignment jig built and protractor-calibrated (see 03_DATA.md)
  [ ] Zero-toe bias test passes: 0 ± 0.1° over 20 passes
  [ ] 200 jig passes captured with full metadata
  [ ] Unrolled tread map pipeline produces clean output

Week 7+ — v2 (only if v1 data looks good)
  [ ] Load-tested plate
  [ ] Frame installed flush
  [ ] Break-beam trigger
  [ ] First real-car pass
```

Everything in this repo runs under `conda activate cv_conda`.
