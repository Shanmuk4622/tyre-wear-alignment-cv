# 11 — Optional Camera-Connected Application

> **Scope note:** the app is an engineering deliverable, not a research contribution. It is **first on the cut list** (`07_ROADMAP.md`). Build it only after the pipeline and evaluation are done. This document exists so the design is ready when there is time — not to justify starting it early.

---

## 1. What it is for

The app connects to the rig camera, triggers an inspection, and presents the **evidence** — not a verdict alone. Its purpose is to make the system's reasoning and its uncertainty legible to a non-expert.

**Design principle:** every number is shown with its interval, and `UNABLE_TO_MEASURE` is displayed as prominently as any result. An app that always produces a confident answer would undo the honesty the rest of the project is built on.

---

## 2. Architecture

```
┌─────────────┐   RTSP / USB / WebSocket   ┌──────────────────┐
│  Rig camera │ ─────────────────────────► │  Inference host  │
│  + LEDs     │ ◄───────────────────────── │  (laptop/Jetson) │
└─────────────┘   trigger, LED sequence    └────────┬─────────┘
                                                    │ REST + WebSocket
                                           ┌────────▼─────────┐
                                           │  App (Flutter    │
                                           │  or React PWA)   │
                                           └──────────────────┘
```

**Recommended split:** models run on the **host**, never on the phone. The phone is a viewer and controller.

Reason: the app cannot reproduce the rig's photometric-stereo illumination, cross-polarisation or calibration. A phone-only inference mode is a **fundamentally weaker sensor** and must not present the same confidence numbers. If you build one anyway, **re-run conformal calibration separately for it and let the intervals be honestly wider** (`04_MODEL.md §10`).

---

## 3. Screens

| Screen | Contents |
|---|---|
| **Capture** | Live preview · alignment guides for framing · quality gate feedback *(focus / exposure / both shoulders visible)* · trigger button |
| **Result** | Verdict badge · tread condition per position · wear-pattern chips · camber & toe with intervals · circumference coverage % |
| **Evidence** | Segmentation overlay · wear heatmap · damage boxes · anomaly heatmap · landmark overlay — toggleable layers |
| **History** | Per-tyre timeline; wear trend over visits; "remaining life" estimate from the longitudinal fit |
| **Refusal** | Why the system could not measure, and what to change *(clean the tyre, reposition, better light)* |

### Result card

```
┌──────────────────────────────────────────────┐
│  ⚠  INSPECT                    coverage 62%  │
│                                              │
│  WEAR      significant  (class 2 of 3)       │
│    outer shoulder   ███░░░░  most worn       │
│    centre           ██████░                  │
│    inner shoulder   ███████                  │
│    pattern: outer_edge_wear · feathering     │
│                                              │
│  DAMAGE    1 crack detected (outer shoulder) │
│  ANOMALY   none flagged                      │
│                                              │
│  ALIGNMENT (screening — not a rack)          │
│    camber  −1.4°  ±0.4°     ⚠ out of range   │
│    toe     screening: LIKELY OUT OF SPEC     │
│            (AUROC-calibrated, not measured)  │
│                                              │
│  CROSS-CHECK                                 │
│    Geometry and wear agree → chronic         │
│    misalignment likely. Confidence: high.    │
│                                              │
│  ⓘ Screening only. Confirm with a gauge      │
│    and a wheel-alignment machine.            │
└──────────────────────────────────────────────┘
```

---

## 4. Non-negotiable UI rules

1. **Never show a point estimate without its interval.** Toe especially — display it as a screening verdict, not a number, unless the interval is genuinely tight.
2. **`UNABLE_TO_MEASURE` is a full-screen state**, not a greyed-out field.
3. **Always show circumference coverage.** "62% observed" prevents a user assuming the whole tyre was checked.
4. **Unseen tread is `unknown`, never `healthy`.** Render it as hatched grey, not green.
5. **A persistent disclaimer**: screening only, confirm with a gauge and an alignment machine.
6. **No generative fill, ever.** Do not inpaint unobserved tread to make a prettier visualisation.

---

## 5. Stack

| Layer | Choice | Why |
|---|---|---|
| Backend | FastAPI + Uvicorn | Same Python env as the models |
| Inference | ONNX Runtime / TensorRT | Portable, fast |
| Transport | REST for results, WebSocket for live preview | Simple |
| App | **Flutter** (or React PWA) | One codebase, Android + iOS. A PWA avoids store friction entirely |
| Storage | SQLite (local) or Postgres | Per-tyre history |
| Demo fallback | **Gradio on HF Spaces** | Upload a clip, get the report card — a working demo in an afternoon |

> **If time is short, build the Gradio Space and nothing else.** It demonstrates the same capability, takes an afternoon instead of three weeks, and is linkable from the report and the HF model card.

---

## 6. API sketch

```
POST /inspect              multipart clip or trigger live capture
  → { job_id }

GET  /inspect/{job_id}
  → { verdict, wear:{severity, per_position, patterns},
      damage:[...], anomaly:{score, regions},
      alignment:{camber:{value, lo, hi}, toe:{screening, auroc_band}},
      coverage_pct, quality:{...}, evidence_urls:{...},
      calibration_id, model_version }

GET  /tyre/{tyre_id}/history
  → [ { date, severity, depth_estimate, verdict } ]
```

Include `model_version` and `calibration_id` in **every** response. Without them you cannot reproduce or debug a field result six months later.

---

## 7. Build order

1. Gradio Space — upload clip → report card *(afternoon)*
2. FastAPI wrapper around the same inference code *(1–2 days)*
3. Live camera preview + quality-gate feedback *(2–3 days)*
4. Flutter/PWA result and evidence screens *(1 week)*
5. History and trend view *(2–3 days)*

**Stop after step 1 unless the research work is genuinely complete.**
