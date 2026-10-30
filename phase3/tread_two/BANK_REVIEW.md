# ATLAS v1 — training-only reference-bank review

**29 October 2026. Review decision: keep the deterministic prototype bank for
this one mechanism pilot; freeze it before fitting.** This is an engineering
appearance review of existing training material, not human groove/physical-wear
annotation. No validation/test image or model score chooses replacement cases.

The actual [contact sheet](prototype_report/TREAD_TWO_Atlas_Concept.png) and
[source-case CSV](pilot/assets/prototype_bank.csv) were inspected. The bank has
18 real patches, six per proxy class, from one source photo per original train
tyre. Independent source support is **1 low / 3 mid / 4 high**.

| Cases | Visible concern / interpretation | v1 treatment |
|---|---|---|
| Low 0–5 | Visible grooves, plus white/blue/green paint or marker lines in several cases | Retain actual pixels and expose the artifact risk; do not call colour evidence physical wear |
| Mid 6–11 | Strong variation in contrast, blur and stretched/diagonal detail | Retain; mask coverage is not a sharpness/groove-quality label |
| High 12–17 | Low contrast, glare/texture variation and partial support in some cases | Retain support/clipping statistics and all failures; no reliable-depth interpretation |

This choice keeps the pilot's mechanism and source provenance simple and avoids
polishing an explanation gallery against old held-out outcomes. It does **not**
mean the 18 cases are optimal or artifact-free. Positive and negative examples
remain actual model inputs and can be inspected individually.

The frozen bank SHA is
`374445591bec46bcc7da45223ad73d18da25034cafdf0c5e0b61fb79ac78918c`.
Exact float32 native pixels and all pixel-addressed CSV values are retained.
The eight bank photographs remain support inputs but are excluded as classifier
queries; same-tyre dependence remains. There is no query-label-conditioned bank.

All selected/final endpoint images receive predeclared grayscale-both,
brightness ×0.8/×1.2 and top-signed-patch occlusion diagnostics. Their raw scores,
patch/reference evidence and exact replacement values are retained. These
probes can expose sensitivity; they cannot prove shortcut elimination or
causal wear reasoning. A later bank change requires a versioned new protocol,
not an in-place edit during this fit. No additional tyre photos are required.
