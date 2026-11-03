# Implementation sequence and acceptance matrix

**User asked for the complete plan. Model code/notebooks and measured training remain pending.** This file fixes what will be built and in which order. Engineering gates are automatic checks in implementation; they are not additional user permission requests.

## Planned isolated directory

```text
phase3/tread_two/geometry_first/
  plan/                  # delivered design documents and JSON proposals
  runtime/               # new model/data/objective/coordinator modules; pending
  validation/            # fixtures, receipts and acceptance; pending
  reports/               # actual future derived science; pending
phase3/notebooks/
  phase3_NB16_TREAD_TWO_GEO_Preparation_And_Contract_Gate.ipynb   # planned
  phase3_NB17_TREAD_TWO_GEO_Resumable_Pilot.ipynb                # planned
  phase3_NB18_TREAD_TWO_GEO_Completed_Review.ipynb               # conditional plan
```

| Order | Planned source owner | Deliverable / condition |
|---|---|---|
|1 |`data_geo.py`, `schema_geo.py` |Pinned split/annotation adapter, typed 40-family schema, exact field reductions |
|2 |`anchor_geo.py` |Strict a 5b8… checkpoint loader, original features/raw points exposed, zero-update equivalence |
|3 |`model_geo.py`, `atlas_geo.py` |Bounded adapters, quality head, detached atlas, encoder/context/masked head; shapes/parameter inventory |
|4 |`objective_geo.py`, `schedule_geo.py` |Loss sufficient statistics, separate gradients/clips, full-coverage deterministic 392 draw plan |
|5 |`store_geo.py`, `telemetry_geo.py` |Atomic units, native chunks/indexes, append-prefix and schema verification |
|6 |`worker_geo.py`, `evaluate_geo.py` |Two-GPU ownership, optimizer/evaluation barriers, selection and predeclared controls |
|7 |`hub_geo.py`, `coordinator_geo.py` |Versioned transport copy/private SDK, source/runtime binding, visible stop/resume/rate behavior |
|8 |Notebook builder +NB16 |Self-contained actual Kaggle preparation/contract gate and disposable seam |
|9 |NB17 builder |One measured .ipynb only after local and actual live checks pass |
|10 |Review/export +NB18 |Raw-file audit, every loss/geometry/latent calculation, interactive atlas and model-card status |

Keep new module names distinct from completed `pilot/`, `lc_pilot/`, `context_pilot/`, their reviews and both workstation trees. A future FILES.json freezes only the new runtime/config/schema and exact copied source identities. PLAN.json remains the design record, not the numeric measured contract.

## Required tests before measured fitting

| Area | Meaningful acceptance test |
|---|---|
|Data |386/81/103 roles, hashes, original ignore policy, human 70/weak 152/none 164 train point support, component consistency |
|Schedule |Many seeded epochs: all 386 primary IDs once,6 padding,49×8, within-batch uniqueness and≥2 components, identical replay |
|Warm start |Zero adapters reproduce anchor masks/points/boxes/features under matched CPU and matchedT4 math paths |
|Gradient isolation |Texture-only and error-only backward give zero geometry/anchor grads; geometry-only gives zero texture grads; group clipping independence |
|Point/mask math |Ordered bounds; correct no-target reduction; independent overlap labels; ignored cells excluded; reference eligibility preserved |
|Masking leakage |Change hidden tile pixels after encoding replacement: masked path's visible tokens/inputs unchanged; patch-local LC prevents boundary leakage |
|SSL |Constant-vector fixture triggers variance/collapse checks; covariance matches manual calculation; N<4/one-component skip preserves texture optimizer state |
|Loss accounting |Recompute every term from CSV/native arrays including ramp, covariance denominator, padding/mask availability and group counts |
|Error head |No weak-point target used as human error truth; targets detached; compare training-median baseline without new calibration promises |
|Atlas |Reconstruct exact grid from retained row parameters; empty/clipped/misaligned rows logged; no input/grid gradient path |
|Resume |Same-environment update 1→4 direct versus update 1 save/restore→4: parameters/moments/counters/RNG/CSV/native outputs equal except declared timing fields |
|Crash points |Before/after optimizer groups, CSV append, native flush, pointer swap, upload success-with-lost-response and evaluation chunk boundary |
|Storage |Actual chunk sizes, float precision roundtrip, active prefix copies, closed-file hard-link immutability, fallback-copy peak,≥2 GB reserve |
|HF |Live owned throwaway namespace, actual restore/continue, no oldnamespace writes, idempotent completed rerun and clear quota wait |
|Notebook |Fresh extracted setup in empty Kaggle-style directory, no hidden old variables,40-family source payload matches readable source |
|Preservation |All completed notebooks/catalogs, original 103M arkdown files and both workstation source hashes unchanged |

Disposable fixtures use declared training-only images and a separate namespace. Their updates/time/metrics are tagged `engineering_fixture`, excluded from the measured fit and never selected as the final model. Actual dual-T4 proof is required, not inferred from a localGTX1650 test.

## Freeze checklist and measured-run start

Before NB17 is delivered: executable dtypes/null handling and numerical reductions complete; all sources/checkpoint/input/overlay hashes bound; exact parameter counts measured; schema/native indexes and raw capacity projection pass; actual deterministic resume seam and live HF gate pass; complete training schedule/augmentation/mask seeding policy fixed; validation thresholds/selection/diagnostics fixed; Notebook instructions/stop/error recovery tested; all retention omissions explicit.

No source, objective, loss coefficient, sampler or logging family changes mid-fit. A necessary repair must preserve numeric intent/history or become a separately identified protocol; never update FILES.json just to bypass a source mismatch. Compatible completed run exits/display-only. New measured run creates initial checkpoint/source/environment/coverage/reference-mean records and publishes them before fitting.

## Notebook handoff and later review

User instructions must say acceleratorT4×2, Internet ON, exact expanded Kaggle input, authorized HF_TOKENSecret, measured free-space requirement, Run All order, expected phase messages, local/native/remote outputs, restart route and known failure actions. Return a compact self-contained .ipynb plus readable source/module docs. No training-script-only handoff.

After user execution, inspect the supplied executed notebook without regenerating it; preserve exact SHA/output; verify pinned public publication bytes/native/source/CSV prefixes; independently recompute geometry, quality, every SSL objective and collapse/selection rule; inspect figures/interactive images; update project trackers/model card and pass/fail capability flags. A failed research hypothesis is retained honestly and does not trigger another automatic fit.

## What happens immediately after this plan

Implement modules 1–5 and local fixtures first, then build NB16. This is the concrete next engineering milestone. NB17 is not yet runnable and no GPU fit should be started from this design-only folder. Workstation/application changes remain deferred. The existing completed ATLAS research hub stays intact and is contextual evidence for the new objective.
