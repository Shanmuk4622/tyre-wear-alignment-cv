# Phase 3 — repository map

## TREAD-TWO implementation addition — 29 October 2026

| Location | Ownership |
|---|---|
| [tread_two/](tread_two/README.md) | Original concept/prototype plus the implemented ATLAS pilot and structured guides |
| [tread_two/pilot/](tread_two/pilot/FILES.json) | Isolated frozen source/native-bank inventory; no edits to prior experiments/workstations |
| [NB08](notebooks/phase3_NB08_TREAD_TWO_ATLAS_Resumable_Pilot.ipynb) | Self-contained Kaggle delivery; six code cells, actual T4/HF gates, one resumable measured fit |
| [pilot_validation/](tread_two/pilot_validation/HANDOFF.json) | Final source/notebook/local-acceptance/stop/extracted-worker receipts; large local fixture directories are ignored |
| [Pilot implementation guide](tread_two/PILOT_IMPLEMENTATION.md) | Actual modules, two-GPU ownership, state machine and direct evidence flow |

The runtime uses existing data only. Both workstation trees and the prior
selected models remain preserved; actual measured fitting is pending Kaggle.

<!-- NB07 CURRENT -->
## NB07 source / output ownership — 29 October 2026

| Path | Purpose |
|---|---|
| `data_gate/` | Seven CPU audit/report/publisher modules + two JSON source/policy contracts |
| `data_intake/` | Three empty collection CSVs, existing-clip resolution control, version marker and incoming native-file folder |
| `notebooks/phase3_NB07_TREAD_ONE_Data_And_Protocol_Gate.ipynb` | New self-contained Kaggle CPU audit, not training |
| `data_gate_outputs/csv/`, `figures/` | Stable verified copies of 22 current CSVs and two figures; [index](data_gate_outputs/CSV_INDEX.md) |
| `data_gate_outputs/runs/`, `cache/`, `publication/` | Retained local versioned results, completed-image/download cache and immutable direct publication preparation |
| `data_gate_validation/` | Unit/CSV/notebook/fresh-CPU/offline-publication receipts; synthetic/fresh fixtures excluded from scientific counts |
| `analysis/` | NB07 builder, targeted checks, replay/handoff validators and documentation exporter |
| `csv_tools/` | Local Artifact Tool CSV round-trip verifier; no Kaggle dependency |

Original datasets, frozen 27 experiment sources and executed NB05/NB06 are preserved.
<!-- END NB07 CURRENT -->

Inspected: 27 October 2026. Existing paths are preserved.

| Location | What belongs here / entry points |
|---|---|
| `prototype/` | Active native Tread Station. `app.py`/`theme.py`: UI; `engine.py`/`classification.py`: inference; `phase2_adapter.py`: current-model switch; `learned_geometry.py`: point overlays; `alignment.py`/`alignment_ui.py`: calibrated-target workflow; `video.py`/`media_export.py`: media; `evidence.py`: records |
| `prototype/Launch Tread Station.cmd` | Existing Windows launcher; activates `cv_conda` |
| `prototype/.vendor/`, `.cache/`, `checkpoints/`, `results/` | Isolated dependencies, cached artifacts, preserved original weights and local outputs; not new experiment namespaces |
| `phase2/` | Separate extraction, labels, dataset release, training, notebook builders, audits and progress |
| `phase2/phase2_training_data.py`, `phase2_models.py` | Phase 2 data/split/sampling and model contracts |
| `phase2/phase2_worker.py`, `phase2_run.py`, `phase2_hub.py` | Optimizer-update recovery, process scheduling, HF persistence/ownership |
| `phase2/phase2_report.py`, `phase2_build_training_notebooks.py` | Reporting/export selection and self-contained notebook generation |
| `phase2/notebooks/` | NB00–NB07 plus historical three-worker YOLO migration copies; not the original-study notebook sequence |
| `phase2/workstation/` | `ACTIVE` marker, immutable export registry, selected model files, validation records, enable/restore launchers and backups |
| `phase2/data/`, `annotation_work/`, `review/`, `manifests/`, `releases/` | Prepared sources, raw labels, galleries, provenance and verified release ZIPs |
| `phase2/completion/`, `completion_seed3/` | Earlier partial audit and final 15-run completion evidence |
| `tyrelib/` | Original study runtimes and notebook generators, including HRNet, matched SegFormer, S5, reporting and original visual replay |
| `notebooks/` | Original NB00–NB33, recovery variants and NBT1; `execution_archives/` preserves earlier execution/error provenance |
| `scripts/` | Dataset/mask preparation, shortcut probes, audit/fault checks and report builders/verifiers |
| `docs/` | Numbered concepts/protocols/audits 00–37, index, repository/status guidance, logbook and Git policy |
| `docs/report/` | September 15 report source, generated report, claims/reproducibility/checklists, figures and pinned small evidence |
| `ACCESS_latex_template_20260513/` | Newer IEEE Access manuscript: `main.tex`, `sections/`, bibliography, analysis/build utilities, PDF/source ZIP and verification records |
| `ACCESS_latex_template_20260513/.build/` | Earlier extracted/build-check manuscript copies; compare against maintained source before interpreting them |
| `outputs/` | Local annotation packages, diagnostics, preview/report/presentation artifacts; not automatically remote training truth |
| `Videos/` | Three workshop videos and sample images; Phase 2 training-cohort sources |
| `tyre_study/` | Legacy directory, empty at the inspected top level; do not assign it a current runtime role |
| `.chart-data-*` | Existing chart artifact folders; preserve until their provenance is needed |
| Root `README.md`, `PROGRESS.md`, `undertand3.md` | Entry point, long historical progress, and earlier September 19 understanding; later Phase 2/paper records supersede parts |
| `ENVIRONMENT.md`, `environment.yml` | Local `cv_conda` requirements; historical proposed directory layout and version expectations are not an exact current Kaggle lock |
| `.env` | Local secret file; its contents were not read |
| `.git`, `.gitignore`, `LICENSE`, `CITATION.cff` | Version history, content policy, code licence and citation metadata |
| `phase3/` | Current understanding, instructions, trackers and orientation audit snapshots; future Phase 3 work starts here |
| `phase3/runtime/` | New custom TREAD-ONE model/data/loss/evaluation/worker/publisher/coordinator sources and exact reused Phase 2 data contracts |
| `phase3/notebooks/`, `releases/` | Generated NB00 CPU preflight, NB01 dual-T4 custom training/evaluation and compact Kaggle ZIP |
| `phase3/validation/` | Local model, full-size fresh-process seam, fault/lifecycle and delivery evidence; fixture outputs are not 60-epoch study results |
| `phase3/completion_20261028/` | Pinned completed-run audit, exact executed-notebook copies, legacy evidence downloaded read-only, direct recovered CSV/schema, derived plots and new review validation |
| `phase3/completion_20261028/nb02_publication/` | Exact successful NB02 archive, publication receipt, downloaded immutable 25-file package and independent verification; raw files ignored in Git but retained locally |
| `phase3/analysis/` | Separate read-only audit/reanalysis/direct-CSV recovery-publication utilities; original r1 training runtime is preserved |
| `phase3/diagnostic/` | Separate NB03 inference/intervention/gradient, direct-file persistence and standalone explorer sources; imports the frozen r1 model/preprocessing |
| `phase3/diagnostics/` | Diagnostic delivery/check receipts, immutable local complete units/snapshots, native maps, explorer and derived summaries; raw run trees are Git-ignored but preserved locally |
| `phase3/diagnostics/nb03_publication/` | Exact executed NB03 archive, pinned 27-file publication, read-only verification and environment comparison; raw archive/downloads ignored in Git |
| `phase3/training/` | Separate NB04 engineering logger, native/CSV transaction store, worker, publisher, coordinator and report; frozen r1 math imported without edits |
| `phase3/training_gates/` | Compact acceptance/delivery/fault/control receipts and durable figures/viewer; large per-contract cases, native states and snapshots ignored in Git but retained locally |
| `phase3/training_gates/nb04_publication/` | Preserved executed NB04, both pinned HF namespaces/source fixtures, cached objects and independent verification; large raw files ignored in Git |
| `phase3/revision2/` | Separate NB05 paired-pilot specification, shared-gradient projector, full-cohort worker, transactional journal, bounded immutable publisher, automatic acceptance and report |
| `phase3/revision2_validation/` | Compact delivery/offline/actual-model checks and later figures; large temporary engineering cases ignored in Git |
| `phase3/revision2_runs/` | Local new-protocol engineering or measured run state/journal/CSV/native snapshots; kept separate from r1 and NB04, Git-ignored |

## Where to make later changes

`revision2_validation/nb06_publication/` preserves executed NB06 and the
independently verified 35-file Kaggle review under `remote/`, with compact
publication/hash/comparison receipts alongside it. The original clean delivery
and fresh-CPU acceptance stay in their earlier folders. See
[completion](phase3NB06Completion.md) and the
[planned next data/protocol gate](phase3ProspectiveDataGate.md).

Completed NB05 evidence is cached under `revision2_validation/nb05_publication/`:
`remote/` and content-addressed `objects/` retain all pinned final files;
`preflight/` retains native/CSV source proofs; `analysis/` holds ten derived
CSV tables, figures and the complete comparison HTML. Large raw caches and
HTML are Git-ignored locally; compact verification receipts and CSV/figures
remain visible. `review_bundle/` is the prepared additive HF review.
`nb06_delivery/` and `nb06_validation/` distinguish clean payload, fresh CPU
acceptance and later live execution. [NB06 guide](phase3NB06Review.md).
Empty prospective templates live in [data_intake/](data_intake/README.md).

NB05's notebook-only lease handling is readable at
[analysis/nb05_launcher.py](analysis/nb05_launcher.py) and embedded by its builder.
It is separate from the 27 frozen experiment files. The supplied failed NB05,
inspection and regression receipts are in `revision2_validation/nb05_repair/`;
see the [repair record](phase3NB05Repair.md).

App behavior: inspect the active prototype adapter/engine/UI. Phase 2 training behavior: inspect Phase 2 sources and regenerate only the relevant Phase 3 deliverable. Original-study reporting: preserve its original namespace and endpoints. Newer paper prose: use `main.tex`/`sections/`; older report prose: edit `manuscript.source.md` and rebuild the generated report.

Do not move notebooks, datasets, evidence or model caches as an orientation cleanup. Existing relative references and source hashes are part of reproducibility.

## After NB07 completion — 29 October 2026

| Path | Current role |
|---|---|
| `data_gate_validation/nb07_publication/` | Exact executed NB07 archive, public-read pinned downloads, notebook/CSV/hash/visual comparison receipts |
| `collection_review/` | Separate local review builder and JavaScript export rules; completed gate source unchanged |
| `collection_review_outputs/` | 44-preview HTML, 18-image contact sheet, three planning CSVs plus unresolved control, hashes/schema; local preparation |
| `data_gate_validation/collection_review/` | Ten review/export checks and isolated synthetic validation files, excluded from scientific data |

See [NB07 completion](phase3NB07Completion.md) and [review guide](phase3CollectionReview.md).

## TREAD-ONE Studio — separate app, 29 October 2026

| Path | Role |
|---|---|
| `phase3/workstation/` | New native app, private model/Qt, launch script, own data/cache; [guide](workstation/README.md) |
| `phase3/workstation/models/` | Exact copied declared seed-3/epoch-20 export and frozen architecture identity |
| `phase3/workstation/data/` | This app's session CSV, completed captures and settings |
| `phase3/workstation_validation/` | 47-file original baseline, model/UI/native-window proof, previews and isolated engineering fixtures |
| `prototype/`, `phase2/workstation/` | Original application/selected registry retained byte-for-byte in the recorded baseline |

[Integration and ownership](phase3TreadOneWorkstation.md). Acquisition-dependent
future plans are superseded for current work by the user's existing-data-only scope.
