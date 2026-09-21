# Phase 2 working understanding

> **Final status — 21 September 2026:** [All 15 runs and NB06 are verified complete](phase2Seed3Completion.md). Installed selections still match the final registry. Earlier partial/deferred/run-next statements below describe historical snapshots. No retraining or extra labeling is required.


## Runnable training revision — 20 September 2026

**NB00 has passed on Kaggle and its HF report is independently verified. NB01–NB06 are implemented; see [phase2RunNotebooks.md](phase2RunNotebooks.md) for the current run order and settings.**

The user delegated the remaining decisions. The runnable revision keeps all three original mid tyres and all video frames in training, avoiding overlap under every possible unknown mapping. The fixed split is **386 train / 81 validation / 103 test images (8/2/2 tyres)**. Held-out evaluation covers old low/high tyres only. Exact video mapping is no longer a blocker for this revision and is not fabricated.

SegFormer ignores conflicts; YOLO uses a documented training-only tyre/tread union and disables the incompatible exclusive-class auxiliary semantic branch. HRNet uses new polygon-derived points with weight 0.25 as weak training labels; only existing human points serve as validation/test truth. No further manual labeling is required to run this revision. The original ZIP and raw annotations remain unchanged.

Default: 15 combined-data runs (five models × three seeds), 60 epochs each. Optional old-only comparison adds 15 runs with matched update budgets. Two local GPU processes and optional static multi-session workers are implemented. Real-model GPU resume tests run automatically before long training. Actual Kaggle T4 results, completed training, reporting and workstation integration remain execution stages, not missing preparation.

The earlier 6/3/3 stratified split, mandatory exact identity matching, mandatory acceptance of training point proposals, and default 30-run study below are historical design proposals superseded by this section. Source-v1 still correctly records unassigned splits; the separate training overlay supplies the runnable revision.


19 September 2026. This note continues the context in the existing `../undertand3.md` without modifying it.

User objective: improve deployed workstation models using old data plus two frames per second from three workshop videos; label only new images; maintain documentation/progress; keep every new artifact in a Phase 2 namespace and leave the existing project unchanged.

Confirmed: all three clips show different physical tyres, all mid-mileage, all from the original 12. They are not three new tyres. Exact matching to `mileage_040000__session_001`, `mileage_070000__session_001`, `mileage_090000__session_001` is still unconfirmed. Do not guess from appearance or video ordering.

Prepared: 152 native portrait PNGs (27/93/32), timestamp/hash manifests, gallery, plan, persistence contract and a new-frame-only LabelMe launcher. Frames occupy about 415 MiB. All 152 user labels are now received and audited. Source release v1 contains 570 verified image/mask sets; archive and CPU preflight notebook are ready. Exact identity mapping, split freeze, conflict handling for YOLO and acceptance of 912 proposed geometry points remain pending. No training or new accuracy results yet.

Critical inference: all three original mid sessions are in the old HRNet train split. Warm-starting that tyre checkpoint would compromise any claim that a held-out mid-video tyre was unseen. Primary Phase 2 measured runs should use general-pretrained initialization and the new frozen physical-tyre split. Existing workstation checkpoints remain useful operational baselines with their known training exposure disclosed.

Proposed experiment: five model families (MobileNetV4, ResNet50, YOLO26m-seg, SegFormer-B0, HRNet-W18), old-only vs combined, three seeds, fixed step/epoch budgets. Medium YOLO must qualify on GTX1650 latency/memory; fallback is explicit, never silent. New images all being mid requires balanced class/tyre/domain sampling.

Classification, two-region segmentation and six-boundary geometry require different supervision. Reuse old masks/points as supplied; new polygon labels provide masks and boxes, then reviewed guide intersections provide points with invalid states. Derived labels are not independent human truth until reviewed. No damage, depth-mm or alignment labels should be invented.

Main documents: `phase2Plan.md`, `phase2LabelingGuide.md`, `phase2TrainingContract.md`, `pahse2Progress.md`. Future work updates this note and progress within Phase 2, not the protected original understanding document.

## Installed state — 21 September 2026

User authorized replacing current workstation models. Five exports installed with validation-only selection; existing app now uses Phase 2 while preserving old checkpoint files and a reversible mode switch. GPU and three-video UI checks passed. 14 runs completed; YOLO seed 3 explicitly deferred. See workstation/README.md and phase2CompletionAudit.md. Classifier weakness remains explicit; no new-data benefit or unseen-video claim.
