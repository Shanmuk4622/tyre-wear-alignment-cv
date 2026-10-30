# Phase 3 — documentation conflicts and precedence

Reviewed: 27 October 2026. Historical files are preserved; this ledger supplies the current interpretation within each scope.

**Latest amendments, 28 October 2026:** [TREAD-ONE completion](phase3CompletionAudit.md) supersedes epoch-13/pending-execution statements. [The user publication policy](phase3DataRetentionPolicy.md) forbids future HF ZIP bundles; original r1 guides are historical release snapshots. [NB02](notebooks/phase3_NB02_Audit_And_Direct_CSV.ipynb) recovers/publishes direct CSV without changing trained weights.

**NB02 completion supersedes its earlier pending status:** all five code cells ran, and the 25-file / 18-CSV publication at `91ccd3315bd296a76a8b583a6d0f2515ccb3d370` is independently verified. Generation-only `CSV_NOTEBOOK_VALIDATION.json` and the original repair regression retain their historical `false` execution flags. [The newer completion record](phase3NB02Completion.md) governs current status. The local 20-table / 30-file plan includes supplements absent from the 18-table / 25-file core publication; these are distinct scopes, not an unaccounted loss.

**NB03 is complete locally:** earlier “diagnosis not implemented” statements are superseded by [the diagnostic results](phase3ClassifierDiagnosticResults.md). Its new HF publication is pending Kaggle NB03, while NB02 publication remains complete. The new 100% nominal training accuracy is post-hoc FP32 inference, not a replacement for unlogged historical augmented-batch accuracy or evidence of improved held-out test performance.

| Older or conflicting wording | Use this interpretation | Controlling local record |
|---|---|---|
| Phase 2 has only 14 completed runs; YOLO seed 3 is deferred | Superseded: all 15 runs and final NB06 completed; installed selections unchanged | [Seed-3 completion](../phase2/phase2Seed3Completion.md), `phase2/completion_seed3/verification.json` |
| Phase 2 exact mid-video identity matching/additional point review blocks training | Superseded for runnable training-r1: all possible mid identities are training-only; weak new points weight 0.25 | [Training contract](../phase2/phase2TrainingContract.md), [run guide](../phase2/phase2RunNotebooks.md) |
| Packaged Phase 2 dataset says every split is UNASSIGNED | Correct for immutable source-v1; the separate training overlay defines 386/81/103 | [Dataset readme](../phase2/phase2DatasetReadme.md), [working understanding](../phase2/phase2Understanding.md) |
| Normal workstation uses YOLO Nano, original CORAL classifiers or seed-1/final-60 geometry | Historical legacy mode; active Phase 2 uses Medium YOLO, softmax classifiers and validation-selected exports | [Workstation guide](../phase2/workstation/README.md), `phase2/workstation/ACTIVE`, registry and `prototype/phase2_adapter.py` |
| UI/evidence key says `yolo26n_seg` or learned mode `matched` | Backward-compatible internal names; consult active adapter/provenance for actual model | [Workstation guide](../phase2/workstation/README.md) |
| App unbuilt; HRNet integration or all alignment work deferred | Native app and learned/calibrated-target software already exist; physical accuracy remains unvalidated | [Current status](../docs/CURRENT_STATUS.md), [prototype](../prototype/README.md), [alignment](../prototype/CALIBRATED_ALIGNMENT.md) |
| Old “run now NB24/NB28/NB31,” recovery or three-worker YOLO instructions | Reproduction/repair history; do not rerun completed work from these headings | Dated completion audits 21, 24, 31, 34, 36; final Phase 2 completion |
| Older docs permit epoch-only recovery | Historical implementation limitation; user's current training requirement is completed-update recovery within the epoch | [Training contract](../phase2/phase2TrainingContract.md), [Phase 3 instructions](AGENTS.md) |
| Original “zero annotation” or SAM2-supervised S5 plan | Executed S5 used 418 manual masks; optional SAM2 comparison/blind consistency deferred | [S3 scope](../docs/22_S3_MASK_COMPARISON.md), [S5 completion](../docs/24_S5_MANUAL_DENSE_TASKS.md) |
| Folds 0/2 conclusively share a physical tyre | Historical visual suspicion/flags; later user confirmed 12 distinct tyres and membership audit found recorded groups disjoint | [Earlier understanding](../undertand3.md), newer paper review/analysis; identity not independently established |
| `undertand3.md` says 31 paper references; old build copies say 16 | September 24 manuscript records 34 references and 11 related application studies | [Newer manuscript README](../ACCESS_latex_template_20260513/README.md), [revision](../ACCESS_latex_template_20260513/SUBMISSION_REVISION_20260924.md) |
| NB33 guide/local verification says the full visual replay has not been tested | That reflects the earlier local implementation check; later paper evidence records a completed, hash-verified 28×9 replay | `ACCESS_latex_template_20260513/revision_evidence/visual/AUDIT.json`, September 24 revision |
| Phase 2 NB07 is the original-paper visual comparison | Different scope: original-paper illustrations use NB33/original checkpoints | [Original visual guide](../docs/37_ORIGINAL_STUDY_VISUAL_COMPARISON.md) |
| 20 GB output versus approximately 1 TB temporary space | Different storage scopes; measure actual paths/capacity and never assume persistence | [Training contract](../phase2/phase2TrainingContract.md) |
| HF limit is universally 128/hour; stop always publishes everything | User's conservative planning concern; implementation must respect actual server limits. Catchable stop attempts publication; hard kills cannot run cleanup | [Training contract](../phase2/phase2TrainingContract.md) |

## Resolution rule

Identify the experiment/model namespace first, then the exact artifact and endpoint. Prefer frozen evidence and later relevant completion records over historical plans. A newer paper revision does not redefine Phase 2's experiment; an installed Phase 2 model does not replace original-paper results. Record remaining uncertainty instead of deleting its history.
