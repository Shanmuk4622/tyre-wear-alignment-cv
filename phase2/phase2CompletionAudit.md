# Phase 2 completion audit — 21 September 2026

> **Final status — 21 September 2026:** [All 15 runs and NB06 are verified complete](phase2Seed3Completion.md). Installed selections still match the final registry. Earlier partial/deferred/run-next statements below describe historical snapshots. No retraining or extra labeling is required.

## Verified result

Hugging Face revision: `be0eadb2bbb698091b563d699e2519b1e262a31c`. Fourteen of fifteen combined runs completed 60 epochs. YOLO seed 3 is resumable after 51 completed epochs, cursor 193, 10,036 optimizer updates (during epoch 52). The uploaded NB06 report is correctly partial, not failed. No final weights are published for seed 3.

Verified all run JSON/console hashes, remote LFS hashes for checkpoints/exports, actual T4 smoke PASS, finite training losses, history/update counts, all 60 validation epochs for completed runs, held-out image/tyre/label membership, recalculated test metrics and NB06 report hashes. Downloaded five selected weights and independently SHA256-verified their bytes.

## Installed selection

Selection uses highest validation score with the existing first-seed tie rule; test scores do not select weights. YOLO selection is provisional among its two completed seeds.

| Model | Seed | Validation | Test |
|---|---:|---:|---:|
| hrnet | 3 | 0.422% | 0.859% |
| mobilenetv4 | 1 | 100.000% | 50.000% |
| resnet50 | 2 | 85.000% | 50.000% |
| segformer | 2 | 99.373% | 98.957% |
| yolo26m | 1 | 99.216% | 98.790% |

Classifier figures are balanced accuracy on the observed low/high classes only; segmentation figures are mean two-region Dice at training input resolution; HRNet figures are mean absolute boundary-position error as a fraction of image width (lower is better).

MobileNet seeds 1 and 3 both reached 100% validation; the frozen registry selects seed 1. Their test balanced accuracies differ sharply (50% versus 94.35%). All three ResNet seeds scored 50% on test. These classifiers are experimental replacements, not demonstrated improvements. Segmentation scores do not certify classifier quality or measured tyre wear.

The split remains 386 train / 81 validation / 103 test images from 8/2/2 physical tyres. All three known mid-tyre videos are training material. No unseen-video or held-out-mid claim is supported. HRNet test uses 25 human-labeled photos. No new annotation is needed for this revision.

## Remaining work, accurately classified

- Required for full 15-run completion: resume only YOLO Worker2/Seed3 to epoch 60, then rerun NB06 and recheck its registry. Preserve the original runtime/protocol. Do not rerun completed training jobs.
- Optional research: 15 old-only control runs were not requested as the default execution and are not complete; no paired new-data benefit claim can be made.
- Optional deployment refit and independent new-tyre videos need separate experiments/data. They are not concealed unfinished preparation.
- Pixel-level manual approval of derived points was superseded by explicit weak training supervision; exact video matching was superseded by all-mid-in-training grouping.
- Original broader study/reporting tasks outside Phase 2 are not automatically completed by these runs.

## Evidence

- `completion/verification.json`: independent checks and all 14 measured results.
- `completion/audit.json`: manifests and per-artifact verification.
- `completion/reports/1789959917/`: downloaded, hash-verified NB06 report and registry.
- `workstation/registry.json`: five validation-selected exports, immutable revision and hashes.
- `workstation/models/`: verified raw state dictionaries and export metadata.
- `workstation/backup-20260921/`: pre-change application sources and readmes; legacy checkpoint files remain untouched.

No remote training/checkpoint/lease mutations were made during this audit. Local workstation changes are explicitly authorized by the 21 September replacement request, superseding the earlier no-existing-file-change instruction for this integration.

**User decision, 21 September:** YOLO seed 3 will be run later. Proceed with completed exports now; its completion and the NB06 refresh are deferred, not blockers for this installation.
