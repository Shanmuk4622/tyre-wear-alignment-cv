# Active four-notebook audit

> **Final status — 21 September 2026:** [All 15 runs and NB06 are verified complete](phase2Seed3Completion.md). Installed selections still match the final registry. Earlier partial/deferred/run-next statements below describe historical snapshots. No retraining or extra labeling is required.

Saved notebook outputs were inspected, not live Kaggle sessions. HF was read at immutable revision `9278e2a65ab45f1831810505d66523f7a6cd1930`, protocol `661e97642e10bf8cfcbb8f581142069733789eb8f413dc70654aa00dd7854af3`. No running jobs, notebook settings, runtime sources or remote files were changed.

| Notebook | Saved-output state, seeds 1 and 2 | Verified HF checkpoint |
|---|---|---|
| NB02 | MobileNetV4: 27 completed epochs, updates 1333/1334 | Epoch 16, updates 814/819 |
| NB03 | SegFormer: 16 completed epochs, updates 3229/3244 | Epoch 10, updates 2076/2073 |
| NB04 | YOLO26m: 3 completed epochs, updates 586/585 | Epoch 2, updates 403/403 |
| NB05 | HRNet: 9 completed epochs, updates 1095/1088 | Epoch 7, updates 786/782 |

All eight active runs have passing T4 fresh-process resume reports. Uploaded history lengths match checkpoint update counts; losses are finite. SHA-256 of downloaded smoke/history/validation JSONs matches each manifest, and HF large-file SHA-256 metadata matches all eight state.pt entries. No model checkpoint payload was downloaded or restored during this read-only audit. The three numerical-retry events in MobileNetV4 were handled; other inspected histories have zero retries.

Each notebook has NUM_WORKERS=1, WORKER_ID=0. This is valid because NB02–NB05 select disjoint model jobs. It does not mean one GPU: each output confirms two GPU processes. Do not switch these four different notebooks to four-way sharding during their runs.

NB02 owns six jobs in order: MobileNetV4 seeds 1,2,3, then ResNet50 seeds 1,2,3. Its two slots currently run MobileNetV4 seeds 1/2. As slots free, seed 3 and then ResNet50 start. ResNet50 is queued, not omitted. Its remote seed-1 smoke artifact can exist from NB01 while training remains not_started. The two MobileNet seeds have different sampled batches and losses; similar epoch counters are not duplicate-job execution.

NB02's saved output lacks an HF success line, but its two uploaded checkpoints and matching histories are present. Remote progress lagging local output is expected for periodic snapshots. The status resumable describes a recoverable checkpoint and does not mean the active worker is paused.

## NB02 quality concern

At the inspected epoch-16 remote snapshot, BOTH MobileNet seeds label all 40 low validation photos as mid, while all 41 high validation photos are correctly high: balanced accuracy 0.50. Training loss falls to roughly 0.169. This indicates a generalization concern, not evidence of a healthy classifier simply because optimization proceeds.

Seed 1 achieved balanced accuracy 1.0 at epoch 4 before falling; the training implementation preserves validation-best weights. Seed 2's best score in the inspected history is 0.50. Validation contains only two old low/high tyres, no mid tyre or video, so neither 100% nor 50% establishes full three-class/video performance. No automatic recipe change or restart was made during this fixed experiment.

Other last-uploaded validation scores: SegFormer Dice 0.9930/0.9930; YOLO Dice 0.9903/0.9178; HRNet mean normalized width errors 0.00508/0.00558. These are partial validation results, not final test/workstation results. YOLO seed 2's early variability should not be concealed by an overall 'all good' claim.

Evidence: `manifests/phase2_running_notebook_audit.json` and `manifests/phase2_running_hf_audit.json`. Executed notebook snapshots were archived without replacing current files.
