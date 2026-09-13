# Reproducibility appendix

## 1. Three different reproduction tasks

1. **Read the report:** open [REPORT.html](REPORT.html). All scientific figures are local; no internet, token or GPU is needed. Keep the `assets/` and `evidence/figures/` folders beside it.
2. **Rebuild documentation:** use the cached small evidence package and the scripts below. This regenerates tables, the endpoint plot, Markdown and HTML. It does not retrain, download datasets or publish anything.
3. **Reproduce scientific training:** follow the frozen notebooks, protocol and checkpoint runtime records. This is expensive, is not required for documentation, and should not overwrite completed namespaces. Independent reruns need an explicitly authorised new run namespace and compatible environment.

## 2. Frozen source map

Public dataset repository: [Shanmuk4622/tyre-wear-study](https://huggingface.co/datasets/Shanmuk4622/tyre-wear-study).

| Artifact layer | Revision / identifier | Meaning |
|---|---|---|
| Reporting publication | `22d5a6bc9f953ba3bf2a75919edc7db3193b317b` | NB20 output snapshot downloaded for this report |
| NB19 study inputs | `35a178b5a94878bdee95a0aa9ed8cb25cd1aeb6b` | Completed exploratory fusion and preceding study inputs |
| NB20 evidence input | `e388dd4ec0e9f522ac8a8b107f86fbeabe5c6a96` | Frozen evidence collection before report publication |
| S5 completed report audit | `a3b29a71f8e6af6c50e68eb64a5bbae9ccf6d1c5` | 81 completed dense-task runs and NB17 report |
| S10 namespace | `s10/reporting-r1/2a333e2a6469905ad8cb822821ea46a357364e6d17bdd53c153c8b7e51fd217e/` | All small source/report package paths |

[The evidence manifest](evidence/evidence_manifest.json) is the exact per-file map, including original repo path, source revision, bytes and SHA256. It contains 38 entries: 25 tables, ten figures and three status records. [REPORT_STATUS](evidence/REPORT_STATUS.json) hashes seven report artifacts. Four additional report figures make 14 public scientific figures. Upstream records remain byte-for-byte unchanged in `evidence/`.

## 3. Local report build

Run from the repository root with a Python environment containing `markdown-it-py`, `matplotlib`, `numpy` and `Pillow`. `requests` is needed only for the optional download. The build does not require PyTorch or CUDA. Actual documentation-build package versions are recorded in `BUILD_PROVENANCE.json` after generation.

```powershell
python scripts/build_project_report.py
python scripts/verify_project_documentation.py
```

On this machine the existing interpreter is `C:\Users\shanm\dev\envs\cv_conda\python.exe`; using it does not install or alter training packages. The authoring file is [manuscript.source.md](manuscript.source.md). Edit that source and the supporting documents, then rebuild; direct edits to generated `REPORT.md` or `REPORT.html` would be overwritten.

If the evidence cache is missing, this **optional read-only** command retrieves only the pinned report package:

```powershell
python scripts/fetch_report_evidence.py
```

The downloader enforces an 8 MiB per-file ceiling and a 20 MiB total safety budget. The initial verified fetch used approximately 3.02 MiB. Existing hash-matching evidence is reused. It never requests model checkpoints or dataset images. A token is read from the process environment or project `.env` if present, never printed. The repository is public; no upload permission is used by this workflow. Do not include `.env` in a submission or archive.

## 4. Scientific notebook map

This is an interpretation/run-order map, **not a request to rerun completed work**. Check exact current notebook instructions and frozen runtime before any future execution. Existing outputs and repair archives are evidence, not disposable clutter.

| Phase | Notebooks | Role and dependency |
|---|---|---|
| Data/baselines | NB00, NB01, NB01A, NB01B, NBT1 | Preparation, baseline/recovery and annotation replay; NB01B is the matched random-init control |
| Classification | NB02–NB05, NB03A | Classic/modern/transformer/foundation sweep and identity audit; nine Small substitutions excluded |
| Explanation then OFAT | NB07, then NB06 | NB07 selects candidates for Stage B despite notebook numbering |
| Stress/calibration/reporting | NB08, NB09, NB10, NB10R | Existing classifier diagnostics and reporting recovery |
| Optional annotation comparison | NB11 | Deferred; not a prerequisite for the manual-supervised S5 route |
| Confirmation | NB12, NB12R | 18 S4b runs followed by paired report |
| Dense protocol | NB13 | Freeze data/model/protocol identities |
| Dense training | NB14, NB15, NB16 | Semantic, YOLO and RT-DETRv2; all completed |
| Dense report | NB17 | Verify all 81 evaluations before reporting |
| Exploratory fusion | NB18 | Consume saved S5 predictions, not new training |
| Evidence/report | NB19, NB20 | CPU-only reporting collection and draft generation |

See [the notebook directory](../../notebooks/) and [S5 execution specification](../24_S5_MANUAL_DENSE_TASKS.md) for exact filenames and the current-position section. Older repair instructions below historical headings are not fresh rerun requests.

## 5. Environment and identity

The root `environment.yml` and `ENVIRONMENT.md` describe local development, not a universal exact Kaggle replay lock. The authoritative training environment is recorded in each frozen protocol, pilot, checkpoint and worker runtime. S5 library pins include Ultralytics 8.4.20, Transformers 4.51.3, segmentation-models-pytorch 0.5.0, timm 1.0.15 and pycocotools 2.0.11. CUDA/PyTorch and NumPy compatibility must match the run's recorded state; an earlier pilot is not automatically the runtime of every later checkpoint.

Model identity includes architecture implementation, head, parameter/tensor identity, initial checkpoint source and revision. A name alone was insufficient for the quarantined Small runs. Genuine RT-DETRv2-R18 is explicitly different from Ultralytics RT-DETR-L. YOLO's repaired `flip-only-r2` augmentation policy is recorded as a runtime amendment rather than retroactively pretending earlier pilots used the same recipe.

## 6. Data and metric reconstruction

- Retain original image IDs, clean-image hashes, derivative-to-parent mapping, manual masks and the frozen fold assignment. Do not resplit while calling a result a reproduction.
- Classification selected and fixed-final metrics stay separate. The matched random-init recovery uses the final endpoint for the comparison reported here.
- S5 uses clean originals, two overlapping tyre/tread regions, native-coordinate evaluation and final-epoch weights. Detector-only mask quantities remain undefined.
- Recompute ROI and fusion class decisions using the saved `decision_rule`: CORAL threshold count versus softmax argmax. Use identical class order and image coverage.
- Fusion is fixed and paired to each run's full-frame prediction. The 405 rows are not 405 independent training runs.
- Report seed SD as seed SD. Do not infer population confidence intervals or p-values from repeated images and shared classifiers.
- Conformal empty sets require explicit handling. The inherited `abstain_rate` field alone is not a complete rejection policy.

## 7. Persistence and interruption guarantees

The parent uploader batches normal publication at approximately 30 minutes and requests a flush at major completion or catchable Stop. Server backoff still applies. There are no claim/heartbeat commits in the corrected static-owner workflow. Each active worker/account must have unique static ownership; changing worker count requires stopping old copies first.

Resume uses the latest **HF-published completed epoch** and compatible model/optimiser/scheduler/scaler/RNG state where supported. It is not exact mid-batch recovery. A forced OS kill cannot execute a flush, and unpublished work lost with a session cannot be recovered from HF. Hash-checked immutable snapshots and journal recovery protect against mismatched weights/status, while evaluated mismatches fail closed.

Temporary scratch capacity is checked, not assumed to be 1 TB. The workflow uses attached Kaggle data, per-job scratch and verified cleanup of generated files. No blanket deletion of user data or completed HF artifacts is part of the procedure.

## 8. Verification levels and limitations

The documentation verifier checks the 38 cached sources against their hashes/byte counts, seven report artifact hashes, table cardinalities, report links, image decoding, citation range, unresolved template markers and generated provenance. This provides evidence-to-document consistency. It does not independently rerun every numerical computation in the training pipeline.

Earlier audits recorded deeper verification: NB17 checks all 81 dense statuses, checkpoint/artifact hashes, histories and downstream coverage; NB18 checks fusion predictions and recomputes metrics. Recovery checkpoint presence and full tensor inspection are different checks and must not be described as identical. See the dated audit documents for exact depth.

## 9. Release contents and secret hygiene

Distribute `REPORT.html` together with `assets/` and `evidence/`, or the full documentation folder. Include `REPORT.md`, references, reproducibility appendix, claims ledger and submission checklist. Include source/build scripts for reproducibility. Do not bundle `.env`, authentication caches, model weights, raw datasets, unrelated output caches or training logs containing credentials. The small reporting evidence contains the source images already rendered into figures, not a new raw-image release.

The local manuscript is not automatically published to HF. Its build provenance is separate from upstream `REPORT_STATUS`; the latter remains the original notebook report's status. A later public manuscript release requires deliberate publication and a new revision rather than silently modifying the frozen evidence snapshot.
