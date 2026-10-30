# Run TREAD-ONE on Kaggle

Version r1, 27 October 2026. **Two notebooks, one architecture, three seeds.** No new manual labels or model-weight downloads are required. The existing workstation remains as it is.

## Files

- `notebooks/phase3_NB00_Data_Preflight.ipynb` — CPU integrity/overlay check and small HF report.
- `notebooks/phase3_NB01_TREAD_ONE.ipynb` — dual-T4 resume gates, three 60-epoch runs, evaluation, explanations, report and export.
- `releases/TREAD_ONE_Kaggle.zip` — both self-contained notebooks, this guide, design, experiment contract and delivery hashes.

Read [the design](TREAD_ONE_DESIGN.md) and [experiment contract](TREAD_ONE_EXPERIMENT.md) for every model, split, loss, selection and persistence choice. Current checks/limitations are in [validation](TREAD_ONE_VALIDATION.md).

## 1. Kaggle setup

1. Import `phase3_NB00_Data_Preflight.ipynb` using Kaggle's notebook import.
2. Add the existing dataset **[Tire Dataset Prepared phase2](https://www.kaggle.com/datasets/shanmuk4622/tire-dataset-prepared-phase2)**. It already contains the original 418 photos plus 152 frames and labels. Do not attach the original-only dataset in its place or upload the whole project.
3. Enable **Internet**.
4. Add/enable Kaggle secret **HF_TOKEN**, with write access to dataset repository **Shanmuk4622/tyre-wear-study**. Do not paste a token into any cell. The notebook never prints its value.
5. Set accelerator **None / CPU** for NB00. Run All in order. Expect source checks, 570 valid image/mask sets, the safe 386/81/103 overlay, and a verified HF commit. Its source report still says source-v1 `training_allowed=false`; the separate Phase 3 overlay report records readiness. That is intentional.

NB00 and NB01 embed their small runtime. They do not need a source ZIP as another Kaggle input. The setup retains Kaggle's CUDA PyTorch and installs only the tested Hub/locking package versions when different.

## 2. Train

1. Import `phase3_NB01_TREAD_ONE.ipynb` into a **single** new notebook.
2. Attach the same Phase 2 dataset; enable Internet and HF_TOKEN.
3. Select **GPU T4 ×2**.
4. Keep the defaults: `DATA_ROOT=''`, `RUN_TAG='treadone-r1'`, `TAKE_OVER=False`, `ALLOW_SINGLE_GPU=False`.
5. Run All. The notebook verifies source data, binds the protocol/runtime, claims ownership and checks HF for saved work.

Two independent seed jobs run on the two T4s. Seed 3 starts when a slot frees. This is one custom model architecture trained three times, not a five-model sweep. Each seed first checks real transformed batches, gradients, full-size memory and a fresh-process optimizer-update restore. A failed check blocks that seed's measured training.

Training is fixed at 60 epochs, batch eight, 49 updates/epoch and 2,940 updates/seed. The notebook logs separate classification, ordinal, segmentation, point, containment, row-consistency and background-consistency terms. Each epoch reports validation task metrics and the frozen joint selection utility. There is no early stopping or automatic smaller model/resolution fallback.

The local GTX1650 test is not a Kaggle timing guarantee. Read the actual smoke timing/estimate; it excludes data decoding, checkpoint writes, validation and upload costs. The source verifier may spend time hashing/decoding before GPU processes start.

## 3. Stop and resume

**Press Stop once and wait.** The launcher forwards the interrupt to the coordinator, which signals seed workers. They finish or abandon the incomplete update, preserve durable progress and attempt an immediate HF flush. Look for `HF snapshot verified: <revision>` and retain that revision. A server backoff or failed network can delay/fail the flush; local checkpoints and the last verified revision are reported.

To continue, start a fresh dual-T4 session with **the same notebook/runtime, settings, input dataset, RUN_TAG and compatible environment**. Run All. The code downloads hash-verified bundles before restoring the next update cursor. Completed updates are not deliberately replayed; verified completed seeds are skipped. An incomplete GPU update is replayed from the preceding durable state. Validation/test interrupted partway can be rerun.

The default publish interval is 30 minutes. Seeds finishing and final report generation also trigger commits. It does not upload each optimizer step. A forced session kill, second Stop during cleanup, driver crash or lost temporary filesystem cannot publish pending local state; up to the work since the last verified remote snapshot may be lost.

Do not change code/package versions or RUN_TAG to get past a resume error. The anchor is intended to stop accidental retraining under a changed protocol. Retain the traceback and saved contract for a compatibility repair.

## 4. Ownership and takeover

Use **one NB01 copy**. The coordinator already uses both GPUs and queues all three seeds. Do not make worker copies or run another notebook against the same experiment.

If a new session says the study is leased, first stop the old notebook and wait for its upload. If it is definitely dead and cannot release, set `TAKE_OVER=True` **once** in the new session; after successful startup restore it to False for later launches. A live old session must not be taken over. CAS updates and fencing prevent a stale publisher from silently overwriting another owner.

Other studies using the same HF account still consume its request allowance. Avoid running publication-heavy jobs simultaneously. The application budget is conservative but not a universal count of HTTP requests; server delays are authoritative.

## 5. Inspect outputs

After all three seeds complete, the final cell prints selected/final metrics, shows the fixed illustrative gallery and creates **`/kaggle/working/TREAD_ONE_Results.zip`** for download.

Important local output paths:

```text
/kaggle/working/treadone_train/
  CONTRACT.json / OVERLAY.json / ENVIRONMENT.json / DATA_PREFLIGHT.json
  LAST_REMOTE.json
  report/
    REPORT.json               three seeds; selected and epoch-60 summaries
    EVALUATION_seed1..3.json   per-image/per-tyre predictions and diagnostics
    EXPORT.json / weights.pt  one validation-selected compact model
    gallery/*.png / *.npz     signed evidence, masks, points, provenance

/tmp/treadone_<protocol>/runs/seed1..3/
  state.pt                    rolling full recovery state
  SMOKE.json / IDENTITY.json / console.log
  TEST_SELECTED.npz / TEST_FINAL.npz
```

HF uses the printed `RUN_PREFIX`, under `phase3/tread-one-r1/<protocol>/`. Seed `latest.zip` bundles contain complete state, detailed history/validation, smoke/identity, errors/logs and raw test arrays. `INDEX.json` and `LAST_REMOTE.json` identify committed hashes/revisions. Earlier published versions remain in repository history.

The gallery shows original image, masks/points and signed predicted-class evidence. Red supports the class logit; blue opposes it. Bias is separate. The middle-class example is training-cohort data. Errors remain visible. Region/point outputs do not receive a fictitious shared classification accuracy.

## Troubleshooting

| Message / symptom | Action |
|---|---|
| Wrong/missing dataset or checksum mismatch | Attach the exact Phase 2 source release; inspect `VERSION.json`. Do not bypass its checksum or substitute the original-only dataset. |
| Secret/401/403 error | Enable HF_TOKEN for this Kaggle notebook and verify repository write access; retain token secrecy. No automatic repository creation is attempted. |
| HTTP 429 / reset delay | Wait for the logged server backoff. Do not spam reruns or parallel publishers. Local progress is retained. |
| Study already leased | Follow the stop/flush/takeover procedure above. |
| RUN_TAG bound to different protocol | Restore the original generated notebook and compatible environment; inspect source/package differences before any migration. |
| Fresh-process resume differences | Stop before long training. Save SMOKE.json/console output and report the executing Torch/GPU/runtime; do not loosen tolerance to continue. |
| Unsupported deterministic kernel | Retain the traceback and request a deterministic implementation repair. Do not disable deterministic mode. |
| Nonfinite loss/gradient | Bounded same-batch retries are automatic. Persistent failure keeps the preceding checkpoint; inspect errors before rerunning. |
| CUDA OOM | Retain the original protocol and checkpoint. Do not silently lower batch/resolution; use the report for an explicit compatibility/resource repair. |
| Storage or cgroup memory guard | Progress is retained. Check actual available paths/capacity; do not assume a 1 TB scratch disk. |
| Final publication fails | Keep the session alive if possible and retain local outputs. `LAST_REMOTE.json` shows the last verified snapshot; unchanged restarts restore from it. |
| High segmentation score but poor classification | Different tasks; inspect each separately. No accuracy improvement is promised by this model. |
| Empty/clipped mask rows or point disagreement | Raw predictions and heuristic flags are preserved. These are review cases, not a calibrated physical verdict. |

## Local development

Every Python command uses `cv_conda`. On this host, the documented workaround is:

```powershell
.\phase3\run_cv.cmd -B phase3/test_treadone.py
.\phase3\run_cv.cmd -B phase3/test_persistence.py
.\phase3\run_cv.cmd -B phase3/test_lifecycle.py
.\phase3\run_cv.cmd -B phase3/build_notebooks.py
```

`run_cv.cmd` activates `cv_conda` through Conda's Python entry point because its `conda.exe` wrapper failed here. Existing OpenCL activation-hook warnings were observed while the intended interpreter/CUDA checks succeeded. No global package reinstall was performed.

Edit readable runtime sources, rerun relevant checks and regenerate the notebooks. Preserve an already-started experiment's sources/protocol. Delivery outer-file hashes are in `releases/DELIVERY.json`; save/share the executed Kaggle notebook outputs as the execution record.
