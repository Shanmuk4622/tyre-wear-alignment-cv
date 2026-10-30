# TREAD-ONE — checks, failures and execution status

Updated: 27 October 2026. Machine-readable evidence lives under `validation/`.

## Current state

Implemented and locally checked. **No 60-epoch training results, Kaggle dual-T4 execution, live Phase 3 HF publication or installed-app replacement are claimed.** The delivered notebook performs its own real-device resume gate and publication checks when run.

## Checks

| Check | Result / practical limit | Evidence |
|---|---|---|
| Active local environment | `cv_conda` interpreter, Torch 2.5.1+cu121, GTX1650 CUDA detected; no global dependency replacement | `check_runtime.py`, environment/session notes |
| Actual architecture | 1,372,159 parameters; real original and new-frame samples; finite forward/backward, ordered points, exact logit accounting and detached segmentation gate | `validation/model_contracts.json` |
| Fresh-process training seam | **Passed with zero differences** in next losses, model, optimizer, scaler, scheduler and RNG; full 512×384 input, **batch eight**, counterfactual branch and noninitial scheduler | `validation/LOCAL_TESTS.json`, `fresh_process/SMOKE.json` |
| Local GPU allocation | Batch-eight smoke observed about 1.17 GiB allocated at peak; excludes driver/context/reserved allocations, other GPU processes and sustained workloads | Same smoke JSON, exact byte count 1,255,147,008 |
| Publication fault fixture | Real publisher against offline revisioned service fixture; CAS conflicts, lost claim response, stale owner, deduplication, restore, corruption, interrupted save and 429/header cases | `validation/PERSISTENCE_TESTS.json` |
| Actual short lifecycle | Paused after update 1, restored without repeating it, finished two-update fixture; final/selected states, scheduler, optimizer, RNG and losses matched uninterrupted execution | `validation/LIFECYCLE_TESTS.json` |
| Raw output/evaluation/export | Test NPZ shapes and additive evidence reconstruction checked; separate selected/final arrays and three-panel gallery generated in lifecycle fixture | Same lifecycle JSON and local fixture outputs |
| Dataset/source verification | Full 2,882-file / 570-image-mask verification is part of delivery verification and each Kaggle training launch | `validation/DATA_PREFLIGHT.json` |
| Notebook delivery | Both schemas/cell syntax, exact embedded source round trips, ZIP/hash round trips, document links and original-file preservation checked by delivery verifier | `validation/DELIVERY_VALIDATION.json` |

The tiny lifecycle's scores are fixture diagnostics, not scientific results. The local timing sample excludes data decoding, durable saves, validation and upload time. No T4 runtime claim is extrapolated from it.

## Troubleshooting recorded during implementation

1. **Conda wrapper failed:** direct `conda.exe` activation returned an execution error. `run_cv.cmd` activates `cv_conda` via the existing base Python's Conda entry point. The interpreter and CUDA then worked. OpenCL activation-hook access/file warnings remain visible; no environment reinstall was made.
2. **CUDA cumulative sum was nondeterministic:** strict mode rejected the ordinal CDF implementation. Replaced the three-class cumulative sum with explicit `p0` and `p0+p1`, keeping strict determinism. The subsequent real-model GPU seam passed.
3. **Resume scheduler warning:** an epoch-boundary resume can legitimately step the scheduler after updates from the previous process; PyTorch's process-local first-step flag is not serialized. Restore reconstructs that flag from the durable update count. The lifecycle checks compare actual scheduler/optimizer states and subsequent losses.
4. **Final versus selected checkpoint aliasing:** final training tensors are copied before loading selected weights for test evaluation. Both endpoints remain distinct in saved state; the lifecycle compares the restored final state against uninterrupted execution.
5. **Immutable index revision:** newly published bundles acquire the containing commit on fresh-session restoration; previous bundles retain pinned revisions. The offline fixture tests restoration through actual publication/index code.
6. **Stable snapshot hashes:** ZIP timestamps are fixed so unchanged state is not uploaded again just because snapshot creation time changed. The fixture checks binary upload counts.
7. **Ambiguous successful claim:** a lost server response may leave the caller's own lease already committed. Claim retries recognize their own owner instead of blocking self-resume. A simulated successful write/lost response exercises this case.
8. **Intervention precision:** base and altered-image diagnostics use the same autocast policy, so a precision change is not accidentally mixed into the intervention effect.
9. **Autocast boundaries:** attention-score math and additive evidence reduction explicitly disable autocast around FP32 operations; converting an operand to float alone does not enforce that boundary. The final full-batch seam uses these exact sources.
10. **Test isolation:** a rerun of the offline service test initially reused a deliberately newer restored checkpoint, correctly retaining it instead of replacing it with the older fixture. Test fixtures now receive unique directories; the final 13 fault cases passed.
11. **Untraced local process exit:** one GPU check exited with code 1 and no Python traceback. Its cause is unproven. Fault tracing/progress markers were enabled in the test driver; the unchanged-runtime rerun passed the full fresh-process seam. This remains recorded rather than labelled as a diagnosed/fixed model defect.
12. **Pre-update failures:** diagnostic-only bundles preserve failed smoke/error/log files even when measured training never produced a checkpoint. Their restore path does not fabricate epoch/update progress; the fault fixture tests it.

## What remains to establish on Kaggle

1. Run NB00 with the real HF secret/repository and verify its pinned published report.
2. Run NB01 on dual T4. Its batch-eight, fresh-process real-model gates must pass on both executing devices; errors prevent long training.
3. Exercise one catchable Stop, wait for its verified snapshot, and resume the same protocol in a fresh session. Local mock tests do not replace this live cross-session check.
4. Complete all three 60-epoch runs and inspect `REPORT.json`, both endpoints and failure examples. Record actual accuracy/resource results instead of treating code completion as model success.
5. Decide on workstation evaluation/promotion only after those results; the current app is unchanged.

Future fixes should append their failure, exact change, affected protocol/source identity and verification. Do not loosen a failed resume assertion or manufacture a completed result.
