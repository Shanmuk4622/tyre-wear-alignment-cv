# ATLAS v1 — validation record and acceptance boundaries

**Superseding status, 30 October 2026:** the repaired Kaggle measured pilot is
complete on HF at 2,940 updates, including both endpoints and diagnostics.
[Pinned publication/report verification](../phase3NB08CompletionStatus.md)
records the check scope and remaining scientific review. Local engineering
receipts below keep their original, narrower claims; no historical fixture is
relabeled as a measured training run.

**30 October execution/repair update:** the supplied notebook completed the
actual dual-T4/pinned-HF engineering gate. Its published proof was fetched and
hash-checked. A local request-policy bug then stopped the measured launcher;
the [external transport repair](../phase3NB08Repair.md) preserves every frozen
runtime/asset byte and adds separate regression/packaged-worker checks.
The earlier local/live boundaries below describe the original 29 October
delivery. A repaired Kaggle run and measured result are still pending.

**29 October 2026.** Local engineering checks precede the user's measured
Kaggle run. The final receipts are
[LOCAL_VALIDATION.json](pilot_validation/LOCAL_VALIDATION.json),
[STOP_VALIDATION.json](pilot_validation/STOP_VALIDATION.json) and
[NOTEBOOK_DELIVERY.json](pilot_validation/NOTEBOOK_DELIVERY.json).
The [final handoff receipt](pilot_validation/HANDOFF.json) binds the delivered
notebook/source, extracted-worker check, preservation, figures and local links.
The final handoff also binds exact source/notebook identities and preservation
checks. Engineering fixture scores are not TREAD-TWO study results.

## Checks and scope

| Check | Evidence / acceptance |
|---|---|
| Original architecture proof | [15 checks](prototype_report/PROOF.json), unchanged original prototype; no optimizer steps in that proof |
| Full-batch training | Actual new model, batch eight, two fixture epochs ×two updates; real existing training images |
| Pause/restore | Update-1 complete native/CSV source restored in a fresh process; remaining trajectory compared with uninterrupted reference |
| Abrupt termination | Real worker exits after new native save but before durable pointer; old update recovered and incomplete update replayed |
| Stop marker | Same marker path used by catchable notebook signal; pause at complete update and resume; kernel UI itself is not locally exercised |
| Scientific equality | All 21 transactional CSV families match, excluding nondeterministic timing/resource fields |
| Retained fixture observations | 900,380 rows per complete trajectory, including exact source pixels/dense probabilities; not 900,380 independent examples |
| Native equality | Model, optimizer, scheduler, disabled scaler, RNG and native evidence tensor values match exactly |
| Publication faults | 17 offline checks: chunk corruption, cursor rollback, rotation, immutable staging, history extension, CAS/lease fencing, ambiguous success, deduplication and rate/backoff |
| Storage | Measured CSV/native fixture extrapolation under the 16 GB guard; live Kaggle recalculation still required |
| Actual local estimate | 12,801,172,262 bytes conservative peak, about 12.8 GB; not a guarantee of a Kaggle quota |
| Original-image evidence | Maximum local native-projection residual 5.63e-9 logit units across 12 class checks; tolerance 2e-6 |
| Notebook | Schema/AST, exact embedded source/bank inventory and fresh extracted-worker check |
| Isolation | Original/new workstation sources, declared weights, original Markdown, frozen experiments and executed NB02–NB07 preserved |

The local device is the existing **GTX 1650**, not dual T4. No HF token is read,
no live upload occurs, and no measured 60-epoch fit is started by these checks.
The notebook's automatic gate must prove the actual T4 and pinned-HF path.
HTML payload/JavaScript syntax and generated figures are checked; browser
interaction/rendering is not claimed as a completed test here.

## Development faults caught and repaired

| Fault found before delivery | Repair / meaning |
|---|---|
| Pre-update evidence referenced the mutable bias parameter after optimizer step | Snapshot the forward bias tensor; saved pre-update evidence now matches its original logits |
| Windows optional peak-memory counter used an incorrectly typed process handle | Explicit Win32 handle/signature declarations; unavailable optional resource counters remain blank rather than stopping training |
| Working state and Store state shared mutable references | Deep-copy working state; crash decisions use the immutable committed unit ledger |
| Source changed during an in-progress development check | Export correctly rejected stale source identity; those development attempts are not final acceptance evidence |
| Repeated restore could repeatedly prefix log filenames | Keep stable content-addressed prior-session log names; preserve control logs too |

Failed development attempts remain under the ignored local fixture directories;
they are not deleted or passed off as completed model training. Only the final
hash-bound passing receipt supports delivery. The untouched original prototype
proof remains a separate earlier artifact.

## What is not yet established

- Actual Kaggle dual-T4 execution, network publication and pinned-state restore
  are automatic notebook gates, not locally claimed successes.
- No measured TREAD-TWO accuracy, variance across seeds or improvement exists yet.
- The historical low/high endpoint was already inspected. The frozen anchor
  saw existing mid training data; reshuffling it cannot create an unseen test.
- Patch cases can encode paint, identity and lighting; conservation of evidence
  is algebraic accountability, not physical/causal wear proof or calibration.
- Forced kernel death cannot publish unsent local work. Resume guarantees apply
  to the last verified durable local/remote boundary, with gaps recorded.

## Reproducing local engineering checks

Use the project's cv_conda wrapper. These commands run fixtures, not the long fit:

```powershell
.\phase3\run_cv.cmd -B phase3\analysis\check_tread_two_training.py
.\phase3\run_cv.cmd -B phase3\analysis\check_tread_two_stop.py
```

Do not run them to replace Kaggle acceptance or to select another model by
fixture score. They preserve current workstations and use no credential fallback.
