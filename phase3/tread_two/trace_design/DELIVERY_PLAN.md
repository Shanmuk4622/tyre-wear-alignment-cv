# Implementation and notebook delivery plan

<!-- NB24_COMPLETION -->
## Actual execution update — 2 November 2026

NB24 completed the original one-seed fit and review. Original selected=None and
all source/contract identities are preserved. [Completion](../../phase3NB24Completion.md)
and [illustrated findings](../../phase3TreadTwoTRACEFindings.md) supersede pending
execution statements below. No rerun or new fit is required for this cycle.
<!-- END_NB24_COMPLETION -->

<!-- TRACE_BATCH_SUCCESSOR -->
## Executable successor delivered

**2 November 2026:** this file retains the original design-stage record. The
separate [TRACE notebook batch](../../phase3TreadTwoTRACEBatch.md) is now implemented
with NB21–NB24,671 executable fields and local acceptance. Use NB24 for automatic
preparation→gates→pilot→review. Actual Kaggle/HF and measured results remain
pending; the original prototype/code/check receipts remain unchanged.
<!-- END_TRACE_BATCH_SUCCESSOR -->

**Design authorised; no additional permission needed for the engineering work.**
The local prototype is complete for this design stage. The remaining work below
is explicitly tracked; it is not disguised as a trained model or notebook that
already exists.

| Stage | Concrete deliverable | Completion condition | Current status |
|---|---|---|---|
| 1. Diagnosis and design | Architecture, fixed math, provisional experiment and telemetry/resource contracts | Evidence-linked choices and implementation scope | Complete |
| 2. Core prototype | Target extraction, 71,022-parameter branch, losses, frozen geometry wrapper | Numeric/leakage/gradient/geometry/CPU restore checks | Complete |
| 3. Actual train-only target audit | Full 386-image target maps, controls, spectra and interactive examples | Original/saved inputs verified; no held-out fitting | Complete |
| 4. Transactional runtime | Dataset binding/cache, deterministic schedule, update ledger, CSV/native writers | Every promised schema field has an exercised writer; truncation/crash/corruption/skip proofs | Pending implementation |
| 5. Resource and HF integration | Real formatted-shard estimate, both GPUs, parent publisher, last-event receipt | ≤15 GB logical /≤18 GB peak; actual fresh-worker HF continuation | Pending |
| 6. Notebook batch | Preparation, one pilot, saved-evidence review and automatic master | Embedded identities/fresh sessions/stop/resume/remote completion skip pass | Planned, not delivered |
| 7. Measured candidate | One separately frozen fit, no sweep | Finite complete run and independent HF verification | Not started |
| 8. Scientific decision | Selection, controls, original-role spectra and research passport | Honest eligible/rejected result; no automatic promotion | Not started |

## Proposed notebook batch

Numbers/names below are reserved in the design only; there are no links to
nonexistent files. Deliver the complete justified batch together, with one
recommended Run All master and sequential internal gates.

| Proposed notebook | Purpose | Accelerator |
|---|---|---|
| NB21 TRACE preparation/contract gate | Verify small pinned geometry input, construct fixed targets, prove math/storage/dual-T4/native+HF recovery; disposable updates only | T4×2 |
| NB22 TRACE resumable pilot | One seed after all engineering gates; full 48-family direct evidence; skip completed work | T4×2 |
| NB23 TRACE completed review/passport | Recompute losses/selection from saved CSV/native, render residual/retrieval atlas and publish limitations | CPU / None |
| NB24 TRACE all-in-one master | Automatically sequence preparation→pilot→review with one owner and compatible shared cache | T4×2; final analysis CPU |

NB24 is planned to be sufficient as the sole user-run notebook, without manual
handoffs between dependent stages. A failed engineering/scientific gate must
publish its reason and stop the dependent stage. Successful execution must not
manufacture a selected model when scientific gates reject every epoch.

## Immediate engineering work

Build the new data/target store and deterministic eight-distinct-image scheduler,
then the complete-update transaction and all schema writers. Measure actual
formatted CSV/native shards before freezing the measured recipe. Reuse proven
transport components through explicit vendoring and regression tests, preserving
all completed packages. Only then generate the notebooks from editable source.

The user has nothing new to run at this design checkpoint. No NB16/NB17/NB20
rerun, new photos, workstation edit or local HF token setup is needed.
