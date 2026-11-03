"""Document the finite GEO preparation handoff and keep live claims pending."""
import json
import re
import sys
from pathlib import Path

assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'tread_two/geometry_first'
def write(path,text):
    path.write_text(text.strip()+'\n',encoding='utf-8')

write(BASE/'README.md', '''# TREAD-TWO / GEO — implemented preparation

**1 November 2026. NB16 is ready for Kaggle; actual T4/HF execution is pending.**
The user approved the geometry-first plan and requested notebooks. This package
implements the next prerequisite: a real-model engineering gate. The measured
60-epoch experiment has not started. No additional photographs are requested.

| Entry | Purpose / status |
|---|---|
| [NB16 notebook](../../notebooks/phase3_NB16_TREAD_TWO_GEO_Preparation_And_Contract_Gate.ipynb) | Self-contained Kaggle preparation; run first |
| [Runbook](RUNBOOK.md) | Exact settings, cells, output, safe stop and recovery |
| [Implementation status](IMPLEMENTATION_STATUS.md) | Actual modules, tensor/parameter counts and remaining fit work |
| [Validation](VALIDATION_AND_RECOVERY.md) | Mathematical checks, complete-update rollback and evidence boundaries |
| [Retention scope](CSV_AND_NATIVE_SCOPE.md) | Actual engineering CSV/native records versus the approved full-fit schema |
| [Master plan](../../phase3TreadTwoGeometryPlan.md) | Original approved geometry-first scientific design |
| [Local proof](validation/DELIVERY.json) | CPU/offline evidence, not an actual two-T4/HF claim |

```mermaid
flowchart LR
  P[Approved design] --> R[Implemented model and engineering store]
  R --> L[Local numerical / restored-state checks]
  L --> K[Run NB16 on two T4s]
  K --> H[Verify published HF receipt independently]
  H --> F[Finish full 40-family shard integration and freeze NB17]
  F --> T[One measured 60-epoch fit]
  T --> V[NB18 saved-evidence review]
```

There are **355,879 trainable / 1,728,038 total parameters** including the frozen
1,372,159-parameter selected TREAD-ONE anchor. Geometry, texture and experimental
error-estimation gradients are isolated. No mileage classifier is fitted here.

NB16 has three code cells: private transport/source extraction; all actual checks
and direct HF publication; receipt and interactive gallery. It embeds 27 files
and stays below 1 MB. Readable source is in `runtime/`; the notebook is generated
from that source. Once the user executes NB16, preserve its exact source/notebook
identity and inspect its outputs rather than regenerating it.

NB17 and NB18 remain reserved next-stage names. The approved acceptance sequence
requires actual two-T4/HF evidence before measured fitting; no empty placeholder,
simulated pass or auto-starting training notebook is supplied. Both workstations
and all completed classifier experiments stay preserved.
''')

write(BASE/'RUNBOOK.md', '''# NB16 — Kaggle runbook

**Use the new preparation notebook; do not rerun NB08–NB15.**

| Setting | Required value |
|---|---|
| Notebook | [phase3_NB16_TREAD_TWO_GEO_Preparation_And_Contract_Gate.ipynb](../../notebooks/phase3_NB16_TREAD_TWO_GEO_Preparation_And_Contract_Gate.ipynb) |
| Accelerator | GPU **T4 ×2** |
| Internet | ON |
| Data | Add expanded `shanmuk4622/tire-dataset-prepared-phase2` |
| Secrets | Enable `HF_TOKEN`, authorized to write `Shanmuk4622/tyre-wear-study` |
| Output space | At least 4 GB free initially; a 2 GB free reserve is checked between updates |
| Start | Run All in order, in a fresh kernel if another model runtime/HF SDK is already imported |

The images are read directly from Kaggle input. Only the pinned 5.5 MB selected
anchor is downloaded from HF. Source extraction and the private HF SDK live in
working output; the notebook does not replace Kaggle Torch/CUDA or install a new
backbone. The optional 1 TB scratch assumption is not used.

## Three cells and expected progress

| Code cell | Work | Expected evidence |
|---|---|---|
| 1 | Private `huggingface_hub==0.36.2` with no dependency replacement; verify/extract every embedded file | 27 matching source/config files |
| 2 | Full dataset SHA/overlay checks; exact anchor warm start; math/isolation/crash tests; batch8 update1→4 and local restored comparison; publish unit1 plus latest completed evidence; actually download and continue unit1→4 in a fresh worker; final publication/release | `ENGINEERING_PROOF.json`, `HF_CONTINUATION_PROOF.json`, `FINAL_GATE.json`, verified HF revision |
| 3 | Print result, show the interactive 32-case gallery and download links | `RUN_RESULT.json`, `Engineering_Passport.html` |

GPU1 owns immutable anchor features, one source image at a time. GPU0 owns all
355,879 trainable parameters. There is one intended model; direct/local/HF paths
are disposable equivalence checks, not three measured candidate fits. No held-out
image enters the optimizer or engineering fixture selection.

The coordinator prints the last committed stage/update every 30 seconds while
waiting. Worker output identifies checksum verification, inference, backward,
CSV/native commit and actual downloaded-state continuation. HF quota/backoff
messages are visible. Long checksum/upload waits are not silently labelled a
numerical failure. Actual speed/memory is recorded; no T4 wall-time promise.

## Stop and restart

```mermaid
sequenceDiagram
  participant U as Kaggle Stop
  participant C as Coordinator
  participant W as Secret-free worker
  participant H as Hugging Face
  U->>C: Catchable interrupt
  C->>W: STOP_REQUESTED marker
  W->>W: Finish/commit current unit; stop before next
  C->>H: Immediate direct-file flush attempt
  H-->>C: Verified revision or visible backoff/failure
  C->>C: Keep last verified/pending receipt
```

Keep the existing Kaggle files. Rerun the unchanged notebook. Partial local
fixtures restore the complete pointer and continue; scientific evidence must
match. A compatible completed run displays its existing receipt with no new
optimizer step or HF write. A fresh empty session can repeat these disposable
engineering checks, then prove a true published-checkpoint continuation. That is
distinct from the required fresh-session auto-resume of the future measured fit.

Forced kernel/process/session death cannot execute cleanup. Preserve saved
Kaggle output where available; bytes never uploaded cannot be guaranteed. Never
reset a protocol/hash, delete a live owner or create a second writer to bypass
an incompatible checkpoint or quota wait.

| Failure | What to do |
|---|---|
| Wrong dataset/GPU/secret | Correct the setting and rerun unchanged |
| Different source/environment | Preserve output and the exact notebook; report the mismatch |
| Worker stops with traceback | Keep `WORKER_STOP.json`, output and notebook transcript |
| HF publication pending | Keep output; inspect `HF_PENDING.json` and `LAST_HF_PUBLICATION.json` |
| Live owner conflict | Stop the other copy cleanly or wait for its 45-minute lease |
| HTTP429 / local budget | Allow the server-directed wait; never reset request history |
| Invalid support / determinism check | Preserve evidence; do not disable the check or change loss/data |

## Finish and return

Success prints `engineering_gate_passed`, with actual dual-T4 and actual
HF-restored continuation true. `measured_fit_ready=false` is intentional: this
notebook proves preparation, while the complete measured telemetry/shard/staging
integration still requires acceptance. Save the executed notebook with outputs.
The next step is independent HF verification, documentation update and final
NB17 integration/freeze. Do not rerun old training or start another classifier.
''')

write(BASE/'IMPLEMENTATION_STATUS.md', '''# GEO implementation and module ownership

**Implemented preparation, not a completed measured trainer.** The original
[architecture](plan/ARCHITECTURE.md), [experiment](plan/EXPERIMENT.md) and
[retention](plan/RETENTION.md) remain the scientific design. This dated record
distinguishes executable work from the next integration milestone.

## Actual parameter inventory

| Group | Parameters | Ownership |
|---|---:|---|
| Frozen original TREAD-ONE | 1,372,159 | GPU1, immutable selected seed3/epoch20 |
| Mask/raw-point refinements | 27,068 | GPU0, geometry loss only |
| Texture encoder/context/projector/decoder | 326,632 | GPU0, SSL loss only |
| Experimental annotation-error head | 2,179 | GPU0, detached observed-error targets |
| Trainable total | **355,879** | Three disjoint AdamW groups, separate norm1 clips |
| Packaged total | **1,728,038** | Below both approved parameter ceilings |

The checkpoint remains `a5b872ed…`, source `d16cd4ee…`. Hooks expose the existing
48×128×96 decoder features, 256D guide features and raw point parameters without
rewriting the original forward/state keys. Zero-initialized residual heads
reproduce original masks/points exactly on the matched local math path.

| Source | Implemented responsibility |
|---|---|
| `data_geo.py` | Exact overlay/source/checksum adapter; original ignore targets; seeded photometric views; no flips |
| `schedule_geo.py` | 386 primary +6 marked padding draws,49×8, unique IDs/batch and≥2 conservative groups |
| `anchor_geo.py` | Strict selected-weight SHA/load, frozen eval, microbatch1 and detached feature transfer |
| `atlas_geo.py` | Detached predicted-support warp, original canonicalizer agreement and replayable row parameters |
| `model_geo.py` | Bounded refinements, patch-local four-channel LC, axial context, VIC projector, masked decoder and detached quality head |
| `objective_geo.py` | Explicit geometry/quality/VIC/masked Huber reductions and human-only point-error targets |
| `store_geo.py` | Finite engineering complete-update transaction, immutable unit CSV/NPY, two rolling native states, verified snapshot |
| `engineering_geo.py` | Real batch8 four-update local/HF seams, independent group clips, native observations and extensive telemetry |
| `checks_geo.py` | Independent loss calculations, actual hidden-RGB test, missing-target/dense confusion checks and crash injection |
| `worker_geo.py` | Separate numeric subprocess without HF_TOKEN |
| `coordinator_geo.py` | Kaggle ownership/status/stop, 30-minute/major publication, verified download/continuation, finite storage report |
| `report_geo.py` | Interactive update filter and 32 predicted-mask cases |
| Nine copied dependencies | Exact frozen anchor/data/geometry/LC/common/HF transport copies; provenance in `VENDORED.json` |

`runtime/FILES.json` binds the new source/config package. Its 40-family schema is
the approved **design proposal**, not a claim that all measured-fit emitters are
already wired. Existing executed source catalogs are untouched.

## Pre-freeze technical clarification

Texture photometric contrast uses a **per-tile mean** before the per-tile 9×9 LC
operation. A global atlas mean would let hidden tile pixels affect visible tile
inputs. The actual hidden-RGB intervention now leaves visible LC values and the
masked-context prediction exactly unchanged. Geometry photometric contrast stays
per-image/per-channel, with brightness→contrast→gamma and clipping at each stage.
This clarification is recorded before a measured fit; no completed experiment
was changed.

## Still required before NB17

| Work | Why it remains a prerequisite |
|---|---|
| Actual NB16 T4/HF receipt and independent audit | CPU/offline proof does not establish real two-GPU/transport behavior |
| All40 CSV/443-field measured emitters | Engineering tables intentionally have a narrower finite schema |
| At-most8-update raw stream→NPY shard sealing/rotation | Engineering uses one immutable NPY per tensor/unit and never claims a scalable 2,940-update layout |
| Full measured storage and staging peak | Current fixture measures body/vector/dense sizes; other40-family paths and copy/hardlink staging are incomplete |
| Train-only mean predictor, G0 baseline and fixed81-image validation | Required original selection/non-collapse rules; not needed for disposable optimizer seams |
| Full60-epoch coordinator and fresh-session HF auto-resume | Engineering fresh-session reruns are separate from the future measured-fit guarantee |
| Epoch/selected/final/intervention/export review | No candidate was trained or selected here |

NB17 is created after these checks/integration, then one seed/60epochs/2,940updates
may be measured. NB18 follows completed retained evidence. No additional
photography, hidden model sweep, production promotion or workstation changes.
''')

write(BASE/'CSV_AND_NATIVE_SCOPE.md', '''# NB16 data retention — actual engineering scope

The user prioritizes detailed training evidence and directly browsable CSV.
NB16 retains disposable engineering observations and recovery, with no ZIP/NPZ
or database envelope. The future measured contract remains **40 CSV families /
443 fields** in [SCHEMAS.json](plan/SCHEMAS.json). It is not silently reduced to
the smaller engineering fixture.

## Actual engineering unit tables

| CSV family | Unit rows / meaning |
|---|---|
| `samples` | Eight source IDs, historical train roles, components and point provenance |
| `objectives` | All geometry numerator/count terms, VIC batch terms, masked Huber sum/count, texture eligibility/ramp and quality sum/count |
| `gradients` | Three preclip group norms, clip limit and completed group counters |
| `parameter_gradients` | Every trainable tensor: preclip norm/max/mean/zero fraction, availability, count and finite flag |
| `geometry_points` | Six predictions/source plus available human/weak targets and weights |
| `region_metrics` | Both masks' exact full-grid TP/FP/FN and Dice |
| `quality_errors` | Three raw error predictions, detached targets and availability; point target human-only |
| `vectors` | Every96D global and64D projected coordinate for both views |
| `tile_support` | Both views/all32tiles: valid coverage, geometric mass, selected mask and eligibility |
| `masked_tiles` | Exact packed native slot→source tile mapping |
| `view_plans` | Actual brightness/contrast/gamma, masked count, eligibility and valid loss mass |

Additional direct files include all570 source-role/component records, the full
60-epoch23,520-draw schedule, actual device peak memory in `RESOURCES.csv`, and
safe HF transport-event CSV. The engineering schema is emitted as
`ENGINEERING_SCHEMA.json`. The private SDK's safe JSONL diagnostic events are
also exported into directly browsable CSV; no credential/header values beyond
safe rate-limit metadata are logged.

Each unit saves uncompressed NPY mask logits, atlas row parameters/validity, both
views' pre/post-context tokens, and up-to8 masked predictions, targets and valid
loss weights. Unused packed slots are identified by `masked_tiles`, not counted
as real observations. Actual forward-support masks/grids can be reconstructed
from source SHA/preprocessing and retained logits/row parameters. Points and
embeddings are additionally preserved in roundtrippable direct CSV.

## Complete state and transaction boundaries

```mermaid
flowchart TD
  A[Compute all group gradients] --> B[Separate clips / apply complete optimizer unit]
  B --> C[Flush immutable CSV and NPY unit]
  C --> D[Write next rolling state with optimizer / RNG / cursor]
  D --> E[Verify hash metadata and unit chain]
  E --> F[Atomic POINTER selects complete state]
  F --> G[Immutable publication snapshot]
  G --> H[Publish direct objects then selecting manifest]
```

Native PT contains trainable state, all Adam moments/group counters, Python,
NumPy, CPU/CUDA RNGs, explicit cosine cursor convention, FP32/no-scaler status,
contract and complete unit list. Two state files rotate; the pointer changes
only after evidence/state durability. An interrupted uncommitted unit is replayed
after restoring the previous complete state. Pending stale files are excluded
from selected publication snapshots, never mistaken for measured evidence.

The coordinator retains a dedicated unit1 snapshot for the actual HF continuation
test **and** the latest committed engineering progress. Thirty-minute/major/stop
flushes select immutable snapshots. It does not repeatedly publish only an old
unit1 while hiding later committed work.

## Measurement versus full-fit budget

The finite report measures actual vector CSV widths, native observation bodies,
rolling-state bytes and a dense-row export probe. It extrapolates to the approved
29,520 source/59,040 view upper bounds, with explicitly provisional allowances.
`complete_40_family_measurement=false` and
`measured_fit_storage_gate_passed=false` remain honest until measured emitters,
8-update shard lifecycle and staging-copy peaks are integrated. The design's
15GB logical/18GB peak/2GB reserve within20GB remains the future acceptance rule.
No existing training evidence is removed to make a projection pass.
''')

write(BASE/'VALIDATION_AND_RECOVERY.md', '''# NB16 validation and recovery proof

**Local proof is CPU/offline. Actual dual-T4/HF proof is pending user execution.**
See the [delivery receipt](validation/DELIVERY.json) and
[local mathematical receipt](validation/LOCAL_ENGINEERING.json).

| Check | Local result / precise scope |
|---|---|
| Data/point/component identity | Every dataset checksum;386/81/103 roles;70human/152weak/164none train point kinds; six conservative groups2/3/5/60/67/249 |
| Warm start | Exact refined/anchor mask logits and points with zero adapters |
| Atlas | Original canonicalizer RGB/support exact; row-parameter reconstructed grid exact |
| Gradient isolation | Geometry-only, texture-only and quality-only backward reach only their own trainable group; anchor has no gradient |
| Hidden tile leakage | Changed hidden RGB gives identical visible LC and masked-context predictions; token replacement also tested |
| Loss arithmetic | Independent loop covariance and FP64 NumPy masked-Huber numerator/count reconstruct the texture loss |
| Missing/ignored targets | Zero supervision counts produce finite reductions; point order remains valid; quality point targets are human-only |
| Dense evidence | Coarse foreground/valid subpixel counts reconstruct original-grid TP/FP/FN exactly |
| Texture skip | No texture AdamW parameter, decay or moment update when support is unavailable |
| Resume | Actual batch8 model, four updates direct versus native restore after update1; parameters, moments, counters, RNG and scientific files exact |
| Separate process | Byte-verified offline snapshot copy, new worker continuation1→4 matches direct model/moments/counters/scientific observations |
| Transaction crashes | Failure after evidence, after state-before-pointer, and after pointer: restore exactly the prior or newly durable unit |
| Float serialization |10,000 FP32 values roundtrip through9-digit CSV without a bit change |
| Notebook | All27 embedded files match readable source; code parses; Jupyter notebook schema validates; no outputs/execution counters shipped |
| Preservation | All24/30/32 frozen candidate files,47+18 workstation entries,27 paired-runtime entries,103 original Markdown records and completed notebook hashes preserved |

The gallery has32 actual predicted-mask cases and an update filter. Numerical
receipts are engineering evidence; no classification, physical wear, depth,
safety or generalisation result is inferred from these disposable updates.

## What NB16 must establish in Kaggle

```mermaid
sequenceDiagram
  participant D as Direct fixture
  participant H as HF verified snapshot
  participant N as Fresh secret-free worker
  D->>D: Commit update1, keep immutable seam
  D->>D: Complete update4 reference
  D->>H: Publish unit1 plus latest progress / sources / CSV
  H->>N: Download and SHA-verify selected files
  N->>N: Load real update1 optimizer/RNG/cursor
  N->>N: Continue updates2–4
  N->>D: Compare model/moments/counters/RNG/files
```

Only a live run through Publisher.download and actual twoT4s establishes the
live gate. Offline copies are explicitly marked in local receipts. Scientific
file comparison excludes timing/environment differences; same-environment RNG
and deterministic math are bound to the protocol. Cross-release/CPU-versus-GPU
bit equality is not promised; [PyTorch reproducibility guidance](https://docs.pytorch.org/docs/2.11/notes/randomness.html)
motivates the matched-environment checks.

## HF operation

The new namespace is `phase3/tread-two/geo-r1/engineering/<contract>` in
`Shanmuk4622/tyre-wear-study`. The coordinator is the sole owner/token holder;
workers make no Hub calls. The unchanged copied transport stages immutable
direct files in batches≤80, verifies tree blob/LFS hashes and selects a final
manifest. Owner compare-and-swap fencing and a45-minute lease prevent competing
writers. Completed runs release ownership.

Normal flush cadence is30minutes plus major stages and catchable stop/error.
The shared local request ledger imposes96 commit attempts/hour and obeys server
Retry-After/RateLimit policies. Cross-host account traffic cannot be atomically
counted by one session; server headers remain authoritative. HF documents
[separate API/resolver/page buckets](https://huggingface.co/docs/hub/en/rate-limits),
so the historical128/hour concern is not treated as a universal quota.

Catchable stop waits for a complete worker unit then attempts publication;
forced death cannot upload unsent bytes. NB16 fresh-session disposable reruns
must not be confused with the future NB17 automatic measured-state restore.

## Reproduce locally

Use the existing environment without package replacement:

```powershell
.\\phase3\\run_cv.cmd -B phase3\\analysis\\build_nb16_geo.py
.\\phase3\\run_cv.cmd -B phase3\\analysis\\validate_nb16_delivery.py
```

The builder refuses to overwrite a notebook with execution outputs. Do not run
it against the user's subsequently executed NB16. Fixture/cache/native files are
ignored by Git; readable sources, compact receipts and the current notebook are
kept visible. No local token discovery or HF write is performed by validation.
''')

write(ROOT/'phase3NB16Preparation.md', '''# NB16 — TREAD-TWO / GEO preparation delivered

**1 November 2026.** The approved geometry-first model is implemented for the
next engineering gate. [Open NB16](notebooks/phase3_NB16_TREAD_TWO_GEO_Preparation_And_Contract_Gate.ipynb)
with **T4×2, Internet ON, the existing expanded Kaggle input and HF_TOKEN**, then
Run All. Start with4GBfree output. [Exact runbook](tread_two/geometry_first/RUNBOOK.md).

## Delivered versus pending

| State | Evidence |
|---|---|
| Implemented | Strict original anchor, bounded geometry heads, detached atlas, custom texture/axial/masked branches, experimental error head |
| Locally checked | Exact four-update restored model/moments/counters/RNG, independent loss math, leakage/isolation/skip tests and crash rollback |
| Notebook ready | Three code cells,27 embedded matching files, private HF SDK, visible coordinator, direct CSV/native publication and32-case interactive gallery |
| Actual Kaggle/HF | Pending NB16 execution; no local credentials or remote write used during delivery |
| Measured fit | Not started; NB17 awaits actual gate and remaining40-family shard/staging integration |
| Completed model review | NB18 follows a completed measured candidate; no new model result exists today |

```mermaid
flowchart LR
  A[Frozen TREAD-ONE1372159] --> B[Geometry refinements27068]
  B --> C[Detached atlas]
  C --> D[Texture / axial / masked modules326632]
  A --> E[Detached error inputs]
  B --> E
  E --> F[Experimental error head2179]
```

Total **1,728,038 parameters**, of which **355,879 trainable**. Texture/error
losses cannot reshape geometry through gradients. The first pilot has no mileage
classification objective. Existing labels remain proxies, not physical wear,
depth or safety measurements.

The finite engineering fixture saves more than scalar loss: all mask logits,
pre/post-context tokens, masked predictions/targets/weights, all global/projected
coordinates, tensor/group gradients, point/region/error records, tile support,
view factors, full schedule/source identities and complete recovery state.
All tabular evidence is directly browsable CSV. The approved future40-family /
443-field schema stays separate from the actual11-family unit engineering schema.
See [retention scope](tread_two/geometry_first/CSV_AND_NATIVE_SCOPE.md).

The actual hidden-RGB test exposed the need for tile-local photometric contrast
means as well as tile-local LC kernels. That clarification is implemented before
measured freeze and verified exactly. No prior experiment source was modified.

Local acceptance uses CPU in `cv_conda`; the installed GTX1650 is not passed off
as dual-T4 evidence. [Delivery receipt](tread_two/geometry_first/validation/DELIVERY.json)
and [technical validation](tread_two/geometry_first/VALIDATION_AND_RECOVERY.md).
Both workstations,103 original Markdown files and all executed prior notebooks
are preserved. No new photos or independent tyres were added.

**Next:** run NB16, independently verify its public HF revision, inspect actual
two-GPU/storage results, finish the complete measured telemetry/shard protocol,
then deliver/freeze NB17 for one60-epoch fit. No old notebook rerun is required.
''')

block='''<!-- GEO_NB16_20261101 -->
## Current — GEO model implemented; run NB16 preparation

**1 November 2026:** the user approved the complete geometry-first plan and requested notebooks. [NB16 preparation](notebooks/phase3_NB16_TREAD_TWO_GEO_Preparation_And_Contract_Gate.ipynb) is implemented and locally checked: **355,879 trainable / 1,728,038 total parameters**, frozen original selected anchor, isolated geometry/texture/error branches and complete-update native/CSV recovery. Real batch8 four-update restore, loss/leakage/gradient/skip/crash checks pass locally. [Delivery and scope](phase3NB16Preparation.md).

**Next: T4×2, Internet ON, existing expanded Kaggle input, HF_TOKEN, at least4GBfree; Run All in NB16.** It publishes direct CSV/native engineering evidence, proves actual HF download/continuation in a fresh secret-free worker, and displays32interactive mask cases. No local HF credentials/writes were used. Actual dual-T4/HF proof remains pending.

NB16 is preparation, not the measured60-epoch fit. **NB17/NB18 are not yet delivered**: actual NB16 verification and complete40-family/443-field measured logging, shard/staging and validation integration precede the one measured candidate. The original plan JSON is retained as a dated design snapshot. No new photos, classifier sweep, production promotion or workstation change. Preserve executed notebooks/sources after the user runs them.
<!-- END_GEO_NB16_20261101 -->
'''
for name in ['README.md','AGENTS.md','phase3Progress.md','phase3NextSteps.md','phase3OpenItems.md',
             'phase3Understanding.md','phase3DataRetentionPolicy.md','phase3ModelRationaleAndLogging.md']:
    path=ROOT/name
    text=path.read_text(encoding='utf-8')
    text=re.sub(r'<!-- GEO_NB16_20261101 -->.*?<!-- END_GEO_NB16_20261101 -->\s*','',text,flags=re.S)
    text=text.replace('## Current — new geometry-first TREAD-TWO design planned completely','## Historical — complete geometry-first plan before implementation')
    first,rest=text.split('\n',1)
    write(path,first+'\n\n'+block+'\n'+rest.lstrip())

def prepend_section(name,section):
    path=ROOT/name
    text=path.read_text(encoding='utf-8')
    text=re.sub(r'<!-- GEO_NB16_DETAILS -->.*?<!-- END_GEO_NB16_DETAILS -->\s*','',text,flags=re.S)
    first,rest=text.split('\n',1)
    write(path,first+'\n\n<!-- GEO_NB16_DETAILS -->\n'+section.strip()+'\n<!-- END_GEO_NB16_DETAILS -->\n\n'+rest.lstrip())

prepend_section('phase3Decisions.md','''## NB16 implementation decisions — 1 November 2026

| ID | Decision | Reason / boundary |
|---|---|---|
| P3-D135 | Deliver the actual prerequisite NB16 now | The approved sequence requires real two-T4/HF proof before measured NB17 |
| P3-D136 | Use tile-local photometric contrast means | Actual hidden-RGB leakage check; no cross-tile mean shortcut |
| P3-D137 | Keep engineering and measured telemetry scopes explicit |11 actual unit families versus40/443 planned fit schema; no fabricated full-fit pass |
| P3-D138 | Retain unit1 seam and latest complete progress | Actual remote-continuation proof must not hide later committed engineering state |
| P3-D139 | Keep measured_fit_ready false until full integration | Raw stream/shards, full validation/selection and staging need separate acceptance |

[Implementation and precise remaining work](tread_two/geometry_first/IMPLEMENTATION_STATUS.md).''')
prepend_section('phase3EvidenceRegister.md','''## NB16 engineering evidence — 1 November 2026

| Evidence | Established scope |
|---|---|
| [DELIVERY.json](tread_two/geometry_first/validation/DELIVERY.json) | Actual local model math, offline restored continuation, notebook/source identity, preservation |
| [LOCAL_ENGINEERING.json](tread_two/geometry_first/validation/LOCAL_ENGINEERING.json) | Independent loss arithmetic, exact warm start/atlas, isolated gradients, hidden-RGB and state recovery |
| [NOTEBOOK.json](tread_two/geometry_first/validation/NOTEBOOK.json) | Current unexecuted self-contained NB16 identity, source count and size |
| [Vendored sources](tread_two/geometry_first/runtime/VENDORED.json) | Nine unchanged copied dependencies and source SHAs |
| [Runbook](tread_two/geometry_first/RUNBOOK.md) | Actual Kaggle settings, stop/recovery/output and next-step boundaries |

Actual T4/HF gate pending. No measured model outcome or new independent tyre.''')
prepend_section('phase3DocumentationInventory.md','''## NB16 implementation documentation — 1 November 2026

| Document | Purpose |
|---|---|
| [NB16 delivery](phase3NB16Preparation.md) | Concrete next notebook and implemented/local/live state distinction |
| [GEO entrypoint](tread_two/geometry_first/README.md) | Package flow and evidence/navigation |
| [Runbook](tread_two/geometry_first/RUNBOOK.md) | Exact Kaggle operation and troubleshooting |
| [Implementation](tread_two/geometry_first/IMPLEMENTATION_STATUS.md) | Parameters/modules, leakage fix and remaining measured fit work |
| [Retention](tread_two/geometry_first/CSV_AND_NATIVE_SCOPE.md) | Actual11 unit families/native state versus planned40-family measured contract |
| [Validation/recovery](tread_two/geometry_first/VALIDATION_AND_RECOVERY.md) | Independent numerical/transaction/live transport gates |
''')
prepend_section('phase3RepositoryMap.md','''## Implemented GEO preparation — 1 November 2026

| Path | State / purpose |
|---|---|
| `tread_two/geometry_first/runtime/` | New isolated model, losses, data, engineering store/worker/coordinator, report and copied frozen transport |
| `notebooks/phase3_NB16_TREAD_TWO_GEO_Preparation_And_Contract_Gate.ipynb` | Current self-contained preparation delivery; unexecuted |
| `tread_two/geometry_first/validation/` | Compact plan/local/notebook/delivery receipts; heavy fixture_ directories ignored |
| `analysis/build_nb16_geo.py` | Regenerates only an unexecuted NB16 from verified readable source |
| `analysis/validate_nb16_delivery.py` | Local math, source payload, offline fresh-process continuation and preservation |
| `analysis/document_nb16_geo.py` | Maintains current implementation documentation/tracker blocks |

NB17/NB18 are future stage names. No placeholders claim a runnable full fit.''')
prepend_section('phase3SessionLog.md','''## NB16 preparation implementation — 1 November 2026

User approved the complete GEO plan and requested notebooks. Implemented the frozen-hook anchor, bounded zero-init geometry heads, detached original-equivalent atlas, four-channel local-contrast patch encoder, two axial blocks,64D VIC projector, masked-target decoder and experimental detached error head. Measured355,879 trainable /1,728,038 total parameters, split27,068 geometry /326,632 texture /2,179 error.

Implemented exact existing-data overlay/component checks and a full-source coverage schedule; separate group clipping; loss numerator/count records; dense original-grid confusion reconstruction; finite engineering native/CSV complete-update persistence; owner-fenced HF transport copying; secret-free worker; visible30-second status;30-minute/major/stop flush; actual published-unit1 download/continuation design. Added individual tensor gradients, complete vectors/tokens/logits/masked targets/weights and view/support/provenance records. The live notebook creates a32-case interactive predicted-mask passport.

Local full-model tests passed warm-start/atlas equality, gradient isolation, hidden-token/actual hidden-RGB leakage, independent covariance/Huber reconstruction, missing/ignored targets, human-only point quality, texture-skip moment/decay invariance, dense block counts, interrupted pointer states and FP32 CSV roundtrip. Direct four-update and restored paths match model/moments/counters/RNG/scientific records; a fresh process continues a verified offline snapshot identically. Local CPU proof is explicitly distinct from actual T4/HF proof. No credential lookup or remote write occurred.

Built self-contained NB16 with27 embedded files/three code cells, under1MB; notebook schema and source payload checks pass. Clarified per-tile photometric contrast means before measured freeze, preventing a hidden-pixel shortcut. Preserved all completed runtime/catalog/notebook identities, both workstations and103 original Markdown records. The approved design snapshots remain historical; added six structured implementation/run/retention/validation documents and current trackers.

Next user action is NB16 T4×2/Internet/expanded Kaggle input/HF_TOKEN/RunAll. Measured_fit_ready remains false until its actual receipt is independently verified and the full40-family shard/staging/validation pipeline is integrated. NB17/NB18 are not yet delivered; no measured fit or new photos.''')

path=ROOT/'phase3DocumentationInventory.md'
text=path.read_text(encoding='utf-8').replace('All 40 families / 435 fields','All 40 families / 443 fields')
write(path,text)
for path,link in [(ROOT/'phase3TreadTwoGeometryPlan.md','phase3NB16Preparation.md'),
                  (BASE/'plan/README.md','../../../phase3NB16Preparation.md')]:
    text=path.read_text(encoding='utf-8')
    notice='<!-- NB16_STATUS -->\n**Implementation update, 1 November 2026:** [NB16 preparation is delivered]('+link+'). Local model/recovery checks pass; actual Kaggle/HF and full measured-fit integration remain pending. The design below is the original planning snapshot.\n<!-- END_NB16_STATUS -->\n'
    text=re.sub(r'<!-- NB16_STATUS -->.*?<!-- END_NB16_STATUS -->\s*','',text,flags=re.S)
    first,rest=text.split('\n',1)
    write(path,first+'\n\n'+notice+'\n'+rest.lstrip())
path=ROOT/'tread_two/README.md'
text=path.read_text(encoding='utf-8')
notice='<!-- NB16_STATUS -->\n**1 November 2026:** the new GEO model/preparation notebook is [implemented and locally checked](geometry_first/README.md). Run NB16 next; actual two-T4/HF gate and measured NB17 remain pending. Earlier ATLAS experiments stay closed/preserved.\n<!-- END_NB16_STATUS -->\n'
text=re.sub(r'<!-- NB16_STATUS -->.*?<!-- END_NB16_STATUS -->\s*','',text,flags=re.S)
first,rest=text.split('\n',1)
write(path,first+'\n\n'+notice+'\n'+rest.lstrip())
print(json.dumps({'new_Markdown_documents':6,'updated_current_trackers':8,'measured_fit_started':False,'live_gate_pending':True}))
