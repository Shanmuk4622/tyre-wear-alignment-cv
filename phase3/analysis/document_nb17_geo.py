"""Current measured-pilot handoff docs; distinguish NB16 completion from pending NB17."""
import json,re,sys,time,os
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'tread_two/geo_pilot';VALID=ROOT/'tread_two/geo_validation'
sys.path.insert(0,str(SRC))
from common import digest,read,sha
from schema_fit import SCHEMA,specification
note=read(VALID/'NOTEBOOK.json');resume=read(VALID/'RESUME.json');assert resume['status']=='passed'
assert resume['source_inventory_sha256']==digest(read(SRC/'FILES.json'))
assert sha(note['notebook'])==note['sha256']
budget=resume['storage'];cols=specification()['columns'];assert cols==468
def humanize(text):
    result=[];fenced=False
    for line in text.splitlines():
        if line.startswith('```'):fenced=not fenced;result.append(line);continue
        if fenced:result.append(line);continue
        parts=re.split(r'(`[^`]*`|\]\([^)]*\)|https?://\S+)',line)
        for i in range(0,len(parts),2):
            part=parts[i]
            part=re.sub(r'(?<=\d)(?=GB|MB|MiB|KiB|seconds|minutes|epochs|updates|images|families|fields|CSV|native|payload|scientific|matching|draws|marked|human|weak|absent|components|parameters|training|source|selected|paired|targets|scalar|direct|finite|views|files|tokens|groups|hour|case|point|sample|observations|draft|pre-fit)', ' ',part)
            part=re.sub(r'\b(Torch|Python|NumPy|Pillow|CUDA|cuDNN|epoch|update|unit|per|all|every|one|at|under|within|above|below|seed|rank|high|mid|low)(?=\d)',r'\1 ',part)
            part=part.replace('twoT4','two T4').replace('at least19','at least 19').replace('with at least19','with at least 19')
            part=re.sub(r'(?<!\d),(?=[^\s])',', ',part)
            part=re.sub(r';(?=\S)','; ',part)
            parts[i]=part
        result.append(''.join(parts))
    return '\n'.join(result)
def write(path,text):
    if path.parent==SRC or path.name=='phase3TreadTwoGEOProtocol.md':text=humanize(text)
    temporary=path.with_name(path.name+'.nb17.tmp');temporary.write_bytes((text.strip()+'\n').encode('utf-8'))
    for attempt in range(8):
        try:os.replace(temporary,path);return
        except OSError:
            if attempt==7:raise
            time.sleep(min(1.,.1*(attempt+1)))

write(SRC/'README.md',f'''# TREAD-TWO / GEO — NB17 measured pilot

**1 November 2026. Ready to run; measured training has not been executed here.**
[NB16 passed and is independently verified](../../phase3NB16Completion.md).
Run [NB17](../../notebooks/phase3_NB17_TREAD_TWO_GEO_Resumable_Pilot.ipynb) next:
**T4×2, Internet ON, the existing expanded Kaggle input, HF_TOKEN, Run All**.
Start a fresh run with at least **19 GB free**. Compatible restarts reuse local
or verified HF progress automatically. No NB16 rerun or new photography.

| Item | Status |
|---|---|
| Source/runtime |28 frozen files plus FILES.json;29 embedded notebook files |
| Notebook |Three code cells; under1 MB; unexecuted |
| Model |355,879 trainable /1,728,038 total parameters |
| Measured fit |One seed /60epochs /49updates per epoch /2,940updates |
| Logging |40 executable CSV families /{cols} fields, plus direct native tensors and recovery |
| Local tests |Full disposable lifecycle, fresh-process recovery, independent losses, rotation crashes, typed transport/replay and selected-export provenance |
| Actual NB17 live gate |Automatically runs in Kaggle before measured fitting; remains pending |

```mermaid
flowchart TD
  N[Verified NB16] --> G[Actual new-store HF / T4 / capacity gate]
  G --> R[Check one-experiment registry + restore newest compatible state]
  R --> C[Train-only mean / quality median + G0 validation controls]
  C --> T[One60-epoch fit / full CSV and native retention]
  T --> V[Each epoch: original geometry / collapse eligibility]
  V --> S[Selected checkpoint or explicit unavailable]
  S --> E[Selected/final all570 source-cohort endpoints]
  E --> P[Predeclared diagnostic probes and interactive overlays]
  P --> H[Final verified HF publication]
```

[Protocol](../../phase3TreadTwoGEOProtocol.md) · [Operations](OPERATIONS.md) ·
[Actual schema/retention](EVIDENCE_SCHEMA.md) · [Validation](VALIDATION.md).
The original approved [design](../geometry_first/plan/README.md) remains a dated
snapshot; its443 draft fields are refined to{cols} executable fields before fit.

No new classification head, mileage loss, depth/safety claim or workstation
change. Representation similarity, masked residuals and error estimates remain
research diagnostics. NB18 follows completed NB17 evidence; no extra fit starts
automatically if this hypothesis fails.
''')

write(ROOT/'phase3TreadTwoGEOProtocol.md',f'''# TREAD-TWO / GEO — implemented measured protocol

**1 November 2026. NB16 completed; NB17 ready, unexecuted.**
This is the separately frozen implementation of the approved geometry-first
objective. [NB16 verification](phase3NB16Completion.md) precedes
[NB17](notebooks/phase3_NB17_TREAD_TWO_GEO_Resumable_Pilot.ipynb).
The completed ATLAS classifier experiments and both workstations remain intact.

## Model and learning paths

| Component | Implemented behavior |
|---|---|
| Anchor | Exact selected TREAD-ONE seed3/epoch20,1,372,159 parameters, frozen eval; weights a5b872ed… |
| Geometry |27,068 parameters; zero-init mask-logit delta±1 and raw-point delta±.25; original ordered6-point decoder |
| Texture |326,632 parameters; predicted-support512×256atlas,32local64×64tiles,4 LC channels,96D encoder,2axial blocks,64D VIC projector and masked4×16×16 target decoder |
| Error estimates |2,179 parameters; detached48 frozen features +16geometry scalars; three annotation-error estimates, human-only point-error targets |
| Separation |No shared trainable task parameters; atlas/quality boundaries detached; independent group norm1 clipping |
| Classification |No mileage CE/ordinal objective or new trained classifier |

```mermaid
flowchart LR
  I[RGB512x384] --> A[Frozen project anchor]
  A --> G[Bounded geometry adapters]
  G --> O[Masks / boxes / points]
  O --> W[Detached predicted-support atlas]
  W --> T[Paired LC / axial texture learning]
  T --> M[Masked prediction + VIC loss]
  A --> Q[Detached quality input]
  O --> Q
  Q --> U[Experimental observed-error regression]
```

Geometry loss is masked BCE + soft Dice,2×weighted point SmoothL1(beta.02),
.1containment,.1eligible row alignment and.1anchor drift. Quality contributes
.25×SmoothL1(beta.02). Texture contributes a min(1,epoch/5) ramp times
25invariance +25(sum of view variance penalties) +sum of covariance penalties
+.5masked Huber(beta.1). Exact mask/point drift denominators are separate CSV
fields; no component is hidden inside a scalar total only.

The hidden-RGB check is conditional on a fixed predicted support/warp. Tile
photometric contrast and LC are local, and hidden tokens are replaced before
context. The frozen geometry anchor still sees the full source image; this is
not a claim that geometry carries no information about hidden regions.

## Data, schedule and optimization

| Item | Frozen choice |
|---|---|
| Input | Existing expanded shanmuk4622/tire-dataset-prepared-phase2; checksum06c6c9d0… /overlay f81a80d2… |
| Roles |386train /81validation /103historical development test; no reassignment |
| Conservative components |Training sizes2/3/5/60/67/249; low/mid/high1/1/4, not8 independent batch tyres |
| Point provenance |70human /152weak /164absent train images; weak loss weight.25 |
| Schedule |Every386 primary IDs once/epoch +6marked padding draws;49×8; within-batch unique IDs and≥2components |
| Optimizer |AdamW FP32, group LRs1e-4/1e-3/3e-4,wd1e-4,betas(.9,.999),eps1e-8,foreach/fused false |
| LR convention |Cosine at the previous completed global update; no LR warmup |
| Texture skip |Need≥4eligible distinct images/≥2components; otherwise every texture gradNone, no moment/decay-only step |
| Devices |GPU1 frozen inference/evaluation; GPU0 trainable modules; one candidate, no DDP |

Training covers23,520draws, including360padding draws; more telemetry is not more
independent photographs. Existing tyre identity/support limits remain relevant.

## Controls, validation and selection

Before optimization, compute the deterministic identity training-atlas mean
predictor from386images, valid counts, and training-only quality-target medians.
Unsupported mean coordinates predict0 with their count retained; zero is not
observed truth. Compute original G0 geometry on the81validation images once.

Every epoch uses the same seed-addressed pair of validation photometric views.
Mask RNG seeds stay fixed; tile eligibility follows current predicted support,
so selected tile identity can change when geometry support changes. Every actual
mask/weight/target is saved. Candidate and mean predictor are compared under the
same current targets and masks.

| Selection gate | Required |
|---|---|
| Mean two-mask validation Dice |≥G0−.002 |
| Human-point width MAE |≤G0+.001 |
| Invalid atlas fraction |≤G0+.01 |
| Median projector std |≥.1 |
| Covariance effective rank |≥4, one fixed view per eligible validation image |
| Mean predictor loss |Available and≥1e-6 |

Among eligible epochs minimize masked_Huber/mean_Huber +.1paired cosine distance,
earliest tie. The historical test never selects a checkpoint. If none qualify,
selected is unavailable; final/all epoch weights and rejection reasons remain.
Predeclared usefulness uses **the selected epoch's original validation metrics**
(≥5%masked-loss gain, paired distance≤.05) plus≥90%training draw loss coverage.
It does not pool training/test endpoint scores to claim a validation success.

Selected/final identity evaluation covers all570images with original cohort
roles; human/weak/missing targets remain distinct. Experimental error estimates
are compared with training medians using MAE/MSE/RMSE and fixed bins. Retrieval
uses eligible training references and excludes the same full source SHA and
entire conservative component; similarity is descriptive, not wear calibration.

Ten historical-cohort diagnostics are brightness.8/1.2,gamma.8/1.2,grayscale,
known5×5blur,explicit GT-background neutralization,reverse bands,reverse columns,
and top valid observable masked-residual tile removal. Structural mappings and
unavailable probes are recorded. These are not new photos or damage labels.

## Retention and capacity

All40families /{cols} executable fields are documented in
[schema/retention](tread_two/geo_pilot/EVIDENCE_SCHEMA.md). Every train/validation/
endpoint mask logit, point, quality output, row parameter, token and masked
prediction/target/weight is retained; vectors, boxes, losses, gradients, exposure
and diagnostics stay CSV. All60epoch weights, selected/final native exports and
complete rolling optimizer/RNG states are saved. No ZIP/NPZ/database envelope.

The final local fixture's adjusted measured forecast is
**{budget['adjusted_measured_projection_bytes']/1e9:.3f} GB**, with typed planning
upper estimate{budget['typed_upper_logical_bytes']/1e9:.3f} GB and projected peak
{budget['projected_peak_bytes']/1e9:.3f} GB. The actual Kaggle gate must establish
same-filesystem hardlinks and pass15GBlogical/18GBpeak/2GBreserve limits within
the20GBassumption. Local Windows copy tests do not prove Kaggle hardlinks.
Runtime guards stop safely rather than discard telemetry.

The measured model/result remains untested until the user executes NB17. Local
fixtures are engineering evidence only. No physical wear/depth/safety, three-class
unseen-tyre transfer or production promotion is established by this protocol.
''')

write(SRC/'OPERATIONS.md','''# NB17 — execution, publication and recovery

## Run settings

| Setting | Required value |
|---|---|
| File | [NB17 GEO resumable pilot](../../notebooks/phase3_NB17_TREAD_TWO_GEO_Resumable_Pilot.ipynb) |
| Kaggle accelerator | GPU T4 ×2 |
| Internet / Secret | ON /HF_TOKEN enabled for Shanmuk4622/tyre-wear-study |
| Data | Existing expanded shanmuk4622/tire-dataset-prepared-phase2 |
| Fresh output |≥19GBfree; no assumed1TBscratch |
| Kernel | Fresh if NB16/another runtime has been imported |
| Start | Run All, all three code cells in order |

The verified NB16 environment was Python3.13.15,Torch2.11.0+cu128,
NumPy2.1.3,Pillow13.0.0,CUDA12.8/cuDNN91900,two Tesla T4s. A different numerical
environment fails clearly rather than starting a second fit or claiming
cross-release bit equivalence. Preserve any existing progress. The private
HF SDK0.36.2 uses --no-deps; global Torch/CUDA are not replaced.

## Automatic sequence

```mermaid
stateDiagram-v2
  [*] --> VerifyNB16AndSources
  VerifyNB16AndSources --> LiveNewStoreGate
  LiveNewStoreGate --> StableExperimentRegistry: actual T4/HF/hardlink/space checks pass
  StableExperimentRegistry --> RestoreOrInitialize
  RestoreOrInitialize --> PublishLocalAhead
  PublishLocalAhead --> ControlsAndTrain
  ControlsAndTrain --> EpochValidation
  EpochValidation --> ControlsAndTrain: more epochs
  EpochValidation --> Endpoints:60epochs complete
  Endpoints --> ProbesAndExport
  ProbesAndExport --> FinalVerifiedPublication
  FinalVerifiedPublication --> [*]
```

The disposable gate uses training-role controls only, two epochs/two batches,
and a separate pilot-gate namespace. It publishes optimizer update1 within the
epoch, downloads that actual snapshot, continues in a fresh secret-free worker,
and compares final model/moments/RNG/cursor/scientific shards. The storage report
measures actual CSV widths, native sizes and the current filesystem. Matching
published certificates are reused; the measured initializer still verifies
snapshot creation before another optimizer step.

A stable geo-r1/EXPERIMENT.json binds the one measured source/environment/seed
intent using a parent-commit compare-and-swap. The fit namespace additionally has
one45-minute leased writer. Source/environment changes do not silently create
a second candidate. Completed remote RESULT exits without optimization.

## Complete-update native and CSV restore

Native recovery holds model, all optimizer groups/moments/counters, disabled
scaler status, explicit LR cursor, Python/NumPy/CPU/CUDA RNGs, next phase/image/
draw, partial mean/validation/endpoint accumulators, best/history, CSV/native
prefix hashes, immutable catalogs and idempotent service acknowledgements.

On restart, compare the local/remote unit history at the shared cursor. Download
the newer remote state directly into working run paths, or publish verified
local-ahead state before optimization. Refuse a divergent chain. Verify every
closed artifact and active prefix, truncate only uncommitted append tails, then
continue the next complete update or evaluation image. A validation interruption
does not restart an epoch. A completed compatible run has no extra optimizer step.

Two native state files rotate. Active data is copied into immutable CSV/NPY
shards before the atomic selecting POINTER changes; older active generations
remain long enough for the previous state. At most8training updates/64source
observations enter a native shard; close early at phase/epoch boundaries. CSV
chunks close at8MiB or phase boundaries. Closed files are immutable/hardlinked
for staging; active prefixes are bounded copies. The measured path refuses a
full-copy staging fallback that could exceed20GB. Engineering-only copy fixtures
remain separately labelled.

## HF cadence and stop behavior

One coordinator owns HF_TOKEN; numeric workers get no token and make no Hub
calls. Sources, input and anchor are verified before fitting. Publish initialization
before any measured update, initial-control completion, every30minutes, and
major/final completion. A catchable stop/error requests a complete worker unit,
then immediately attempts a consistent flush. A forced kernel/session kill
cannot execute cleanup or guarantee upload of unsent bytes.

Heartbeats identify worker phases and long transfer/verification waits every
30seconds. Server Retry-After/RateLimit remains authoritative. Defaults stay
conservative until observed headers; then API/resolver limits use at most80%
of that observed quota with ceilings800/4,000 per300seconds. The independent
96commit-attempt/hour guard is unchanged. Cross-host traffic cannot be counted
by one local ledger; server remaining/reset still fences requests. NB16's
observed1,000/5,000 headers motivate this pre-fit transport adaptation.

Direct immutable objects are batched≤80, verified by blob/LFS hashes, then a
selecting manifest commits the snapshot. Lost successful responses reconcile
their marker. Final publication releases ownership. Old studies are never
rewritten. Fresh-session restore reads the full selected data with visible status;
it may take time for a large retained history.

| Problem | Action |
|---|---|
| Wrong input/GPU/secret | Correct settings, rerun unchanged |
| Source/env/experiment registry mismatch | Preserve notebook and files; do not bypass identities |
| Concurrent owner | Stop the other writer cleanly or wait for its displayed lease |
| HTTP429 / budget wait | Keep session open; do not reset request history |
| Space/hardlink gate fails | Preserve its report; measured fitting has not started |
| Catchable stop/error | Keep working output; inspect HF_PENDING/WORKER_ERROR and rerun unchanged |
| Forced death | Restore surviving files/latest verified HF state; unsent bytes are not guaranteed |
| Nonfinite update | Restore last complete state, retain error event, stop; no automatic numerical retry |
| No eligible selected model | Keep final/all evidence; do not choose a test winner |

After execution, save NB17 with outputs. Inspect it without regeneration, pin the
HF revision, independently audit CSV/native histories/selection, update project
documents and create NB18's retained-evidence review. Both workstations stay
deferred and no additional photos/seed/sweep is automatically requested.
''')

schema='''# GEO executable evidence schema and retention

**40 families /468fields before measured freeze.** The original443-field proposal
is preserved. The executable contract corrects14 draft types and adds25fields
for disaggregated drift, complete SSL accounting, masking policy, state digest,
quota/time metadata, timing, box/gap availability and structural-probe mapping.
HF event/publication identities are replay-safe text IDs across fresh sessions;
training/sample/unit IDs remain integers and seeds retain uint64 precision.

CSV bools are0/1, missing values are empty with availability/count/reason context,
raw FP32 vectors/probabilities use9significant digits, derived FP64 uses17.
The actual full specification is embedded in CONTRACT.json.schema and emitted
for local inspection in geo_validation/EXECUTABLE_SCHEMA.json.

## All families and actual columns

| Family | Cadence | Columns |
|---|---|---|
'''
for family,meta in SCHEMA.items():
    schema+='| `'+family+'` | '+meta['cadence']+' | '+', '.join('`'+c['name']+'` ('+c['dtype']+')' for c in meta['columns'])+' |\n'
schema+='''
## Native bodies and reconstruction

All train/validation/identity endpoint source passes retain FP32 two-mask logits,
six points, three error estimates, atlas row parameters/validity, both views'
pre/post-context32×96tokens and up-to8masked4×16×16predictions/targets/weights.
Padded slots are explicitly absent in the mask plan, never fake observations.
Original RGB/annotation data is read-only on Kaggle, not duplicated per draw.

spatial_maps_index identifies a logical stream and global sample key. Resolve
that key through native_chunks_index's first/last sample bounds and NPY header.
For active unsealed streams, native state holds exact shape/dtype/first sample/
byte prefix/hash; the snapshot includes only that committed raw prefix. The
complete native state is CPU-readable and requires no model forward. Structural
probe mappings additionally define band/column permutations; they are not new
monotonic geometry warps.

Selected/final endpoints also retain dense CSV: each128×96cell stores probability,
prediction and original4×4foreground/valid subpixel counts. These exactly rebuild
full512×384TP/FP/FN. Boxes enclose native valid masks in half-open pixel coordinates.
All human/weak/missing point targets and target availability remain distinct.

The probe allowance is deliberately narrower: retain full geometry native,
points/error estimates/row parameters and full scalar/vector/mapping CSV, but
not probe texture token/prediction/target native bodies. The native keys for
those unsaved probe bodies are blank. Training/validation/identity endpoint
texture bodies are fully retained. This boundary is fixed before fitting.

## Loss/ledger semantics

Geometry CSV contains independent mask and point anchor-drift sums/counts because
their denominators differ. Batch VIC variance/covariance is recorded once per
batch; each64D projector coordinate permits independent recomputation. Masked
Huber preserves raw sums and channel-weight counts; the mean predictor is
evaluated under identical masks. Quality targets are detached; human-point
availability never includes weak proposals. Every trainable tensor and each
group's pre/post clip norm/count/step state is retained.

Sample predictions record the parameter update that actually generated them,
separately from the global completed training update. A selected epoch20 endpoint
therefore records parameter_update980 and completed_update2940, rather than
mislabeling its weights as final. Controls are Geo epoch0. Native chunk epoch
metadata follows the observation's checkpoint/epoch, not the next cursor.

unit_ledger.native_state_sha256/state_semantic_sha256 identifies the semantic
model/optimizer/RNG/cursor content. The selecting POINTER separately hashes the
actual serialized PT bytes. A self-referential file hash is not fabricated.
The ledger's kind describes the executed operation; the state cursor describes
the next operation. Transport batches are acknowledged inside native state so
replayed queue files do not duplicate records.

## Explicit measurement boundaries

Pipeline record_seconds is whole observed-unit wall time before native commit.
Feature time includes frozen inference/transfer, so separate transfer_seconds is
unavailable rather than invented. Geometry time includes refinements/warp/quality;
texture time covers paired forward paths; backward_optimizer_seconds is separate.
Previous commit time is lagged and unavailable immediately after restore. For
interventions the component timers describe the reference forward; whole-unit
time includes the actual intervention. Current allocated/reserved memory is
sampled per unit; these samples do not claim every transient micro-peak.

Transport CSV captures safe HTTP policy/status/time and publication receipts;
session-scoped request IDs and receipt-scoped publication IDs avoid fresh-session
counter collisions. The final commit's own acknowledgement is in the HF
PUBLICATIONS marker and local result/transport log; it cannot be recursively
included in the very manifest it acknowledges. Failed-network bytes not uploaded
remain local. No credentials/alternate token files are recorded.

Store rolling state after every complete update, keep all60epoch weights and
selected/final exports, retain every verified30-minute/major/stop HF snapshot.
Do not claim2,940permanent full optimizer copies: the two rolling states and
published snapshots have that role, while scientific observations are permanent.
'''
write(SRC/'EVIDENCE_SCHEMA.md',schema)

write(SRC/'VALIDATION.md',f'''# GEO pilot validation and evidence boundaries

| Check | Local evidence / status |
|---|---|
| Full streamed lifecycle |Disposable train-only mean/baseline,2epochs×2batches,validation,final export/probes;52complete units |
| Fresh process restore |Published-format update1 seam copied/byte-verified offline; model,optimizer,RNG,cursor exact;{resume['restore']['scientific_shards_exact']}scientific shards match |
| Independent objective reconstruction |Geometry/quality/SSL/total from typed CSV sufficient statistics, with declared FP32 tolerance |
| Native/CSV rotation |Crash after evidence, after state, after pointer while sealing64samples: prior/new complete state restored |
| Selected-export provenance |Explicitly injected fixture branch only: epoch1/parameter update2 while completed update4; not an actual selected model |
| HTTP budgeting |Observed policy80%under ceiling;96/hour commit guard unchanged; no remote write |
| Transport integration |Real read-only pinned NB16 logical file resolves through its manifest; request IDs distinct across sessions; replayed service batch idempotent |
| Notebook/package |Three unexecuted code cells,29embedded files; matching readable source and Jupyter schema |
| Preserved studies |Executed NB16 SHA/26source files, earlier candidate catalogs,103original Markdown records and both workstations unchanged |

Current receipts: [RESUME](../geo_validation/RESUME.json),
[loss accounting](../geo_validation/LOSS_ACCOUNTING.json),
[rotation/quota tests](../geo_validation/TRANSACTIONS.json),
[selected branch](../geo_validation/SELECTED_BRANCH.json),
[transport](../geo_validation/TRANSPORT.json),
[notebook](../geo_validation/NOTEBOOK.json).
Local tests are CPU/offline engineering checks, not a measured candidate.

Adjusted fixture forecast: **{budget['adjusted_measured_projection_bytes']/1e9:.3f}GB**;
typed planning upper:{budget['typed_upper_logical_bytes']/1e9:.3f}GB;
projected peak:{budget['projected_peak_bytes']/1e9:.3f}GB. Actual Kaggle hardlinks,
new-store HF restore and live resource/space checks remain automatic prerequisites
inside NB17. Local Windows permits finite copies for testing and does not prove
the measured same-filesystem staging path. Runtime space guards remain active.

The original NB16 actually passed two-T4/HF recovery. NB17's extended writer,
40-family emitters and phase cursor have separate checks. Source/state/numeric
environment mismatch never becomes permission to reset or start a second fit.
No model outcome, original-selection winner, physical-depth/safety claim or
production promotion has been observed for the unexecuted measured notebook.

For local reproduction use `phase3/run_cv.cmd -B` with the analysis builder,
validate_geo_gate.py,verify_geo_loss_accounting.py,test_geo_transactions.py,
test_geo_transport_integration.py and test_geo_selected_branch.py. They use the
existing cv_conda environment; no global package/credential lookup or remote
publication. Once NB17 is executed, preserve it and its sources instead of
regenerating it. Heavy fixtures/mirrors/native files stay ignored by Git.
''')

block='''<!-- NB16_COMPLETE_NB17_READY_20261101 -->
## Current — NB16 HF-verified; run NB17 GEO measured pilot

**1 November 2026:** executed NB16 is complete and independently verified at HF revision9f991635…: **174 direct files /60 CSVs /42,217 rows /65 native arrays**, actual twoT4s and HF-restored continuation,96matching scientific files, released owner. All27payloads match; earlier context/review/readiness manifests are unchanged. **Do not rerun NB16.** [Completion and insights](phase3NB16Completion.md).

**Next: [NB17 GEO resumable pilot](notebooks/phase3_NB17_TREAD_TWO_GEO_Resumable_Pilot.ipynb)** — T4×2,Internet ON, existing expanded Kaggle input,HF_TOKEN, fresh kernel and≥19GBfree for a new run; Run All. The new integrated storage/HF gate runs automatically before one seed/60epochs/2,940updates. Local complete lifecycle,307-shard recovery, loss/rotation/quota/transport and selected-export provenance tests pass. Actual NB17 live gate/measured results remain pending. [Protocol](phase3TreadTwoGEOProtocol.md) and [run guide](tread_two/geo_pilot/OPERATIONS.md).

Implemented40CSVfamilies/468executable fields (443draft +25pre-fit accounting/provenance fields;14type corrections), bounded direct NPY/CSV storage, all epoch weights and full rolling recovery. Stable experiment registry prevents accidental refits; source/env/history mismatch stops. Preserve executed notebooks/sources. Existing photos only; no classifier loss, physical-depth/safety claim or workstation change. NB18 follows completed NB17 evidence.
<!-- END_NB16_COMPLETE_NB17_READY_20261101 -->
'''
block=humanize(block)
for name in ['README.md','AGENTS.md','phase3Progress.md','phase3NextSteps.md','phase3OpenItems.md','phase3Understanding.md','phase3DataRetentionPolicy.md','phase3ModelRationaleAndLogging.md']:
    path=ROOT/name;text=path.read_text(encoding='utf-8')
    text=re.sub(r'<!-- NB16_COMPLETE_NB17_READY_20261101 -->.*?<!-- END_NB16_COMPLETE_NB17_READY_20261101 -->\s*','',text,flags=re.S)
    text=text.replace('## Current — GEO model implemented; run NB16 preparation','## Historical — NB16 preparation delivery before execution')
    first,rest=text.split('\n',1)
    if name=='phase3NextSteps.md':first='# Next steps — run NB17 TREAD-TWO / GEO'
    write(path,first+'\n\n'+block+'\n'+rest.lstrip())

def add_section(name,section):
    path=ROOT/name;text=path.read_text(encoding='utf-8');text=re.sub(r'<!-- NB17_DETAILS -->.*?<!-- END_NB17_DETAILS -->\s*','',text,flags=re.S)
    first,rest=text.split('\n',1);write(path,first+'\n\n<!-- NB17_DETAILS -->\n'+humanize(section.strip())+'\n<!-- END_NB17_DETAILS -->\n\n'+rest.lstrip())
add_section('phase3Decisions.md','''## NB16 completion / NB17 freeze — 1 November 2026

| ID | Decision | Evidence / limit |
|---|---|---|
| P3-D140 | Close NB16 without rerun | All174files/60CSV and actual T4/HF continuation verified |
| P3-D141 | Freeze a separate GEO measured runtime | Completed NB16 bytes/sources preserved; full40-family lifecycle added separately |
| P3-D142 | Correct14draft types and add25sufficient-statistic/provenance fields |468executable fields before measured fit; original443-field draft preserved |
| P3-D143 | Bounded raw prefixes→NPY plus immutable CSV shards and strict staging | No ZIP/NPZ; measured hardlink/space gate before optimization |
| P3-D144 | Header-driven API/resolver budgets, separate96/hour commit guard | NB16 actually advertised1,000/5,000 per300seconds; preserve server backoff |
| P3-D145 | Stable single-experiment registry and shared-history resume | Source/env changes cannot silently start another fit |
| P3-D146 | Utility evaluated on original selected validation metrics only | No training/test endpoint pooling or test-based checkpoint substitution |

[Implemented protocol](phase3TreadTwoGEOProtocol.md).''')
add_section('phase3EvidenceRegister.md','''## NB16 completion and NB17 acceptance

| Evidence | What it establishes |
|---|---|
| [NB16 completion](phase3NB16Completion.md) | Executed identity, pinned174-file public audit, resource plot and actual two-T4/HF proof |
| [NB17 resume](tread_two/geo_validation/RESUME.json) | New streamed lifecycle,307scientific shards and exact fresh-process model/moments/RNG/cursor |
| [Loss accounting](tread_two/geo_validation/LOSS_ACCOUNTING.json) | Independent per-batch geometry/quality/SSL/total reconstruction |
| [Transactions](tread_two/geo_validation/TRANSACTIONS.json) |64-sample seal crash points and observed quota/commit-budget behavior |
| [Transport](tread_two/geo_validation/TRANSPORT.json) | Actual pinned manifest resolution and replay-safe cross-session CSV IDs; no writes |
| [Selected branch](tread_two/geo_validation/SELECTED_BRANCH.json) | Injected test branch only: selected epoch/update provenance remains distinct from final counter |

Actual NB17 Kaggle/new-HF gate and measured model outcome remain pending.''')
add_section('phase3SessionLog.md','''## NB16 verified; NB17 implemented — 1 November 2026

User reported NB16 complete and requested HF verification, documentation and continuation. Preserved executed SHAea4f6784588f… with three complete cells/no errors and27matching payload files. Pinned public audit at9f991635… checked all173selected objects/174logical files,24,341,191bytes,60CSV/42,217rows,65finiteNPY arrays, both complete-unit chains and32-case viewer behavior. Original context-fit/review/readiness manifests remain unchanged and owner released.96scientific files match after actual two-T4/HF restore; independent raw-vector/target reconstruction agrees within FP32 tolerances. Diagnostic scalar warnings are not failed checks. The visible wait was conservative API-budget handling.

Implemented the isolated NB17 runtime: every40family emitter,468typed fields, train-only mean/quality-median controls, G0 validation, all60epoch weights, original eligibility/selection, same-source/component-excluded retrieval, selected/final all570image endpoints,10diagnostics and standalone interactive mask overlays. Corrected draft field types and added explicit loss denominators/provenance. Selected predictions retain their actual checkpoint epoch/update, distinct from final training counter. Utility uses original selected validation metrics plus training coverage. No classifier/depth/safety promotion is introduced.

Added bounded CSV/raw native prefix transactions,64-observation/8-update NPY sealing, two rotating full native states, source/env/history verification, local-ahead publication and fresh-session full HF restore. A stable registry binds one measured experiment. Single-owner transport has30-minute/major/stop flush, visible waits, actual-header80%API/resolver budgets and unchanged96/hour commit guard. Replay-safe request/publication IDs and native service acknowledgement prevent metadata duplication across sessions. Pinned logical files resolve through immutable manifests rather than assuming root-level paths.

Disposable local tests cover full lifecycle, exact fresh-process model/moments/RNG/cursor and307scientific shards, independent total-loss reconstruction, seal crash points, cross-session transport IDs, service replay and injected selected-export provenance. These are CPU/offline fixtures, not a measured model. Actual new live storage/hardlink/HF gate runs in NB17 before fitting. Built29-file self-contained notebook with three code cells under1MB; updated schema/run/protocol/acceptance docs and current trackers. No credentials read or remote write locally. All completed notebooks/runtime/workstations/original Markdown remain preserved.

Next: user runs NB17 T4×2/Internet/expanded input/HF_TOKEN/RunAll. After execution, independently verify actual publication and create NB18's retained-evidence review. No NB16 rerun, new photographs or automatic additional candidate fit.''')
add_section('phase3DocumentationInventory.md','''## NB16 completion / full GEO pilot docs

| Document | Purpose |
|---|---|
| [NB16 completion](phase3NB16Completion.md) | Pinned public audit, diagrams, table and real resource figure |
| [GEO protocol](phase3TreadTwoGEOProtocol.md) | Implemented model/data/loss/selection/retention and measured-run boundaries |
| [Pilot README](tread_two/geo_pilot/README.md) | Current runnable notebook and evidence navigation |
| [Operations](tread_two/geo_pilot/OPERATIONS.md) | Kaggle cells, live gate, stable registry, resume/stop/quotas/troubleshooting |
| [Evidence schema](tread_two/geo_pilot/EVIDENCE_SCHEMA.md) | Every468field/type, actual scope, gaps and reconstruction semantics |
| [Validation](tread_two/geo_pilot/VALIDATION.md) | Local tests versus pending actual new live gate |
''')
add_section('phase3RepositoryMap.md','''## Completed NB16 and measured GEO implementation

| Path | State / owner |
|---|---|
| `tread_two/nb16_completion/ea4f6784588f/` | Compact inspection/manifest/verification; ignored executed archive and full mirror |
| `tread_two/nb16_report/figures/` | Curated actual dual-T4 resource figure |
| `tread_two/geometry_first/runtime/` | Frozen executed NB16 source; preserved |
| `tread_two/geo_pilot/` | Isolated frozen NB17 source/data/schema/transport plus human documentation |
| `tread_two/geo_validation/` | Compact acceptance/schema/notebook receipts; ignored fixture_ directories |
| `notebooks/phase3_NB17_TREAD_TWO_GEO_Resumable_Pilot.ipynb` | Current unexecuted measured-pilot handoff |
| `analysis/build_nb17_geo.py` | Rebuild only an unexecuted new notebook; never executed NB16 |

No Git staging/push, completed-source rewrite or workstation change.''')

for path in [ROOT/'phase3NB16Preparation.md',ROOT/'phase3TreadTwoGeometryPlan.md',ROOT/'tread_two/geometry_first/README.md',ROOT/'tread_two/geometry_first/IMPLEMENTATION_STATUS.md',ROOT/'tread_two/geometry_first/plan/README.md']:
    relative=Path(__import__('os').path.relpath(ROOT/'phase3NB16Completion.md',path.parent)).as_posix()
    text=path.read_text(encoding='utf-8');text=re.sub(r'<!-- NB16_DONE_STATUS -->.*?<!-- END_NB16_DONE_STATUS -->\s*','',text,flags=re.S)
    first,rest=text.split('\n',1);notice=f'<!-- NB16_DONE_STATUS -->\n**Current update,1November2026:** [NB16 is complete and HF-verified]({relative}); no rerun. NB17 is delivered in the isolated geo_pilot branch. Earlier pending/preparation statements below are dated history.\n<!-- END_NB16_DONE_STATUS -->\n'
    write(path,first+'\n\n'+notice+'\n'+rest.lstrip())
path=ROOT/'tread_two/README.md';text=path.read_text(encoding='utf-8');first,rest=text.split('\n',1)
notice='<!-- NB17_READY -->\n**1November2026:** [NB16 completed](../phase3NB16Completion.md); [NB17 GEO pilot is ready](geo_pilot/README.md). Run NB17 next. Both workstations/old experiments remain preserved; no new photos.\n<!-- END_NB17_READY -->\n'
text=re.sub(r'<!-- NB17_READY -->.*?<!-- END_NB17_READY -->\s*','',text,flags=re.S);first,rest=text.split('\n',1);write(path,first+'\n\n'+notice+'\n'+rest.lstrip())
(VALID/'EXECUTABLE_SCHEMA.json').write_text(json.dumps(specification(),indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':'documented','CSV_families':40,'executable_fields':cols,'NB17_notebook':note['sha256'],'actual_measured_fit_pending':True}))
