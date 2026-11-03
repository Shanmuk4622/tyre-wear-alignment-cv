"""Structured completion evidence and durable, consistent next-step trackers."""
import json,re,sys
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,s):
    for old,new in {'13GBfree':'13 GB free','10.31GB':'10.31 GB','13GB':'13 GB','8GBrecommended':'8 GB recommended','60epochs':'60 epochs','2,940updates':'2,940 updates','2940updates':'2,940 updates','0/60eligible':'0/60 eligible','8455shards':'8,455 shards','60epochweights':'60 epoch weights','9867sampled':'9,867 sampled','9,867sampled':'9,867 sampled','all11636selectedobjects':'all 11,636 selected objects','40CSVfamilies':'40 CSV families','468fields':'468 fields','one seed/60':'one seed / 60','30-minute/':'30-minute / ','CPU/None':'CPU / None'}.items():s=s.replace(old,new)
    Path(p).write_text(s.rstrip()+'\n',encoding='utf-8')
folder=Path(read(ROOT/'tread_two/nb17_completion/LATEST.json')['path']);proof=read(folder/'VERIFICATION.json');assert proof['status']=='passed'
f=read(ROOT/'tread_two/nb17_report/FINDINGS.json');v=f['final_validation'];files=read(ROOT/'tread_two/nb17_report/FILE_VERIFICATION.json');native=read(ROOT/'tread_two/nb17_report/NATIVE_VERIFICATION.json')
delivery=read(ROOT/'tread_two/geo_postrun_validation/DELIVERY_CHECKS.json');assert delivery['status']=='passed'
notes=read(ROOT/'tread_two/geo_postrun_validation/NOTEBOOKS.json')
report=f'''# NB17 complete — GEO evidence and research outcome

**2 November 2026. The attached notebook is NB17, although the message says NB16.**
Its three code cells completed without saved errors. All 29 embedded files
match the frozen measured runtime. NB16 was already verified; neither notebook
needs a training rerun. The executed NB17 is preserved unchanged.

## Independent publication verification

| Evidence | Verified value |
|---|---|
| Executed NB17 SHA-256 | `{proof['executed_notebook_sha256']}` |
| HF revision | `{proof['revision']}` |
| Original fit protocol | `{proof['contract']}` |
| Logical publication files | 11,637, including the selecting manifest |
| Selected content objects | 11,636; all local downloaded bytes/SHA verified |
| Retained content bytes | 10,307,839,796 (10.31 decimal GB) |
| Direct CSV | {files['CSV_files']:,} files / {files['CSV_rows']:,} rows |
| Executable scientific schema | 40 families / 468 fields |
| Native arrays | 8,457: 8,455 indexed measured shards + 2 controls |
| Native checkpoint files | 63: all 60 epoch weights, final export, anchor and selected recovery state |
| Measured fit | One seed / 60 epochs / 2,940 updates |
| Training observations | 23,520 = 23,160 primary + 360 scheduled padding draws |
| All source observations | 30,447, including controls, validation, endpoint and probes |
| Final complete unit | 9,877; complete cursor, optimizer/RNG state and final digest verified |
| Publication owner | Released |
| Original NB13/NB14/NB15/NB16 publications | Exact selecting manifests unchanged |

[Verification receipt](tread_two/nb17_completion/4b92d8fd7081/VERIFICATION.json),
[executed-notebook inspection](tread_two/nb17_completion/4b92d8fd7081/INSPECTION.json),
[pinned manifest](tread_two/nb17_completion/4b92d8fd7081/REMOTE_MANIFEST.json),
[file/schema proof](tread_two/nb17_report/FILE_VERIFICATION.json) and
[native proof](tread_two/nb17_report/NATIVE_VERIFICATION.json).
The verification used public HF reads, no local credentials and zero remote writes.

## What the model actually achieved

| Original validation metric | G0 baseline | Final epoch 60 | Interpretation |
|---|---:|---:|---|
| Mean tyre/tread Dice | {f['G0']['mean_dice']:.9f} | {v['mean_dice']:.9f} | Floor preserved; delta {f['Dice_delta']:.9f} |
| Human point MAE / image width | {f['G0']['human_point_MAE']:.9f} | {v['human_point_MAE']:.9f} | {100*f['point_relative_gain']:.4f}% lower retrospectively |
| Invalid atlas fraction | 0 | 0 | Support remained available |
| Masked Huber | — | {v['masked_huber']:.9f} | Same final masked observations |
| Train-mean control Huber | — | {v['mean_huber']:.9f} | Train-only identity control |
| Relative masked loss gain | — | {100*v['masked_gain']:.4f}% | Below declared 5% utility requirement |
| Paired embedding cosine distance | — | {v['pair_cosine']:.9f} | Strong consistency alone is insufficient |
| 64D projector rank | {f['G0']['rank']:.6f} | {v['projector_rank']:.6f} | Original eligibility requires rank ≥ 4 |
| 96D pooled encoder rank | 14.780426 | {v['global_rank']:.6f} | Separate descriptive audit also near rank one |
| Projector median std | {f['G0']['std_median']:.6f} | {v['projector_std_median']:.6f} | Scale improved; diversity did not follow |

![Validation geometry, masked control and latent-rank trajectories](tread_two/nb17_report/figures/GEO_Validation.png)

**All 60 epochs meet the geometry floor; zero meet representation eligibility.**
At the model's 384-pixel input width, the mean human point error changes from
{384*f['G0']['human_point_MAE']:.4f} to {384*v['human_point_MAE']:.4f} pixels:
about {384*(f['G0']['human_point_MAE']-v['human_point_MAE']):.4f} pixels lower.
The 5.23% figure is a relative error reduction, not a physical tread-depth gain.
The projector validation rank ranges {f['rank_min']:.6f}–{f['rank_max']:.6f}.
The final leading covariance direction explains 99.44% of projector variance and
99.71% of pooled-encoder variance. These are specific held-out-vector findings;
they do not prove every per-tile tensor is constant or identify a unique cause.
The original checkpoint-selection rule correctly leaves `selected=None`.

The 570-image pooled rank is 3.607015 for the projector and 2.025142 for the
pooled encoder. Test-only ranks are 1.015306 / 1.029422. Pooling roles can hide
held-out degeneracy; none of these descriptive values can replace the original
validation rule. No test-based reselection or promotion is performed.
The native pre/post-context tokens and component scatter are also audited:
99.0481%/99.3859% of validation projector/pooled variation lies between its two
conservative components, and local encoder diversity is already limited.
[Detailed postmortem, tables and mathematical constraints](phase3TreadTwoGEOLessons.md).

![Role-specific latent spectra and saved intervention sensitivity](tread_two/nb17_report/figures/GEO_Representation_And_Probes.png)

## Further insights and limits

| Finding | Evidence and implication |
|---|---|
| Strong pixel segmentation | Native coarse logits and all 14,008,320 final dense CSV rows checked against existing PNG masks; full-grid confusion counts reconstruct exactly |
| Human-only point evaluation | Validation human count 25; weak/absent targets stay excluded. A point MAE improvement on this small development cohort is not an independent deployment result |
| Error head has a weak control result | On final validation all three learned error estimates have higher MAE than the train-only constant median. Do not describe them as calibrated confidence |
| Appearance sensitivity remains | Mean embedding change under blur ≈0.522342, background-neutral ≈0.223975, grayscale ≈0.009521; brightness/gamma changes are much smaller |
| Spatial reversals have small pooled changes | Band reversal ≈0.000086; column reversal ≈0.000145. This measures the saved collapsed pooled representation, not proof of a causal architectural defect |
| Full source exposure | Every one of 386 training images appears once per epoch before six explicit padding draws; all 23,520 draws SSL-valid |
| Limited independent support | 570 photos: 386 train / 81 validation / 103 test; only six conservative train components, low/mid/high = 1/1/4; no middle-class held-out cohort |
| No new class/depth/safety evidence | GEO fitted no mileage-classification head; no measured physical-depth/safety targets exist in this run |

![Experimental annotation-error controls and independent task gradient norms](tread_two/nb17_report/figures/GEO_Quality_And_Gradients.png)

The generated [interactive local atlas](tread_two/nb17_report/GEO_Review_Atlas.html)
contains 570 source images, 3,420 predicted point markers and 720 human-target markers.
Filter by original role/component, sort by mask Dice, human error or masked-control
ratio, search an image, page through cases and export the filtered metrics as CSV.
Pink circles are predictions; green rings appear only for actual human targets.
The HTML is a generated local/HF artifact; Git retains its source and curated figures.

The completed 2,940 objectives reconstruct from retained sufficient statistics
and raw CSV vectors (maximum total difference about 6.93e-6, within declared
FP32/FP64 tolerances). Native patch predictions/targets/weights reconstruct
masked and mean-control losses, native pooling matches CSV vectors, and all
60 epoch weights are finite. The final export equals epoch 60 and the last state;
the frozen anchor remains unchanged.

Explicit gaps remain: historical optimizer states are rolling, not all permanently
retained; historical ledger link continuity is verified, but not every historical
unit digest can be re-derived without its discarded prefix maps. The final unit
digest and complete semantic state are independently verified. Anchor drift uses
retained sums/counts; there are no new anchor forwards. The original mean map
was built from full train-only targets, while only selected masked target bodies
are retained per observation; its entire construction cannot be independently
recomputed without additional forwards. Probe raw texture bodies were deliberately
not retained; their compact geometry, CSV vectors/statistics and metadata remain.

Resource telemetry contains 9,867 sampled observations:

| Quantity | Maximum sampled bytes | Interpretation |
|---|---:|---|
| GPU 0 allocated / reserved | 201,397,760 / 723,517,440 | Completed-unit samples, not within-step peaks |
| GPU 1 allocated / reserved | 58,809,344 / 96,468,992 | Completed-unit samples, not within-step peaks |
| Host RSS | 2,327,363,584 | Sampled resident process memory |
| Logical artifacts | 10,300,607,571 | Sampled local artifact total |

These are **not true within-step peaks**. The
`staging_bytes` field is blank throughout; do not invent a measured staging
series. The publication total is 10,307,839,796 bytes.
Component timing summaries are not end-to-end wall
time. Transport rows capture events ingested before the selecting snapshot;
its own final commit/verification cannot recursively be included in that same
snapshot. The pinned server revision and released owner establish publication.

## What happens next

```mermaid
flowchart LR
  A[NB17 complete and frozen] --> B[NB18 complete saved-evidence audit]
  B --> C[Direct CSV + native proof + image atlas]
  C --> D[Publish NB18]
  D --> E[NB19 capability passport and research handoff]
  E --> F[Publish; close this pilot as unpromoted research]
  G[NB20 Run All master] --> B
  G --> E
```

All justified postrun notebooks are supplied together. **Run
[NB20](notebooks/phase3_NB20_TREAD_TWO_GEO_All_In_One_Postrun.ipynb) once**; it
chains NB18 and NB19 automatically. Separate stage notebooks are also available.
Use CPU/None, Internet ON, existing expanded prepared Kaggle input and HF_TOKEN.
Fresh start needs at least 13 GB free; the 10.31 GB input is shared once.
[Full instructions](phase3TreadTwoGEOPostrunBatch.md).

Local full-data analysis, fresh notebook setup, stage interruption/hash detection,
offline publication/handoff/restore and interactive-script controls are checked.
**Actual new NB18/NB19 Kaggle/HF publication remains pending.** No new fit,
photo collection or workstation change is launched. This closes the GEO pilot,
not every future research direction or the broader application.
'''
write(ROOT/'phase3NB17Completion.md',report)
rows='\n'.join('| '+str(n)+' | ['+item['notebook']+'](notebooks/'+item['notebook']+') | '+item['mode']+' | '+str(item['bytes'])+' |' for n,item in zip([18,19,20],notes))
guide=f'''# GEO postrun batch — all notebooks delivered together

**Recommended: run NB20 once with Run All.** It executes NB18, publishes the
review, then executes/publishes NB19 without a manual handoff. The individual
notebooks are alternatives for separate runs; launching all three concurrently
duplicates work and competes for ownership.

| NB | Notebook | Mode | Delivered bytes |
|---|---|---|---:|
{rows}

## Kaggle setup and sequence

| Setting | Value |
|---|---|
| Accelerator | None / CPU |
| Internet | ON |
| Existing input | Expanded `shanmuk4622/tire-dataset-prepared-phase2` |
| Secret | Enable `HF_TOKEN`, write access to `Shanmuk4622/tyre-wear-study` |
| Fresh free space | At least 13 GB; full pinned input 10.31 GB, shared once |
| RAM | About 8 GB recommended; stream dense CSV and native shards |
| Action | Fresh kernel, preserve compatible session files, **Run All in NB20** |

NB19 also automatically satisfies a missing NB18 dependency. All three files
embed the same {notes[0]['payload_files']}-file verified runtime package. No hidden
local Python files, earlier notebook variables or new dataset are required.
Compatible NB17 files can be reused through exact hashes and immutable hardlinks;
otherwise the pinned HF content is restored into one shared cache.

```mermaid
sequenceDiagram
  participant K as Kaggle NB20
  participant H as Hugging Face
  participant W as Secret-free CPU worker
  K->>H: Restore pinned NB17 evidence or verify cache
  K->>W: NB18: schemas, metrics, native tensors and existing masks
  W-->>K: Completed source-bound stage receipts + visuals
  K->>H: Publish verified direct NB18 evidence
  K->>W: NB19: capability passport + research handoff
  W-->>K: Model card + CSV limits + interactive passport
  K->>H: Publish NB19; release owner
```

## Retention, recovery and troubleshooting

| Event | Behavior |
|---|---|
| Completed input download | Size/SHA retained; reused on rerun |
| Finished CPU stage | Exact artifact hashes and result receipt retained; skip when matching |
| Interrupted CPU stage | Replay from the preceding complete stage; no optimizer exists |
| Major boundary | Immediate direct HF publication |
| Active run | 30-minute publication cadence, safe request-budget ledger |
| Catchable Stop/error | Terminate numeric worker, attempt immediate completed-evidence/status flush |
| Server limit | Respect Retry-After/reset headers; bounded retries, retain local work |
| Forced process/session death | Cannot upload unsent bytes; restore published stages and available local cache |
| Existing matching completed HF result | Restore and skip; no new analysis or training |
| Wrong data/source/content | Stop at the technical gate; preserve evidence and report the mismatch |
| No selected model | Expected original scientific rejection; continue the research passport |

The original measured snapshot remains immutable. New namespaces are
`phase3/tread-two/geo-r1/review/<postrun-contract>` and
`phase3/tread-two/geo-r1/passport/<postrun-contract>`. Their contract binds the
pinned fit revision, exact new sources and scope. Every stage publishes direct
CSV and appropriate JSON/MD/PNG/HTML/NPY files; no ZIP/NPZ envelope. The original
10.31GB is not duplicated into a new HF report publication.

The parent reads only Kaggle Secret HF_TOKEN. Numeric workers receive no token.
HF SDK 0.36.2 is private and installed without altering Torch/CUDA/NumPy. One owner
and local lock, immutable object batches and selecting manifests protect writes.
The 96/hour commit guard and server-observed API/resolver windows remain separate;
cross-host account traffic cannot be globally counted by one local ledger.

## Scope and verification status

| Area | Status |
|---|---|
| NB17 completed training and source identity | Independently verified |
| Full local saved-evidence analysis | Passed on actual downloaded NB17 bytes and existing masks |
| Three notebooks: schema, source hashes, fresh extraction | Passed |
| Stage interruption/invalidation and automatic handoff | Passed offline with explicitly injected transport/worker fixtures |
| Actual new Kaggle/HF report publication | Pending user execution |
| Interactive report script controls | Node DOM fixture passed; no claim of full native-browser rendering |
| Physical depth/safety or GEO classification | Not validated / not fitted |
| Extra fit, new photos, workstation changes | None |

[Runtime and operations](tread_two/geo_postrun/README.md),
[delivery checks](tread_two/geo_postrun_validation/DELIVERY_CHECKS.json),
[notebook identities](tread_two/geo_postrun_validation/NOTEBOOKS.json) and
[NB17 findings](phase3NB17Completion.md). The passport records this pilot as
completed, unselected and unpromoted; no automatic next full fit follows it.
'''
write(ROOT/'phase3TreadTwoGEOPostrunBatch.md',guide)
tag='NB17_COMPLETE_POSTRUN_BATCH_20261102'
def block(prefix=''):
    return f'''<!-- {tag} -->
## Current — NB17 verified; all postrun notebooks delivered together

**2 November 2026:** the attachment is completed NB17 (message says NB16).
HF revision `1113bf8a…`: **11,637 logical files / 3,080 direct CSVs /
{files['CSV_rows']:,} rows / 8,457 native arrays / 10.31 GB**. One seed,
60 epochs and 2,940 updates; all selected bytes, schemas, native evidence and original masks
independently checked. Owner released; earlier four publication manifests unchanged.
**Do not rerun NB16/NB17 training.** [Completion and illustrated insights]({prefix}phase3NB17Completion.md).

The geometry floor passes 60/60; human point MAE improves 5.23% retrospectively.
Representation eligibility passes 0/60: final validation projector rank 1.038659,
pooled 96D rank 1.020990; masked gain 4.0337% is below 5%. **Selected unavailable;
final remains research-only, unpromoted.** No new classification/depth/safety claim.

**Next: [NB20 Run All master]({prefix}notebooks/phase3_NB20_TREAD_TWO_GEO_All_In_One_Postrun.ipynb)**.
It automatically runs/publishes NB18 full review then NB19 evidence passport.
All three notebooks are delivered now; individual stage files are optional.
CPU / None, Internet ON, existing expanded Kaggle input, HF_TOKEN, at least 13 GB free
on a fresh start. One shared verified cache, complete-stage recovery, publication
every 30 minutes and at major/stop boundaries. [Full batch/run guide]({prefix}phase3TreadTwoGEOPostrunBatch.md).
Local actual-data audit and offline source/handoff/restore checks pass; new
NB18/NB19 Kaggle/HF publication remains pending. Existing photos only, no extra
fit or workstation change. User preference: supply all justified next notebooks
together with an automatic sequential master; preserve scientific gates.
<!-- END_{tag} -->
'''
trackers=['AGENTS.md','README.md','phase3Progress.md','phase3NextSteps.md','phase3OpenItems.md','phase3Understanding.md','phase3DataRetentionPolicy.md','phase3ModelRationaleAndLogging.md']
for name in trackers:
    path=ROOT/name;s=path.read_text(encoding='utf-8');s=re.sub(r'<!-- '+tag+r' -->.*?<!-- END_'+tag+r' -->\n?', '',s,flags=re.S)
    s=s.replace('## Current — NB16 HF-verified; run NB17 GEO measured pilot','## Historical — NB16 HF-verified; NB17 pre-execution delivery')
    lines=s.splitlines();lines[0]={'phase3NextSteps.md':'# Next steps — Run All in NB20 / GEO postrun batch'}.get(name,lines[0]);s='\n'.join(lines)
    i=s.find('\n');write(path,s[:i+1]+'\n'+block()+s[i+1:])
for relative,prefix in [('tread_two/README.md','../'),('tread_two/geo_pilot/README.md','../../'),('phase3TreadTwoGEOProtocol.md','')]:
    path=ROOT/relative;s=path.read_text(encoding='utf-8');s=re.sub(r'<!-- '+tag+r' -->.*?<!-- END_'+tag+r' -->\n?','',s,flags=re.S)
    s=s.replace('## Current — new geometry-first TREAD-TWO design planned completely','## Historical — geometry-first design before implementation')
    i=s.find('\n');write(path,s[:i+1]+'\n'+block(prefix)+s[i+1:])
def section(name,marker,content):
    path=ROOT/name;s=path.read_text(encoding='utf-8');s=re.sub(r'<!-- '+marker+r' -->.*?<!-- END_'+marker+r' -->\n?','',s,flags=re.S)
    i=s.find('\n');write(path,s[:i+1]+'\n<!-- '+marker+' -->\n'+content.strip()+'\n<!-- END_'+marker+' -->\n'+s[i+1:])
section('phase3Decisions.md','NB17_BATCH_DECISIONS','''## NB17 completion and batch delivery — 2 November2026

| ID | Decision | Evidence / boundary |
|---|---|---|
| P3-D147 | Identify completion from the attached NB17 bytes | Message says NB16; NB16 already verified; archive executed NB17 unchanged |
| P3-D148 | Accept completed execution and failed representation eligibility separately |60epochs/2940updates;0/60eligible; no selected checkpoint invented |
| P3-D149 | Audit64D projector and96D pooled vectors by original role | Held-out near-rank-one; pooled570 does not replace validation |
| P3-D150 | Preserve richer direct CSV/native evidence and exact masks | All11636selectedobjects; objective/native/dense reconstruction and provenance |
| P3-D151 | Deliver NB18, NB19 and recommended NB20 together | Master sequences dependent stages without manual handoff; avoid competing writers |
| P3-D152 | Separate finite CPU stage recovery from training resumption | Replay incomplete analysis stage; completed fit never receives another update |
| P3-D153 | Keep GEO final as unselected research artifact | Geometry maintained, texture gate failed, error head loses to constant control |
| P3-D154 | No automatic next full fit, new photos or workstation change | User existing-data/deferred-workstation scope; a future fit needs a justified protocol |
''')
section('phase3EvidenceRegister.md','NB17_BATCH_EVIDENCE',f'''## NB17 and postrun batch — 2 November2026

| Evidence | Location | Status |
|---|---|---|
| Executed notebook +29payloads | [Inspection](tread_two/nb17_completion/4b92d8fd7081/INSPECTION.json) | Preserved; all match |
| Independent public HF audit | [Verification](tread_two/nb17_completion/4b92d8fd7081/VERIFICATION.json) | All selected bytes and scientific checks pass |
| 40CSVfamilies/468fields | [File/schema proof](tread_two/nb17_report/FILE_VERIFICATION.json) | {files['CSV_rows']:,} rows |
| Native and final-state audit | [Native proof](tread_two/nb17_report/NATIVE_VERIFICATION.json) |8455shards+2controls;60epochweights |
| Scientific result | [Findings](tread_two/nb17_report/FINDINGS.json) | Geometry60/60; representation0/60 |
| Runnable batch | [Guide](phase3TreadTwoGEOPostrunBatch.md) | NB18/NB19/NB20 delivered together |
| Delivery/local checks | [Receipt](tread_two/geo_postrun_validation/DELIVERY_CHECKS.json) | Pass; actual new Kaggle/HF publication pending |
''')
section('phase3SessionLog.md','NB17_BATCH_SESSION','''## 2 November2026 — NB17 verification and uninterrupted postrun batch

1. Archived supplied executed NB17; distinguished its filename/output from the message's NB16 wording.
2. Independently downloaded/verified pinned HF content without local credentials or remote writes; checked original earlier selecting manifests unchanged.
3. Reconstructed all objectives, original validation criteria, raw native evidence, final masks and retained recovery state. Recorded geometry improvement and failed latent rank honestly.
4. Built a separate saved-data runtime and NB18/NB19/NB20; one master runs both stages sequentially, caches once and publishes direct evidence.
5. Checked actual-data outputs, notebook schema/fresh setup, completed-stage recovery, offline handoff/restore and interactive controls. Updated structured docs and Git ignores; both workstations/completed sources preserved.
6. Next user action: Run All in NB20 on Kaggle CPU/None with existing prepared input, Internet and HF_TOKEN. No extra full fit/new photos; actual report publication pending.
''')
section('phase3DocumentationInventory.md','NB17_BATCH_DOCS','''## GEO completion and batch additions

| Document | Purpose |
|---|---|
| [NB17 completion](phase3NB17Completion.md) | Pinned evidence, illustrated metrics, findings and explicit gaps |
| [Postrun batch guide](phase3TreadTwoGEOPostrunBatch.md) | All notebook links, Kaggle setup, automatic order, retention and troubleshooting |
| [Runtime guide](tread_two/geo_postrun/README.md) | Module/ownership/cache/recovery behavior and scientific scope |
| [GEO postmortem](phase3TreadTwoGEOLessons.md) | Component/token-space diagnosis, small-batch covariance math and bounded future questions |
| [Research model card](tread_two/geo_passport_report/MODEL_CARD.md) | Capability limits and final/unavailable-selected provenance |
''')
section('phase3RepositoryMap.md','NB17_BATCH_MAP','''## GEO completion / postrun locations

| Path | Role |
|---|---|
| `tread_two/nb17_completion/4b92d8fd7081/` | Executed archive/public HF mirror (ignored), compact inspection/verification/manifest receipts |
| `tread_two/nb17_report/` | Actual-data derived CSV/native/HTML review (large outputs ignored), curated figures and summaries |
| `tread_two/geo_postrun/` | Separate self-contained embedded package, numeric audit modules, publisher/coordinator and run guide |
| `tread_two/geo_postrun_validation/` | Source/notebook identities and local delivery/check receipts; disposable fixtures ignored |
| `tread_two/geo_passport_report/` | Local verified research model card, capability/evidence/provenance CSVs |
| `notebooks/phase3_NB18…`, `phase3_NB19…`, `phase3_NB20…` | Review, passport and recommended automatic master |
''')
section('phase3DataRetentionPolicy.md','GEO_RETENTION_LESSONS','''## NB17 retention lessons and future requirements — 2 November 2026

The user wants richer evidence from existing data, direct CSV, continuous
documentation and all justified next notebooks delivered together. The 570-photo
boundary remains: richer telemetry is not new independent tyre photography.

| Retained now | Explicit gap | Requirement for a later training protocol |
|---|---|---|
| Complete objective sufficient statistics, raw vectors and masked bodies | Full identity target-map construction cannot be recomputed solely from selected masked bodies | Retain the full mean-building target maps/weights once, with train-only provenance and an explicit space budget |
| 9,867 resource samples | Post-unit samples are not true within-step memory peaks; staging_bytes is blank | Record per-device peak counters, reset intervals and actual staging/unique-byte measurements as direct CSV |
| All 60 epoch weights and complete rolling optimizer/RNG/cursor state | Every historical optimizer state/prefix map is not permanently retained | State exact recovery versus historical-reconstruction coverage; do not promise unlimited checkpoints within 20 GB |
| Ingested HF request/publication rows and pinned server receipt | A selecting snapshot cannot recursively contain its own final commit | Retain an additive transport/publication receipt stream with bounded commits and explicit final-event coverage |
| Full source/native/scientific evidence plus probes | Probe raw texture bodies are absent under the original policy | Declare retention granularity and omissions before a future fit; budget fuller probe capture if justified |

Do not retrofit or invent missing historical measurements. NB18/NB19/NB20 add
derived direct CSVs, native/dense reconstruction, token/component diagnostics,
resource/timing summaries, figures and interactive evidence. They perform no new
training or photo collection. A future fit must retain the existing complete-update
resume, HF every-30-minute/major/stop attempts, server-aware backoff, source/data
identity checks and clear Kaggle instructions.
''')
print('Structured NB17 completion, batch guide and15 current/history trackers updated.')
