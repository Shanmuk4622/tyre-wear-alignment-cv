"""Current notebook delivery documentation, based on passed local receipts."""
import csv,json,sys
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'tread_two/trace_batch';VALID=ROOT/'tread_two/trace_validation'
sys.path.insert(0,str(SRC))
from common import read,atomic,file_meta
from schema import SCHEMA

def block(path,tag,body):
    text=path.read_text(encoding='utf-8');begin='<!-- '+tag+' -->';end='<!-- END_'+tag+' -->';value=begin+'\n'+body.strip()+'\n'+end
    if begin in text:
        i=text.index(begin);j=text.index(end,i)+len(end);text=text[:i]+value+text[j:]
    else:
        i=text.index('\n')+1;text=text[:i]+'\n'+value+'\n'+text[i:]
    path.write_text(text,encoding='utf-8')

proof=read(VALID/'LOCAL_LIFECYCLE.json');assert proof['status']=='passed';fixture=Path(proof['fixture']);facts=read(VALID/'FULL_CACHE.json')
for name in ['TRANSACTIONS.json','COORDINATOR.json','PAYLOADS.json','PREPARATION_RECOVERY.json']:assert read(VALID/name)['status']=='passed',name
source=read(SRC/'FILES.json');assert proof['source_inventory']==source
for name in ['TRANSACTIONS.json','COORDINATOR.json','PREPARATION_RECOVERY.json']:assert read(VALID/name)['source_inventory']==source,name
full_core=['DATA_PIN.json','SCHEMAS.json','SOURCE_COMPONENTS.csv','common.py','cache.py','data.py','targets.py','frozen_anchor.py','frozen_geometry.py','geometry.py','atlas_geo.py','schema.py','metrics.py']
assert all(facts['source_inventory'][n]==source[n] for n in full_core)
atomic(VALID/'FULL_DATA_SOURCE_EQUIVALENCE.json',dict(status='passed',full_cache_receipt='FULL_CACHE.json',unchanged_scientific_data_modules=full_core,
    distinction='All570 targets were constructed before later parent/storage/allocator-diagnostic fixes; these scientific data modules and inputs remain byte-identical. The updated allocator startup and chunk recovery are separately tested in PREPARATION_RECOVERY.',
    all570_existing_images=True,full_cache_bytes=facts['cache_bytes'],new_photos=0,new_training_updates=0))

# Presence/field coverage across natural stages, not fabricated normal errors.
rows=[]
controller_dirs=sorted(VALID.glob('fixture_controller_*/all/transport_tail'),key=lambda p:p.stat().st_mtime)
error_root=Path(read(VALID/'TRANSACTIONS.json')['fixture'])/'worker_error/run'
roots=[fixture/'direct/run',fixture/'recovered/run',fixture/'cache',fixture/'review',Path(facts['cache']),error_root]
if controller_dirs:roots.append(controller_dirs[-1])
for family,spec in SCHEMA.items():
    count=0;filled=set();locations=[]
    for root in roots:
        for path in root.glob('csv/'+family+'/*.csv'):
            n=0
            with path.open(encoding='utf-8',newline='') as stream:
                for r in csv.DictReader(stream):n+=1;filled.update(k for k,v in r.items() if v!='')
            if n:locations.append(str(path.relative_to(root)));count+=n
        for path in root.glob('active/csv_'+family+'_*.csv'):
            # Active generations not selected by a receipt are excluded here.
            pass
    rows.append(dict(family=family,declared_fields=len(spec['columns']),fixture_rows=count,fields_nonempty_in_exercised_records=len(filled),exercised=count>0,
        scope='CPU/single_GPU_numerical_and_injected_transport_or_error_fixtures; not_measured_training',locations=';'.join(sorted(set(locations)))))
assert all(r['exercised'] for r in rows),[r['family'] for r in rows if not r['exercised']]
with (VALID/'WRITER_COVERAGE.csv').open('w',newline='',encoding='utf-8') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)

budget=read(fixture/'direct/STORAGE_PROJECTION.json');logical=budget['logical_bytes']-budget['cache_bytes']+facts['cache_bytes'];peak=int(logical*1.2+128*1024**2)
atomic(VALID/'LOCAL_STORAGE_ESTIMATE.json',dict(status='within_design_ceiling',scope='formatted_8_train_fixture_extrapolation_adjusted_for_actual_all570_cache_not_live_Kaggle_measurement',
    logical_bytes=logical,projected_peak_bytes=peak,actual_full_cache_bytes=facts['cache_bytes'],live_projection_required=True))
scientific=read(fixture/'review/SCIENTIFIC_AUDIT.json')
validation=f'''# TRACE notebook acceptance — local proof and pending live gates

**2 November 2026.** [NB21–NB24 and Run All instructions](phase3TreadTwoTRACEBatch.md)
are delivered together. This page records actual local checks; no measured
60-epoch TRACE fit or new HF publication has been performed here.

| Check | Actual result | Evidence |
|---|---|---|
| All notebooks |4valid clean notebooks,33exact embedded payloads each; fresh extracted setup/imports execute | [Payload proof](tread_two/trace_validation/PAYLOADS.json) |
| Actual model lifecycle |8existing train images,2disposable optimizer updates, all phases/endpoints/probes/retrieval | [Lifecycle](tread_two/trace_validation/LOCAL_LIFECYCLE.json) |
| Complete-update restart | Fresh worker from the within-epoch native seam matches model/optimizer/RNG/cursor and scientific files | Same receipt; CPU, not a live HF proof |
| Atomic faults | Evidence/state/pointer interruptions, appended tails, committed corruption, worker error and injected selected export | [Fault proof](tread_two/trace_validation/TRANSACTIONS.json) |
| Preparation restart |16actual train images; stop after8; original chunk hashes preserved; review partial output replay and receipt skip/corruption rejection | [Proof](tread_two/trace_validation/PREPARATION_RECOVERY.json) |
| All existing inputs |570images /72chunks;386/81/103roles;all60source schedules checked | [Full cache](tread_two/trace_validation/FULL_CACHE.json) |
| Source scope of full cache | Scientific input/target modules unchanged after later parent/storage/allocator fixes | [Equivalence](tread_two/trace_validation/FULL_DATA_SOURCE_EQUIVALENCE.json) |
| Per-view prediction audit |{scientific['all_prediction_values_CSV_native_exact']:,}fixture prediction values and{scientific['all_moment_values_CSV_native_exact']:,}moment values match native FP32 exactly | Full lifecycle review |
| Objective and selection audit | Every raw local/masked loss and both fixture training objectives reconstructed; original selected=None reproduced | Same review; no claim about future TRACE accuracy |
| Writer availability | All48families exercised across their natural/injected fixture stages;671declared fields, explicit nullable gaps | [Coverage CSV](tread_two/trace_validation/WRITER_COVERAGE.csv) |
| Parent orchestration | Preparation→gate→fit→review order, completed skip, owner release, stop flush, failed-budget block and finite tail | [Offline injected-service proof](tread_two/trace_validation/COORDINATOR.json) |
| Interactive passport | Actual report JavaScript, role/search/paging, PNG signatures and injected missingness renderer | [Viewer](tread_two/trace_validation/VIEWER.json); synthetic DOM, not native screenshot |
| Preservation / Git | Completed source/notebooks and both workstations checked; caches/fixtures ignored, current deliverables visible | [Handoff](tread_two/trace_validation/DELIVERY.json) |

The full scientific-data construction ran locally on one GTX1650 with CPU target
statistics and zero optimizer updates. Other model/transaction fixtures use CPU;
the host's one CUDA RNG is also retained where available. These do not substitute
for two-T4 arithmetic or remote HF continuation. The actual parent test replaces
HF/numeric services with explicit local fixtures; it makes zero network writes.

## Capacity evidence

The actual all570 cache is **{facts['cache_bytes']:,} bytes**. Actual formatted
fixture CSV/native sizes, adjusted for this full cache and the declared full
training/validation/two-potential-endpoint/probe schedule, project approximately
**{logical/1e9:.2f}GB logical /{peak/1e9:.2f}GB peak**. This is a local planning
estimate. NB21 repeats the calculation on the real Kaggle full cache/formatter
and requires actual remaining free space plus the2GBreserve before enabling
measured initialisation. It never silently drops promised evidence to fit.

## Issues found before delivery

| Issue | Resolution |
|---|---|
| Some Windows fixtures cannot create hardlinks | Permit bounded copies only in offline engineering; measured Kaggle snapshots still require hardlinks |
| NumPy boolean in a selection record | Convert eligibility gates to native bool before canonical recovery serialization |
| Empty CSV shards in review | Skip empty bodies while preserving their declared headers |
| A secondary GPU allocator may be uninitialised before peak reset | Non-random four-byte context warm-up, synchronise, then reset peak counters |
| Final snapshot acknowledgement could race worker exit | Parent records the completed publication/released owner in its callback and reuses that verified receipt |
| Partial review files pre-exist after interruption | Explicit declared artifact sets select complete stage outputs; directory-difference discovery removed |
| Over-frequent phase publication could consume commit quota |30-minute cadence plus major scientific stages; epochs remain durable locally |
| Source changed during development | Rebuilt all new payloads and source-bound checks; no executed historical notebook was regenerated |

## What remains for Run All

Actual T4×2 availability, Kaggle capacity, HF Secret/authentication, a verified
published checkpoint downloaded into a fresh worker, and identical continuation
must pass before the long fit. Then one seed /60epochs is measured and the
complete review publishes its observed result. A scientific rejection is still
a valid completed run; final/all epoch weights are retained, selected may remain
unavailable, and no physical-depth/safety or improved-classification claim follows.
'''
(ROOT/'phase3TreadTwoTRACEValidation.md').write_text(validation,encoding='utf-8')

status='''## Current — NB21–NB24 TRACE batch delivered; run NB24 once

**2 November2026:** all four self-contained TRACE notebooks are implemented.
**[NB24 Run All master](notebooks/phase3_NB24_TREAD_TWO_TRACE_All_In_One_Batch.ipynb)**
automatically sequences fixed-target preparation, actual dual-T4/HF continuation
and storage gates, one seed /60epochs /2,940updates, then the saved-evidence
review and interactive passport. Individual NB21/NB22/NB23 files are supplied too.

Use a dedicated Kaggle **T4×2**, Internet ON, the existing expanded
`shanmuk4622/tire-dataset-prepared-phase2` input and **HF_TOKEN**. Start cold with
at least19GBfree; actual capacity must also preserve the2GBreserve. Run one
master, not competing writers. [Full setup/recovery guide](phase3TreadTwoTRACEBatch.md).

Local real-model lifecycle/restart/fault/review and fresh payload tests pass.
All570 existing photos and60source schedules were exercised. The executable
schema has48CSVfamilies /671fields with native predictions/tokens, all epoch
weights and complete-update recovery. [Acceptance and boundaries](phase3TreadTwoTRACEValidation.md).

**Actual Kaggle/live HF and measured training remain pending user execution.**
The notebooks enforce those live gates automatically. No local token/HF write,
new photo or workstation change. Original GEO remains closed/unselected; no old
completed notebook needs a rerun. Preserve this new source/notebook/protocol
once execution starts.
'''
for name in ['AGENTS.md','README.md','phase3Progress.md','phase3NextSteps.md','phase3OpenItems.md','phase3Understanding.md','phase3DataRetentionPolicy.md','phase3ModelRationaleAndLogging.md','phase3TreadTwoTRACEDesign.md','phase3TreadTwoGEONextResearch.md']:
    block(ROOT/name,'TRACE_BATCH_20261102',status)
    path=ROOT/name;text=path.read_text(encoding='utf-8').replace('## Current — TRACE design and local prototype checked','## Earlier design checkpoint — TRACE prototype checked')
    path.write_text(text,encoding='utf-8')
block(ROOT/'tread_two/README.md','TRACE_BATCH_20261102',status.replace('](notebooks/','](../notebooks/').replace('](phase3TreadTwoTRACEBatch.md)','](../phase3TreadTwoTRACEBatch.md)').replace('](phase3TreadTwoTRACEValidation.md)','](../phase3TreadTwoTRACEValidation.md)'))
for name in ['README.md','DELIVERY_PLAN.md','EXPERIMENT.md','EVIDENCE.md','OPERATIONS.md','FINDINGS.md']:
    block(ROOT/'tread_two/trace_design'/name,'TRACE_BATCH_SUCCESSOR','''## Executable successor delivered

**2 November2026:** this file retains the original design-stage record. The
separate [TRACE notebook batch](../../phase3TreadTwoTRACEBatch.md) is now implemented
with NB21–NB24,671executable fields and local acceptance. Use NB24 for automatic
preparation→gates→pilot→review. Actual Kaggle/HF and measured results remain
pending; the original prototype/code/check receipts remain unchanged.
''')
block(ROOT/'phase3Decisions.md','TRACE_BATCH_DECISIONS','''## TRACE batch implementation — 2 November2026

| ID | Decision | Evidence / boundary |
|---|---|---|
| P3-D168 | Deliver NB21/NB22/NB23 and recommended sequential NB24 together | No manual notebook handoff or concurrent writers |
| P3-D169 | Keep prototype/completed packages immutable; use separate trace_batch |14exact helper/core copies; explicitly adapted new store |
| P3-D170 | Freeze48families /671fields, adding prediction-update and paired-cosine provenance | All natural/injected fixture writers exercised; unavailable fields remain nullable |
| P3-D171 | Keep batch8 unique-source coverage, six explicit padding draws and one measured seed | All60schedules checked; no new independent photos |
| P3-D172 | Require actual full formatting/capacity and HF-restored dual-T4 continuation before long fit | Local proofs are not mislabeled live Kaggle evidence |
| P3-D173 | Use complete-update prefixes, owner fencing and finite final transport receipt | Atomic fault, stop-flush and final-acknowledgement tests |
| P3-D174 | Preserve research rejection and original GEO selected=None | No checkpoint substitution, workstation deployment or new safety/classification claim |
''')
block(ROOT/'phase3EvidenceRegister.md','TRACE_BATCH_EVIDENCE','''## TRACE executable batch evidence

| Evidence | Location |
|---|---|
| Notebook identities | [Four notebooks](tread_two/trace_validation/NOTEBOOKS.json) |
| Actual local lifecycle and independent review | [Receipt](tread_two/trace_validation/LOCAL_LIFECYCLE.json) |
| Complete-update/error/corruption/selected-branch fixtures | [Transactions](tread_two/trace_validation/TRANSACTIONS.json) |
| All570 source cache /all60 schedules | [Full data](tread_two/trace_validation/FULL_CACHE.json) |
| Preparation and stage recovery | [Receipt](tread_two/trace_validation/PREPARATION_RECOVERY.json) |
| Parent sequence/stop/skip with injected services | [Controller](tread_two/trace_validation/COORDINATOR.json) |
|48-family writer exercise | [CSV](tread_two/trace_validation/WRITER_COVERAGE.csv) |
| Fresh setup and delivery preservation | [Payloads](tread_two/trace_validation/PAYLOADS.json) / [handoff](tread_two/trace_validation/DELIVERY.json) |
''')
block(ROOT/'phase3SessionLog.md','TRACE_BATCH_SESSION','''## 2 November2026 — complete TRACE notebook implementation

1. Built separate cache/data/schedule, transaction, telemetry, resource/selection and parent/review modules without changing the checked prototype or completed experiments.
2. Delivered four self-contained notebooks with one sequential Run All master. Added actual HF seam/restore/continue and capacity gates before one60-epoch candidate.
3. Exercised actual8-image CPU lifecycle/within-epoch continuation, raw prediction/objective/selection review, seven fault/selected/error boundaries,16-image preparation restart and all570-image frozen cache with60schedules.
4. Tested real orchestration with explicit injected HF/numeric services and actual notebook setup in fresh interpreters. No local credential lookup, remote write or measured fit occurred.
5. Fixed allocator initialisation, final publication acknowledgement, partial-stage artifact selection and serialization/review issues caught before delivery. Updated documentation, source provenance and Git ignores.
6. Next user action: Run All in NB24 on Kaggle T4×2 with existing prepared input, Internet/HF_TOKEN and sufficient measured space. Live gates and measured outcome remain pending.
''')
block(ROOT/'phase3DocumentationInventory.md','TRACE_BATCH_DOCS','''## TRACE notebook batch documents

| Document | Purpose |
|---|---|
| [Run All guide](phase3TreadTwoTRACEBatch.md) | All four notebooks, settings, sequence, recovery, retention and troubleshooting |
| [Acceptance](phase3TreadTwoTRACEValidation.md) | Actual local proofs, fixes, source scope, capacity and pending live gates |
| [Runtime map](tread_two/trace_batch/README.md) | Modules, exact source/recipe boundaries and HF namespaces |
''')
block(ROOT/'phase3RepositoryMap.md','TRACE_BATCH_MAP','''## TRACE executable batch locations

| Path | Role |
|---|---|
| `tread_two/trace_batch/` |32source/configuration files +selecting FILES inventory; isolated from prototype/completed studies |
| `tread_two/trace_validation/` | Compact acceptance receipts, notebook hashes and writer coverage |
| `tread_two/trace_validation/fixture_*/` | Disposable local data/model/cache/publication tests; ignored by Git |
| `notebooks/phase3_NB21…` through `NB24…` | Preparation, measured pilot, CPU review and recommended automatic master |
| `phase3TreadTwoTRACEBatch.md` | Current user run instructions |
''')

# Prose-only spacing: do not alter code, formulas, file paths or old dated blocks.
replacements={'2 November2026':'2 November 2026','all570':'all 570','All570':'All 570','570images':'570 images','570 existing':'570 existing','60source':'60 source','all60':'all 60','All60':'All 60','8existing':'8 existing','16actual':'16 actual','after8':'after 8','33exact':'33 exact','4valid':'4 valid','2disposable':'2 disposable','All48':'All 48','all48':'all 48','671declared':'671 declared','671executable':'671 executable','48CSVfamilies':'48 CSV families','48families':'48 families','671fields':'671 fields','60epochs':'60 epochs','2,940updates':'2,940 updates','19GBfree':'19 GB free','least19':'least 19','the2GBreserve':'the 2 GB reserve','60schedules':'60 schedules','386train':'386 train','71,022new':'71,022 new','14exact':'14 exact','one60-epoch':'one 60-epoch','/81':'/ 81','/103':'/ 103','/60':'/ 60','batch8':'batch 8','GPU0':'GPU 0','GPU1':'GPU 1','All386':'All 386','all386':'all 386','0.0001,clip1':'0.0001, clip 1','weight decay0.0001':'weight decay 0.0001','LR0.001':'LR 0.001','clip1':'clip 1','Seed1':'Seed 1','once/epoch +6':'once/epoch + 6',';8 distinct':'; 8 distinct','than98':'than 98','about1.09':'about 1.09','the25-second':'the 25-second','and60':'and 60','has48':'has 48','and48':'and 48','/671':'/ 671','/2,940':'/ 2,940',';671':'; 671'}
new_docs=[ROOT/'phase3TreadTwoTRACEBatch.md',ROOT/'phase3TreadTwoTRACEValidation.md',SRC/'README.md']
for path in [*ROOT.glob('*.md'),ROOT/'tread_two/README.md',*(ROOT/'tread_two/trace_design').glob('*.md'),SRC/'README.md']:
    original_text=path.read_text(encoding='utf-8');text=original_text
    if path in new_docs:
        for a,b in replacements.items():text=text.replace(a,b)
    else:
        import re
        def polish(m):
            part=m.group(0)
            for a,b in replacements.items():part=part.replace(a,b)
            return part
        text=re.sub(r'<!-- TRACE_BATCH[A-Z0-9_]* -->[\s\S]*?<!-- END_TRACE_BATCH[A-Z0-9_]* -->',polish,text)
    if text!=original_text:path.write_text(text,encoding='utf-8')
print(dict(status='documented',writer_families_exercised=len(rows),schema_fields=671,full_cache_bytes=facts['cache_bytes'],logical_estimate=logical,peak_estimate=peak),flush=True)
