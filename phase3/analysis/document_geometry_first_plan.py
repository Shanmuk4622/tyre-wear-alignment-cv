"""Advance current project records to the requested new GEO design, preserving history."""
import json,re,sys
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];PLAN=ROOT/'tread_two/geometry_first/plan'
schema=json.loads((PLAN/'SCHEMAS.json').read_text());assert schema['CSV_families']==40 and schema['declared_columns']==443
def insert(name,body):
    path=ROOT/name;old=path.read_text(encoding='utf-8');marker='GEO_PLAN_20261101'
    old=re.sub(r'\n<!-- '+marker+r' -->.*?<!-- END_'+marker+r' -->\n','\n',old,flags=re.S)
    head,rest=old.split('\n',1);rest=re.sub(r'(?m)^## Current — ',r'## Historical — ',rest)
    path.write_text(head+'\n\n<!-- '+marker+' -->\n'+body.strip()+'\n<!-- END_'+marker+' -->\n'+rest,encoding='utf-8')
common='''## Current — new geometry-first TREAD-TWO design planned completely

**1 November 2026:** the user requested the complete geometry-first development plan. [Master plan](phase3TreadTwoGeometryPlan.md) and [technical package](tread_two/geometry_first/plan/README.md) now define one new **TREAD-TWO / GEO** research candidate: frozen selected TREAD-ONE features, bounded supervised geometry refinements, an isolated masked-texture/self-supervised branch, and experimental annotation-error estimates. No mileage-classification loss in the first pilot.

The plan chooses one seed / 60 epochs / 2,940 updates, exact existing data/anchor, source-coverage scheduling, independent gradient clipping, original-role validation, checkpoint eligibility/selection, full raw-tensor retention and **40 CSV families / 443 declared fields**. New HF namespace, dual-T4 ownership, complete-update restore and 30-minute/major/stop publication are specified. Proposed 15 GB logical / 18 GB peak ceilings require actual fixture measurements within the 20 GB assumption; 1 TB scratch is optional.

**Status: design only; runtime, numerical/live resource gates and training notebooks are pending.** Next: implement isolated data/model/store modules and build planned **NB16 GEO engineering/contract gate**, then one NB17 measured pilot if its gates pass. No existing executed notebook is rerun or regenerated. No new photos, model forward/optimizer update, HF write or workstation change occurred during planning.

The completed ATLAS v1/LC/context series remains closed and unpromoted. This new geometry/representation objective is a separate requested direction; earlier statements about no extra notebook to close the unchanged classifier series remain historical and valid in that scope. Both workstations stay deferred.'''
for name in ['README.md','AGENTS.md','phase3Progress.md','phase3NextSteps.md','phase3OpenItems.md','phase3Understanding.md','phase3DataRetentionPolicy.md','phase3ModelRationaleAndLogging.md']:
    insert(name,common)
path=ROOT/'phase3NextSteps.md';text=path.read_text(encoding='utf-8');path.write_text('# Next steps — implement the geometry-first GEO design and NB16 gate\n'+text.split('\n',1)[1],encoding='utf-8')
insert('tread_two/README.md',common.replace('](phase3','](../phase3').replace('](tread_two/geometry_first/','](geometry_first/'))
insert('phase3TreadTwoResearchClosure.md','''## Later requested direction — geometry-first GEO plan

On 1 November 2026 the user requested a complete new development plan after discussing what remains possible with existing photos. [TREAD-TWO / GEO plan](phase3TreadTwoGeometryPlan.md) defines a different geometry/representation objective; implementation is pending. The closure below still applies to the completed ATLAS classifier series. No old fit, model selection, research index or workstation is modified by the plan.''')
insert('phase3DocumentationInventory.md','''## New GEO design package — 1 November 2026

| Document | Purpose |
|---|---|
| [Master plan](phase3TreadTwoGeometryPlan.md) | Complete goal/evidence/architecture/run/acceptance/deliverable summary |
| [Architecture](tread_two/geometry_first/plan/ARCHITECTURE.md) | Tensor shapes, bounded masks/points, error inputs, texture losses and disjoint gradients |
| [Experiment](tread_two/geometry_first/plan/EXPERIMENT.md) | Fixed source roles, deterministic full coverage, augmentation, optimizer and selection |
| [Retention](tread_two/geometry_first/plan/RETENTION.md) | All 40 families / 435 fields, raw native arrays, loss reconstruction, capacity and gaps |
| [Operations](tread_two/geometry_first/plan/OPERATIONS.md) | Two-T4 ownership, Kaggle cells, atomic restore/stop/HF/resource behavior |
| [Implementation](tread_two/geometry_first/plan/IMPLEMENTATION.md) | Module order, tests, engineering freeze and planned NB16–NB18 |
| [Risks/decisions](tread_two/geometry_first/plan/RISKS_AND_DECISIONS.md) | Previous evidence → design → test, failure actions and pending measured facts |

The package also contains PLAN.json, SCHEMAS.json, SOURCES.json and BUDGET.json. These are design proposals and source facts, not a measured runtime protocol or trained model.''')
insert('phase3EvidenceRegister.md','''## GEO design evidence — 1 November 2026

| Evidence | What it establishes |
|---|---|
| [Plan](tread_two/geometry_first/plan/PLAN.json) | Chosen design values/status and planned one-model budget; no measured updates |
| [Schema proposal](tread_two/geometry_first/plan/SCHEMAS.json) | All planned CSV family/field/type/key/cadence definitions |
| [Source record](tread_two/geometry_first/plan/SOURCES.json) | Pinned original anchor/source/annotation/prior-result hashes and primary technical references |
| [Capacity arithmetic](tread_two/geometry_first/plan/BUDGET.json) | Exact tensor-shape byte totals plus explicitly provisional CSV/other/staging allowances |

Evidence inputs include the a5b8… seed3/epoch20 geometry anchor, NB05's shared-gradient findings, NB07 point/identity counts and verified NB15 support limits. New model quality, throughput, exact parameter count, actual disk peak and T4 replay remain pending implementation/live gates. [Master plan](phase3TreadTwoGeometryPlan.md).''')
insert('phase3RepositoryMap.md','''## New planned GEO branch — 1 November 2026

| Path | State / purpose |
|---|---|
| `tread_two/geometry_first/plan/` | Delivered design docs, JSON proposal/schema, pinned sources and capacity arithmetic |
| `tread_two/geometry_first/runtime/` | Planned isolated modules; not yet implemented |
| `tread_two/geometry_first/validation/` | Plan checks now; later numerical/resource/recovery receipts and ignored fixtures |
| `analysis/write_geometry_first_plan.py` | Author design documents and machine proposals only; no trainer/model forward |
| `phase3TreadTwoGeometryPlan.md` | Complete project-level entry point |

Future NB16/NB17/NB18 are planned names only. Old source trees/catalogs and executed notebooks remain unchanged. No generated training native/cache data belongs in Git; only compact plan/source/receipts and curated figures will be kept visible.''')
insert('phase3Decisions.md','''## Geometry-first GEO design — 1 November 2026

| ID | Chosen direction | Reason / boundary |
|---|---|---|
| P3-D128 | Plan a new geometry/representation objective within TREAD-TWO | User explicitly requested complete planning; old ATLAS classifier closure remains preserved |
| P3-D129 | Freeze selected r1 anchor; train bounded geometry adapters and isolated texture/error branches | Existing useful dense results and shared-gradient/point regression evidence; no new measured improvement claimed |
| P3-D130 | Use masked local-target prediction plus variance/invariance/covariance; no first-pilot mileage loss | Avoid repeating confidence sharpening on already-correct training labels; collapse/mean-predictor controls required |
| P3-D131 | Cover all386trainingimages once/epoch plus6documented padding draws | Reduce extreme replacement repetition; ensure structured batch source diversity, while retaining component limitations |
| P3-D132 | One seed/60epochs after automatic gates; geometry/collapse eligibility then representation selection | Existing photos and old historical roles; no test-selected checkpoint or automatic extra sweep |
| P3-D133 | Retain40CSVfamilies/443fields plus every training mask logit/token/masked target/prediction in native shards | User's extensive telemetry requirement;15GBlogical/18GBpeak projections must be measured before fitting |
| P3-D134 | Deliver plan now; implement isolated runtime and NB16 next | Model/notebooks not yet runnable; no workstation/HF/model mutation during planning |

[Full design](phase3TreadTwoGeometryPlan.md), [risks and primary sources](tread_two/geometry_first/plan/RISKS_AND_DECISIONS.md).''')
insert('phase3SessionLog.md','''## Complete GEO planning requested and delivered — 1 November 2026

After discussing remaining work, the user requested the geometry-first direction be planned completely. Read the current project instructions, ATLAS closure/readiness results, original TREAD-ONE architecture, NB05/NB06 gradient/dense failures, the exact frozen anchor/canonicalizer and NB07 annotation-support records. Consulted the original VICReg paper and official PyTorch2.11 grid-sampling/reproducibility and HF rate-limit documentation; sources and local hashes are retained.

Designed TREAD-TWO / GEO: original selected r1 seed3/epoch20 anchor frozen, small zero-initialized bounded mask/raw-point refinements, detached predicted-support atlas, two-view local-contrast96Dtexture encoder/two axial blocks, masked tile reconstruction,64Dvariance/invariance/covariance projector, and three detached-target experimental annotation-error estimates. Parameter groups/gradient paths/clips are separate; no mileage-classification objective enters pilot1. Exact geometry/quality/texture reductions, source grouping, deterministic392draw full-coverage epoch, selection/failure rules and diagnostics are chosen in the plan.

Created eight Markdown design documents and four machine-readable JSON records under the new plan folder/master entry point. Defined40draftCSVfamilies/443fields with proposed types/keys/cadences, full training raw tensor retention, uncompressed native shard/prefix semantics,60epoch weights and complete rolling recovery. Shape arithmetic covers29,520train/validation/endpoint source passes and59,040texture views before controls; adjusted logical estimate14.630GB and same-filesystem peak15.167GB remain provisional because CSV widths/real model writes are unmeasured. Ceilings15GBlogical/18GBpeak preserve2GBreserve within a20GBassumption. No1TBscratch dependence or silent mid-fit evidence reduction.

Specified planned NB16 actual dual-T4 engineering/contract gate, one NB17 seed1/60epoch/2940update measured model, and conditional NB18 review. GPU1 owns frozen feature inference and barrier evaluation; GPU0 trains disjoint heads/texture. Planned HF cadence remains30min/major/catchable-stop with private SDK, immutable objects, owner fencing, visible server backoff and fresh-session restore. Exact model parameter count, actual T4 math/memory/runtime, native/CSV capacity and model outcomes are pending measured gates, not invented results.

Updated current project trackers, decisions, evidence/inventory/map and marked older closure statements by scope. This task produced a design only: no model implementation, .ipynb delivery, optimizer step, image/model forward, dataset expansion, HF write or workstation change. Next engineering milestone is isolated runtime/data/store implementation and NB16. Existing photos, completed experiments/research index and both workstations remain preserved.''')
print(json.dumps(dict(status='documented',new_direction='TREAD-TWO/GEO',implementation_pending=True,CSV_families=40,fields=443),indent=2))
