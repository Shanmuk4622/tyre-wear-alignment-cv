"""Structured NB20 completion, pinned research registry and proposed next direction."""
import re,shutil,sys
from pathlib import Path
import pandas as pd
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tread_two/geo_postrun'))
from common import read,atomic,file_meta,sha
OUT=Path(read(ROOT/'tread_two/nb20_completion/LATEST.json')['path']);P=read(OUT/'VERIFICATION.json');assert P['status']=='passed'
R,S=P['stages']['review'],P['stages']['passport'];F=read(OUT/'remote/review/FINDINGS.json');V=F['final_validation'];REPO='Shanmuk4622/tyre-wear-study'
def write(p,s):Path(p).write_text(s.rstrip()+'\n',encoding='utf-8')
def tree(pin):return f'https://huggingface.co/datasets/{REPO}/tree/{pin["revision"]}/{pin["prefix"]}'
release=ROOT/'tread_two/geo_research_release';release.mkdir(exist_ok=True)
figures=ROOT/'tread_two/nb20_report/figures';figures.mkdir(parents=True,exist_ok=True)
for p in (OUT/'remote/review/figures').glob('*.png'):shutil.copyfile(p,figures/p.name)
pubs=[]
for stage,nb in [('engineering',16),('fit',17),('review',20),('passport',20)]:
    if nb<20:
        folder=Path(read(ROOT/f'tread_two/nb{nb}_completion/LATEST.json')['path']);v=read(folder/'VERIFICATION.json');pin=v;size=v['bytes_verified'];rows=v['CSV_rows'];count=v['CSV_files']
    else:folder=OUT;v=P;pin=P['stages'][stage];size=pin['bytes'];rows=pin['CSV_rows'];count=pin['CSV_files']
    pubs.append(dict(stage=stage,notebook='NB'+str(nb),revision=pin['revision'],prefix=pin['prefix'],files_with_manifest=pin['files'],payload_bytes=size,CSV_files=count,CSV_rows=rows,
                     row_scope='Source/audit/derived/metadata/transport rows; copies across stages are not independent observations',owner_released=True,HF_tree=tree(pin),verification_receipt=str((folder/'VERIFICATION.json').relative_to(ROOT)),verification_sha256=sha(folder/'VERIFICATION.json')))
pd.DataFrame(pubs).to_csv(release/'publications.csv',index=False)
fit=next(p for p in pubs if p['stage']=='fit');fm=read(Path(read(ROOT/'tread_two/nb17_completion/LATEST.json')['path'])/'REMOTE_MANIFEST.json')
manifests={s:read(OUT/(s.upper()+'_MANIFEST.json')) for s in ['review','passport']};artifacts=[]
for stage,name,title in [('review','GEO_Review_Atlas.html','570-image interactive atlas'),('passport','GEO_Evidence_Passport.html','Published capability passport'),('passport','MODEL_CARD.md','Published model card'),('passport','provenance.csv','Actual publication provenance'),('fit','exports/final.pt','Unselected final research model')]:
    pin=fit if stage=='fit' else P['stages'][stage];m=fm if stage=='fit' else manifests[stage];meta=m['files'][name]
    artifacts.append(dict(stage=stage,title=title,logical_file=name,status='research_only',revision=pin['revision'],bytes=meta['bytes'],sha256=meta['sha256'],url=f'https://huggingface.co/datasets/{REPO}/resolve/{pin["revision"]}/{pin["prefix"]}/{meta["object"]}'))
artifacts.append(dict(stage='fit',title='Selected joint GEO model',logical_file='',status='unavailable_original_gate_rejected_all_epochs',revision=fit['revision'],bytes='',sha256='',url=''))
pd.DataFrame(artifacts).to_csv(release/'artifacts.csv',index=False);urls={r['logical_file']:r['url'] for r in artifacts if r['logical_file']}
for name in ['capabilities.csv','handoff_decisions.csv','validation_metric_passport.csv','native_checkpoint_pointers.csv','reconstruction_coverage.csv','provenance.csv','MODEL_CARD.md']:shutil.copyfile(OUT/'remote/passport'/name,release/name)
report=f'''# NB20 completed — review and passport independently HF-verified

**2 November 2026. NB20 passes. No NB18, NB19 or NB20 rerun is needed.**
The master completed both dependent stages on Kaggle CPU. All three code cells
completed without saved errors; all 19 embedded files match the frozen runtime.
The executed notebook is preserved unchanged:
`{P['executed_notebook_sha256']}`.

## Actual publications

| Stage | HF revision | Files including manifest | CSV files / rows | Content bytes | Owner |
|---|---|---:|---:|---:|---|
| NB18 review, through NB20 | `{R['revision']}` | 77 | 34 / 147,869 | 70,863,609 | Released |
| NB19 passport, through NB20 | `{S['revision']}` | 44 | 17 / 35,943 | 3,784,988 | Released |
| Combined postrun publications | Two pinned revisions | 121 | 51 / 183,812 | 74,648,597 | Both released |

[Review]({tree(R)}) · [passport]({tree(S)}) ·
[independent verification](tread_two/nb20_completion/8c9c2991e1bb/VERIFICATION.json) ·
[notebook inspection](tread_two/nb20_completion/8c9c2991e1bb/INSPECTION.json).
All 119 selected content objects were downloaded and checked by size/SHA.
Both selecting manifests are unchanged at the latest checked repository revision.
Verification read no local credentials and made zero HF writes. There are no
ZIP/NPZ/database envelopes.

The 183,812 rows include inventories, derived statistics, copied tables and live
transport records. They add no independent tyres or new training observations.
The notebook's local-root CSV counter has a different scope: publication adds
transport CSVs to the selected HF snapshots. Published manifests are the count
authority. The original fit remains **3,080 CSVs / 27,514,418 rows / 8,457 native
arrays / 10.31 GB**.

```mermaid
flowchart LR
  A[NB17 completed measured fit] --> B[NB20 automatic CPU review]
  B --> C[NB18 published: 610bdb22]
  C --> D[NB19 published: abfec8d6]
  D --> E[Independent verification]
  E --> F[Current GEO cycle closed; no rerun]
```

## Verification coverage

| Check | Verified result |
|---|---|
| Execution and source | Three completed cells, zero saved errors, all 19 payload hashes match |
| Content and ownership | All 119 objects; both manifests and released owners |
| Tables | 48 reference-comparable tables checked; actual provenance checked separately; published transport CSVs inventoried |
| Derived vectors | 64D and 96D arrays exactly match independent reconstruction |
| Recovery stages | All seven completed review receipts bind their exact outputs |
| Original evidence | Published proof covers all 27,514,418 original rows and 8,455 measured native shards plus two controls |
| Original epoch weights | All 60 file hashes checked; 4,020 parameter norms independently recomputed in FP64 |
| Interactive evidence | 570 images, 3,420 predicted point markers, 720 human targets; filtering/sorting/paging/CSV controls pass |
| Passport provenance | Actual review revision replaces the earlier local pending-publication marker |
| Preservation | NB13/NB14/NB15/NB16/NB17 selecting manifests unchanged |
| New model/data work | Zero optimizer updates, image/model forwards or new photos |

### Numerical precision is documented

The largest cross-platform difference in a weight-norm diagnostic is
**1.4901161e-8, one FP32 ULP**. Only that column uses its explicit field-specific
tolerance. Other numeric columns retain the tight original comparison checks;
raw model weights and derived vectors remain unchanged.

For all 4,020 norms, comparison with independent FP64 sums of squares gives at
most **2.2200368e-5 absolute / 1.1799424e-6 relative error**, or 12 FP32 ULPs from
the rounded FP64 value. Every norm satisfies a conservative FP32 reduction error
bound. This concerns diagnostic precision, not altered predictions/weights.
[Direct audit](tread_two/nb20_completion/8c9c2991e1bb/FP64_WEIGHT_NORM_AUDIT.csv).

The HF atlas is seven bytes shorter than the local HTML, but its parsed data,
embedded images, page structure and JavaScript are identical. This is text
serialization, not a scientific discrepancy.
[Comparison](tread_two/nb20_completion/8c9c2991e1bb/HTML_COMPARISON.json).
Interactive controls were tested in a DOM fixture; no native-browser rendering
claim is added. All three published PNGs are curated for these records.

## Scientific findings remain unchanged

| Original validation result | Value | Interpretation |
|---|---:|---|
| Geometry floor | 60/60 epochs pass | Baseline geometry retained |
| Human point MAE / image width | {V['human_point_MAE']:.9f} | 5.23% relative reduction from G0, about 0.1625 input pixels |
| Representation eligibility | 0/60 epochs pass | Selected joint checkpoint unavailable |
| Final projector / pooled rank | {V['projector_rank']:.6f} / {V['global_rank']:.6f} | Near-rank-one held-out vectors |
| Masked prediction gain | {100*V['masked_gain']:.4f}% | Below the 5% utility requirement |
| Error head vs constant control | Worse validation MAE for all three quantities | Experimental error estimates lack a demonstrated control advantage |

![Completed fit: geometry, point error, masked control and latent rank](tread_two/nb20_report/figures/GEO_Validation.png)

Publication success does not change the scientific gate. The final model remains
an **unselected, unpromoted research artifact**. These small validation/test roles
were already inspected, contain two conservative components each and have no
middle-class cohort. No physical-depth, safety or new GEO-classification claim
is established.

[Published interactive atlas]({urls['GEO_Review_Atlas.html']}) ·
[published capability passport]({urls['GEO_Evidence_Passport.html']}) ·
[published model card]({urls['MODEL_CARD.md']}).

## What now

The current cycle is complete: preparation → one measured fit → full review →
passport → independent verification. **There is no remaining notebook to run for
this cycle.** The [compact research handoff](phase3TreadTwoGEOResearchClosure.md)
records actual publications, native pointers and capability limits.

My recommendation is a **separate revision of texture learning**, using fixed
local targets and a diversity objective compatible with the effective batch
size, while reusing saved geometry and existing photos.
[Next-design brief](phase3TreadTwoGEONextResearch.md) explains the reasoning and
required gates. It is proposed, not implemented or trained. No automatic extra
fit starts, and workstation changes remain deferred as requested.
'''
write(ROOT/'phase3NB20Completion.md',report)
write(ROOT/'phase3TreadTwoGEOResearchClosure.md',f'''# GEO research handoff — current cycle complete

**2 November 2026.** NB16 preparation, NB17 fit and NB20's NB18/NB19 review/passport
stages are complete and independently HF-verified. No rerun or extra notebook
is required to close this cycle. [Completion proof](phase3NB20Completion.md).

| Item | Durable pointer | Status |
|---|---|---|
| Measured fit | [Pinned fit]({fit['HF_tree']}) | 60 epochs / 2,940 updates; frozen |
| Completed review | [Pinned review]({tree(R)}) | Seven completed audit stages |
| Capability passport | [Pinned passport]({tree(S)}) | Actual review provenance retained |
| Final native model | [Pinned final artifact]({urls['exports/final.pt']}) | Unselected research artifact |
| Selected joint model | Unavailable | Original rule rejected all 60 epochs |
| Small local index | [Publications](tread_two/geo_research_release/publications.csv) / [artifacts](tread_two/geo_research_release/artifacts.csv) | Exact revisions, paths, hashes and counts |
| Model card | [Exact published card](tread_two/geo_research_release/MODEL_CARD.md) | Capability limits preserved |
| Earlier ATLAS series | [Historical hub](tread_two/research_release/Evidence_Hub.html) | Separate objectives and frozen evidence |

```mermaid
flowchart TD
  A[One completed GEO candidate] --> B[Geometry floors preserved]
  A --> C[Texture eligibility failed]
  A --> D[Error estimates lack validation control advantage]
  B --> E[Retain research evidence]
  C --> E
  D --> E
  E --> F[No selected-model promotion]
  E --> G[Separately justified next design]
```

This index points to already-published evidence; it is local metadata, not a new
HF experiment. [RELEASE.json](tread_two/geo_research_release/RELEASE.json) binds
its source receipt and index hashes. The earlier ATLAS hub is preserved as a
dated separate release and is not regenerated.

The broader application is not declared complete. Both workstations and their
selected checkpoints remain preserved; their UI work is deferred. Existing
photos only. Classification/depth/safety limits remain unchanged.

The [recommended next research](phase3TreadTwoGEONextResearch.md) is a targeted
texture revision. Repeating the same geometry fit after removing texture is not
a justified new accuracy experiment: the completed runtime already isolated
the trainable task paths. Use retained weights/observations for geometry analysis.
''')
write(ROOT/'phase3TreadTwoGEONextResearch.md','''# Next research direction — repair texture learning with existing data

**Proposal only, 2 November 2026.** No new runtime, training notebook, measured
model or accuracy claim is delivered by this brief. NB20 closes the current
GEO cycle. [Completion](phase3NB20Completion.md) · [postmortem](phase3TreadTwoGEOLessons.md).

## Recommendation and motivation

Design one separately versioned **TREAD-TWO GEO texture revision**, keeping useful
saved geometry as an explicitly declared research input. The existing 570 photos
are the photo scope. Only 386 training photos may supply fitted statistics/targets;
held-out roles remain previously inspected development cohorts.

| Observed evidence | Consequence for the proposed revision |
|---|---|
| Local encoder and pooled vectors already have low rank | A larger final projector or transformer alone does not address the measured failure location |
| About 99% of validation variation lies between two components | Track within-component variation and role-specific spectra; pooled rank cannot establish generalisation |
| Eight images / 64 projector dimensions | Centered batch covariance has rank at most seven; use compatible dimensionality or a separately defined estimator |
| Very strong pair consistency with low diversity | Pair consistency needs independent reconstruction and diversity controls |
| Masked gain only 4.03% vs train mean | Preserve this control and investigate targets independent of student updates |
| Geometry passes all epoch floors | Reuse retained geometry with its actual selection provenance |
| Error heads lose to median controls | Keep confidence claims conditional on a demonstrated control advantage |

## Changes to investigate before freezing a protocol

1. **Fixed local targets.** Define deterministic multiscale texture targets or
   descriptors from existing train-role patches. The student cannot collapse
   these targets by changing its encoder. Any fitted normalisation, whitening
   or target transform must use training data only. These remain image-derived
   targets, not measured wear/damage labels.
2. **Batch-compatible diversity.** Investigate a small projection dimension
   (for example six with batch eight), while requiring meaningful rank in both
   projection and pooled features. This removes the specific 64D/rank≤7 conflict;
   it does not guarantee useful learning. A cross-batch queue is an alternative
   needing its own estimator, staleness policy and complete recovery-state proof.
3. **Declared geometry input.** The selected original TREAD-ONE anchor and
   unselected final GEO adapters have different provenance. State which is used
   and why; preserve the original selected=None outcome.
4. **One bounded measured candidate after engineering gates.** Freeze the target
   math, weights, architecture, source, schema, original roles and eligibility
   before fitting. Keep the familiar Kaggle/HF workflow. Objective weights and
   duration need numerical/resource measurements before they become a contract.

```mermaid
flowchart LR
  A[Verified current evidence] --> B[Fixed-target and diversity design]
  B --> C[Train-only numerical and leakage checks]
  C --> D[Resource, resume and live HF gates]
  D --> E[One separately frozen candidate]
  E --> F[Original-role review and controls]
  F --> G[Publish evidence, including failures]
```

## Gates and retention before a long fit

| Gate | Required evidence |
|---|---|
| Targets and leakage | Train-only provenance; hidden-pixel changes cannot leak into visible student inputs through preprocessing |
| Objective arithmetic | Independent reduction from direct CSV/native sufficient statistics; explicit missingness |
| Task separation | Gradient checks and unchanged declared geometry behavior |
| Representation | Per-role/component spectra, within/between scatter, local-token and pooled diagnostics |
| Storage | Actual 20 GB working-area budget; optional 1 TB scratch never assumed |
| Recovery | Exact completed-update optimizer/RNG/queue/cursor restore; actual HF continuation before long fitting |
| Publication | Direct CSV/native records; 30-minute/major/stop attempts; server backoff and single-owner writes |
| Fuller evidence | Complete target-map construction once, true memory peak counters, populated staging measurements and explicit final-event coverage |
| Delivery | All justified notebooks together with an automatic sequential master |

Training still has six conservative components (low/mid/high = 1/1/4), and each
held-out role has two components with no middle class. A new objective cannot
manufacture independent tyres or physical-depth/safety truth. Any improved
development result must retain those limitations.

This recommendation launches no fit, requests no new photos and leaves deferred
workstation work unchanged. Preserve every completed source/weight/publication,
and distinguish proposed, engineered, locally checked, Kaggle-completed and
independently verified states in any later implementation.
''')
write(release/'README.md','''# Completed GEO research registry

Small local CSV pointers to four verified HF stages. Native weights and full
telemetry remain at their pinned HF object paths; no new experiment is created.

| File | Purpose |
|---|---|
| publications.csv | Revisions, file/row counts, released ownership and verification receipts |
| artifacts.csv | Atlas, passport, model card and final/unavailable-selected pointers |
| capabilities.csv / reconstruction_coverage.csv | Exact published capabilities and evidence gaps |
| handoff_decisions.csv | Published research closure, no selected promotion or automatic fit |
| provenance.csv | Actual fit/review revisions from the published passport |
| MODEL_CARD.md | Exact published model card |
| RELEASE.json | Local index hashes and verification receipt |

[Completion](../../phase3NB20Completion.md) ·
[handoff](../../phase3TreadTwoGEOResearchClosure.md) ·
[proposed next research](../../phase3TreadTwoGEONextResearch.md).
''')
atomic(release/'RELEASE.json',dict(status='local_index_of_verified_publications',verification_receipt=str((OUT/'VERIFICATION.json').relative_to(ROOT)),verification_sha256=sha(OUT/'VERIFICATION.json'),publications=4,selected_available=False,promotion='research_only',remote_writes=0,files={p.name:file_meta(p) for p in release.iterdir() if p.is_file() and p.name!='RELEASE.json'}))
TAG='NB20_VERIFIED_GEO_CLOSED_20261102'
def block(prefix=''):
    return f'''<!-- {TAG} -->
## Current — NB20 HF-verified; GEO research cycle complete

**2 November 2026:** NB20 completed both NB18 review and NB19 passport stages.
All 19 payloads match; three cells complete without saved errors. Actual HF
revisions **610bdb22… / abfec8d6…** contain **121 logical files, 51 CSVs /
183,812 rows**. All 119 objects, released owners, 48 reference-comparable tables,
exact vector arrays and actual passport provenance check out. Earlier NB13–NB17
selecting manifests are unchanged. **No NB18/NB19/NB20 rerun needed.**
[Completion and numerical/evidence notes]({prefix}phase3NB20Completion.md).

The current GEO cycle is closed as **unselected, unpromoted research**. Geometry
passes 60/60 epochs; representation eligibility passes 0/60. The original
27,514,418 CSV rows and native/epoch evidence remain on HF.
[Compact research handoff]({prefix}phase3TreadTwoGEOResearchClosure.md).

**Recommended next direction:** separately design a texture revision with fixed
local targets and batch-compatible diversity, reusing saved geometry and existing
photos. [Design brief]({prefix}phase3TreadTwoGEONextResearch.md). This is proposed,
not implemented or trained. No new notebook/fit is required to finish this cycle.
Both workstation implementations remain deferred and preserved.
<!-- END_{TAG} -->
'''
paths=['AGENTS.md','README.md','phase3Progress.md','phase3NextSteps.md','phase3OpenItems.md','phase3Understanding.md','phase3DataRetentionPolicy.md','phase3ModelRationaleAndLogging.md','phase3NB17Completion.md','phase3TreadTwoGEOPostrunBatch.md','phase3TreadTwoGEOLessons.md','phase3TreadTwoGEOProtocol.md','tread_two/README.md','tread_two/geo_pilot/README.md','tread_two/geo_postrun/README.md']
for relative in paths:
    p=ROOT/relative;s=p.read_text(encoding='utf-8');s=re.sub(r'<!-- '+TAG+r' -->.*?<!-- END_'+TAG+r' -->\n?','',s,flags=re.S)
    s=s.replace('## Current — NB17 verified; all postrun notebooks delivered together','## Historical — NB17 verification and batch delivery before NB20 execution')
    if relative=='phase3NextSteps.md':s=s.replace(s.splitlines()[0],'# Next steps — GEO cycle closed; proposed texture revision',1)
    prefix='../../' if relative.count('/')==2 else '../' if relative.count('/')==1 else '';i=s.find('\n');write(p,s[:i+1]+'\n'+block(prefix)+s[i+1:])
def section(name,marker,text):
    p=ROOT/name;s=p.read_text(encoding='utf-8');s=re.sub(r'<!-- '+marker+r' -->.*?<!-- END_'+marker+r' -->\n?','',s,flags=re.S);i=s.find('\n')
    write(p,s[:i+1]+'\n<!-- '+marker+' -->\n'+text.strip()+'\n<!-- END_'+marker+' -->\n'+s[i+1:])
section('phase3Decisions.md','NB20_DECISIONS','''## NB20 verification and closure — 2 November 2026

| ID | Decision | Evidence / boundary |
|---|---|---|
| P3-D155 | Close NB18/NB19 through actual NB20 execution | Both stage publications verified; no fictitious standalone executions |
| P3-D156 | Use published inventories as the count authority | 51 HF CSVs include publication-time transport records; local-root counts differ in scope |
| P3-D157 | Measure numerical precision explicitly | One cross-platform FP32 norm ULP; FP64 original-weight audit; parsed HTML identical despite seven-byte serialization difference |
| P3-D158 | Preserve selected=None and research-only status | Publication does not change 0/60 representation eligibility |
| P3-D159 | Add a small local registry of published GEO evidence | Exact revisions/hashes; no new HF experiment or model duplication |
| P3-D160 | Recommend a separately justified texture revision | Fixed local targets and batch-compatible diversity; proposed only; existing data and no automatic fit |
''')
section('phase3EvidenceRegister.md','NB20_EVIDENCE','''## Completed NB20 — independent evidence

| Evidence | Location | Result |
|---|---|---|
| Executed master +19 payloads | [Inspection](tread_two/nb20_completion/8c9c2991e1bb/INSPECTION.json) | Three complete cells; no saved errors |
| Both HF stages | [Verification](tread_two/nb20_completion/8c9c2991e1bb/VERIFICATION.json) | Objects, owners, scientific values and provenance pass |
| Table comparison | [CSV](tread_two/nb20_completion/8c9c2991e1bb/SCIENTIFIC_COMPARISON.csv) | 48 comparable tables; copied passport tables match review bytes |
| Numeric precision | [FP64 summary](tread_two/nb20_completion/8c9c2991e1bb/FP64_WEIGHT_NORM_SUMMARY.json) / [HTML](tread_two/nb20_completion/8c9c2991e1bb/HTML_COMPARISON.json) | Explicit diagnostics; raw weights/vectors unchanged |
| Current pointers | [Registry](tread_two/geo_research_release/README.md) | Four pinned stages and actual published provenance |
''')
section('phase3SessionLog.md','NB20_SESSION','''## 2 November 2026 — NB20 verified and documented

1. Archived executed NB20 unchanged; matched all 19 payloads and three completed cells.
2. Downloaded/verified both actual HF publications, released owners and unchanged manifests. Checked 48 comparable tables, exact vectors, original/native audit records and actual passport provenance.
3. Quantified one-ULP cross-platform norm diagnostics and 4,020 FP64 original-weight norms. Parsed atlas and script are identical despite a seven-byte serialization difference.
4. Added completion report, pinned research registry and proposed next-design brief. Completed sources and both workstations are preserved.
5. No current-cycle rerun remains. A texture revision is proposed; no new training, photos or workstation changes started.
''')
section('phase3DocumentationInventory.md','NB20_DOCS','''## NB20 completion and handoff additions

| Document | Purpose |
|---|---|
| [NB20 completion](phase3NB20Completion.md) | Actual HF counts, independent checks, precision notes, figures and scientific status |
| [GEO research closure](phase3TreadTwoGEOResearchClosure.md) | Pinned model/evidence pointers and cycle completion |
| [Proposed next research](phase3TreadTwoGEONextResearch.md) | Evidence-driven texture revision and required engineering gates |
| [Registry](tread_two/geo_research_release/README.md) | Small direct CSV index of published artifacts |
''')
section('phase3RepositoryMap.md','NB20_MAP','''## NB20 completion locations

| Location | Role |
|---|---|
| `tread_two/nb20_completion/8c9c2991e1bb/` | Safe inspection, manifests, inventories and verification; execution copy and remote mirror ignored |
| `tread_two/nb20_report/figures/` | Exact published PNGs curated for Markdown |
| `tread_two/geo_research_release/` | Small local registry and exact published model-card/capability copies |
| `phase3NB20Completion.md` | Current verified stages and scientific outcome |
| `phase3TreadTwoGEONextResearch.md` | Proposed next direction; no new runnable experiment |
''')
print('NB20 completion, research registry, proposed next direction and current trackers updated.')
