"""Close verified NB07 and advance the notes to actual identity/capture review."""
import json,re,sys
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];PUB=ROOT/'data_gate_validation/nb07_publication'
sys.path.insert(0,str(ROOT/'data_gate'))
from common_gate import read,atomic
def write(name,text):(ROOT/name).write_text(text.rstrip()+'\n',encoding='utf-8')
def update(name,old,new):
 p=ROOT/name;s=p.read_text(encoding='utf-8')
 if new in s:return
 assert old in s,(name,old[:100]);write(name,s.replace(old,new,1))
def append(name,text):
 p=ROOT/name;s=p.read_text(encoding='utf-8')
 if text.strip() not in s:write(name,s.rstrip()+'\n\n'+text)
def main():
 r=read(PUB/'VERIFICATION.json');review=read(ROOT/'data_gate_validation/collection_review/VALIDATION.json');assert r['status']==review['status']=='passed'
 url=f'https://huggingface.co/datasets/Shanmuk4622/tyre-wear-study/tree/{r["revision"]}/{r["destination"]}'
 write('phase3NB07Completion.md',f'''# NB07 complete — verified data audit and next collection work

**29 October 2026 · Asia/Calcutta.** All five NB07 code cells completed without
saved errors. The independent read-only audit verifies **46 direct files**,
including **22 structured CSV tables / 7,614 rows** plus four raw/control CSVs.
HF ownership is released. **No NB07 rerun is needed for these unchanged inputs.**

[Published snapshot]({url}) ·
[Independent verification](data_gate_validation/nb07_publication/VERIFICATION.json) ·
[Final preservation/link checks](data_gate_validation/nb07_publication/HANDOFF_VALIDATION.json) ·
[Published source explorer](data_gate_validation/nb07_publication/remote/TREAD_ONE_Data_Gate.html) ·
[Next identity/capture review](phase3CollectionReview.md)

## Completed gates

| Gate | Verified result | Scope |
|---|---|---|
| Notebook | Cells 1–5, zero saved errors; 232,871 bytes | Exact supplied execution archived; 42 embedded files match accepted delivery |
| HF | 46 logical direct files including selecting manifest | Git/LFS identities and every payload SHA-256 checked at pinned revision |
| CSV | 22 structured tables, 7,614 rows including 206 dictionary rows | Four input/control CSVs add three unresolved control rows; 26 CSV files total |
| Existing data | 570 images / 12 original tyres | 418 original photos + 152 frames of existing tyres |
| Historical grouping | Pass | Ten conservative co-assignment components; no historical split crossing |
| Annotation provenance | 720 human / 912 weak point rows | Weak proposals remain separate from human truth |
| Mask conflict handling | 45 raw disagreement images; zero unignored | Existing masks/ignore policy retained; no repair invented |
| Training exposure | 47,040 saved occurrences | 23,520 per condition from 120 pinned previous CSV shards; no new training |
| Prospective readiness | **False, expected** | Zero new intake, 12 explicit gaps, draft policy and subject minima undeclared |
| Intake errors | 0 | Empty intake is not a passing prospective cohort |
| Publication format | Zero ZIP/NPZ/database uploads | Native CSV/JSON/PNG/HTML/source |

```mermaid
flowchart TD
    A["NB06 complete<br/>saved-model failure atlas"] --> B["NB07 complete on Kaggle CPU<br/>source / identity / quality / exposure audit"]
    B --> C["HF verified<br/>46 direct files / 26 CSV files"]
    C --> D["Source integrity PASS<br/>historical grouping PASS"]
    C --> E["Prospective readiness FALSE<br/>actual new intake / policy missing"]
    E --> F["Prepared now: local image comparison<br/>identity CSV export + capture priorities"]
    F --> G["User-assisted new independent tyres<br/>reviewed native labels / annotations"]
    G --> H["Freeze prospective experiment<br/>then one justified later revision"]
```

## Exact publication identity

| Item | Identity |
|---|---|
| HF repository | Shanmuk4622/tyre-wear-study |
| Pinned revision | `{r['revision']}` |
| Publication contract | `{r['publication_contract']}` |
| Kaggle data run | `{r['run_id']}` |
| Executed notebook SHA-256 | `{r['executed_notebook_sha256']}` |
| Clean delivery SHA-256, historical | `aabf20e532daeaa8d349fd5ac66be5bd1d9ee2d2f739ca30f994becda5b85220` |
| Local preflight run, retained | `a0df0c048f177fddd64622f648a214d56f621584f0ec13900297abc75f432263` |
| Local prepared publication, retained | `6f5306b473bd29f59de4184362ddc08e1acf38d9e3364afc6632bffe53ce8522` |
| Verifier's remote writes | 0 |

The Kaggle contract differs from local preparation because environment is part
of the identity. Source code, source data/overlay, raw inputs and policy agree.
Do not overwrite the local contract to disguise this distinction.

## Published CSV inventory

| Tables | Rows | Meaning |
|---|---:|---|
| source_images / source_tyres / source_clips | 570 / 12 / 3 | Existing image, physical tyre and existing clip registries |
| video_parent_review | 9 | Three clips × three possible original mid parents, unconfirmed |
| image_quality | 570 | Fixed-grid capture descriptors and native conflicts |
| region_geometry | 1,140 | Image × overlapping tyre/tread native geometry |
| point_geometry / point_coverage | 1,632 / 570 | Human/weak source points and coverage, including no-point images |
| similarity_candidates / exact_duplicates | 1,710 / 0 | Three directed neighbours per source / byte-identical pairs |
| leakage_components / historical_class_support | 10 / 9 | Conservative split constraints / role × class support |
| capture_descriptors | 7 | Nonempty historical role × class descriptive means |
| training_image_exposure | 1,140 | Condition × source image, aggregated from saved occurrences |
| intake_tyres / intake_sessions / intake_images / intake_issues | 0 / 0 / 0 / 0 | No actual new intake supplied |
| prospective_support / readiness_gaps / gate_checks | 9 / 12 / 5 | Declared role/class support, missing requirements and scoped results |
| csv_schema | 206 | Column metadata; not new empirical subjects/images |

[Complete direct table links and schemas](data_gate_outputs/CSV_INDEX.md).
Empty tables retain headers. Unknown/unavailable values remain blank; observed
zero remains zero. The four separately published input/control CSVs are the
three empty collection registries and three-row unresolved clip control.

## Cross-environment comparison and visuals

| Check | Result |
|---|---|
| Byte-equal tables | **20 of 22**; identities, source hashes, geometry/truth, support, similarity, exposure and readiness agree exactly |
| image_quality | 3,430 changed numeric cells; no text/identity changes |
| Largest image-quality brightness difference | 0.001900 intensity units for foreground; foreground-minus-background maximum 0.002007 |
| Focus descriptor maximum difference | 0.125977 intensity squared; this is a capture descriptor, not model accuracy |
| capture_descriptors | 26 changed numeric cells; maximum 0.009059 in mean focus variance |
| Embedded previews | 570 decoded 144×192 JPEGs; maximum pixel difference from local = **0** |
| HTML | JavaScript syntax valid; browser interaction/layout not verified |

These numerical differences are environment-sensitive descriptors; the exact
implementation cause was not isolated. The published environment is CPU,
Python 3.13.15, NumPy 2.1.3, Pillow 13.0.0, Torch 2.11.0+cu128,
huggingface_hub 1.29.0 and filelock 3.32.5. The local preflight uses different
package versions. No descriptor threshold or new model was fitted here.
Native truth, identity and all readiness conclusions agree exactly.

![Published class/subject coverage](data_gate_validation/nb07_publication/remote/figures/NB07_class_and_subject_support.png)

![Published quality and ignore-policy audit](data_gate_validation/nb07_publication/remote/figures/NB07_quality_and_annotation_review.png)

Both published figures were inspected. The CPU run uses no optimizer updates
or neural forward passes; the presence of a CUDA Torch build is not GPU use.

## Insights and continuation

| Evidence | Next implication |
|---|---|
| Only one low training tyre | Collect additional independent low identities, not repeated old views alone |
| Mid held-out support is zero | Obtain additional reserved mid subjects before claiming unseen-mid transfer |
| High train photos are 3 / 2 / 5 / 60 across four tyres | Subject counts alone hide uneven appearance coverage; broaden capture/identity support |
| Three mid clip parents remain unresolved | Use actual capture records; similarity and visual resemblance cannot confirm parent identity |
| No new intake or frozen policy | A correct completed audit remains training-not-ready |

Continued work is already prepared as the [identity/capture review pack](phase3CollectionReview.md):
44 existing source previews, an 18-image mid contact sheet, 56 planning/review
CSV rows plus three unresolved controls, and evidence-based resolution export.
Ten export/validator checks pass. This new pack is **local preparation, not
another HF publication**. No mapping is assigned by the agent.

Actual independent acquisition still awaits the user's planned help. No new
tyre target, split ratio, r3 architecture or training date is invented. Keep
the current workstation and validation selections. After actual collection,
review identity/annotations, declare meaningful class/role minima and freeze
full model-selection/task/failure rules before the next fit.
''')
 latest='''**Latest, 29 October 2026: NB07 is complete on Kaggle and independently HF-verified.**
[Completion and insights](phase3NB07Completion.md) ·
[22 structured CSV tables / 7,614 rows](data_gate_outputs/CSV_INDEX.md) ·
[Prepared identity/capture review](phase3CollectionReview.md).
All five cells passed; 46 direct files / 26 CSV files and released ownership
are verified. Source integrity/historical grouping pass. Prospective training
readiness remains **false**: actual new intake and frozen policy are missing.
The new local review pack helps compare existing identities and export reviewed
mapping CSVs. Zero additional tyres, model fits or inference runs were added.
No unchanged-input NB07 rerun is needed.'''
 for name in ('README.md','phase3Progress.md','phase3Understanding.md','phase3OpenItems.md','phase3DataGrowth.md','phase3DataRetentionPolicy.md'):
  p=ROOT/name;s=p.read_text(encoding='utf-8');pattern=r'<!-- NB07 CURRENT -->.*?<!-- END NB07 CURRENT -->';assert re.search(pattern,s,re.S);write(name,re.sub(pattern,'<!-- NB07 CURRENT -->\n'+latest+'\n<!-- END NB07 CURRENT -->',s,count=1,flags=re.S))
 update('README.md','| NB07 data/protocol gate | Implemented and locally verified; Kaggle/HF pending | [22-table CSV index](data_gate_outputs/CSV_INDEX.md), [run guide](phase3NB07DataGate.md); source pass, prospective ready=false |','| NB07 data/protocol gate | Complete on Kaggle / independently HF-verified | [Completion](phase3NB07Completion.md); 46 direct files, source pass, prospective ready=false |\n| Identity/capture preparation | Local review pack implemented / checked | [Review guide](phase3CollectionReview.md); 44 existing previews, mapping CSV export; actual new subjects still missing |')
 update('README.md','    F --> G["NB07 implemented / locally checked<br/>next: Kaggle publication + actual intake"]','    F --> G["NB07 complete + HF verified<br/>next: identity review + actual independent intake"]')
 update('phase3Progress.md','The review/publication is complete; NB07 is implemented locally. Next is its Kaggle publication, actual independent tyre coverage and a frozen prospective protocol.','The review/publication and NB07 Kaggle/HF audit are complete. Next is the prepared identity/capture review, actual independent tyre coverage and a frozen prospective protocol.')
 update('phase3OpenItems.md','The [NB07 data/protocol gate](phase3NB07DataGate.md) is implemented locally; its Kaggle/HF execution and independent\nuser-assisted tyre collection remain open.','The [NB07 data/protocol gate](phase3NB07Completion.md) is Kaggle-complete and independently HF-verified. Identity review and independent\nuser-assisted tyre collection remain open; the local review/export tool is ready.')
 update('phase3OpenItems.md','| Next classification milestone | [NB07](phase3NB07DataGate.md) implemented/locally verified; execute its CPU Kaggle publication, then actual intake/frozen protocol; no new subjects or r3 fit |','| Next classification milestone | [NB07](phase3NB07Completion.md) complete/HF-verified; use [identity/capture review](phase3CollectionReview.md), then actual intake/frozen protocol; no new subjects or r3 fit |')
 update('phase3DataGrowth.md','step is the [prospective data/protocol gate](phase3ProspectiveDataGate.md).','step was the [prospective data/protocol gate](phase3ProspectiveDataGate.md), now complete/HF-verified as NB07. Actual independent acquisition remains open.')
 p=ROOT/'phase3NextSteps.md';s=p.read_text(encoding='utf-8');history=s[s.index('The remaining sections preserve'):]
 write(p.name,'''# Next steps — actual identity review and independent intake

**29 October 2026. [NB07 is complete and independently HF-verified](phase3NB07Completion.md).**
All five cells and 46 direct files pass. No rerun is needed for unchanged inputs.

| Next action | Concrete deliverable / state |
|---|---|
| Review known identities | [Offline comparison/export tool](collection_review_outputs/TREAD_ONE_Identity_And_Capture_Review.html), [guide](phase3CollectionReview.md); three clips remain unresolved unless supported by actual records |
| Plan actual captures | [Class priorities CSV](collection_review_outputs/capture_priorities.csv); low training support 1, no reserved mid, high photos concentrated 3/2/5/60 |
| Acquire new independent tyres | User help later; actual sources and reviewed metadata in [intake pack](data_intake/README.md) |
| Audit changed intake | Run NB07 only when actual files/mappings/policy change; preserve completed execution and new contract separately |
| Freeze prospective experiment | Meaningful subject minima, tyre-disjoint roles, reviewed labels/annotations, model hypothesis, selection and dense-task preservation before reserved predictions |
| Later custom revision | One justified fit after data/protocol readiness; no r3 architecture/training selected now |

The new review pack is local preparation, not another live HF publication.
Its 44 existing previews and 56 planning rows add zero independent tyres.
The unchanged gate correctly reports prospective readiness=false. No new model
notebook needs to run at this stage; actual collection is the next scientific input.

'''+history)
 update('phase3NB07DataGate.md','**Implemented and executed locally, 29 October 2026.**','**Complete on Kaggle and independently HF-verified, 29 October 2026.**\n[Completion audit, publication and insights](phase3NB07Completion.md).')
 update('phase3NB07DataGate.md','training. Kaggle execution and live HF publication of NB07 are still pending.','training. All five Kaggle cells and the 46-file direct publication are now independently verified. No unchanged-input rerun is needed.')
 update('phase3NB07DataGate.md','| Publication | 46 direct files prepared locally; **no live NB07 HF upload claimed** |','| Publication | **Complete / independently HF-verified**: 46 direct files at `1b6d7122ed47350a9623476975912aff54c690a1` |')
 update('phase3NB07DataGate.md','## Run NB07 on Kaggle','## Historical Kaggle instructions — rerun only for changed inputs')
 update('phase3NB07DataGate.md','No remote revision is claimed; actual publication records `PUBLICATION.json`.','That is the historical local preparation identity. The actual Kaggle publication\nis independently verified at revision `1b6d7122ed47350a9623476975912aff54c690a1`\nwith contract `dbb3cf3634d1c2b19034b32e7182a4deaab1fdb5dc094a605f733514009d6f91`.\n[The completion report](phase3NB07Completion.md) explains the environment-bound identities.')
 update('phase3NB07DataGate.md','Next: run NB07 on Kaggle for publication, review the video-parent queue, then\npopulate actual user-assisted independent tyre intake. Freeze the prospective\nexperiment before choosing one justified later custom revision.','Next: use the [identity/capture review pack](phase3CollectionReview.md), then\npopulate actual user-assisted independent tyre intake. NB07 is complete; rerun\nonly for changed inputs. Freeze the prospective experiment before a later fit.')
 update('phase3ProspectiveDataGate.md','**29 October 2026: implemented as NB07 and executed locally.**','**29 October 2026: NB07 is complete on Kaggle and independently HF-verified.**\n[Completion and exact publication](phase3NB07Completion.md).')
 update('phase3ProspectiveDataGate.md','| Kaggle / HF execution | Notebook ready; not yet executed there | CPU/Internet/prepared input/HF_TOKEN; 46 direct files prepared locally |','| Kaggle / HF execution | Complete / independently verified | Five successful cells, 46 direct files, 26 CSVs; released writer |\n| Identity/capture review tool | Implemented/checked locally | [Comparison and export guide](phase3CollectionReview.md); actual new data still needed |')
 update('phase3ProspectiveDataGate.md','B --> H["NEXT: Kaggle NB07<br/>direct HF evidence publication"]','B --> H["DONE: Kaggle NB07 + HF verification<br/>next: identity/capture review"]')
 update('phase3NB06Completion.md','implemented the CPU source/intake audit; its Kaggle publication is the current execution step.','implemented the CPU source/intake audit; [NB07 completion](phase3NB07Completion.md) now verifies its Kaggle/HF publication. Actual identity/collection work follows.')
 # Preserve local preflight exports. Published versions are linked separately.
 p=ROOT/'data_gate_outputs/CSV_INDEX.md';s=p.read_text(encoding='utf-8')
 if '<!-- PUBLISHED NB07 -->' not in s:
  h,rest=s.split('\n',1);s=h+'\n\n<!-- PUBLISHED NB07 -->\n**Kaggle publication is complete and independently verified.**\n[Completion audit](../phase3NB07Completion.md) · [Pinned published files]('+url+') ·\n[Cached published CSV folder](../data_gate_validation/nb07_publication/remote/csv).\nThe table links below retain the original local preflight exports: 20 CSVs\nare byte-identical to publication, while image_quality/capture_descriptors\nhave small environment-sensitive numerical differences. See the completion\nreport for exact magnitudes. The local run/contract remains preserved.\n<!-- END PUBLISHED NB07 -->\n'+rest;write(p.relative_to(ROOT),s)
 atomic(ROOT/'data_gate_outputs/PUBLISHED.json',dict(status='independently_verified',revision=r['revision'],destination=r['destination'],publication_contract=r['publication_contract'],run_id=r['run_id'],published_files=46,CSV_files=26,structured_rows=7614,local_preflight_preserved=True,verification='data_gate_validation/nb07_publication/VERIFICATION.json'))
 append('phase3EvidenceRegister.md',f'''## NB07 completed Kaggle publication — 29 October 2026

| Evidence | Verified state |
|---|---|
| [Publication verification](data_gate_validation/nb07_publication/VERIFICATION.json) | 46 direct files / 26 CSVs; SHA-256 plus Git/LFS checks; released owner |
| Pinned revision | `{r['revision']}` |
| Kaggle publication contract | `{r['publication_contract']}` |
| Executed notebook | SHA `{r['executed_notebook_sha256']}`; five successful cells; 42 source files match |
| [Cross-environment comparison](data_gate_validation/nb07_publication/CROSS_ENVIRONMENT_COMPARISON.json) | 20 CSVs byte-identical; two numeric quality tables differ; 570 preview decoded pixels exact |
| [Local continuation pack](collection_review_outputs/CONTRACT.json) | 44 selected existing sources, 56 planning rows plus three unresolved controls; zero mappings assigned/new tyres |
| [Review tool checks](data_gate_validation/collection_review/VALIDATION.json) | Ten export/validator checks; source/previews/JS verified; browser rendering not verified |

The verifier made no remote writes. The continuation pack is local preparation,
separate from the completed 46-file HF snapshot. [Completion](phase3NB07Completion.md).''')
 append('phase3RepositoryMap.md','''## After NB07 completion — 29 October 2026

| Path | Current role |
|---|---|
| `data_gate_validation/nb07_publication/` | Exact executed NB07 archive, public-read pinned downloads, notebook/CSV/hash/visual comparison receipts |
| `collection_review/` | Separate local review builder and JavaScript export rules; completed gate source unchanged |
| `collection_review_outputs/` | 44-preview HTML, 18-image contact sheet, three planning CSVs plus unresolved control, hashes/schema; local preparation |
| `data_gate_validation/collection_review/` | Ten review/export checks and isolated synthetic validation files, excluded from scientific data |

See [NB07 completion](phase3NB07Completion.md) and [review guide](phase3CollectionReview.md).''')
 append('phase3DocumentationInventory.md','''## NB07 completion and continuation — 29 October 2026

[phase3NB07Completion.md](phase3NB07Completion.md) records the actual five-cell
Kaggle execution, pinned 46-file HF verification, 26 CSVs, cross-environment
differences, source/subject findings and readiness limits.
[phase3CollectionReview.md](phase3CollectionReview.md) documents the implemented
local comparison/export tool, CSVs, source hashes, tests and actual collection
priorities. Current status/next-action notes were synchronized; the original
local preflight identities and 103-file Markdown baseline remain preserved.''')
 p=ROOT/'phase3Decisions.md';s=p.read_text(encoding='utf-8');decisions='''| P3-D048 | 2026-10-29 | Close NB07 as Kaggle-complete and independently HF-verified; preserve executed notebook/source | Five cells, 46 direct files / 26 CSVs, released owner and source/manifest/Git/LFS checks; [completion](phase3NB07Completion.md) |
| P3-D049 | 2026-10-29 | Keep source-audit success separate from prospective training readiness after completion | Zero new intake, unfrozen policy and undeclared subject minima correctly retain 12 gaps; do not trigger another fit |
| P3-D050 | 2026-10-29 | Continue with local evidence-based identity/capture review rather than assign video parents by similarity | 44 existing previews and export validation help human review; actual acquisition remains necessary; [guide](phase3CollectionReview.md) |
| P3-D051 | 2026-10-29 | Preserve local and published descriptor records separately | 20 CSVs byte-identical; two quality tables have small numeric environment differences; do not replace contracts or claim new measurements |
'''
 if 'P3-D048' not in s:write(p.name,s.replace('\n\nAppend a new decision','\n'+decisions+'\nAppend a new decision'))
 append('phase3DataRetentionPolicy.md','''## Completed audit / manual review boundary — 29 October 2026

NB07 now has an independently verified direct-file publication. Retain its
executed notebook and published/local contract distinction; quality differences
are recorded per column, not hidden. Human review exports start unresolved,
require actual evidence to confirm and remain distinct from authoritative
project inputs until reviewed. The new local capture-review pack is not yet
published to HF. No repeat model training or invented acquisition is needed
to preserve this continuation evidence.''')
 append('phase3SessionLog.md',f'''## 29 October 2026 — verify executed NB07 and prepare actual capture review

The supplied executed NB07 has five successful code cells and SHA
`{r['executed_notebook_sha256']}`. Archived it without modification and verified
all 42 embedded accepted files. Independently fetched the pinned public HF
snapshot at `{r['revision']}` using read-only public requests and bounded retry,
verified 46 direct logical files / 26 CSVs, every manifest SHA-256 and Git/LFS
identity, publication contract and released owner. No credential fallbacks or
remote writes were used.

| Finding / action | Result |
|---|---|
| Actual audit | 570 images / 12 tyres, 7,614 structured rows; source and historical grouping pass |
| Prospective readiness | False as expected; zero intake/new subjects, 12 explicit gaps |
| Cross-environment records | 20 CSVs byte-identical; quality/capture numeric differences recorded per field; all text/truth/identity/exposure/support unchanged |
| Source gallery | 570 decoded previews, pixel difference zero; JS syntax valid; published figures inspected |
| Next work implemented | Local identity comparison/export and capture priorities; 44 native source hashes, 56 planning rows plus three unresolved controls, contact sheet |
| Review logic | Ten checks pass; actual project mappings remain unresolved; synthetic export validates in a temporary folder only |
| Coverage detail | High training tyres have 3/2/5/60 photos; viewer includes the two available views for the smallest group without duplicating a third |
| Publication boundary | NB07 is complete on HF; new review pack remains local preparation, no new upload or mapping claim |
| Preserved | Original 103 Markdown, 27 frozen experiment files, nine completed gate sources and all executed notebooks |

Updated completion/run/next-step/context/decision/evidence/intake-retention notes.
No model inference, optimizer update, physical acquisition or promotion occurred.
The next scientific input is actual user-assisted independent acquisition and
reviewed provenance/annotations, followed by a complete prospective protocol.
''')
 print('NB07 completion and next identity/capture review documented; historical preflight preserved.')

if __name__=='__main__':main()
