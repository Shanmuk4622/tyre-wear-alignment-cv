"""Publish stable local CSV/figure links and synchronize the Phase 3 trackers."""
import json,re,shutil,sys
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(name,text):(ROOT/name).write_text(text.rstrip()+'\n',encoding='utf-8')
def section(name,text):
 p=ROOT/name;s=p.read_text(encoding='utf-8');heading,rest=s.split('\n',1)
 marker='<!-- NB07 CURRENT -->';end='<!-- END NB07 CURRENT -->'
 if marker in rest:rest=re.sub(re.escape(marker)+'.*?'+re.escape(end)+'\n*','',rest,flags=re.S)
 write(name,heading+'\n\n'+marker+'\n'+text.strip()+'\n'+end+'\n\n'+rest.lstrip())
def replace(name,old,new):
 p=ROOT/name;s=p.read_text(encoding='utf-8')
 if new in s:return
 assert old in s,(name,old[:80]);write(name,s.replace(old,new,1))
def append_once(name,text):
 p=ROOT/name;s=p.read_text(encoding='utf-8')
 if text.strip() not in s:write(name,s.rstrip()+'\n\n'+text)

def main():
 base=ROOT/'data_gate_outputs';latest=read(base/'LATEST.json');run=Path(latest['path']);summary=read(run/'SUMMARY.json');schema=read(run/'CSV_SCHEMA.json')
 validation=read(ROOT/'data_gate_validation/VALIDATION.json');assert validation['status']=='passed' and validation['run_id']==latest['run_id']
 publication=read(base/'PUBLICATION_DELIVERY.json');delivery=read(ROOT/'data_gate_validation/DELIVERY.json')
 for name in schema:
  target=base/'csv'/(name+'.csv');target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(run/'csv'/target.name,target)
 for p in (run/'figures').glob('*.png'):
  target=base/'figures'/p.name;target.parent.mkdir(exist_ok=True);shutil.copy2(p,target)
 for name in ('SUMMARY.json','CONTRACT.json','CSV_SCHEMA.json','POLICY.json'):shutil.copy2(run/name,base/name)
 roles={
 'source_images':'One existing source image/frame: IDs, hashes, domain, label, historical split and parent constraint',
 'source_tyres':'One recorded original physical tyre; missing measurements remain blank',
 'source_clips':'One existing clip; no new independent subject',
 'video_parent_review':'One clip × possible original mid parent; similarity is review evidence only',
 'image_quality':'One existing image, fixed-grid quality and native conflict counts',
 'region_geometry':'One existing image × tyre/tread region, valid native geometry',
 'point_geometry':'One retained human/weak source point, native and normalized coordinates',
 'point_coverage':'One image, including images without points',
 'similarity_candidates':'One image × directed nearest-neighbour candidate, three per image',
 'exact_duplicates':'One byte-identical image pair; empty with headers when none',
 'leakage_components':'One conservative split co-assignment constraint; not a new subject count',
 'historical_class_support':'One historical role × proxy class, including absent classes',
 'capture_descriptors':'One nonempty historical role × class, descriptive image-weighted means',
 'training_image_exposure':'One condition × source image; aggregates saved augmented occurrences',
 'intake_tyres':'One supplied tyre record; currently none',
 'intake_sessions':'One supplied capture session; currently none',
 'intake_images':'One supplied native image; currently none',
 'intake_issues':'One input validation issue; zero rows does not make empty intake ready',
 'prospective_support':'One proposed role × class, including missing minima',
 'readiness_gaps':'One missing prospective requirement',
 'gate_checks':'One scoped historical/prospective check',
 'csv_schema':'One column definition, units and provenance meaning; metadata rather than new observations'}
 rows=['| CSV | Rows | Record grain / interpretation |','|---|---:|---|']
 for name in roles:rows.append(f'| [{name}.csv](csv/{name}.csv) | {schema[name]["rows"]:,} | {roles[name]} |')
 write('data_gate_outputs/CSV_INDEX.md',f'''# NB07 direct CSV evidence — open here

**29 October 2026. Locally completed audit.** {len(schema)} structured tables /
{summary['csv_rows']:,} rows, including 206 schema rows. These are derived from
570 existing images / 12 original tyres and existing logged training evidence.
**Zero new independent tyres or images supplied; prospective training ready=false.**

[Full technical/run guide](../phase3NB07DataGate.md) ·
[Interactive source explorer](runs/{run.name}/TREAD_ONE_Data_Gate.html) ·
[Intake guide](../data_intake/README.md) ·
[Validation receipt](../data_gate_validation/VALIDATION.json)

## Direct table inventory

'''+ '\n'.join(rows)+f'''

Empty tables retain headers. Blank means unknown/not applicable; zero means
observed zero. The 1,140 exposure rows summarize 47,040 saved augmented
occurrences; they are not 47,040 independent photos. Points distinguish
720 human from 912 weak polygon-derived records. Similarity rows never assign
identity. Ten leakage components retain 12 physical tyres.

## Contracts and source identities

| Artifact | Meaning |
|---|---|
| [SUMMARY.json](SUMMARY.json) | Completed audit, historical pass and separate prospective-readiness result |
| [CSV_SCHEMA.json](CSV_SCHEMA.json) | Exact table headers, row counts, byte sizes and SHA-256 values |
| [CONTRACT.json](CONTRACT.json) | Source, code, environment, policy and intake identities |
| [POLICY.json](POLICY.json) | Actual draft policy used; no invented subject minima |
| [LATEST.json](LATEST.json) | Versioned original run path; these stable CSV/PNG copies preserve its exact bytes |
| [Publication preparation receipt](PUBLICATION_DELIVERY.json) | 46 direct files prepared, not a live HF revision |

Run ID: `{latest['run_id']}`.
Four raw input/control CSVs are preserved in the versioned run and separately
in the publication snapshot; their three unresolved clip rows are not included
in the structured-row total. Publication therefore contains 26 CSV files.

![Class and physical-tyre support](figures/NB07_class_and_subject_support.png)

![Capture descriptors and mask policy](figures/NB07_quality_and_annotation_review.png)

The source gallery was generated and its JavaScript/previews checked. Browser
interaction/layout was not verified. Full-resolution incoming data are hash-bound
by future audits but are not automatically uploaded by this evidence publisher.
''')
 # Replace the former active plan, preserving earlier model evidence elsewhere.
 write('phase3ProspectiveDataGate.md','''# TREAD-ONE — prospective data and protocol gate

**29 October 2026: implemented as NB07 and executed locally.** The source-data
audit is complete; actual independent collection and protocol freeze remain
open. Read [NB07 technical/run guide](phase3NB07DataGate.md),
[22-table CSV inventory](data_gate_outputs/CSV_INDEX.md) and
[intake instructions](data_intake/README.md).

| Gate | State | Evidence / next action |
|---|---|---|
| Existing source and historical grouping | Pass locally | 570 images / 12 tyres; 10 conservative components; hashes and split constraints checked |
| Richer CSV collection | Implemented and executed locally | 22 tables / 7,614 rows, including schema; 47,040 old augmented occurrences summarized |
| Identity review | Queue implemented; mapping unresolved | Three distinct existing mid clips, nine possible parent combinations; no automatic identity assignment |
| New independent subjects | Missing | Zero additional tyres/images supplied; user plans to help later |
| Prospective data readiness | False, expected | 12 explicit gaps; actual intake, role/class minima and frozen policy missing |
| Kaggle / HF execution | Notebook ready; not yet executed there | CPU/Internet/prepared input/HF_TOKEN; 46 direct files prepared locally |
| Later custom revision | Not started | Choose one justified hypothesis after data/protocol review; no r3 fit or promotion |

## Evidence behind the order

| Observation | Consequence |
|---|---|
| Both NB05 selected models miss all 62 low test photos | Low transfer must remain an explicit endpoint |
| Training contains only one low tyre | Add independent subjects rather than counting repeated photos as evidence of transfer |
| No mid tyre in historical validation/test | Acquire new reserved mid subjects before claiming unseen-mid generalization |
| Projection validation BA 83.75%, selected test BA 17.073% | Validation spikes do not replace representative independent evaluation |
| Dice remains near 97.8%; candidate point error rises | Declare dense-task preservation before the next fit |
| Labels are mileage proxies | Physical wear/depth requires separate actual measurement truth |

This motivates a data/representation investigation; it does not establish a
single cause of failure or a universal optimizer conclusion.

```mermaid
flowchart TD
    A["DONE: NB06 review / HF verified"] --> B["DONE LOCALLY: NB07 source audit<br/>CSV / identity queue / quality / exposure"]
    B --> H["NEXT: Kaggle NB07<br/>direct HF evidence publication"]
    B --> C["User-assisted independent capture<br/>low / mid / high + documented sessions"]
    C --> D["Validate native files / annotations<br/>resolve identity + label provenance"]
    D --> E["Freeze prospective tyre-disjoint protocol<br/>before reserved predictions"]
    E --> F["One motivated custom revision<br/>explicit dense-task constraints"]
    F --> G["Later resumable Kaggle training<br/>rich direct-HF evidence"]
```

## Rules before another model fit

Keep all views/derivatives of a tyre together. The old inspected test set is
development evidence; new reserved validation/test subjects must be independent.
Leave unknown readings/identities blank with a reason. Human and weak point
labels remain distinct. Record proxy sources, mask review and capture conditions.
Choose actual positive per-class/per-role subject minima against the study
budget; no sample size, recruitment campaign or split ratio is invented here.

Passing the software data gate is not approval of a complete scientific
experiment. Document model hypothesis, changed parameters/objectives, selection
rule, task/failure endpoints, dense preservation and budget before training.
A tread-focused classification path remains a candidate design question,
not a selected or validated architecture. If data changes too, comparison to
old models does not isolate architecture effects.

Future training follows [standing instructions](AGENTS.md): Kaggle notebook,
measured storage, documented dual-T4 usage, native completed-update recovery,
extensive direct CSV, explicit HF_TOKEN and bounded periodic/stop publication.
NB07 itself is a finite CPU audit with completed-stage publication.
''')
 latest_text='''**Latest, 29 October 2026: NB07 is implemented and locally verified.**
[Data/protocol gate and run guide](phase3NB07DataGate.md) ·
[22 direct CSV tables / 7,614 rows](data_gate_outputs/CSV_INDEX.md) ·
[Kaggle CPU notebook](notebooks/phase3_NB07_TREAD_ONE_Data_And_Protocol_Gate.ipynb).
The audit covers 570 existing images / 12 tyres and summarizes 47,040 saved
training occurrences. Source integrity and historical grouping pass;
prospective training readiness is **false** because actual new intake and
the frozen policy are missing. Zero additional tyres were collected. There are
46 direct publication files prepared; Kaggle execution/live NB07 HF upload
remain pending. No new fit, inference or workstation promotion.'''
 for name in ('README.md','phase3Progress.md','phase3Understanding.md','phase3OpenItems.md','phase3DataGrowth.md','phase3DataRetentionPolicy.md'):
  section(name,latest_text)
 # Explicitly relabel the former front-page status as historical, avoiding two active next actions.
 replace('README.md','**Latest — 29 October 2026: NB06 is complete and independently HF-verified.**','**Previous milestone — 29 October 2026: NB06 is complete and independently HF-verified.**')
 replace('README.md','The NB05 selected projection result remains worse than baseline; no model was\ntrained or promoted by NB06. Next is the','The NB05 selected projection result remains worse than baseline; no model was\ntrained or promoted by NB06. Its planned next milestone is now implemented as the')
 replace('README.md','| Further training / workstation promotion | No next measured fit launched; installed models retained | Independent subject coverage and a prospective protocol are the next data gate |','| NB07 data/protocol gate | Implemented and locally verified; Kaggle/HF pending | [22-table CSV index](data_gate_outputs/CSV_INDEX.md), [run guide](phase3NB07DataGate.md); source pass, prospective ready=false |\n| Further training / workstation promotion | No next measured fit launched; installed models retained | Actual independent collection and frozen prospective protocol remain open |')
 replace('README.md','    E --> F["NB06 review + publication complete<br/>next: independent tyre data / protocol gate"]','    E --> F["NB06 review + publication complete"]\n    F --> G["NB07 implemented / locally checked<br/>next: Kaggle publication + actual intake"]')
 replace('README.md','The next milestone is the [prospective data/protocol gate](phase3ProspectiveDataGate.md).','The [prospective data/protocol gate](phase3ProspectiveDataGate.md) is now implemented locally as NB07.')
 replace('phase3Progress.md','**Latest, 29 October 2026:** NB05 is complete','**Earlier NB05/NB06 completion, 29 October 2026:** NB05 is complete')
 replace('phase3Progress.md','independently verified: 35 direct files / ten CSV tables. Next is the\n[data/protocol gate](phase3ProspectiveDataGate.md); no further fit is launched.','independently verified: 35 direct files / ten CSV tables. The\n[data/protocol gate](phase3ProspectiveDataGate.md) has since been implemented as NB07; no further fit is launched.')
 replace('phase3Progress.md','The review/publication is complete; next is independent tyre coverage and a prospective protocol.','The review/publication is complete; NB07 is implemented locally. Next is its Kaggle publication, actual independent tyre coverage and a frozen prospective protocol.')
 replace('phase3Understanding.md','**Latest, 29 October 2026:** NB05 completed','**Earlier NB05/NB06 completion, 29 October 2026:** NB05 completed')
 replace('phase3Understanding.md','outcomes. Next is the [data/protocol gate](phase3ProspectiveDataGate.md); no','outcomes. The [data/protocol gate](phase3ProspectiveDataGate.md) is now implemented as NB07; no')
 replace('phase3OpenItems.md','**Latest, 29 October 2026:** NB05\'s later execution','**Earlier NB05/NB06 completion, 29 October 2026:** NB05\'s later execution')
 replace('phase3OpenItems.md','The [prospective data/protocol gate](phase3ProspectiveDataGate.md) and independent\nuser-assisted tyre collection remain open.','The [NB07 data/protocol gate](phase3NB07DataGate.md) is implemented locally; its Kaggle/HF execution and independent\nuser-assisted tyre collection remain open.')
 replace('phase3OpenItems.md','| Next classification milestone | [Data/protocol gate](phase3ProspectiveDataGate.md); no new subjects collected or r3 training started |','| Next classification milestone | [NB07](phase3NB07DataGate.md) implemented/locally verified; execute its CPU Kaggle publication, then actual intake/frozen protocol; no new subjects or r3 fit |')
 p=ROOT/'phase3NextSteps.md';s=p.read_text(encoding='utf-8');remaining=s.index('The remaining sections preserve')
 write('phase3NextSteps.md','''# Next steps — NB07 publication and actual independent intake

**29 October 2026.** [NB07 is implemented and verified locally](phase3NB07DataGate.md).
It collects 22 structured CSV tables / 7,614 rows from existing evidence, plus
an image explorer and explicit prospective readiness checks. Source integrity
and historical grouping pass. No new model fit or inference was launched.

| Next action | State / instruction |
|---|---|
| Run NB07 on Kaggle and publish | Open [NB07](notebooks/phase3_NB07_TREAD_ONE_Data_And_Protocol_Gate.ipynb); CPU, Internet ON, prepared Phase 2 input, HF_TOKEN, ≥1 GiB free; Run All |
| Review the output | [CSV index](data_gate_outputs/CSV_INDEX.md); current false readiness is expected with empty intake |
| Resolve clip-parent provenance where possible | Three existing distinct mid clips; candidate distances do not prove identity |
| Collect actual additional tyres | User-assisted low/mid/high capture, including independent reserved mid subjects; [intake format](data_intake/README.md) |
| Freeze the prospective protocol | Actual subject minima, tyre-disjoint roles, labels/annotations, selection and dense-task criteria before reserved predictions |
| Later custom model | One justified revision after data/protocol review; no r3 selected or started |

The 46-file direct publication snapshot is prepared locally. NB07 has not yet
been run on Kaggle or independently HF-verified. Existing NB02–NB06 publications
remain complete. Current collection adds zero new independent tyres.

'''+s[remaining:])
 replace('phase3NextSteps.md','Their next-run wording is superseded by the completed audit and actions above;','Their next-run wording is superseded by the completed audit and NB07 actions above;')
 replace('phase3NB06Completion.md','The next scientific milestone is the\n[classification generalization/data-and-protocol gate](phase3ProspectiveDataGate.md):','The next scientific milestone was the\n[classification generalization/data-and-protocol gate](phase3ProspectiveDataGate.md), now implemented locally as [NB07](phase3NB07DataGate.md):')
 replace('phase3NB06Completion.md','review notebook was launched by this documentation update.','review notebook was launched by the NB06 documentation update. NB07 subsequently\nimplemented the CPU source/intake audit; its Kaggle publication is the current execution step.')
 section('phase3RepositoryMap.md','''## NB07 source / output ownership — 29 October 2026

| Path | Purpose |
|---|---|
| `data_gate/` | Seven CPU audit/report/publisher modules + two JSON source/policy contracts |
| `data_intake/` | Three empty collection CSVs, existing-clip resolution control, version marker and incoming native-file folder |
| `notebooks/phase3_NB07_TREAD_ONE_Data_And_Protocol_Gate.ipynb` | New self-contained Kaggle CPU audit, not training |
| `data_gate_outputs/csv/`, `figures/` | Stable verified copies of 22 current CSVs and two figures; [index](data_gate_outputs/CSV_INDEX.md) |
| `data_gate_outputs/runs/`, `cache/`, `publication/` | Retained local versioned results, completed-image/download cache and immutable direct publication preparation |
| `data_gate_validation/` | Unit/CSV/notebook/fresh-CPU/offline-publication receipts; synthetic/fresh fixtures excluded from scientific counts |
| `analysis/` | NB07 builder, targeted checks, replay/handoff validators and documentation exporter |
| `csv_tools/` | Local Artifact Tool CSV round-trip verifier; no Kaggle dependency |

Original datasets, frozen 27 experiment sources and executed NB05/NB06 are preserved.''')
 section('phase3EvidenceRegister.md',f'''## NB07 local implementation evidence — 29 October 2026

| Evidence | Recorded result / identity |
|---|---|
| [Run contract](data_gate_outputs/CONTRACT.json) | `{latest['run_id']}`; source/data/intake/policy/environment bound |
| [Structured CSV index](data_gate_outputs/CSV_INDEX.md) | 22 tables / 7,614 rows including 206 dictionary definitions; zero new subjects |
| [Source summary](data_gate_outputs/SUMMARY.json) | 570 images / 12 tyres; source and historical split pass; prospective false |
| [NB07 payload delivery](data_gate_validation/DELIVERY.json) | Notebook SHA `{delivery['notebook_sha256']}`; 42 embedded files |
| [Targeted unit checks](data_gate_validation/UNIT_TESTS.json) | 19 passed; isolated synthetic fixtures, not scientific records |
| [CSV Artifact Tool receipt](data_gate_validation/ARTIFACT_CSV_VALIDATION.json) | All 22 CSVs exact after import/recalculation/export |
| [Fresh CPU / publication check](data_gate_validation/VALIDATION.json) | Three extracted-source CPU stages, network/CUDA disabled, four actual source-image recomputations, all CSV/HTML byte equal; offline 46-file publication recovery |
| [Prepared publication](data_gate_outputs/PUBLICATION_DELIVERY.json) | `{publication['contract']}`; 46 direct files, 26 CSV; no live upload yet |

Raw mask disagreements 45 vs unignored 0; human/weak points 720/912. Existing
training occurrences summarized 47,040; no neural inference/optimizer update.
Browser rendering and actual Kaggle/HF execution are not claimed.''')
 decisions='''| P3-D044 | 2026-10-29 | Implement NB07 as a CPU data/protocol gate before another fit | One low training tyre, no held-out mid subjects and failed selected transfer require explicit subject/provenance evidence; no architecture benefit invented |
| P3-D045 | 2026-10-29 | Preserve 22 direct structured CSV tables plus raw intake/control snapshots, native figures/HTML and contracts | 7,614 rows describe existing data/metadata; additional physical tyre count stays zero; [NB07](phase3NB07DataGate.md) |
| P3-D046 | 2026-10-29 | Keep audit completion separate from prospective readiness; leave minima and protocol unfrozen until actual collection | A valid source audit can publish false readiness; synthetic validator fixtures are not a sample-size recommendation |
| P3-D047 | 2026-10-29 | Canonicalize CSV column union across cached and recomputed descriptors | Fresh CPU replay exposed insertion-order drift in cached point records; canonical exports restore byte-identical resume results |
'''
 p=ROOT/'phase3Decisions.md';s=p.read_text(encoding='utf-8')
 if 'P3-D044' not in s:write('phase3Decisions.md',s.replace('\n\nAppend a new decision', '\n'+decisions+'\nAppend a new decision'))
 append_once('phase3DocumentationInventory.md','''## NB07 implementation documentation — 29 October 2026

[phase3NB07DataGate.md](phase3NB07DataGate.md) now owns the implemented CPU gate,
code/data flow, descriptor definitions, recovery limits, tests, Kaggle setup and
publication status. [CSV_INDEX.md](data_gate_outputs/CSV_INDEX.md) inventories all
22 current tables and diagrams. [Prospective gate](phase3ProspectiveDataGate.md)
now distinguishes locally completed implementation from missing new intake.
[Intake guide](data_intake/README.md) documents exact fields and annotation JSON.
Progress/Understanding/NextSteps/OpenItems, DataGrowth/Retention, Decisions,
EvidenceRegister, RepositoryMap and SessionLog were synchronized. The initial
103-file Markdown baseline is unchanged; these are Phase 3 working notes.''')
 append_once('AGENTS.md','''## NB07 data-retention reaffirmation — 29 October 2026

The user authorized implementation and explicitly requested more CSV collection.
Keep source/intake registries, descriptor definitions, labels/identity review,
annotation provenance, per-image exposure, coverage, validation issues and gaps
as direct CSV with hashes/schema. Richer derived rows do not add independent
tyres. The shipped empty intake must never manufacture training-ready status.
Keep future training's complete-state recovery and HF cadence requirements;
document the finite CPU audit's narrower publication/recovery guarantees.''')
 append_once('phase3SessionLog.md',f'''## 29 October 2026 — implement NB07 and collect richer CSV evidence

The user approved the proposed data/protocol gate and requested more CSVs and
maintained Markdown. Implemented seven CPU modules, two JSON contracts and a
self-contained Kaggle NB07 notebook (42 embedded files). Executed the audit on
570 existing images: 22 structured CSVs / 7,614 rows, including schema, plus
four preserved raw input/control CSVs. New independent subjects/images remain
zero. Aggregated 120 pinned NB05 shards / 47,040 saved training occurrences
without inference or training.

| Finding / action | Outcome |
|---|---|
| Historical support | Train low/mid/high tyres 1/3/4; validation/test each 1/0/1 |
| Identity | 12 original tyres; three distinct existing mid clips, unresolved parents; 10 conservative co-assignment components |
| Annotation audit | 45 raw tread-outside-tyre images, zero after existing ignore masks; 720 human / 912 weak point rows |
| Actual prospective intake | Empty; 12 readiness gaps; correct ready=false |
| Outputs | Two figures, 570-preview filterable HTML, 46-file direct publication preparation; no ZIP/native incoming-media upload |
| Resume defect found by fresh replay | Cached point key order changed CSV headers; canonicalized column union; complete replay now byte-identical |
| Validation | 19 targeted unit checks; 22 exact Artifact Tool CSV round-trips; three fresh CPU stages with network/CUDA disabled; four real descriptor recomputations; offline ambiguous-response publication recovery |
| Preserved | Original 103 Markdown sources, frozen 27 experiment sources and executed NB05/NB06 |
| Not claimed | Kaggle execution/live NB07 HF publication, browser rendering, new training, inference or new acquired tyres |

Run identity `{latest['run_id']}`; notebook SHA
`{delivery['notebook_sha256']}`; prepared publication contract
`{publication['contract']}`. Figures/CSV preview visually inspected; HTML syntax
and 570 embedded previews checked. [Technical/run guide](phase3NB07DataGate.md).
Next user action: Kaggle NB07 CPU/Internet/prepared input/HF_TOKEN, Run All; later
actual independent intake and full prospective protocol before another fit.
''')
 # Refresh the exact prepared publication identity in the technical guide.
 p=ROOT/'phase3NB07DataGate.md';s=p.read_text(encoding='utf-8');s=re.sub(r'Local publication contract:\n`[0-9a-f]{64}`',f'Local publication contract:\n`{publication["contract"]}`',s);write(p.name,s)
 print('NB07 stable CSV/figure copies and 16 documentation trackers synchronized.')

if __name__=='__main__':main()
