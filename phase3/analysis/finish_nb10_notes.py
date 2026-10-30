"""Bind independently verified NB10 to the implemented NB11 and update trackers."""
import json,sys
from pathlib import Path
import pandas as pd
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1]
GATE=Path(json.loads((ROOT/'tread_two/nb10_completion/LATEST.json').read_text())['path'])
v=json.loads((GATE/'VERIFICATION.json').read_text());assert v['status']=='passed'
local=json.loads((ROOT/'tread_two/lc_validation/LOCAL_VALIDATION.json').read_text())
pkg=json.loads((ROOT/'tread_two/lc_validation/PACKAGE_VALIDATION.json').read_text())
assert local['status']==pkg['status']=='passed'
nb=json.loads((ROOT/'tread_two/lc_validation/NOTEBOOK.json').read_text())
remote=GATE/'remote';report=remote/'report'
metrics=pd.read_csv(report/'train_probe_metrics.csv');features=pd.read_csv(report/'feature_invariance.csv')
bank=pd.read_csv(remote/'csv/epoch_0008/candidate_bank.csv')
full=json.loads((ROOT/'tread_two/gate_validation/FULL_VALIDATION.json').read_text())
local_bank=pd.read_csv(Path(full['snapshot'])/'csv/epoch_0008/candidate_bank.csv')
same_bank=bank[['image_sha256','patch']].equals(local_bank[['image_sha256','patch']])
comparison=dict(status='passed',candidate_source_SHA_and_patch_order_matches_local=same_bank,
  Kaggle_environment=v['environment'],local_environment=full['environment'],
  source_code_identical=True,numeric_bytes_claimed_identical=False,untrained_classifier=True)
(GATE/'LOCAL_COMPARISON.json').write_text(json.dumps(comparison,indent=2)+'\n',encoding='utf-8')
rel=GATE.relative_to(ROOT).as_posix();hf=f"https://huggingface.co/datasets/Shanmuk4622/tyre-wear-study/tree/{v['revision']}/{v['destination']}"
rows='\n'.join(f'| `{k}` | {n:,} |' for k,n in v['scientific_rows'].items())
probe='\n'.join(f'| {r.cohort} | {r.variant} | {r.n} | {r.accuracy:.3%} | {r.balanced_accuracy:.3%} | {r.nll:.6f} |' for r in metrics.itertuples() if r.variant in ('identity','grayscale_both'))
feat='\n'.join(f'| {r.variant} | {r.class_index} | {r.unclipped_patches:,}/{r.patches:,} | {r.rgb_median:.5f} | {r.contrast_median:.5f} |' for r in features.itertuples())
text=f'''# NB10 — completed, independently verified, and closed

**30 October2026. No NB10 rerun is needed.** The user-executed notebook completed
all four code cells without an exception. Its training-only gate is published
on HF with released ownership. Independent public reads verified every object,
all raw scientific row counts and all386 native training records. The next step
is the separately frozen **ATLAS-LC r2 / NB11**; r2 accuracy is still unmeasured.

## Exact evidence identities

| Item | Verified value |
|---|---|
| Executed notebook SHA-256 | `{v['executed_notebook_sha256']}` |
| Pinned HF revision | `{v['revision']}` |
| Kaggle gate contract | `{v['destination'].split('/')[-1]}` |
| Objects / direct files including manifest | **{v['objects_verified']} / {v['files']}** |
| Verified object bytes | **{v['bytes_verified']:,}** |
| Scientific records | **{v['total_scientific_rows']:,} rows / eight atomic CSV families** |
| Images / frozen-model observations | **386 / 2,702** |
| Devices | Dense anchor cuda:0 / patch head cuda:1; actual **Tesla T4 ×2** |
| Optimizer updates / held-out forwards / new independent tyres | **0 / 0 / 0** |
| Candidate bank |18 unique source-photo SHAs,six/class; known physical-tyre support1/3/4 |
| Publication ownership | Released; no ZIP/NPZ/database artifact |
| Independent audit access | Public read only; no local credentials or remote writes |

[Pinned HF publication]({hf}), [preserved execution]({rel}/executed.ipynb),
[safe inspection]({rel}/INSPECTION.json), [independent verification]({rel}/VERIFICATION.json),
[remote manifest]({rel}/REMOTE_MANIFEST.json), [local/Kaggle distinction]({rel}/LOCAL_COMPARISON.json).

## What the audit checked

```mermaid
flowchart LR
    A["Executed NB10<br/>4 complete cells"] --> B["36-file payload matches delivery"]
    B --> C["Pinned HF revision<br/>owner released"]
    C --> D["494 objects<br/>size + SHA-256"]
    D --> E["8 CSV families<br/>1,918,052 rows"]
    E --> F["386 native train records<br/>source SHA + finite descriptors"]
    F --> G["18 distinct reference photos<br/>untrained proposal"]
    G --> H["Separate NB11 hypothesis<br/>local engineering passed"]
```

The audit recomputed the canonical contract digest, checked source-code hashes,
cursor/state hash, probability normalization/argmax/correctness, seven probes per
image, role/source consistency, native descriptor finiteness, candidate array
shape/finite values and the complete386-case/18-reference report payload.
Hash verification is exhaustive for the publication. This audit does **not**
claim an independent rerun of every T4 forward, every feature statistic or every
logit reconstruction; the earlier local full gate reconstructed8,106 logits.
The newly downloaded Kaggle viewer's controls are separately checked in handoff.

| Scientific family | Independently recounted rows |
|---|---:|
{rows}

No fitting occurred. `epoch_0000…0008` partition names are reusable-store image
blocks here, not training epochs. Nonfatal setup dependency warnings did not
stop execution; no Torch/CUDA/package repair is needed for completed NB10.

## Scientific reading: train probes are not held-out success

| Cohort | Probe | Images | Accuracy | Observed-class BA | NLL |
|---|---|---:|---:|---:|---:|
{probe}

These are the frozen v1 head's **training-only** probes. The earlier preserved
selected/final development endpoint still has BA50%,62low→mid/41high→high.
Strong training-probe scores do not repair that transfer failure. Training low
support remains one physical tyre; neither seven views nor repeated draws adds
independent subjects. The18-photo bank passes structural rules, not a human
artifact-free or clinically meaningful texture assessment.

## Fixed LC representation probe

Relative L2 is summarized only over the explicitly **unclipped patch subset**.
The low brightening subset is small; do not extend it to every patch. These are
fixed-feature stability measurements, not a trained LC classifier's scores.

| Probe | Proxy class | Unclipped / all patches | RGB median relative L2 | LC median relative L2 |
|---|---|---:|---:|---:|
{feat}

![Actual Kaggle proposed reference cases]({rel}/remote/report/candidate_bank.png)

[Interactive386-image Kaggle gate explorer]({rel}/remote/report/Training_Gate_Explorer.html).
The local and Kaggle candidate source-SHA/patch order match: **{same_bank}**.
Numeric environments differ; no cross-environment byte-identity claim is made.
See the retained report CSVs for exact environment-specific measurements.

## Decision and next action

The gate justifies testing a separately versioned representation/reference
policy. Simply excluding all18photos would reduce queries378→368 and leave two
high groups with one original query photo. r2 instead keeps all386 train images,
uses full-source SHA self exclusion with6/n support normalization, sqrt-size
group sampling and paired appearance losses. This combines motivated changes;
it is not an isolated causal ablation or a promised accuracy improvement.

| Work item | Current state |
|---|---|
| NB08 measured fit / NB09 review / NB10 gate | Completed and independently HF-verified |
| ATLAS-LC r2 model,data,loss,telemetry,recovery | Implemented in a separate source/namespace |
| Local actual batch8 recovery | Four updates; uninterrupted/pause/abrupt/native restore equal across24CSV families |
| Packaged notebook |31embedded files; actual extracted worker matches reference; five unexecuted code cells |
| Local output projection |15,015,649,644bytes,passes16GB guard; Kaggle must remeasure |
| Actual r2 T4×2/live-HF preflight and60epoch fit | Pending user Kaggle execution |
| Workstations | Preserved; no candidate installed |

Run [NB11](notebooks/phase3_NB11_TREAD_TWO_ATLAS_LC_Resumable_Pilot.ipynb) with
T4×2,Internet,existing expanded Kaggle input and HF_TOKEN. Details:
[architecture and illustrated motivation](phase3TreadTwoLCArchitecture.md),
[protocol/recovery/storage](phase3TreadTwoLCProtocol.md),
[run guide](tread_two/lc_pilot/README.md), [24-family/285-column schema](tread_two/lc_pilot/SCHEMA.md).
'''
(ROOT/'phase3NB10Completion.md').write_text(text,encoding='utf-8')
marker='<!-- NB10_LC_UPDATE_20261030 -->'
def front(name,body):
    p=ROOT/name;s=p.read_text(encoding='utf-8');heading,sep,rest=s.partition('\n')
    if marker in rest:rest=rest.split(marker,1)[0]+rest.split('<!-- END_NB10_LC_UPDATE -->',1)[1]
    p.write_text(heading+'\n\n'+marker+'\n## Latest status — NB10 closed; NB11 ready, 30 October2026\n\n'+body.strip()+'\n\n<!-- END_NB10_LC_UPDATE -->\n'+rest.lstrip('\n'),encoding='utf-8')
common='NB10 is completed and independently HF-verified:495 direct files,1,918,052 scientific rows,386native training records,T4×2,zero fitting/held-out forwards and released ownership. No NB10 rerun is needed. [Completion proof](phase3NB10Completion.md).'
next='Next is [NB11 / ATLAS-LC r2](notebooks/phase3_NB11_TREAD_TWO_ATLAS_LC_Resumable_Pilot.ipynb): one separately versioned seed,paired LC views,18-photo source-SHA exclusions,all386queries and sqrt-size group sampling. Local batch8 recovery/package checks pass; actual Kaggle/live-HF preflight and measured accuracy remain pending. Use T4×2,Internet,the existing expanded Kaggle dataset and HF_TOKEN. [Architecture](phase3TreadTwoLCArchitecture.md), [protocol](phase3TreadTwoLCProtocol.md).'
front('README.md',common+'\n\n'+next+'\n\nExisting tyre data only. Both workstations and completed experiments are preserved.')
front('phase3Progress.md',common+'\n\n| Stage | State |\n|---|---|\n| NB08 / NB09 / NB10 | Complete; independently HF-verified |\n| ATLAS-LC r2 implementation | Done; isolated source/recipe/namespace |\n| Local recovery + package | Passed;24CSV families,285columns;15.02GB projected peak |\n| NB11 actual T4×2/live-HF/fit | Pending Kaggle |\n\n'+next)
front('phase3Understanding.md',common+'\n\nThe gate supports an LC hypothesis, not a generalization claim. Avoid losing scarce queries to a larger disjoint bank: keep all386,block identical full source SHAs per query,normalize remaining class support and soften group repetition. The next model remains TREAD-TWO; dense outputs still come from the frozen TREAD-ONE anchor.\n\n'+next)
front('phase3NextSteps.md',common+'\n\n1. Run NB11 unchanged on Kaggle:T4×2,Internet,expanded prepared input,HF_TOKEN.\n2. Allow its actual live recovery/storage gate before the60epoch measured candidate. Preserve executed notebook and direct CSV/native evidence.\n3. Audit the pinned completed publication,exposure,reference exclusions,loss accounting and selected/final failures before promotion.\n\nNo new photo collection or workstation work. '+next)
front('phase3OpenItems.md','| Item | State |\n|---|---|\n| NB10 supplied execution/publication | Closed; all494objects and1,918,052rows checked |\n| NB10 proposal vs scarce queries | Resolved in r2 source-SHA exclusion/all386query policy |\n| r2 implementation/local recovery/package | Closed;24families/285columns |\n| Actual r2 Kaggle dual-T4/live HF preflight | Pending NB11 |\n| r2 measured classification/generalization | Unknown; no accuracy improvement claimed |\n| Independent low/mid cohort limits | Remain; no acquisition requested |\n| Workstation changes | Deferred by user |\n\n'+next)
front('AGENTS.md',common+'\n\n- Preserve the executed NB10 (`499c84cf…`) and all earlier frozen studies. NB10 is a zero-update training-only gate; its bank proposal is not installed in v1.\n- ATLAS-LC r2 is separately implemented/frozen under `tread_two/lc_pilot/`,new HF namespace `phase3/tread-two/atlas-lc-r2`. Use NB11; do not rewrite active source/assets/contracts.\n- Existing data only; all386training queries remain eligible with mandatory full-source SHA self exclusion and6/n reference scaling. Never substitute label-conditioned references.\n- Log both appearance views and all24CSV families/285columns,complete recovery state and declared gaps. Keep all60epoch weights; direct CSV/native publication,never ZIP.\n- Local fixture/package checks passed. Distinguish those from pending actual T4×2/live-HF/fit and unmeasured accuracy. No workstations changed.\n\n'+next)
front('phase3EvidenceRegister.md','| Evidence | Verified scope |\n|---|---|\n| [NB10 publication audit]('+rel+'/VERIFICATION.json) | All494objects,8CSV families,386native records,contract/source/owner |\n| [NB10 local comparison]('+rel+'/LOCAL_COMPARISON.json) | Actual environments and reference-SHA/patch order |\n| [LC math check](tread_two/lc_validation/MATH_VALIDATION.json) | Source exclusions,normalization,additive logits,paired loss and deterministic sampler |\n| [LC local recovery](tread_two/lc_validation/LOCAL_VALIDATION.json) | Four actual batch8 updates,24CSV families,native state equality,17offline checks,15.02GB projection |\n| [NB11 package check](tread_two/lc_validation/PACKAGE_VALIDATION.json) |31embedded files; extracted worker scientific/native equality |\n\n'+common)
front('phase3DocumentationInventory.md','| Document | Purpose |\n|---|---|\n| [NB10 completion](phase3NB10Completion.md) | Exact pinned HF proof,actual gate findings and status boundaries |\n| [LC architecture](phase3TreadTwoLCArchitecture.md) | Observation→design table,flow diagram,equations,actual feature/sampler illustrations |\n| [LC protocol](phase3TreadTwoLCProtocol.md) | Frozen recipe,selection,recovery state diagram,HF/capacity contracts and planned evidence counts |\n| [LC run guide](tread_two/lc_pilot/README.md) | Kaggle settings,run/resume/troubleshooting and module ownership |\n| [LC schema](tread_two/lc_pilot/SCHEMA.md) |24families/285columns,keys,units,denominators,native formats and explicit unsaved fields |\n| [LC validation](phase3TreadTwoLCValidation.md) | Actual tests,packaged worker,scope and preservation handoff |')
front('phase3TreadTwoAppearanceGate.md',common+'\n\nThe older local results below remain a dated local record. Actual Kaggle results and environment-specific values are in [NB10 completion](phase3NB10Completion.md). The next step is the separate LC model,not repeating this gate or installing its bank into v1.\n\n'+next)
front('phase3TreadTwoFindings.md',common+'\n\nThe preserved v1 selected/final endpoint remains BA50%,62low→mid/41high→high. A positive train-only gate does not change that result. '+next)
front('phase3SessionLog.md','Completed this user-requested follow-up: archived/inspected executed NB10; verified all public pinned HF objects,raw CSV counts,native training roles/hashes and released owner; froze NB10 references for a new model; implemented r2 LC/source-exclusion/paired-loss/sampler/telemetry/recovery; passed local actual batch8/fault/restore/package checks; wrote structured diagrams/tables/schema/figures. No remote writes from local work,no new photos,no workstation changes.\n\n'+next)
p=ROOT/'phase3Decisions.md';s=p.read_text(encoding='utf-8')
if 'P3-D084' not in s:s+='''

## NB10 closure and ATLAS-LC r2 — 30 October2026

| ID | Decision | Evidence / limit |
|---|---|---|
| P3-D084 | Close NB10 after exhaustive pinned-object verification; do not rerun |495direct files,1,918,052rows,386native training records,owner released; zero fitting |
| P3-D085 | Implement a separately frozen ATLAS-LC r2 candidate | v1 appearance sensitivity and NB10 LC/reference proposal motivate a hypothesis,not a measured improvement |
| P3-D086 | Keep all386queries and block identical full source-SHA references with6/n normalization | Avoid two singleton groups produced by disjoint18-photo exclusion; no query target enters support mask |
| P3-D087 | Use sqrt-size group sampling and two-view CE/JS/texture consistency | Reduce extreme expected repetition and constrain appearance response; replacement/cohort limits remain |
| P3-D088 | Retain24CSV families/285columns,both views,permissions,pair losses and full native recovery | User's extensive direct-CSV retention priority; keep explicit unmeasured fields |
| P3-D089 | Deliver one-seed NB11 after local batch8/recovery/package and capacity checks | Actual T4×2/live-HF preflight remains automatic before measured fit; no workstation promotion or new photos |
'''
p.write_text(s,encoding='utf-8')
print('Updated11entry/tracker notes,decisions and NB10 completion; actual next model NB11 ready')
