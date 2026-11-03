"""Record completed NB15 and the consolidated, unpromoted research handoff."""
import json,re,sys
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1]
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
done=Path(read(ROOT/'tread_two/nb15_completion/LATEST.json')['path']);v=read(done/'VERIFICATION.json');validation=read(ROOT/'tread_two/research_release_validation/VALIDATION.json')
assert v['status']==validation['status']=='passed'
all_exact=all(x['byte_identical'] for x in v['table_comparisons']);maximum=max(x['maximum_numeric_difference'] for x in v['table_comparisons'])
def write(name,text):(ROOT/name).write_text(text.strip()+'\n',encoding='utf-8')
def insert(name,text):
    path=ROOT/name;old=path.read_text(encoding='utf-8');marker='NB15_CLOSED_20261101'
    old=re.sub(r'\n<!-- '+marker+r' -->.*?<!-- END_'+marker+r' -->\n','\n',old,flags=re.S)
    title,rest=old.split('\n',1);rest=re.sub(r'(?m)^## Current — ',r'## Historical — ',rest)
    path.write_text(title+'\n\n<!-- '+marker+' -->\n'+text.strip()+'\n<!-- END_'+marker+' -->\n'+rest,encoding='utf-8')
write('phase3NB15Completion.md',f'''
# NB15 complete — verified existing-data readiness and evidence passport

**1 November 2026. The supplied executed NB15 is complete and independently HF-verified. No rerun is needed.** All four code cells completed without saved errors; all 40 source/metadata payload files match the accepted delivery. The expected scientific readiness result is **false**, while execution and evidence integrity **pass**.

## Exact publication and audit

| Item | Verified result |
|---|---|
| Executed notebook SHA-256 | `98837b907abc332fd2eaf745b578656f1c70df594ede28e13c02ca678b43aa07` |
| Pinned HF revision | `f4b44d3d8e309db28d42171f1c5210881aa7b079` |
| Gate contract | `1bbb0142fc67619b74b3fda9f8ad8924d836f1ba6eaf67c53ca58fa02fe58f49` |
| Direct publication | **38 logical files**, including manifest; all **37 objects** checked by byte count and SHA-256 |
| Verified payload | **58,797,180 bytes**; no ZIP/NPZ/database publication |
| Scientific tables | **16 CSVs / 196,905 rows**, including 188,160 algebraic correction records |
| Independent reconstruction | All 16 schemas/counts/text/missing values/numbers compared; all byte-identical: **{all_exact}**; maximum numeric difference **{maximum:g}** |
| Visual artifacts | Three valid PNGs and interactive passport with 1,440 component/epoch/variant summaries |
| Capability record | Exact SUMMARY.json and CAPABILITIES.json agreement with independent local analysis |
| Previous publications | All **1,502** context-fit entries and **79** context-review entries unchanged at the new revision |
| Ownership | Released, `owner=null`, `expires=0` |
| Actual environment | Python 3.13.15 / NumPy 2.1.3 / pandas 2.3.3; private HF SDK publication completed |
| Analysis scope | Zero model forwards, zero optimizer updates, zero new independent tyres, zero held-out prediction rows used |
| Verifier scope | Public reads only; no local credential files read or remote writes |

[Independent verification](tread_two/nb15_completion/98837b907abc/VERIFICATION.json) · [executed inspection](tread_two/nb15_completion/98837b907abc/INSPECTION.json) · [published manifest](tread_two/nb15_completion/98837b907abc/REMOTE_MANIFEST.json) · [HF gate at the verified revision](https://huggingface.co/datasets/Shanmuk4622/tyre-wear-study/tree/f4b44d3d8e309db28d42171f1c5210881aa7b079/phase3/tread-two/readiness-gate/v1/1bbb0142fc67619b74b3fda9f8ad8924d836f1ba6eaf67c53ca58fa02fe58f49).

The private SDK installation ran successfully from the delivered setup. Kaggle used NumPy 2.1.3 / pandas 2.3.3; local reconstruction used NumPy 2.4.6 / pandas 3.0.3. The largest retained numeric difference is approximately 1e-12, while table schemas, counts, text and missingness agree. All published object bytes match their own pinned SHA-256 records. The actual published passport JavaScript passes epoch/class/variant/weighted-curve checks in a synthetic DOM. Figure files decode correctly. These are functional/artifact checks; no native browser screenshot claim is made. The original local held-out perturbation and training-SHA rejection tests remain preserved in [package acceptance](tread_two/readiness_validation/PACKAGE_VALIDATION.json).

```mermaid
flowchart LR
    A[NB15 user execution\n4 completed cells] --> B[Pinned public HF audit\n37 objects / 16 CSVs]
    B --> C[Integrity PASS\nreadiness FALSE]
    C --> D[Close unchanged experiments\nclassifier remains research-only]
    D --> E[Consolidated research handoff\n3 model cards / 9 publications]
    E --> F[Future application work\nworkstation changes still deferred]
```

## What false readiness means

| Training proxy class | Existing images | Known original tyres | Conservative components |
|---|---:|---:|---:|
| Low | 67 | 1 | 1 |
| Mid | 249, including 152 existing video frames | 3 | 1 |
| High | 70 | 4 | 4 |

Withholding the only low component leaves zero low fit images. Withholding the conservative mid component leaves zero mid fit images. These are two of six conditional holdouts. No fold model was trained, and the available grouping does not support a complete class-supported leave-component-out comparison. The mid co-assignment preserves unresolved clip-parent constraints; it does **not** assert that three known mid tyres are physically identical.

This is a support limitation, not a notebook failure or proof that every alternative model must fail. Existing photos were sufficient for this completed audit. More views, repetitions or CSV rows do not increase independent tyre support, and an identical rerun cannot change that fact.

![Independent support and conditional fold counts](tread_two/readiness_report/figures/independent_support.png)

The frozen LC prior already predicts all **47,040 sampled training views** correctly. Full context rescues **0** prior errors and transiently spoils **44** prior-correct view observations early in training. On the same final-epoch 784 views / 214 distinct images, prior NLL is 0.636877 versus context 0.031365, with confidence rising from 0.530181 to 0.969124. The spatial term supplies almost all that loss/confidence change; removing the global bias barely changes it. These are algebraic parts of saved changing online states, not independently trained ablations or a fixed final-model cohort test.

![Algebraic correction utility on the same saved training views](tread_two/readiness_report/figures/correction_parts.png)

The historical low failure remains: all 62 low images become mid and all 41 high images remain high. Original validation-selected epochs are preserved. The held-out endpoint has already informed development and lacks mid targets. Labels remain mileage proxies; physical wear, depth and safety targets are unavailable.

## Completed next step: consolidated research handoff

Built and checked the [interactive evidence hub](tread_two/research_release/Evidence_Hub.html), [handoff guide](tread_two/research_release/README.md) and [technical closure](phase3TreadTwoResearchClosure.md). The handoff contains three model cards, seven metadata CSVs / 134 rows, six selected/final metric records, nine verified publications, fourteen pinned checkpoint/explorer/figure descriptors and a comparison figure. It references all original HF science without loading, altering or republishing native weights.

The handoff is a **local report/index**; its indexed scientific evidence is already on HF, and the new index itself has not been uploaded. [Release manifest](tread_two/research_release/RELEASE.json), [validation](tread_two/research_release_validation/VALIDATION.json).

**No NB16 or repeat training/publication notebook is needed to close this unchanged experimental series.** ATLAS v1/LC/context remain completed research candidates, not a deployable classification claim. Application changes remain deferred under the user's earlier instruction, and both workstations are preserved. A later existing-data experiment needs a distinct supported objective and protocol; no new-photo prerequisite is introduced here.
''')
write('phase3TreadTwoResearchClosure.md',r'''
# TREAD-TWO current experiments — research closure and handoff

**1 November 2026. Three candidate fits, their reviews and the existing-data readiness gate are complete and HF-verified. Classification remains research-only/unpromoted.** This closes the current v1 / LC r2 / context r3 experimental series. It does not declare the broader tyre application finished or establish useful classification on unseen tyres.

## Result and decision

| Candidate | Original selected epoch | Selected / final BA | Low / high recall | Selected NLL | Final NLL |
|---|---:|---:|---|---:|---:|
| ATLAS v1 | 18 | 50% / 50% | 0/62; 41/41 | 1.178296 | 1.428298 |
| ATLAS-LC r2 | 7 | 50% / 50% | 0/62; 41/41 | 0.961273 | 1.369872 |
| ATLAS-CONTEXT r3 | 7 | 50% / 50% | 0/62; 41/41 | 2.174813 | 2.358475 |

All comparisons describe the same already-inspected 103-image low/high historical endpoint. No mid target, independent confirmation cohort, causal single-factor comparison or image-independent confidence interval is invented. Each fit completed one seed / 60 epochs / 2,940 updates; NB13A recovered the existing context experiment. Different architectures are not seed replicates.

![Selected/final recognition, probability loss and calibration](tread_two/research_release/figures/candidate_comparison.png)

LC addresses an observed appearance-stability problem and improves selected probability loss. Context adds learned layout sensitivity and correctly conserved internal evidence. Neither repairs low recognition. Context mainly sharpens training confidence and worsens historical probability loss. The choices followed earlier observations, but the measured results do not establish the hypotheses as successful classifiers.

NB15 independently verifies the class-support limit: known original training tyres low/mid/high = 1/3/4; conservative components = 1/1/4. Two conditional holdouts remove a fit class. The three known mid tyres remain physically distinct; their unresolved clip-parent co-assignment is a leakage safeguard. [NB15 completion and proof](phase3NB15Completion.md), [methods](phase3TreadTwoReadiness.md).

```mermaid
flowchart TD
    V[ATLAS v1\nreference evidence] --> L[LC r2\npaired local contrast]
    L --> C[Context r3\nposition-aware bounded residual]
    V --> R[Completed evidence reviews]
    L --> R
    C --> R
    R --> G[Existing-data readiness gate\nclass support insufficient for full group-held-out comparison]
    G --> H[Research handoff\noriginal checkpoints / evidence / capabilities]
    H --> P[Classifier remains unpromoted]
    H --> W[Application work later\nworkstations remain deferred]
```

## Handoff contents and code ownership

| Deliverable | Purpose |
|---|---|
| [Evidence hub](tread_two/research_release/Evidence_Hub.html) | Offline model/type filters, selected/final metrics, capability statements and exact evidence links |
| [v1 card](tread_two/research_release/MODEL_CARD_V1.md), [LC card](tread_two/research_release/MODEL_CARD_LC.md), [context card](tread_two/research_release/MODEL_CARD_CONTEXT.md) | Architecture/results/source/selection/limit records; selected/final native SHA and direct URLs |
| [models.csv](tread_two/research_release/models.csv) | Three model identities, original epochs, unchanged inventory hashes and native pointers |
| [endpoint_metrics.csv](tread_two/research_release/endpoint_metrics.csv) | Six saved endpoint rows; BA, recalls, NLL, Brier, ECE and explicit scope |
| [publications.csv](tread_two/research_release/publications.csv) | Nine pinned publications: three fits, three reviews, three gates; counts, audit scope and receipts |
| [artifacts.csv](tread_two/research_release/artifacts.csv) | Fourteen exact SHA/size/HF object descriptors, including six selected/final native checkpoints |
| [capabilities.csv](tread_two/research_release/capabilities.csv) | Research classification, retained frozen geometry, internal explanations and unavailable depth/safety targets |
| [provenance.csv](tread_two/research_release/provenance.csv), [dictionary](tread_two/research_release/table_dictionary.csv) | Consumed source hashes, field definitions, units and missingness |
| [RELEASE.json](tread_two/research_release/RELEASE.json) | Every local source/output hash and the explicit local-only handoff status |
| [Generator](analysis/build_tread_two_research_release.py) | Builds only metadata, cards, figure and research HTML from completed receipts/manifests/CSV |
| [Acceptance](analysis/validate_research_release.py) | Checks source/artifact hashes, manifest/native pointers, row scopes, cards, figure and actual viewer JS |

Seven CSV tables contain 134 metadata rows. These are not new training observations or independent tyres. Native weights are referenced at immutable revisions rather than copied into a new archive. NB08's raw-row count remains a saved cursor declaration with bounded recount scope; later exhaustive audits and derived-table checks retain their separate scope. NB07 structured counts exclude additional raw/control inputs. There is no combined row total presented as new data.

## Actual checks

[Validation receipt](tread_two/research_release_validation/VALIDATION.json): all source/output hashes match; three original selected epochs are 18/7/7; fourteen artifact descriptors match the original manifests by size/SHA/object URL; nine publication records retain released owners and passed receipts. The actual HTML JavaScript filters for three models and three record kinds, model-card links and capability/artifact display pass a synthetic DOM test. The scientific comparison PNG was visually inspected. Native browser rendering is not claimed.

The independent NB15 audit separately checks all 37 published payloads, all 16 CSV schemas/counts/values, three PNGs, actual passport controls, the capability record, and unchanged context fit/review manifests. No model forward, optimizer update, local credential lookup, HF write or application mutation occurs during closure.

## Current next step

The consolidation step is implemented and ready for review. No further notebook is needed for these unchanged completed studies. Use the handoff as the evidence contract when workstation/application work is resumed; those changes remain deferred. Keep the selected models and both workstations unchanged. Existing photos only; the audit and closure do not require acquisition.

A later model experiment can explore a distinct supported target, representation or protocol, but this closure does not automatically start a fourth full classifier fit, choose a post-hoc blend/threshold, or relabel mileage as physical depth/safety. A false support flag cannot be fixed by an identical rerun. The current conclusion is a completed research series with unresolved low-class transfer.

All indexed scientific evidence is already directly retained on HF. The new local research index itself has not been published to HF; its compact sources/cards/CSV/figure are kept Git-visible. [Handoff guide](tread_two/research_release/README.md).
''')
common='''## Current — NB15 verified; current ATLAS experiments closed as research

**1 November 2026:** executed NB15 is complete and HF-verified at `f4b44d3d…`: **38 direct files / 16 CSVs / 196,905 rows**, three figures and the interactive capability passport. All published bytes, table values, schemas and capability records pass independent checks; owner released. Earlier context fit/review manifests remain unchanged. **No NB13/NB13A/NB14/NB15 rerun is needed.** [Completion and proof](phase3NB15Completion.md).

Scientific readiness remains false as expected: training conservative components low/mid/high = 1/1/4, and two conditional holdouts remove a fit class. All three completed candidate fits remain research-only/unpromoted; historical low recall is 0/62. This is a support/result finding, not an execution error or a claim that every alternative model must fail.

**Next consolidation step is implemented:** [interactive research evidence hub](tread_two/research_release/Evidence_Hub.html), three model cards, original selected/final native pointers, nine pinned publications and seven metadata CSVs. [Research closure and technical handoff](phase3TreadTwoResearchClosure.md). The index is local; its underlying scientific records are already on HF. No NB16 or fourth full classifier fit is needed to close these unchanged studies.

The broader application is not declared complete. Workstation changes remain deferred, both workstations and frozen experiments are preserved, and only existing photos are used. A later experiment needs a distinct supported objective/protocol. Earlier pending-run instructions below are dated history.'''
for name in ['README.md','AGENTS.md','phase3Progress.md','phase3NextSteps.md','phase3OpenItems.md','phase3Understanding.md','phase3DataRetentionPolicy.md','phase3ModelRationaleAndLogging.md','phase3TreadTwoFindings.md']:
    insert(name,common)
path=ROOT/'phase3NextSteps.md';text=path.read_text(encoding='utf-8');path.write_text('# Next steps — completed research handoff; application changes deferred\n'+text.split('\n',1)[1],encoding='utf-8')
insert('phase3TreadTwoReadiness.md','''## Superseding measured completion — 1 November 2026

NB15 is now complete and independently HF-verified: all 38 direct files / 16 CSVs / 196,905 rows, three figures and actual passport controls pass; writer released. [Completion](phase3NB15Completion.md). Readiness false is the expected support finding. Do not rerun NB15 to change it. The next consolidation step is now [implemented as the research handoff](phase3TreadTwoResearchClosure.md); no further notebook is needed for the unchanged studies. Earlier pending-Kaggle wording below records the original delivery.''')
insert('phase3NB14Completion.md','''## Later closure — 1 November 2026

The proposed NB15 has now completed and is independently HF-verified. [NB15 completion](phase3NB15Completion.md). No NB14/NB15 rerun is needed. The current v1/LC/context experimental series is consolidated in the [research handoff](phase3TreadTwoResearchClosure.md); classifier transfer remains unresolved. Earlier next-NB15 wording below is historical.''')
insert('tread_two/readiness_gate/README.md','''## Completed and verified — 1 November 2026

NB15's four cells completed. Independent HF checks pass for all 38 direct files / 16 CSVs / 196,905 rows, three figures and the passport; ownership is released. Expected support readiness remains false. **No rerun is needed.** [Completion](../../phase3NB15Completion.md) · [Consolidated research handoff](../../phase3TreadTwoResearchClosure.md). The original instructions below are historical run/recovery documentation; preserve the executed notebook and four frozen gate modules.''')
insert('tread_two/README.md',common.replace('](phase3','](../phase3').replace('](tread_two/research_release/','](research_release/'))
insert('phase3EvidenceRegister.md','''## NB15 closure and research handoff — 1 November 2026

| Evidence | Scope |
|---|---|
| [NB15 verification](tread_two/nb15_completion/98837b907abc/VERIFICATION.json) | Every published payload hash/size, all 16 tables, capability/summary, figures, actual passport JS, unchanged earlier manifests |
| [Executed inspection](tread_two/nb15_completion/98837b907abc/INSPECTION.json) | Four completed cells, no errors, forty matching payloads and actual private-SDK/publication output |
| [Release manifest](tread_two/research_release/RELEASE.json) | Local handoff only; source/output hashes and three-model/nine-publication scope |
| [Release acceptance](tread_two/research_release_validation/VALIDATION.json) | Model cards, original epochs/native object descriptors, exact receipt links, row-scope distinctions and viewer controls |

[NB15 completion](phase3NB15Completion.md) · [Research closure](phase3TreadTwoResearchClosure.md). No new fit, prediction, independent tyre, HF write or workstation change.''')
insert('phase3DocumentationInventory.md','''## Completed NB15 / current research closure — 1 November 2026

| Document | Purpose |
|---|---|
| [NB15 completion](phase3NB15Completion.md) | Pinned public verification, actual execution, expected false readiness, figures and current next state |
| [Research closure](phase3TreadTwoResearchClosure.md) | Cross-candidate results, experiment decision, diagram, module/data ownership and limits |
| [Handoff README](tread_two/research_release/README.md) | How to inspect/rebuild the local metadata index and preserve audit scope |
| [v1 card](tread_two/research_release/MODEL_CARD_V1.md), [LC card](tread_two/research_release/MODEL_CARD_LC.md), [context card](tread_two/research_release/MODEL_CARD_CONTEXT.md) | Original selection, metrics, source and exact native checkpoint references |

No further notebook is needed to close unchanged studies. Application work remains deferred; all underlying scientific results are already directly retained on HF.''')
insert('phase3RepositoryMap.md','''## NB15 completion / local research handoff paths — 1 November 2026

| Path | Retention / ownership |
|---|---|
| `tread_two/nb15_completion/98837b907abc/` | Small inspection/manifest/verification receipts; ignored executed archive and full HF mirror |
| `tread_two/research_release/` | Three model cards, seven metadata CSVs / 134 rows, one curated PNG, interactive HTML, capability and hashed release manifest |
| `tread_two/research_release_validation/` | Small functional/numeric/provenance acceptance and final preservation receipts |
| `analysis/build_tread_two_research_release.py` | Metadata/report generator; no fitting, inference, credential lookup or publication |
| `analysis/verify_nb15_publication.py` | Pinned public NB15 audit; all37objects and16tables checked |

The local research browser is an evidence report, not a workstation implementation. Both application trees and all completed runtime/notebook identities remain preserved.''')
insert('phase3Decisions.md','''## NB15 completion and current experimental-series closure — 1 November 2026

| ID | Decision | Evidence / boundary |
|---|---|---|
| P3-D122 | Close NB15 without rerun | Four complete cells; all 38 files / 16 CSVs verified, released owner and unchanged upstream manifests |
| P3-D123 | Keep false support readiness separate from failed execution | Two conditional folds remove a fit class; analysis integrity passes; mid co-assignment is not identity equivalence |
| P3-D124 | Close v1/LC/context as completed, unpromoted research candidates | All selected/final low recalls 0/62; probability changes do not establish transfer |
| P3-D125 | Consolidate original selected/final checkpoint and evidence identities | Three cards, nine pinned publications, fourteen artifact descriptors and local provenance release |
| P3-D126 | Avoid an unnecessary repeat/fourth full fit or publication-only NB16 for unchanged studies | Scientific records already directly retained on HF; local handoff is not claimed uploaded |
| P3-D127 | Preserve deferred workstation scope and existing-photo constraint | No application tree changes or acquisition prerequisite; later experiments need a distinct supported protocol |

[Completion](phase3NB15Completion.md) · [Research closure and implemented handoff](phase3TreadTwoResearchClosure.md).''')
insert('phase3SessionLog.md','''## NB15 verification and consolidated research handoff — 1 November 2026

User reported NB15 complete and requested HF verification, Markdown updates and continuation. Preserved executed SHA `98837b907abc…`, all 40 matching source/metadata payloads and four completed code cells with no saved errors. The actual private HF SDK setup and publication completed successfully.

Independent public audit at `f4b44d3d…` verified all 37 selected objects / 38 logical direct files, 58,797,180 bytes, all 16 CSVs / 196,905 rows, three PNG figures, actual interactive passport controls, SUMMARY.json and CAPABILITIES.json. All 1,502 original context-fit manifest entries and 79 context-review entries are unchanged at this revision; owner released. No credential reads or remote writes. Cross-environment numeric/text/schema/missingness comparisons are recorded in the receipt.

Scientific readiness remains false as expected: low/mid/high conservative components 1/1/4 and two conditional holdouts missing a fit class. Original known tyres remain 1/3/4; unresolved mid co-assignment is a leakage constraint, not physical identity equivalence. The training-only correction utility finding is unchanged: prior already correct on all 47,040 sampled views; context rescues 0 prior errors, transiently spoils 44 and mainly increases confidence through spatial correction. All three candidates' historical low recall remains 0/62. No classification promotion or new physical wear/depth/safety claim.

Implemented the next consolidation step as a local research release: three model cards, seven metadata CSVs / 134 rows, six original endpoint records, nine pinned verified publications, fourteen checkpoint/explorer/figure descriptors, comparison PNG and interactive evidence hub. Every consumed receipt/manifest/metric CSV hash and each generated artifact hash is retained. Original validation-selected epochs 18/7/7 are preserved; no model weights loaded or republished. NB08 cursor-declared row counts remain distinguished from later exhaustive raw-row and derived-table audits.

Local acceptance checks all sources/artifacts/tables, native object SHA/size/URL descriptors, original selected epochs, model-card links, three model filters and fit/review/gate controls against actual HTML JavaScript. Comparison figure visually inspected; native browser rendering is not claimed. No further notebook is needed to close unchanged current studies. Underlying science is already on HF; the new index itself is local and not claimed uploaded. Added a narrow Git exception for its curated figure. Both workstations, all frozen fits/gates/reviews and existing source photos remain preserved. Application work remains deferred; no acquisition prerequisite or automatic fourth full classifier fit introduced.''')
print(json.dumps(dict(status='written',NB15='verified_completed',tables=16,rows=196905,all_tables_byte_identical=all_exact,maximum_numeric_difference=maximum,next_step='local_research_handoff_implemented'),indent=2))
