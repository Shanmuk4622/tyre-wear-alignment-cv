"""Close verified NB14 and document measured existing-data support and NB15."""
import json,re,sys
from pathlib import Path
import pandas as pd
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];done=Path(json.loads((ROOT/'tread_two/nb14_completion/LATEST.json').read_text())['path']);v=json.loads((done/'VERIFICATION.json').read_text());assert v['status']=='passed'
p=json.loads((ROOT/'tread_two/readiness_validation/PACKAGE_VALIDATION.json').read_text());assert p['status']=='passed'
summary=json.loads((ROOT/'tread_two/readiness_report/SUMMARY.json').read_text());ep=pd.read_csv(ROOT/'tread_two/readiness_report/online_epoch_variants.csv')
prior=ep[ep.variant.eq('frozen_prior')];full=ep[ep.variant.eq('full_context')]
assert prior.online_accuracy.eq(1).all() and prior.view_observations.sum()==47040
def write(name,text):(ROOT/name).write_text(text.strip()+'\n',encoding='utf-8')
def insert(name,body):
    path=ROOT/name;text=path.read_text(encoding='utf-8');marker='NB14_COMPLETED_20261101'
    text=re.sub(r'\n<!-- '+marker+r' -->.*?<!-- END_'+marker+r' -->\n','\n',text,flags=re.S)
    first,rest=text.split('\n',1);rest=re.sub(r'(?m)^## Current — ',r'## Historical — ',rest)
    path.write_text(first+'\n\n<!-- '+marker+' -->\n'+body.strip()+'\n<!-- END_'+marker+' -->\n'+rest,encoding='utf-8')
write('phase3NB14Completion.md',r'''
# NB14 complete — ATLAS-CONTEXT saved-results review

**1 November 2026. Executed NB14 is complete and independently HF-verified. No rerun is needed.** All four code cells completed without saved errors; all 44 embedded source/metadata payloads match the delivered package. The original NB13 fit remains unchanged.

## Exact publication and verification

| Item | Independently verified result |
|---|---|
| Executed notebook SHA-256 | `356dc7893cc5bfe67368fae758962cd6759afe6f9aa5eecc9d2b5be7d3679c63` |
| HF repository revision | `10c2f16ddf669ca6454cd27ebb306d699dba3d92` |
| Review contract | `793525de3180a2f9025333c4afec219162aaca9a819474c1e60f208bf3e9a5bc` |
| Direct publication | **80 files**, including manifest; all 79 objects verified by size and SHA-256 |
| Verified payload | **217,059,186 bytes**; zero ZIP/NPZ/database artifacts |
| Scientific review | **52 CSVs / 585,825 rows**; all tables byte-identical to independent local reconstruction |
| Numeric comparison | Maximum difference **0** at saved precision; nullable/text values and schemas also checked |
| Visual evidence | Five PNG figures, **1,854** endpoint/probe cases, 18 literal references |
| Attention | 206 full identity endpoint matrices, each 2 layers × 4 heads × 32×32; raw matrices/interactive controls verified |
| Preserved original fit | All **1,502** original manifest entries identical at the review revision |
| Owner | Released; no active writer |
| New work performed by NB14 | Zero model forwards, zero optimizer updates, zero new independent tyres |
| Independent verification | Public reads only; no local credential files read, no remote writes |
| Actual review environment | Python 3.13.15 / NumPy 2.1.3 / pandas 2.3.3 |

[Verification receipt](tread_two/nb14_completion/356dc7893cc5/VERIFICATION.json) · [executed inspection](tread_two/nb14_completion/356dc7893cc5/INSPECTION.json) · [published manifest](tread_two/nb14_completion/356dc7893cc5/REMOTE_MANIFEST.json) · [HF review at the verified revision](https://huggingface.co/datasets/Shanmuk4622/tyre-wear-study/tree/10c2f16ddf669ca6454cd27ebb306d699dba3d92/phase3/tread-two/atlas-context-r3/review/793525de3180a2f9025333c4afec219162aaca9a819474c1e60f208bf3e9a5bc).

Local reconstruction used Python 3.11.15 / NumPy 2.4.6 / pandas 3.0.3. Exact table agreement across these environments was measured, not assumed. Figure files decode correctly. Explorer controls were checked against the actual published JavaScript using a synthetic DOM; this is functional testing, not a claim of native browser screenshot verification.

```mermaid
flowchart LR
    A[NB13A completed original fit\n60 epochs / 2940 updates] --> B[NB14 CPU review\n52 derived CSV tables]
    B --> C[HF verified\nall80files / unchanged fit]
    C --> D[Close NB14\nclassifier stays unpromoted]
    D --> E[NB15 existing-data support\nconditional folds / correction parts]
    E --> F[Direct CSV + evidence passport\nno fourth full fit]
```

## Scientific conclusion

The review confirms the prior findings; it does not provide a new accuracy experiment. The spatial branch responds to layout but does not recover low recognition. The frozen LC prior is unchanged and all signed reference/context/combined evidence reconstructs saved predictions. Correct accounting and attention visualization do not establish physical explanation or unseen-tyre transfer.

| Historical endpoint metric | LC selected prior | Context selected | Context final |
|---|---:|---:|---:|
| Observed-class balanced accuracy | 50.00% | 50.00% | 50.00% |
| Low recall | 0/62 | 0/62 | 0/62 |
| High recall | 41/41 | 41/41 | 41/41 |
| NLL | 0.961273 | 2.174813 | 2.358475 |
| Brier | 0.568355 | 1.105581 | 1.129997 |
| ECE, ten equal-width bins | 0.425969 | 0.552019 | 0.561834 |

The context candidate stays research-only/unpromoted. The original validation rule still selects context epoch 7; no alternate checkpoint or blend is chosen from test. The already-inspected historical endpoint has no mid targets and is development evidence. Mileage proxies are not depth, physical wear or safety labels.

![Saved correction, attention and probability loss](tread_two/context_analysis_report/figures/context_and_attention.png)

All 16 saved endpoint/probe groups have zero argmax flips, while scores move. Final reverse-column probability change reaches 0.087466. This demonstrates learned layout sensitivity; it does not establish that useful wear geometry was learned. Full attention exists only for the 206 identity endpoints. Training/probe matrices remain unavailable; they are not invented from summaries.

The exhaustive **58,176,066 raw training-row / 60-native-epoch audit** belongs to [NB13 completion](phase3NB13Completion.md). NB14 reconstructs its consumed saved-evidence subset and adds the 52 derived tables; it must not be described as rerunning the original fit or independently recalculating every training affinity.

## Execution note and next step

NB14 successfully used HF SDK 0.36.2. Its setup output also contained dependency-conflict warnings for unrelated preinstalled libraries; all four cells nevertheless completed. Preserve this executed notebook/source. New NB15 installs the same tested SDK in a private publication directory and exposes it only to the publishing subprocess, avoiding replacement of Kaggle's global HF SDK. No Torch/CUDA installation changes are introduced; this does not claim to repair unrelated preinstalled-library requirements.

Implemented next: [NB15 existing-data readiness and evidence passport](notebooks/phase3_NB15_TREAD_TWO_Existing_Data_Readiness_And_Evidence_Passport.ipynb). It uses training rows only, quantifies conservative component support and algebraic correction utility, and publishes a structured capability record. [Findings/methods](phase3TreadTwoReadiness.md) · [Kaggle guide](tread_two/readiness_gate/README.md).

Use **CPU / None, Internet ON, HF_TOKEN enabled, no image dataset, 2 GiB free and about 4 GB RAM; Run All**. Local acceptance passes; actual NB15 Kaggle/HF completion is pending. Both workstations remain unchanged, and this step requires no additional photos.
''')
write('phase3TreadTwoReadiness.md',f'''
# TREAD-TWO — existing-data readiness and evidence passport

**1 November 2026. Implemented and executed locally; actual NB15 Kaggle/HF publication pending.** Analysis integrity passes; scientific readiness for a complete three-class component-held-out comparison is **false**. This is a measured support limit, not a runtime error or proof that all possible models must fail.

## Why this is the next step

The [completed NB14](phase3NB14Completion.md) confirms that added positional capacity amplified training confidence without improving the historical low-class failure. Another full seed/architecture run has no supporting transfer evidence. This gate uses existing pinned registries and training observations to identify what can be checked honestly, without new photographs, checkpoint reselection or workstation changes.

```mermaid
flowchart TD
    I[NB07 source hashes + conservative grouping] --> G[386 historical training images]
    R[NB14 saved training predictions + correction parts] --> O[47040 paired-view observations]
    G --> F[Six conditional component holdouts\ncount class / bank support only]
    O --> A[Prior / bias / spatial / full logits\nno fitting or model forward]
    F --> P[Readiness false\nlow/mid withholding removes fit class]
    A --> C[Correction utility CSVs + figures]
    P --> E[Evidence passport / capability JSON]
    C --> E
    E --> H[NB15 additive HF publication\n16 direct CSV tables]
```

## Existing training support

| Proxy class | Images | Original photos | Existing video frames | Known original tyres | Conservative components |
|---|---:|---:|---:|---:|---:|
| Low | 67 | 67 | 0 | 1 | 1 |
| Mid | 249 | 97 | 152 | 3 | 1 |
| High | 70 | 70 | 0 | 4 | 4 |

The mid component conservatively co-assigns unresolved parent candidates and their clips. **It is not an assertion that three known mid tyres are identical.** Known physical tyres, provisional clip groups and conservative components are counted separately. No historical validation/test image is reassigned to training.

| Conditional withheld group | Remaining low/mid/high fit images | Result |
|---|---|---|
| High session 001 | 67 / 249 / 67 | Three classes remain; no fold model is fitted |
| High session 002 | 67 / 249 / 68 | Three classes remain; no fold model is fitted |
| High session 003 | 67 / 249 / 65 | Three classes remain; no fold model is fitted |
| High session 004 | 67 / 249 / 10 | Three classes remain; no fold model is fitted |
| Only training low tyre | 0 / 249 / 70 | Low fit class disappears |
| Conservative mid component | 67 / 0 / 70 | Mid fit class disappears |

This blocks a complete class-supported leave-component-out comparison across the current training grouping. The four high-only conditional holdouts do not validate low/mid transfer. Counting support does not make the existing globally trained encoder group-held-out.

![Existing independent support and conditional folds](tread_two/readiness_report/figures/independent_support.png)

The frozen 18-reference bank is audited after removing the full held component and identical query source SHA. All **1,158 query/class rows** are retained; unsupported `6/n` normalization remains blank. No replacement bank is trained or scored.

## What the correction parts show

For every saved training observation, reconstruct `b`, `b+g`, `b+s`, and `b+g+s`, where `b` is frozen selected LC, `g` bounded global bias and `s` summed spatial correction. All four variants use the identical saved view and actual pre-update model state. These are algebraic diagnostic parts, not independently trained ablations.

The frozen prior already correctly classifies **all 47,040 sampled training-view observations**. Full context rescues **{int(full.rescued_prior_errors.sum())}** prior errors and transiently spoils **{int(full.spoiled_prior_corrects.sum())}** prior-correct view observations early in training. It lowers training NLL mainly through the spatial term. This is a confidence/loss finding on trained images and repeated changing states, not evidence of transfer or a fixed final-cohort evaluation.

| Final training epoch, same 784 saved views / 214 distinct images | NLL | Mean max probability | Argmax accuracy |
|---|---:|---:|---:|
| Frozen prior | 0.636877 | 0.530181 | 100% |
| Bias only | 0.636948 | 0.529765 | 100% |
| Spatial only | 0.031380 | 0.969111 | 100% |
| Full context | 0.031365 | 0.969124 | 100% |

![Algebraic correction parts across training epochs](tread_two/readiness_report/figures/correction_parts.png)

![Same-view loss change and argmax utility](tread_two/readiness_report/figures/utility_by_component.png)

## Code, retention and execution

| Module / artifact | Responsibility |
|---|---|
| [public_inputs.py](tread_two/readiness_gate/public_inputs.py) | Six public pinned inputs, hash/size cache, bounded 1 MiB reads, retries/server waits, no credential lookup |
| [INPUTS.json](tread_two/readiness_gate/INPUTS.json) | Exact NB14/NB07 revisions, prefixes, object hashes/sizes and provenance |
| [run_gate.py](tread_two/readiness_gate/run_gate.py) | Filter training rows before numeric analysis; identity/fold/bank support; four logit parts and view/group aggregates |
| [render_gate.py](tread_two/readiness_gate/render_gate.py) | Three PNG figures, interactive epoch/class/variant passport, methods and table index |
| [publish_gate.py](tread_two/readiness_gate/publish_gate.py) | Immutable additive direct files through unchanged r3 transport, ownership and rerun reconciliation |
| [CAPABILITIES.json](tread_two/readiness_report/CAPABILITIES.json) | Research-only classifier; retained frozen geometry; internal evidence scope; depth/safety unavailable |

Six inputs total **70,218,229 bytes**. Outputs: **16 CSV tables / {p['CSV_rows']:,} rows**, including **188,160** saved-view counterfactual rows, all 2,316 conditional fold assignments and 1,158 bank-support rows, a 166-field dictionary, three figures and interactive passport. [Full CSV inventory](tread_two/readiness_report/CSV_INDEX.md), [methods/denominators/limits](tread_two/readiness_report/METHODS.md), [interactive passport](tread_two/readiness_report/Evidence_Passport.html).

NB15 is self-contained: 40 hashed payload files. CPU/None, Internet, HF_TOKEN for writes, no dataset, 2 GiB free / about 4 GB RAM. Only the HF SDK is added to a private publication folder; global Kaggle libraries and Torch/CUDA are not replaced. Original native model weights are not loaded. Finite report/cache files survive a stopped session when working output survives; unchanged rerun reuses verified inputs and rebuilds derived tables. Complete publication happens after the major analysis cell; it does not claim per-update optimizer recovery or 30-minute snapshots for a partial derived report. Catchable Stop lets the current completed-report upload finish; forced kernel death cannot upload unsent files.

Destination: `phase3/tread-two/readiness-gate/v1/<contract>`, **{p['publication_files']} logical direct files** including manifest. Source/protocol and original fit/review namespaces remain read-only. No ZIP/NPZ/database. Large local input cache, acceptance fixtures and row-level counterfactual table stay outside Git; direct HF publication retains them.

## Actual local proof and decision

[Package acceptance](tread_two/readiness_validation/PACKAGE_VALIDATION.json), [public transport/cache checks](tread_two/readiness_validation/TRANSPORT_VALIDATION.json), [notebook identity](tread_two/readiness_validation/NOTEBOOK.json), [viewer checks](tread_two/readiness_validation/VIEWER.json), [analysis receipt](tread_two/readiness_report/ANALYSIS.json).

- Every table size/schema/row/hash and unsupported reference case checked; actual class/component support and algebraic counts confirmed.
- Changing every held-out numeric prediction/correction and held-out registry label leaves **all scientific CSVs byte-identical**. Changing a training source SHA is rejected.
- Actual 40-file setup cell executed locally with Kaggle paths stubbed: extraction/rerun/drift rejection pass. Fresh extracted scientific code reproduces every CSV without network downloads.
- Actual passport JavaScript epoch/class/variant controls and weighted curves pass a synthetic DOM check; no native browser screenshot claim.
- Offline publication handles a lost successful response, releases ownership, skips an identical rerun and writes only its new gate namespace.

Training-only describes the rows analyzed. The frozen LC prior had already been historically selected on validation, so this is not an untouched end-to-end benchmark. No physical depth/safety/wear truth or independent tyres are created.

**Decision:** retain ATLAS v1/LC/context as documented research candidates; do not automatically launch a fourth full classifier fit or install the failed classifier. Run NB15 once for the requested durable HF evidence publication, then verify and close this gate. A later existing-data experiment needs a distinct supported objective/protocol; an identical rerun cannot repair missing support. Workstation changes remain deferred under the user's instruction; no new-photo collection is required for this completed audit.
''')
write('tread_two/readiness_gate/README.md',r'''
# NB15 — existing-data readiness and evidence passport

**CPU saved-record audit, not training.** [Completed NB14](../../phase3NB14Completion.md) · [readiness findings and module guide](../../phase3TreadTwoReadiness.md).

## Run on Kaggle

1. Import [NB15](../../notebooks/phase3_NB15_TREAD_TWO_Existing_Data_Readiness_And_Evidence_Passport.ipynb).
2. Accelerator **None / CPU**, **Internet ON**, no image dataset attachment.
3. Enable the **HF_TOKEN** Secret with write permission for `Shanmuk4622/tyre-wear-study`.
4. Keep at least **2 GiB free output** and about **4 GB RAM**; inputs total about 70 MB.
5. **Run All**: extract → training-only support/correction analysis → direct HF publication → figures/passport/receipt.
6. Keep the executed notebook and `GATE_PUBLICATION.json` for independent verification.

| Output | Expected evidence |
|---|---|
| `report/SUMMARY.json` | Analysis passed; scientific readiness false; six conditional folds, two missing fit classes |
| Direct CSV | 16 tables / 196,905 rows, including 188,160 algebraic saved-view observations |
| Visuals | Three scientific PNGs and interactive `Evidence_Passport.html` |
| `CAPABILITIES.json` | Classifier research-only/unpromoted; retained geometry/internal-evidence scope; depth/safety unavailable |
| Publication | Released-owner receipt under `phase3/tread-two/readiness-gate/v1/<contract>` |

Working path: `/kaggle/working/tread_two_nb15_readiness/`. The private `publication_sdk/` holds HF SDK 0.36.2 only for the publishing subprocess. No global HF SDK/Torch/CUDA replacement. No original image/model download, model forward, optimizer update or workstation change.

## Stop, restart and troubleshooting

| Situation | Action |
|---|---|
| Analysis interrupted | Preserve files and rerun unchanged. Verified cache is reused; derived tables are atomic and rebuilt. |
| New session | Reimport the same notebook and Run All; approximately 70 MB of pinned evidence downloads again unless preserved output is reused. |
| Scientific readiness false | Expected support finding. Do not rerun a model to change metadata or merge uncertain identities to force a pass. |
| Private SDK install interrupted | Keep files and rerun publication. READY is written only after the private version check succeeds; incomplete installation is retried. |
| HF rate/backoff wait | Allow server delay to finish and keep a single writer. Request/commit budgets and immutable staging preserve completed output. |
| Stop during upload | Catchable Stop lets the completed report flush finish; forced kernel death cannot upload unsent files. Rerun unchanged to reconcile/complete. |
| Secret or compatibility failure | Enable the authorized HF_TOKEN or preserve the exact error. Never paste a token into source or overwrite incompatible manifests/checkpoints. |

No ZIP/NPZ publication. This finite audit publishes after the major analysis completes; it does not claim optimizer recovery for a partially derived report. More observations are richer telemetry from existing data, not additional independent tyres. No NB13/NB13A/NB14 rerun is needed.

[CSV inventory](../readiness_report/CSV_INDEX.md) · [methods](../readiness_report/METHODS.md) · [local acceptance](../readiness_validation/PACKAGE_VALIDATION.json).
''')
common='''## Current — NB14 verified; NB15 existing-data readiness implemented

**1 November 2026:** executed NB14 is complete and HF-verified at `10c2f16d…`: **80 direct files / 52 CSVs / 585,825 rows**, five figures, 1,854 cases and 206 attention matrices. Every table matches independent local reconstruction exactly; all 1,502 original fit entries remain unchanged; owner released. **Do not rerun NB13/NB13A/NB14.** [Completion and findings](phase3NB14Completion.md).

The classifier remains research-only: historical BA 50%, low recall 0/62 and worse context NLL. Implemented next: [NB15 readiness/evidence passport](notebooks/phase3_NB15_TREAD_TWO_Existing_Data_Readiness_And_Evidence_Passport.ipynb). Local analysis passes; support readiness is false: known training tyres low/mid/high = 1/3/4 form conservative components 1/1/4, so two of six conditional holdouts remove an entire fit class. All 47,040 saved training views were already argmax-correct under the frozen LC prior; context mainly strengthens training confidence. [Detailed evidence and limits](phase3TreadTwoReadiness.md).

**Next: CPU/None, Internet ON, HF_TOKEN, no image dataset, 2 GiB free / about 4 GB RAM; Run All in NB15.** It retains 16 CSV tables / 196,905 rows, three figures and an interactive capability passport in a separate HF namespace. No fourth full fit, checkpoint reselection, new photos or workstation changes. [Run guide](tread_two/readiness_gate/README.md). Actual NB15 Kaggle/HF publication remains pending. Earlier pending steps below are dated history.'''
for name in ['README.md','AGENTS.md','phase3Progress.md','phase3NextSteps.md','phase3OpenItems.md','phase3Understanding.md','phase3DataRetentionPolicy.md','phase3ModelRationaleAndLogging.md','phase3TreadTwoFindings.md']:
    insert(name,common)
path=ROOT/'phase3NextSteps.md';text=path.read_text(encoding='utf-8')
# Undo the prior default-Windows-decoding pass on historical lines, if present.
repaired=[]
for line in text.splitlines():
    if 'â' in line or 'Â' in line:
        try:line=line.encode('cp1252').decode('utf-8')
        except (UnicodeEncodeError,UnicodeDecodeError):pass
    repaired.append(line)
text='\n'.join(repaired)+'\n';text='# Next steps — publish the existing-data readiness and evidence passport\n'+text.split('\n',1)[1];path.write_text(text,encoding='utf-8')
for name in ['phase3NB13Completion.md','phase3TreadTwoContextReview.md']:
    insert(name,'''## Later measured status — 1 November 2026

**NB14 is completed and independently HF-verified**, not pending: all80directfiles,52CSVs/585,825rows, released owner and unchanged original fit. [Completion](phase3NB14Completion.md). Do not regenerate or rerun its executed notebook. Next is [NB15 existing-data readiness](phase3TreadTwoReadiness.md), [run guide](tread_two/readiness_gate/README.md); actual NB15 publication remains pending. Older pending-NB14 wording below records the prior delivery.''')
insert('tread_two/context_review/README.md','''## Completed — use NB15 next

NB14 completed and independently HF-verified on1November2026: all80directfiles/52CSVs, no errors, released owner and unchanged fit. No rerun needed. [Completion](../../phase3NB14Completion.md). Next: [NB15 readiness/evidence passport](../readiness_gate/README.md). Earlier run instructions below are preserved as historical guidance.''')
insert('tread_two/context_pilot/README.md','''## Completed family review — 1 November 2026

NB13/NB13A and the NB14 review are complete and independently HF-verified. Preserve these executed notebooks and the original 32-entry runtime. [NB14 completion](../../phase3NB14Completion.md). Current next action is [NB15 existing-data readiness and evidence passport](../readiness_gate/README.md), with CPU / Internet / HF_TOKEN and no image attachment. Earlier training/review run instructions below are historical; no rerun is needed.''')
for name in ['phase3TreadTwoContextArchitecture.md','phase3TreadTwoContextProtocol.md','phase3TreadTwoContextValidation.md','phase3TreadTwoContextEvidenceSchema.md']:
    insert(name,'''## Completed context review — 1 November 2026

The NB13 fit and NB14 review are complete and HF-verified. All 80 review files / 52 CSVs match, and the original fit is preserved. [NB14 completion](phase3NB14Completion.md). The classifier remains unpromoted. Next is the implemented [NB15 existing-data readiness / evidence passport](phase3TreadTwoReadiness.md), not another full fit. Earlier pending-NB14 statements below describe the prior delivery. The original architecture/protocol/schema are unchanged.''')
insert('tread_two/README.md',common.replace('](phase3','](../phase3').replace('](notebooks/','](../notebooks/').replace('](tread_two/readiness_gate/','](readiness_gate/'))
insert('phase3EvidenceRegister.md','''## NB14 closure and NB15 readiness evidence — 1 November 2026

| Evidence | Scope |
|---|---|
| [NB14 verification](tread_two/nb14_completion/356dc7893cc5/VERIFICATION.json) | All79objects/80files,52tables exactly equal, five figures, published viewer/attention, original fit unchanged |
| [Executed NB14 inspection](tread_two/nb14_completion/356dc7893cc5/INSPECTION.json) | SHA356dc789…,four completed cells,44payloads, actual publication receipt/warnings |
| [NB15 pinned inputs](tread_two/readiness_gate/INPUTS.json) | Six source objects/70,218,229bytes at exact NB14/NB07 revisions |
| [Readiness analysis](tread_two/readiness_report/ANALYSIS.json) | 16derivedCSVs/196,905rows; no forward/fitting/held-out numeric use |
| [Capability record](tread_two/readiness_report/CAPABILITIES.json) | Research-only classification, retained frozen geometry, internal evidence, physical depth/safety unavailable |
| [Package validation](tread_two/readiness_validation/PACKAGE_VALIDATION.json) | Held-out perturbation invariance, source mismatch rejection, fresh40-file package equality, actual JS controls and offline publication |

[NB14 report](phase3NB14Completion.md) · [NB15 findings/methods](phase3TreadTwoReadiness.md).''')
insert('phase3DocumentationInventory.md','''## NB14 / readiness documents — 1 November 2026

| Document | Purpose |
|---|---|
| [NB14 completion](phase3NB14Completion.md) | Exact public verification, equality and preserved fit, illustrations, conclusions and execution warning record |
| [Readiness/evidence passport](phase3TreadTwoReadiness.md) | Existing-data support/folds, correction utility, tables, figures, code ownership, resources, limits and checks |
| [NB15 Kaggle guide](tread_two/readiness_gate/README.md) | CPU/Internet/Secret/run order, private SDK, restart/stop/troubleshooting |
| [Derived CSV index](tread_two/readiness_report/CSV_INDEX.md) | 16tables/196,905rows and166-field dictionary |

Current next action is NB15 publication; completed NB14 and all training studies remain preserved.''')
insert('phase3RepositoryMap.md','''## NB14 completion and readiness paths — 1 November 2026

| Path | Retention / owner |
|---|---|
| `tread_two/nb14_completion/356dc7893cc5/` | Versioned safe receipts/manifest; ignored full remote mirror and executed archive |
| `tread_two/readiness_gate/` | Four isolated Python modules and exact six-object INPUTS.json; no model source changes |
| `tread_two/readiness_report/` | 16tables/three figures/passport/methods/capability JSON; heavy counterfactual CSV ignored locally and published on HF |
| `tread_two/readiness_validation/` | Small package/notebook/viewer/transport/handoff receipts; cache/fixtures ignored |
| `notebooks/phase3_NB15_TREAD_TWO_Existing_Data_Readiness_And_Evidence_Passport.ipynb` | Self-contained CPU readiness and direct additive HF publication |

[Completion](phase3NB14Completion.md) · [Readiness design](phase3TreadTwoReadiness.md). Both workstations and all original fit/review modules remain unchanged.''')
insert('phase3Decisions.md','''## NB14 closure / existing-data readiness — 1 November 2026

| ID | Decision | Evidence / boundary |
|---|---|---|
| P3-D116 | Close NB14 without rerun; preserve executed notebook/source and original fit | 80files/52tables verified, exact table equality, released owner, all1502fit entries unchanged |
| P3-D117 | Audit existing independent support before another full classifier fit | One low and one conservative mid training component; two conditional folds remove fit class |
| P3-D118 | Keep mid co-assignment separate from physical identity | Three known mid tyres remain distinct; unresolved clip-parent constraint is preserved |
| P3-D119 | Quantify algebraic correction utility without fitting or new held-out inference | Four variants on47040savedtrainingviews; held-out perturbation leaves scientific outputs identical |
| P3-D120 | Publish structured capability/evidence passport through CPU NB15 | 16CSVtables/196,905rows; model remains research-only, physical depth/safety unavailable |
| P3-D121 | Isolate tested HF SDK in new publication subprocess | NB14 completed despite unrelated dependency warnings; no global SDK/Torch/CUDA replacement in NB15 |

No new-photo prerequisite or workstation work is introduced. Future classification exploration needs a distinct objective/protocol; this gate does not prove all possible models impossible.''')
insert('phase3SessionLog.md',f'''## NB14 verification and existing-data readiness implementation — 1 November 2026

User reported NB14 complete and requested HF verification, Markdown updates and continuation. Preserved executed SHA `356dc789…` and all 44 matching payloads. All four code cells completed without errors. The public-read audit at HF `10c2f16d…` verified 79 objects / 80 logical files, 217,059,186 bytes, 52 CSV tables / 585,825 rows, five PNGs, 1,854 interactive cases, 18 references and 206 full attention matrices. All scientific tables are byte-identical to independent local reconstruction; maximum numeric difference is 0. All 1,502 original fit manifest entries are unchanged; writer released. No local credential reads or remote writes.

Implemented and ran the isolated existing-data readiness and algebraic online utility analysis. Six pinned NB14/NB07 inputs total 70,218,229 bytes. Only 386 training images and 47,040 training-view observations enter numeric calculations. Known original tyres are 1/3/4; conservative components are 1/1/4. Six conditional holdouts are enumerated without fitting: low and mid folds each remove an entire fit class. Unresolved mid co-assignment is a constraint, not physical identity equivalence. Frozen 18-reference bank support is counted after complete component/source-SHA exclusion; unsupported normalization stays blank.

Four algebraic variants produce 188,160 rows. Frozen LC is already argmax-correct on all 47,040 saved training views; full context rescues {int(full.rescued_prior_errors.sum())} and transiently spoils {int(full.spoiled_prior_corrects.sum())} prior-correct view observations. Epoch 60 uses the same 784 views / 214 images: prior NLL 0.636877 versus full 0.031365; bias-only 0.636948 and spatial-only 0.031380. This describes confidence/loss fitting on known images, not unseen-tyre transfer or independently trained ablations. Research-only status retained; no fourth full classifier fit triggered.

Delivered NB15 for CPU / Internet / HF_TOKEN / no image attachment, with a 40-file package and private HF SDK publication directory. Sixteen CSVs / 196,905 rows, three figures, an interactive passport and CAPABILITIES.json are retained. Actual-data tests cover hashes, schemas and support; perturbing all held-out numeric predictions/corrections and registry labels leaves scientific outputs byte-identical; a training SHA mismatch is rejected. Actual setup cell, fresh extracted science, viewer JavaScript and offline publication lost-response/owner-release/idempotence checks pass. Actual NB15 Kaggle/HF completion remains pending. Added narrow Git ignores for cache, fixtures and the heavy counterfactual table; source, summaries and figures stay visible. No additional photos, fitting, inference, checkpoint reselection or workstation changes.''')
print(json.dumps(dict(status='written',NB14=v['status'],NB15_tables=p['CSV_tables'],NB15_rows=p['CSV_rows'],scientific_readiness=summary['scientific_readiness']),indent=2))
