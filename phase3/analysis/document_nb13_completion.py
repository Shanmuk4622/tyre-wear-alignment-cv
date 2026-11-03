"""Document measured NB13A completion and the separate CPU evidence review."""
import json,re,sys
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1]
done=Path(json.loads((ROOT/'tread_two/nb13_completion/LATEST.json').read_text())['path'])
v=json.loads((done/'VERIFICATION.json').read_text());assert v['status']=='passed'
a=json.loads((ROOT/'tread_two/context_analysis_report/ANALYSIS.json').read_text());assert a['status']=='passed'
p=json.loads((ROOT/'tread_two/context_review_validation/PACKAGE_VALIDATION.json').read_text());assert p['status']=='passed'
def write(path,body):
    (ROOT/path).write_text(body.strip()+'\n',encoding='utf-8')
def insert(path,body,marker='NB13_COMPLETED_20261101'):
    target=ROOT/path;text=target.read_text(encoding='utf-8')
    text=re.sub(r'\n<!-- '+marker+r' -->.*?<!-- END_'+marker+r' -->\n','\n',text,flags=re.S)
    first,rest=text.split('\n',1)
    # Retain dated history, but remove ambiguous Current headings in superseded notices.
    rest=re.sub(r'(?m)^## (Current — |Latest status — |Latest superseding status — )',r'## Historical — ',rest)
    target.write_text(first+'\n\n<!-- '+marker+' -->\n'+body.strip()+'\n<!-- END_'+marker+' -->\n'+rest,encoding='utf-8')

write('phase3NB13Completion.md',r'''
# NB13 / NB13A completed — ATLAS-CONTEXT r3

**1 November 2026. The supplied executed NB13A completed recovery and the original fit. Independently verified on Hugging Face; no NB13 or NB13A rerun is needed.** The context candidate remains a research result and is not promoted to either workstation.

## Completion and recovery proof

| Item | Verified evidence |
|---|---|
| Executed NB13A | `8bdf46ec2040a5e69999d38a5d077e8f337702e52f46254efc31e8937e2cf5fa`; two code cells completed, no saved errors |
| Embedded source | All 35 payload files match; all 32 frozen model/runtime entries preserved |
| Measured protocol | `36f473b0886d6c2f49c416436d270198af3982230e3045da9969b0bf42169625` |
| Pinned HF revision | `a44418844ba3f1c98adee2cae76a8bbf9f5f4e9e` |
| Publication | 1,503 direct files including manifest; all 1,502 scientific/control object sizes and SHA-256 values verified |
| Payload | 5,645,673,072 verified bytes; no ZIP publication |
| Scientific CSV | 27 families / **58,176,066 rows**; every raw schema and count verified |
| Completed fit | 60 epochs / **2,940 optimizer updates**; selected epoch 7, update 343 |
| Recovery actually used | HF checkpoint at **2,486 updates**, then 454 remaining updates; original protocol and passed gate reused |
| Native recovery | Model, optimizer, RNG/order/cursors, dense cache and finite-state checks passed; all 60 epoch checkpoints retained |
| Append-only history | 3,913 linked units / 45,852 CSV chunks; hashes, offsets and update sequence 0–2,940 checked |
| Frozen prior | Original validation-selected LC epoch 7 remains bit-identical in every epoch state and final state |
| Ownership | Released: `owner=null`, `expires=0` |
| Actual environment | Tesla T4×2; Python 3.13.15; Torch 2.11.0+cu128; CUDA 12.8; NumPy 2.1.3; Pillow 13.0.0 |
| Independent audit | Public reads, zero local credential reads, zero remote writes, zero model forwards or fitting |

The prior repair documents verified HF update 1,294 on 31 October. That was a dated partial snapshot. The **executed recovery** explicitly restored a newer verified revision at update 2,486. Do not claim that this execution restored 1,294, lost 1,173 updates or started a new experiment. Original freeze cause remains unestablished; the repaired recovery did complete.

[Full byte/raw-row receipt](tread_two/nb13_completion/8bdf46ec2040/VERIFICATION.json) · [ledger/native receipt](tread_two/nb13_completion/8bdf46ec2040/LEDGER_VERIFICATION.json) · [executed output inspection](tread_two/nb13_completion/8bdf46ec2040/INSPECTION.json) · [scientific findings](tread_two/nb13_completion/8bdf46ec2040/SCIENTIFIC_FINDINGS.json) · [published manifest](tread_two/nb13_completion/8bdf46ec2040/REMOTE_MANIFEST.json) · [HF tree at the verified revision](https://huggingface.co/datasets/Shanmuk4622/tyre-wear-study/tree/a44418844ba3f1c98adee2cae76a8bbf9f5f4e9e/phase3/tread-two/atlas-context-r3/fit/36f473b0886d6c2f49c416436d270198af3982230e3045da9969b0bf42169625).

```mermaid
flowchart LR
    A[Stopped original NB13] --> B[NB13A revision 2]
    B --> C[Verified HF restore at 2486]
    C --> D[Same fit completes at 2940]
    D --> E[All published bytes / rows / native history verified]
    E --> F[Saved-data science review\nzero fitting / zero forwards]
    F --> G[NB14 CPU publication\n52 CSVs / figures / interactive explorer]
    F --> H[Research result retained\nno workstation promotion]
```

## What the model achieved

The context branch was motivated by the observed order invariance of the previous local-reference pooling. It now responds to token layout, but **this did not improve the measured classification failure**. The unchanged LC branch provides a useful internal comparison: on validation its BA remains 56.25% and NLL 0.934841, while every trained context epoch has BA 50%. The original selection rule chose context epoch 7 with NLL 1.734916. Do not retrospectively replace this with epoch zero or a test-favored epoch.

| Previously inspected historical endpoint | LC selected, epoch 7 | Context selected, epoch 7 | Context final, epoch 60 |
|---|---:|---:|---:|
| Images | 103 | 103 | 103 |
| Observed-class balanced accuracy | 50.00% | 50.00% | 50.00% |
| Accuracy | 39.806% | 39.806% | 39.806% |
| Low recall | 0/62 | 0/62 | 0/62 |
| High recall | 41/41 | 41/41 | 41/41 |
| NLL, lower is better | 0.961273 | 2.174813 | 2.358475 |
| Brier, lower is better | 0.568355 | 1.105581 | 1.129997 |
| ECE, 10 equal-width bins | 0.425969 | 0.552019 | 0.561834 |

All 62 low images become mid; all 41 high images remain high. Selected/final context mean confidence is 95.01% / 95.99%. The learned correction therefore strengthens the existing wrong low→mid decision while improving confidence on already-correct high cases. This is descriptive evidence of harmful probability behavior; it does not identify a specific physical or acquisition shortcut as its cause.

![Fit, selection and actual confusion](tread_two/context_analysis_report/figures/fit_and_confusion.png)

![Correction, attention and probability loss](tread_two/context_analysis_report/figures/context_and_attention.png)

## What the extra evidence tells us

| Question | Saved evidence / interpretation |
|---|---|
| Did sampling cover the training inputs? | All 386 queries were drawn; 23,520 paired draws / 47,040 view observations; maximum 423 draws per image. Repeated views do not create independent tyres. |
| Was self-reference blocked? | All 967,572 permission rows verified; 4,882 view-reference rows blocked, equivalent to 2,441 paired draws; source-SHA rules and 6/n scaling checked independently of target labels. |
| Does context respond to layout? | All 412 reverse-band/column cases preserve frozen prior logits within tolerance. Learned correction changes; maximum final column-reversal class-probability change is 0.087466. |
| Does stable argmax imply unchanged scores? | No. All 16 endpoint/probe groups have zero flips, while layout/occlusion scores move. Final top-patch occlusion changes a probability by up to 0.078925. |
| Is attention a physical explanation? | No. All 1,687,552 saved endpoint values and 1,648 head summaries pass probability/entropy/masking checks. Mean normalized entropy is broad, roughly 0.81–0.95. This records internal attention, not causal wear truth. |
| Is the accounting faithful? | All 53,754 logit/correction observations checked; 1,854 endpoint/probe decompositions reconstruct logits within 3.39e-7, combined patch values within 6.49e-9. |
| Does the paired loss match? | All 23,520 paired objectives, 752,640 patch distances, 47,040 seeded appearance factors and 2,940 update means checked; maximum pair-loss residual 2.59e-7. |
| Was dense geometry retrained? | No. Frozen dense branch retained for all 570 images. Historical endpoint tyre/tread Dice is 0.98485 / 0.98352; human-point MAE is 0.01196 image widths on 150 points from 25 images. These are retained branch results, not context improvements. |

Online sampled training BA reaches 100% from epoch 3. These predictions come from changing pre-update states and repeated paired draws, not one fixed final-model evaluation of all training images. Do not compare them as an independent train/test experiment.

![Paired objective and exposure](tread_two/context_analysis_report/figures/paired_learning.png)

## Retention, scope and missingness

| Scope | Result |
|---|---|
| Exhaustive original publication audit | All 1,502 objects, all 27 raw CSV schemas/counts, all committed chunk/unit hashes, all 60 native epoch identities and frozen prior equality |
| Independent science reconstruction | 876 consumed inputs / 636,534,073 bytes; **52 derived CSVs / 585,825 rows**, five PNGs, 1,854 explorer cases and 206 full endpoint attention matrices |
| Not independently recalculated | Every training affinity value; full raw bytes and counts were audited, while detailed affinity/logit reconstruction covers all selected/final/probe cases |
| Not saved / not invented | Full per-update or probe attention matrices, unmeasured physical wear/depth/safety labels, unseen independent tyres |
| Cohort limits | 570 existing images; 386 train / 81 validation / 103 historical test. Training known-tyre low/mid/high counts 1/3/4; held-out cohorts have no mid targets. |
| Interpretation | Historical held-out data has already informed development; observed-class BA covers low/high only. Attention/evidence accounting and token interventions are not causal validation. |

Raw families and columns: [frozen schema](phase3TreadTwoContextEvidenceSchema.md). Derived tables/columns/units: [CSV index](tread_two/context_analysis_report/CSV_INDEX.md), [dictionary](tread_two/context_analysis_report/table_dictionary.csv), [methods](tread_two/context_analysis_report/METHODS.md). [Interactive explorer](tread_two/context_analysis_report/ATLAS_Failure_Explorer.html) filters endpoint/probe/class/error and separates frozen LC, context and combined evidence; attention layer/head is selectable for the 206 saved identity cases. Large local tables/HTML/cache stay outside Git; NB14 publishes them directly to HF.

## Next: NB14, without another fit

The isolated [NB14 completed review](notebooks/phase3_NB14_TREAD_TWO_CONTEXT_Completed_Review.ipynb) is implemented and locally validated. Use **CPU / None, Internet ON, HF_TOKEN enabled, no image dataset, 4 GiB free and about 8 GB RAM; Run All**. It reads pinned evidence and publishes the new review namespace without touching the completed fit. [Detailed run guide](tread_two/context_review/README.md), [review design and local proof](phase3TreadTwoContextReview.md).

Actual NB14 Kaggle execution and HF review publication remain pending. After verification, close the review and use its evidence to decide whether another explicitly frozen experiment is justified. This result does not justify another blind seed/epoch run or workstation installation. Existing photos only; both workstations preserved.
''')

write('phase3TreadTwoContextReview.md',r'''
# ATLAS-CONTEXT completed review — NB14

**1 November 2026. Implemented and locally checked; actual Kaggle/HF review publication pending.** The measured fit is complete; see [NB13 completion](phase3NB13Completion.md). This separate CPU review contains no model fitting or model forwards.

## Evidence flow and ownership

```mermaid
flowchart TD
    H[Pinned HF fit manifest + SHA objects] --> R[Verified read cache]
    P[Pinned LC / v1 comparison CSVs] --> A[Saved-data accounting]
    R --> A
    A --> C[52 direct CSV tables]
    A --> V[5 PNG figures + 1854-case explorer]
    V --> T[206 endpoint attention matrices\nlayer / head / tile controls]
    C --> B[Isolated immutable review bundle]
    T --> B
    B --> U[HF additive review publication\noriginal fit read-only]
```

| Source module | Responsibility |
|---|---|
| [review_context.py](tread_two/context_review/review_context.py) | Pinned inputs, size/SHA cache, bounded downloads, probabilities/cohort/dense metrics, orchestration and table inventory |
| [paired_context.py](tread_two/context_review/paired_context.py) | Draw probabilities, self-exclusion, seeded appearance, paired CE/JS/texture and residual penalty, gradients/exposure, historical comparisons |
| [context_checks.py](tread_two/context_review/context_checks.py) | All correction bounds/logit/patch accounting, frozen LC matching, full endpoint attention, layout response and report notes |
| [render_context.py](tread_two/context_review/render_context.py) | Five scientific figures, interactive base/context/combined tiles, source mapping and attention controls |
| [publish_context_review.py](tread_two/context_review/publish_context_review.py) | Additive direct-file contract/publication through unchanged r3 transport; owner release and identical-rerun skipping |
| `assets/` | Pinned v1/LC cohort CSVs, 103 selected LC identity logits and version/hash metadata |

The frozen `context_pilot/FILES.json` still contains the original 32 entries. Review code lives outside it; neither its source nor the measured training protocol is replaced. All 44 notebook payload files, including frozen transport/source, pinned fit manifest and audit receipt, are individually hashed.

## CSV and viewer contract

The [52-table inventory](tread_two/context_analysis_report/CSV_INDEX.md) and [632-column dictionary](tread_two/context_analysis_report/table_dictionary.csv) cover 585,825 derived rows. Important additions to the prior saved-data review:

| Table | Rows | Meaning |
|---|---:|---|
| `context_observations` | 53,754 | Frozen logits, corrections, changes in NLL/predictions and accounting residuals |
| `context_epochs` | 124 | Stage/epoch aggregates; online observations remain distinct from fixed evaluation |
| `context_by_variant` | 44 | Class-conditioned context magnitude and loss by endpoint/probe |
| `endpoint_context_evidence` | 59,328 | 32 signed correction tiles for each of 1,854 endpoint/probe cases |
| `frozen_LC_prior_metrics` | 198 | Metrics recomputed from unchanged saved prior logits |
| `base_context_pairs` | 1,854 | Matched frozen/context accounting for every explorer case |
| `layout_prior_context_delta` | 412 | Separate prior invariance, correction response and probability response for token-order probes |
| `attention_head_audit` | 1,648 | 206 cases × 2 layers × 4 heads, with entropy/mask/probability checks |
| `attention_by_endpoint` | 16 | Endpoint/layer/head summaries |
| `frozen_prior_peer_check` | 206 | Identity-logit equality against selected LC epoch 7 |
| `LC_context_comparison` | 18 | Pinned descriptive endpoint/probe comparisons; missing LC layout probes remain missing |
| `endpoint_patch_accounting` | 1,854 | Frozen-reference plus context contribution conservation |

Viewer filters: endpoint, all eight probes, target class, wrong/flip status, text search and paging. Choose explained class and frozen/context/combined evidence. Select attention layer/head and hover query/key cells. Seven-decimal copies support interaction; original raw CSV precision remains on HF. Full attention is unavailable for training/probes and displayed as unavailable. All thumbnails are original images; layout interventions move saved tokens, not photographs.

## Resources, restart and publication

| Item | Bound / behavior |
|---|---|
| Kaggle | CPU/None; Internet; authorized HF_TOKEN Secret for publication; no dataset attachment |
| Disk/RAM | At least 4 GiB free output; about 8 GB RAM; source evidence subset approximately 637 MB |
| Original fit archive | 5.646 GB verified locally; the CPU review does not download/load all native weights |
| Dependencies | Existing NumPy/pandas/Matplotlib/Pillow/filelock/requests; only HF SDK pinned to 0.36.2 if needed. No Torch/CUDA replacement. |
| Downloads | SHA/size cache; larger inputs assembled in bounded 1 MiB ranges with four reader threads, unique temporary files and full final SHA. Ignored/invalid ranges rejected. |
| Stopped analysis | Cache survives in existing/saved working files; rerun rebuilds finite derived tables atomically. No optimizer or training continuation exists here. |
| HF write policy | Separate `phase3/tread-two/atlas-context-r3/review/<contract>`; direct CSV/JSON/PNG/HTML/source; no ZIP/NPZ. One major completed-review publication using inherited budgets, 80-object batches, server backoff and ownership. |
| Stop / rerun | Catchable stop allows current completed-report publication to finish. Forced termination cannot upload unsent files. Same completed manifest is verified/skipped on rerun; no concurrent writers. |

This finite review publishes when its major analysis cell has completed. It does not claim NB13's 30-minute/complete-update training recovery for partially derived reports. No project `.env` or alternate credentials are read. Server-directed waits remain possible; preserve working files and logs.

## Local acceptance and remaining check

[Package validation](tread_two/context_review_validation/PACKAGE_VALIDATION.json), [setup execution check](tread_two/context_review_validation/SETUP_VALIDATION.json), [notebook identity](tread_two/context_review_validation/NOTEBOOK.json), [viewer checks](tread_two/context_review_validation/VIEWER.json), [analysis receipt](tread_two/context_analysis_report/ANALYSIS.json).

| Check actually performed | Result / limit |
|---|---|
| Derived scientific tables | Every row/column inventory and SHA checked; logit, patch, pair and attention residuals pass |
| Cache behavior | Reuse and corruption replacement; exact range reassembly; ignored-range response rejected before body read |
| Actual explorer JavaScript | Filters/paging/class/components/layer/head/1,024 matrix cells/missingness checked in a synthetic DOM |
| Offline HF service | Additive namespace, lost successful-commit response reconciliation, released owner, identical rerun skipped without another commit |
| Self-contained notebook | All 44 extracted payload hashes match; fresh extracted analysis reproduces every scientific CSV byte-for-byte |
| Actual setup cell | Executed in an empty fixture directory with only Kaggle paths/SDK metadata stubbed; exact extraction, unchanged rerun and changed-source rejection pass |
| Scope | Zero local model forwards/training/remote writes. Synthetic DOM is functional testing, not a claim of browser screenshot QA. |
| Still pending | Actual Kaggle NB14 and independent audit of its future HF review publication |

Review results are development evidence. No new independent tyres, checkpoint reselection, physical wear labels or workstation installation. [Kaggle run guide](tread_two/context_review/README.md).
''')

write('tread_two/context_review/README.md',r'''
# NB14 — ATLAS-CONTEXT saved-results review

The original NB13/NB13A fit is complete and HF-verified. **Run NB14 next; do not repeat training.** [Completion and findings](../../phase3NB13Completion.md) · [technical review guide](../../phase3TreadTwoContextReview.md).

## Kaggle instructions

1. Import [phase3_NB14_TREAD_TWO_CONTEXT_Completed_Review.ipynb](../../notebooks/phase3_NB14_TREAD_TWO_CONTEXT_Completed_Review.ipynb).
2. Accelerator **None / CPU**, Internet **ON**. No tyre dataset is required. Keep at least **4 GiB free** output and about **8 GB RAM**.
3. Enable Kaggle Secret **HF_TOKEN** with write access to `Shanmuk4622/tyre-wear-study`. Do not paste its value into a notebook or file.
4. **Run All**, in order: unpack/prepare → saved-data analysis → direct HF publication → inspect figures/receipt.
5. Preserve the executed notebook and final `REVIEW_PUBLICATION.json`. Return the executed NB14 for independent HF verification.

| Cell | Expected result |
|---|---|
| Setup | 44 verified source/metadata assets; pinned CPU review ready |
| Analysis | 52 CSVs / 585,825 rows, five figures, 1,854-case explorer, 206 endpoint attention cases |
| Publish | Released-owner receipt for isolated `atlas-context-r3/review/<contract>` namespace |
| Inspect | Printed receipt/summary, context figure and explorer/table links |

Working folder: `/kaggle/working/tread_two_nb14_context_review/`. `cache/` contains verified evidence, `report/` derived artifacts, `bundle/` immutable publication staging. Original full native checkpoints are not loaded; this is a CPU evidence review with zero model forwards and zero fitting.

## Resume and troubleshooting

| Situation | Action |
|---|---|
| Analysis interrupted | Preserve working files; rerun unchanged. Valid cache files are reused and finite derived tables rebuilt atomically. |
| Session reset | Reimport the same NB14 and Run All; pinned inputs redownload. Retained working output can avoid redundant downloads. |
| HF asks for backoff / appears quiet during upload | Allow printed server waits to finish. Shared request budgeting and 80-object batches remain active; do not start another writer. |
| Stop during publication | Catchable Stop lets the completed report flush finish. Forced kernel shutdown cannot upload unsent files; rerun unchanged to reconcile/complete publication. |
| Wrong/missing Secret | Enable `HF_TOKEN` for this notebook and rerun publication. No alternate credential file is used. |
| Source/hash mismatch | Preserve files/error output. Do not edit embedded code, original fit sources or manifests to bypass the check. |
| Insufficient output space | Keep at least 4 GiB available. Do not duplicate the original image dataset/full fit mirror into the review folder. |
| Missing attention for a probe | Expected: full matrices were saved only for selected/final identity endpoints. The viewer explicitly marks other cases unavailable. |

The review publishes complete direct CSV/JSON/PNG/HTML/source records, never ZIP/NPZ. It does not promise optimizer recovery for an unfinished derived report. The completed fit remains read-only, the classifier unpromoted, both workstations unchanged and the existing-photo constraint retained.

## Module and evidence references

| Module | Owns |
|---|---|
| `review_context.py` | Pinned read cache, common audit and output tables |
| `paired_context.py` | Paired objective, exclusions, exposure and gradient checks |
| `context_checks.py` | Frozen-prior/context/attention/layout conservation |
| `render_context.py` | Figures and interactive evidence viewer |
| `publish_context_review.py` | Isolated direct-file HF publication |

[Local package proof](../context_review_validation/PACKAGE_VALIDATION.json) · [CSV inventory](../context_analysis_report/CSV_INDEX.md) · [methods](../context_analysis_report/METHODS.md) · [original 27-family schema](../../phase3TreadTwoContextEvidenceSchema.md).
''')

common='''## Current — NB13A complete and HF-verified; NB14 CPU review ready

**1 November 2026:** the supplied NB13A restored the newer HF checkpoint at **2,486 updates** and completed the original **60 epochs / 2,940 updates**. All **1,503 direct files / 58,176,066 scientific CSV rows**, the native/CSV ledger and all 60 frozen-prior epoch states are verified; ownership is released. **Do not rerun NB13/NB13A.** [Completion and illustrated findings](phase3NB13Completion.md).

The context branch responds to layout but does not fix classification: selected/final historical BA **50%**, low recall **0/62**, high **41/41**; NLL worsens to **2.174813 / 2.358475**. Keep it unpromoted. These are previously inspected mileage-proxy cohorts, not independent physical wear/depth/safety validation.

**Next: [NB14 saved-data review](notebooks/phase3_NB14_TREAD_TWO_CONTEXT_Completed_Review.ipynb)** — **CPU / None, Internet ON, HF_TOKEN, no dataset attachment, 4 GiB free and about 8 GB RAM; Run All**. Locally checked: 52 derived CSVs / 585,825 rows, five figures, 1,854 interactive cases and 206 endpoint attention matrices. Zero new fitting/model forwards; additive direct HF review publication. Actual NB14 Kaggle/HF completion remains pending. [Run guide](tread_two/context_review/README.md) · [Review design and checks](phase3TreadTwoContextReview.md).

Preserve all completed notebooks/runtime identities and both workstations. Existing photos only. Earlier recovery positions/pending instructions below are dated history, superseded by this completion proof.'''
for name in ['README.md','AGENTS.md','phase3Progress.md','phase3NextSteps.md','phase3OpenItems.md','phase3Understanding.md','phase3DataRetentionPolicy.md','phase3ModelRationaleAndLogging.md','phase3TreadTwoFindings.md']:
    insert(name,common)

insert('phase3EvidenceRegister.md','''## NB13A completion / NB14 review evidence — 1 November 2026

| Evidence | Role |
|---|---|
| [NB13 receipt](tread_two/nb13_completion/8bdf46ec2040/VERIFICATION.json) | All 1,503 direct files, raw 58,176,066 rows, original protocol, full SHA/size verification |
| [Ledger/native audit](tread_two/nb13_completion/8bdf46ec2040/LEDGER_VERIFICATION.json) | 3,913 units, 45,852 chunks, all 60 finite epoch states/frozen prior, gap-free update sequence |
| [Executed inspection](tread_two/nb13_completion/8bdf46ec2040/INSPECTION.json) | Two complete cells, actual restore 2,486, original gate reused, released writer |
| [Findings](tread_two/nb13_completion/8bdf46ec2040/SCIENTIFIC_FINDINGS.json) | Classification, geometry, probes, exposure and numerical residuals derived without new forwards |
| [Scientific review](tread_two/context_analysis_report/ANALYSIS.json) | 876 evidence inputs; 52 derived CSVs / 585,825 rows; scope and checks |
| [NB14 package proof](tread_two/context_review_validation/PACKAGE_VALIDATION.json) | 44 payload files, fresh table equality, actual viewer JS, cache and offline publication |
| [NB14 setup proof](tread_two/context_review_validation/SETUP_VALIDATION.json) | Actual first cell in an empty fixture, identical rerun and drift rejection; local path/SDK stubs only |
| [NB14 identity](tread_two/context_review_validation/NOTEBOOK.json) | Current clean notebook hash and complete payload hashes; actual Kaggle execution pending |

[Illustrated completion](phase3NB13Completion.md) · [Review methodology](phase3TreadTwoContextReview.md). Old partial checkpoints below are dated observations, not current completion status.''')

insert('phase3DocumentationInventory.md','''## NB13A closed / NB14 documentation — 1 November 2026

| Document | Purpose |
|---|---|
| [NB13 completion](phase3NB13Completion.md) | Pinned HF proof, actual 2,486 recovery point, raw/native ledger, metrics, figures, findings and decision |
| [Context completed review](phase3TreadTwoContextReview.md) | Module ownership, 52-table additions, interactive evidence, resources/restart/HF and actual local checks |
| [NB14 run guide](tread_two/context_review/README.md) | CPU/Internet/HF_TOKEN settings, cells, restart/stop/troubleshooting and expected outputs |
| [Derived CSV inventory](tread_two/context_analysis_report/CSV_INDEX.md) | 52 tables / 585,825 rows and complete column dictionary |

Current task is NB14 CPU review publication; NB13/NB13A are completed and must be preserved. Earlier pending-run instructions below are historical.''')

insert('phase3RepositoryMap.md','''## New verified-fit and CPU-review paths — 1 November 2026

| Path | Ownership / retention |
|---|---|
| `tread_two/nb13_completion/8bdf46ec2040/` | Small versioned receipts; ignored executed archive/full 5.646 GB HF mirror |
| `tread_two/context_review/` | Five isolated analysis/publication modules and four pinned peer assets; frozen model code untouched |
| `tread_two/context_analysis_report/` | 52 derived CSVs, five curated PNGs, methods/index; large row tables, attention JSON and explorer ignored locally but included in HF review |
| `tread_two/context_review_validation/` | Versioned notebook/package/viewer/handoff receipts; ignored acceptance fixtures |
| `notebooks/phase3_NB14_TREAD_TWO_CONTEXT_Completed_Review.ipynb` | Self-contained CPU saved-data audit and additive direct HF publication |
| `analysis/verify_nb13_publication.py`, `verify_nb13_ledger.py` | Exhaustive public-read completion audit; no fit or credentials |

[Completion](phase3NB13Completion.md) · [Review guide](phase3TreadTwoContextReview.md). Keep original 32-entry context package, earlier studies and both workstations unchanged.''')

for name in ['phase3NB13ARepair.md','phase3NB13FreezeAndRecovery.md','phase3TreadTwoContextArchitecture.md','phase3TreadTwoContextProtocol.md','phase3TreadTwoContextValidation.md']:
    insert(name,'''## Superseding measured outcome — 1 November 2026

**NB13A completed and independently HF-verified:** actual recovery restored update **2,486** and reached **60 epochs / 2,940 updates** under the original protocol. All 1,503 direct files / 58,176,066 raw CSV rows, native/CSV history and all 60 unchanged frozen-prior epoch states verified; writer released. [Completion, figures and findings](phase3NB13Completion.md).

Do not rerun NB13/NB13A. Context did not repair low recall and worsened NLL; candidate remains unpromoted. Next is [NB14 CPU saved-data review](notebooks/phase3_NB14_TREAD_TWO_CONTEXT_Completed_Review.ipynb), [run guide](tread_two/context_review/README.md). Actual NB14 execution/publication pending. Earlier partial positions and pending gates in this document are dated history, preserved to explain the experiment and repair.''')
insert('phase3TreadTwoContextEvidenceSchema.md','''## Completed retention audit — 1 November 2026

The frozen planned schema below was retained exactly: **27 raw CSV families / 58,176,066 scientific rows**. Every published byte, schema and count was independently verified. [Completion proof](phase3NB13Completion.md), [raw audit](tread_two/nb13_completion/8bdf46ec2040/VERIFICATION.json). Actual counts:

| Family | Rows |
|---|---:|
'''+''.join(f'| `{name}` | {count:,} |\n' for name,count in v['scientific_rows'].items())+'''
All 60 epoch weights and complete native recovery are retained directly on HF. Full selected/final attention is retained; full per-update/probe matrices were not saved and are not reconstructed from summaries. The separate [NB14 derived review](phase3TreadTwoContextReview.md) adds 52 CSVs / 585,825 rows without training or additional independent tyres. The pre-fit wording below documents the original frozen contract.''')
insert('phase3NB12Completion.md','''## Later status — 1 November 2026

NB12 remains completed and preserved. Its proposed NB13 hypothesis has now been trained and independently verified through NB13A: 60 epochs / 2,940 updates, no low-class improvement and worse probability loss. [Measured completion](phase3NB13Completion.md). Current next action is [NB14 CPU review](notebooks/phase3_NB14_TREAD_TWO_CONTEXT_Completed_Review.ipynb), [run guide](tread_two/context_review/README.md). The NB13 unmeasured/next statements below describe the 30 October decision, not today's status.''')
for path in ['tread_two/README.md','tread_two/context_pilot/README.md']:
    prefix='../' if path=='tread_two/README.md' else '../../'
    insert(path,'''## Current — original context fit completed; use NB14 next

**1 November 2026:** NB13A restored HF update 2,486 and completed all 60 epochs / 2,940 updates. All 1,503 direct files / 58,176,066 raw CSV rows verified; owner released. Do not rerun the completed fit. [Completion and evidence]('''+prefix+'''phase3NB13Completion.md).

The context candidate remains unpromoted: low recall 0/62, historical BA 50%, confidence loss worse. Next is **NB14 CPU / Internet / HF_TOKEN / no image attachment**, Run All. [Review guide]('''+('context_review/README.md' if prefix=='../' else '../context_review/README.md')+'''). Existing photos only; workstations and frozen training source unchanged. Earlier run/pending statements below are historical.''')

decisions='''## NB13A closure and saved-context review — 1 November 2026

| ID | Decision | Evidence / boundary |
|---|---|---|
| P3-D110 | Close original NB13/NB13A without another fit | 60 epochs / 2,940 updates, all published bytes/rows/ledger/native states verified, released owner |
| P3-D111 | Record actual recovery at 2,486; preserve dated 1,294 partial audit | Executed bootstrap/output identifies a newer verified HF revision; no invented lost/replayed-update claim |
| P3-D112 | Keep original selected epoch 7 and leave context unpromoted | Validation worsens vs unchanged LC prior; low recall stays 0/62 and endpoint NLL increases |
| P3-D113 | Separate layout sensitivity and conserved explanation from useful recognition/causal truth | Learned correction responds while frozen prior is invariant; zero probe flips do not mean unchanged scores |
| P3-D114 | Deliver isolated CPU NB14 with richer direct CSV/attention review | 52 tables / 585,825 rows, five figures, 1,854 cases, actual extracted science/viewer/offline publication pass |
| P3-D115 | Preserve original fits/workstations; defer any new model to a separately justified frozen experiment | Existing photos only; actual NB14/HF publication remains pending user execution |

[Completion](phase3NB13Completion.md) · [Review and validation](phase3TreadTwoContextReview.md).'''
insert('phase3Decisions.md',decisions)
insert('phase3SessionLog.md','''## NB13A completed verification and next review — 1 November 2026

User reported NB13A complete and requested HF verification, documentation and continuation. Preserved supplied executed notebook SHA `8bdf46ec…`, both complete code cells and all 35 matching payload entries. Public-read pinned audit at `a4441884…` verified all 1,502 manifest objects (5,645,673,072 bytes), all 27 raw CSV schemas/counts (58,176,066 rows), 3,913 history units and 45,852 CSV chunks. Native final/full order/cursors finite; all 60 epoch identities and original selected LC prior bit equality checked. All updates 0–2,940 retained, original partial-history prefix cryptographically linked, owner released.

Actual executed bootstrap restored newer HF update 2,486, then completed the original experiment at 2,940. Earlier update 1,294 remains historical partial evidence; no current loss/replay count is inferred from it. Actual dual-T4/live-HF gate reused. Exact earlier freeze cause still unproven.

Independent saved-science analysis consumed 876 cached inputs without network refetch, zero forwards or training: 52 derived CSVs / 585,825 rows. Validation selects epoch7/update343 (BA50%, NLL1.734916); unchanged LC prior validation BA56.25%, NLL0.934841. Historical selected/final BA50%, low0/62 high41/41, NLL2.174813/2.358475. Context reinforces low→mid confidence; candidate remains unpromoted. Frozen geometry retained, no new dense improvements claimed. Full correction/attention/layout/pair/self-exclusion accounting and limitations documented.

Implemented five separate review modules and four pinned peer assets; 44-file self-contained CPU NB14. Added direct 52-table publication, five figures, 1,854-case interactive base/context/combined evidence, 206 full endpoint attention cases with explicit missing probe/training matrices. Actual table hashes/counts/residuals, cache corruption/range handling, viewer JS controls, offline additive publication/lost-response/idempotence/owner release, and fresh extracted notebook science reproduction pass. This local evidence is distinct from pending actual NB14 Kaggle/HF completion.

Updated current project trackers, decision/evidence/inventory/map, repair/protocol/architecture outcome notices and detailed completion/review/run guides. Expanded narrow Git ignores for heavy context review rows/attention and acceptance fixtures; compact summaries/source/curated figures remain visible. No local credential reads, remote writes, fitting, additional photos or workstation changes. Next: NB14 CPU, Internet/HF_TOKEN, no data attachment, Run All; return executed notebook for independent review publication audit.''')
print(json.dumps(dict(status='written',date='2026-11-01',measured_updates=v['optimizer_updates'],tables=len(a['csv_tables']),package=p['status']),indent=2))
