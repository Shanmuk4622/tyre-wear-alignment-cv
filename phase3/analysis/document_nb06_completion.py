"""Record NB06's verified execution/publication and the next data decision."""
import json,sys
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'revision2_validation/nb06_publication'

def write(name,text):(ROOT/name).write_text(text,encoding='utf-8')
def update(name,old,new):
 p=ROOT/name;text=p.read_text(encoding='utf-8');assert old in text,(name,old[:90]);p.write_text(text.replace(old,new,1),encoding='utf-8')
def prepend(name,text):
 p=ROOT/name;content=p.read_text(encoding='utf-8');first,rest=content.split('\n',1);p.write_text(first+'\n\n'+text.strip()+'\n'+rest,encoding='utf-8')

def main():
 r=json.loads((OUT/'VERIFICATION.json').read_text(encoding='utf-8'));assert r['status']=='passed'
 contract=json.loads((OUT/'remote/CONTRACT.json').read_text(encoding='utf-8'));url=f'https://huggingface.co/datasets/Shanmuk4622/tyre-wear-study/tree/{r["revision"]}/{r["destination"]}'
 report=f'''# NB06 complete — published TREAD-ONE review and next milestone

**29 October 2026 — Asia/Calcutta.** All six code cells completed successfully.
The independent read-only audit verifies **35 published files**, including
**10 directly readable CSV tables / 758 derived rows**, three figures and the
complete interactive image atlas. The HF writer is released at the pinned
publication. **No NB06 rerun is needed.**

[Open the published review]({url}) or the cached
[published explorer](revision2_validation/nb06_publication/remote/TREAD_ONE_NB05_Comparison_Explorer.html).
The completed notebook is preserved exactly; no training source, checkpoint,
test selection or installed model was changed.

## Completed workflow and evidence

| Gate | Verified result | Evidence |
|---|---|---|
| Notebook | Execution counts 1–6; zero saved errors | [Exact archive](revision2_validation/nb06_publication/executed.ipynb) |
| Embedded input/source files | All 36 equal the accepted delivery/readable files | [Independent verifier](analysis/verify_nb06_publication.py) |
| HF publication | 35 logical files including selecting manifest; all SHA-256 and Git/LFS identities checked | [VERIFICATION.json](revision2_validation/nb06_publication/VERIFICATION.json) |
| Direct tables | 10 tables / 758 rows; headers/order/counts and all numeric values agree with the local audit | [Publication receipt](revision2_validation/nb06_publication/PUBLICATION.json) |
| Ownership | Released at the pinned revision; no forced takeover | Published OWNER.json checked independently |
| Original measured evidence | NB05's pinned publications, 3,515 files and 1,561,734 core rows rechecked by the executed review | Published verification/ and source_metadata/ records |
| Image atlas | 103 source test images, 412 endpoint records, 1,854 image blocks, valid JS syntax | Cached published HTML; mask/heat decoded pixels match local exactly |
| New model/data | Zero optimizer updates, zero model forward passes, zero new independent tyres | Review scope and frozen original inputs |

```mermaid
flowchart LR
    A["NB05 complete<br/>two matched 60-epoch conditions"] --> B["NB06 complete on Kaggle CPU<br/>recheck / derive / visualize"]
    B --> C["HF direct review verified<br/>35 files / 10 CSV tables"]
    C --> D["Next: data + protocol gate<br/>additional independent tyres"]
    D --> E["Later: one motivated custom revision<br/>classification transfer + dense-task preservation"]
```

## Exact publication identity

| Item | Identity |
|---|---|
| Repository | Shanmuk4622/tyre-wear-study |
| Pinned revision | `{r['revision']}` |
| Review contract | `{r['contract']}` |
| Namespace | `{r['destination']}` |
| Executed notebook | {r['executed_notebook_bytes']:,} bytes |
| Executed SHA-256 | `{r['executed_notebook_sha256']}` |
| Clean delivery SHA-256, historical | `6817b3a5d53ae42ca2430d29cfd2d78083e75475b53c008fa41a100f58cd718c` |
| Public-read audit remote writes | 0 |

The Kaggle review contract differs from the earlier local prepared contract
because it includes the actual software environment. The published run used
CPU, Python 3.13.15, Torch 2.11.0+cu128, NumPy 2.1.3 and Pillow 13.0.0. A CUDA
build being installed does not mean this review ran GPU inference. The six
analysis-source hashes and original NB05 publication/checkpoint identities
agree exactly with the accepted source payload.

## Published table inventory

| Direct CSV | Rows | Main purpose |
|---|---:|---|
| endpoint_metrics | 4 | Declared selected/final condition endpoints |
| validation_trajectory | 120 | All 60 validation epochs per condition |
| epoch_training_gradient | 120 | Sampled training accuracy, objectives, conflicts, projection, clipping and timings |
| endpoint_class_metrics | 12 | Class support/predictions/recall/precision |
| endpoint_tyre_metrics | 8 | Physical test-tyre outcomes per endpoint |
| calibration_reliability | 40 | Ten confidence-bin sufficient-statistic rows per endpoint |
| selective_risk | 84 | Descriptive fixed-threshold coverage/error; no fitted deployment rule |
| validation_prediction_stability | 162 | Per-image 60-epoch correctness and prediction flips |
| paired_image_deltas | 206 | Per-image candidate/baseline endpoint deltas |
| resource_timing_summary | 2 | Retained worker-active intervals and sampled resource limits |

Probabilities/accuracy/Dice and error divided by image width are fractions;
durations are seconds and memory is bytes. Blank means unavailable/not
applicable. Repeated images/epochs/points and the 758 derived rows are not new
independent subjects. Raw measured NB05 tables remain in their original pinned
namespaces; this additive review does not replace them.

## Model insight remains the same

NB06 publishes and verifies evidence; it does not train an improved model.
The original validation selections remain fixed:

| Selected endpoint | Epoch | Test balanced accuracy | Mean two-region Dice | Human-point error (% width) |
|---|---:|---:|---:|---:|
| Baseline | 23 | 50.000% | 97.852% | 1.067% |
| Shared projection | 17 | 17.073% | 97.782% | 1.885% |

Both selected models miss all 62 low-class test photos. Projection predicts mid
on 89/103 images and correctly recognizes only 14/41 high-class photos. Its
83.75% selected validation BA did not transfer. This one paired development
result does not establish universal optimizer behavior or a causal shortcut.
The test cohort remains two already inspected tyres, low/high only; human point
truth is available on 25 photos. [Full NB05 interpretation](phase3NB05Completion.md).

![Published endpoint comparison](revision2_validation/nb06_publication/remote/figures/NB05_endpoint_comparison.png)

## Published visuals and audit limits

The published HTML is **{r['remote_explorer_bytes']:,} bytes**, with condition,
selected/final endpoint, wrong-case and human-label filters, masks, points,
signed class evidence and the full validation trajectory. Heat color scale is
per image/class and the numerical maximum is shown. Intrinsic additive
accounting is not physical wear/depth/safety or input-level causal validation.

Its byte size differs from the locally rendered HTML because the rendering
environments differ. All ten CSV tables match numerically with maximum
difference **{r['max_numeric_difference_from_local']:.1f}**, and corresponding decoded predicted-mask and
signed-heat pixels match exactly. All 1,854 embedded blocks have the expected
image shapes. JavaScript syntax passes. The published static figure was
inspected; browser interaction/layout has not been visually verified.

The original study's limitations remain: one paired seed, few independent
tyres, no held-out mid cohort, local-only SQL history, bounded rather than
continuous resources/maps, and no new physical measurements. An audit receipt
does not turn the inspected development cohort into an independent blind test.

## What is next

The next scientific milestone is the
[classification generalization/data-and-protocol gate](phase3ProspectiveDataGate.md):

1. Reconcile physical tyre/parent-video identity and mileage-label provenance.
2. With the user's planned later help, register additional independent tyres
   and varied sessions, including low/mid/high coverage for intended claims.
3. Freeze a prospective tyre-disjoint evaluation protocol before inspecting
   reserved-cohort model outputs; retain native sources and direct HF metadata.
4. Use that evidence to choose one focused custom-model revision with explicit
   segmentation/geometry preservation criteria.

[Empty intake templates](data_intake/README.md) are already prepared. No new
tyre count, acquisition budget, training date or r3 architecture is claimed or
frozen here. Current installed models remain unchanged. No additional model or
review notebook was launched by this documentation update.
'''
 write('phase3NB06Completion.md',report)
 # Current entry points: preserve historical design/run receipts below the latest state.
 p=ROOT/'README.md';text=p.read_text(encoding='utf-8');start=text.index('**Latest');end=text.index('**Current — 28',start)
 text=text[:start]+'''**Latest — 29 October 2026: NB06 is complete and independently HF-verified.**
All six cells passed; 35 direct files include ten CSV tables / 758 derived rows,
three figures and the complete image atlas. [Completion and publication](phase3NB06Completion.md).
The NB05 selected projection result remains worse than baseline; no model was
trained or promoted by NB06. Next is the
[independent-tyre data/protocol gate](phase3ProspectiveDataGate.md), with
[intake templates](data_intake/README.md) ready for the user's later help.

'''+text[end:]
 text=text.replace('| Completed-pair review / NB06 | **Local review complete; CPU notebook and direct publication prepared** | [Run guide](phase3NB06Review.md), [interactive atlas](revision2_validation/nb05_publication/analysis/TREAD_ONE_NB05_Comparison_Explorer.html) |',
  '| Completed-pair review / NB06 | **Complete on Kaggle / independently HF-verified** | [Completion](phase3NB06Completion.md); 35 direct files / ten CSV tables |')
 text=text.replace('E --> F["NB06 CPU review + failure atlas<br/>then independent tyre coverage"]','E --> F["NB06 review + publication complete<br/>next: independent tyre data / protocol gate"]')
 text=text.replace('Next is [NB06](phase3NB06Review.md), a CPU-only saved-evidence review and direct HF publication; no further model training.', '[NB06](phase3NB06Completion.md) has now completed that CPU-only review and direct HF publication. The next milestone is the [prospective data/protocol gate](phase3ProspectiveDataGate.md).')
 write('README.md',text)
 update('phase3Understanding.md','The next [NB06 CPU review/publication](phase3NB06Review.md) is prepared; no extra\nmodel training or installation.', '[NB06](phase3NB06Completion.md) is now Kaggle-complete and independently\nHF-verified: 35 direct files / ten CSV tables, with unchanged measured model\noutcomes. Next is the [data/protocol gate](phase3ProspectiveDataGate.md); no\nextra model training or installation.')
 update('phase3Progress.md','[NB06 CPU review/publication](phase3NB06Review.md) are ready. Its new live review\npublication remains pending; no further model fit is launched.', '[NB06 review/publication](phase3NB06Completion.md) is now complete and\nindependently verified: 35 direct files / ten CSV tables. Next is the\n[data/protocol gate](phase3ProspectiveDataGate.md); no further fit is launched.')
 update('phase3Progress.md','**Current state: r1 training and NB02–NB05 Kaggle execution/publication are complete and independently verified.**', '**Current state: r1 training and NB02–NB06 Kaggle execution/publication are complete and independently verified.**')
 update('phase3Progress.md','Next is a CPU saved-evidence review/publication and independent tyre coverage.', 'The review/publication is complete; next is independent tyre coverage and a prospective protocol.')
 update('phase3Progress.md','| NB06 Kaggle/direct review publication | Prepared; live execution pending | CPU-only notebook; [guide](phase3NB06Review.md); no new training/inference |', '| NB06 Kaggle/direct review publication | **Complete / independently verified** | Six successful cells, 35 hashed files, ten CSV tables / 758 rows; [completion](phase3NB06Completion.md) |')
 prepend('phase3NB06Review.md','''**Completion — 29 October 2026:** all six cells and the 35-file HF publication
are independently verified. [Completion, provenance and findings](phase3NB06Completion.md).
**No NB06 rerun is needed.** The clean-delivery identity and run procedure below
are retained as history. The next milestone is the [data/protocol gate](phase3ProspectiveDataGate.md).''')
 update('phase3NB06Review.md','| Prepared HF review | 35 direct files / 10 CSV tables; **not yet published live** |', '| Published HF review | **35 direct files / ten CSV tables, independently verified**; [receipt](revision2_validation/nb06_publication/PUBLICATION.json) |')
 update('phase3NB06Review.md','| NB06 | 516,785 bytes; 36 embedded files; CPU-only reproducible review/publication |', '| Original clean NB06 delivery | 516,785 bytes; 36 embedded files; historical source-package identity |')
 update('phase3NB06Review.md','| NB06 SHA-256 |', '| Original clean-delivery SHA-256 |')
 update('phase3NB06Review.md','These checks do not claim a completed Kaggle NB06 execution or live publication.', 'Those were the local acceptance boundaries. The later [NB06 completion audit](phase3NB06Completion.md) now independently verifies actual Kaggle execution and live publication.')
 update('phase3NextSteps.md','| Reproduce/publish the review directly to HF | NB06 prepared; live publication pending | [CPU notebook and instructions](phase3NB06Review.md); no new training/inference |', '| Reproduce/publish the review directly to HF | **Complete / independently verified** | [NB06 completion](phase3NB06Completion.md); 35 files / ten CSV tables |')
 prepend('phase3NextSteps.md','''**Latest milestone:** [NB06](phase3NB06Completion.md) completed and published
the review at a pinned verified HF revision. The active next work is the
[classification generalization/data-and-protocol gate](phase3ProspectiveDataGate.md).
No further model fit or notebook execution is scheduled by this update.''')
 update('phase3OpenItems.md','[NB06 CPU review/publication](phase3NB06Review.md) is prepared; its live\npublication and independent user-assisted tyre collection remain open.', '[NB06](phase3NB06Completion.md) has completed and is independently HF-verified.\nThe [prospective data/protocol gate](phase3ProspectiveDataGate.md) and independent\nuser-assisted tyre collection remain open.')
 update('phase3OpenItems.md','| NB06 review publication | Prepared CPU-only notebook; actual Kaggle execution/live review publication pending |', '''| NB06 review publication | **Complete / independently HF-verified:** six cells, 35 files and ten CSV tables |
| Next classification milestone | [Data/protocol gate](phase3ProspectiveDataGate.md); no new subjects collected or r3 training started |''')
 update('phase3OpenItems.md','[implementation/guide](phase3R2PairedPilot.md). No measured result or improvement claim yet', '[completion](phase3NB05Completion.md). Measured selected candidate regresses; no promotion')
 update('phase3DataGrowth.md','The [CPU review](phase3NB06Review.md) is the immediate next portable step.', 'The [CPU review/publication](phase3NB06Completion.md) is complete; the active next\nstep is the [prospective data/protocol gate](phase3ProspectiveDataGate.md).')
 update('phase3DataRetentionPolicy.md','The ten post-hoc review tables and interactive image atlas are prepared for\nadditive publication through [NB06](phase3NB06Review.md); that new review\npublication is pending.', 'The ten post-hoc review tables and interactive atlas are now published and\nindependently verified through [NB06](phase3NB06Completion.md): 35 direct files,\nincluding 758 derived CSV rows.')
 print('Updated current completion and project state; execution/source files preserved.')

if __name__=='__main__':main()
