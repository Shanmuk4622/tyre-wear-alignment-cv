"""Write evidence-linked NB05 completion and the CPU NB06 run guide."""
import csv,json,sys
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];PUB=ROOT/'revision2_validation/nb05_publication'

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def table(name):
 with open(PUB/'analysis'/(name+'.csv'),encoding='utf-8',newline='') as f:return list(csv.DictReader(f))

def main():
 inspection=read(PUB/'NOTEBOOK_INSPECTION.json');science=read(PUB/'SCIENTIFIC_VERIFICATION.json');preflight=read(PUB/'PREFLIGHT_VERIFICATION.json');download=read(PUB/'DOWNLOAD.json')
 assert science['status']==preflight['status']==download['status']=='passed'
 result=read(PUB/'NOTEBOOK_COMPLETION.json');endpoints=table('endpoint_metrics');validation=table('validation_trajectory');classes=table('endpoint_class_metrics');resource=table('resource_timing_summary')
 lines=['# NB05 completed — paired TREAD-ONE result and independent audit','',
 '**29 October 2026 — Asia/Calcutta.** Both matched seed-1 conditions completed 60 epochs and 2,940 optimizer updates each on the two Kaggle T4s. Both actual live preflights passed. The independent audit passes. **The projected candidate does not improve the declared selected development endpoint and is not supported for promotion.**','',
 'The next step has already been implemented locally: a full saved-evidence review, ten derived CSV tables, scientific figures, and an interactive atlas. [NB06 CPU review/publication instructions](phase3NB06Review.md). Independent tyre collection remains the next data gate; [empty intake templates](data_intake/README.md) are ready for the user’s later help. No additional model was trained here.','',
 '## What completed, and what the result means','',
 '| Gate | State | Evidence |','|---|---|---|',
 '| Notebook execution | Three code cells, counts 1/2/3; zero saved errors | [Archived execution](revision2_validation/nb05_publication/executed.ipynb), [inspection](revision2_validation/nb05_publication/NOTEBOOK_INSPECTION.json) |',
 '| Measured pair | Both 60 epochs / 2,940 updates complete | Native states, 60 histories/condition and update-by-update CSV |',
 '| Direct HF retention | **3,515 published logical files**, including two manifests; zero ZIP/NPZ/database uploads | [Download/hash receipt](revision2_validation/nb05_publication/DOWNLOAD.json) |',
 '| Scientific recalculation | **1,561,734 core rows**, both selections, all 412 native endpoint maps | [Independent scientific checks](revision2_validation/nb05_publication/SCIENTIFIC_VERIFICATION.json) |',
 '| Live recovery | Both update-1 pinned sources continue to identical update-4 states; 20 scientific CSV families equal | [Independent preflight verification](revision2_validation/nb05_publication/PREFLIGHT_VERIFICATION.json) |',
 '| Model quality / promotion | Candidate fails this paired selected-endpoint comparison; installed app unchanged | Endpoint / class / tyre tables below |',
 '| New subjects | Zero; richer telemetry does not add physical tyres | Original 12-tyre dataset, 8/2/2 train/validation/test grouping |','',
 '```mermaid','flowchart LR','    A["NB05 complete<br/>both 60-epoch conditions"] --> B["PASS<br/>pinned files / native recovery / CSV recalculation"]','    B --> C["Selected candidate regresses<br/>classification + point error"]','    C --> D["Completed locally<br/>paired failure atlas / calibration / trajectory review"]','    D --> E["NB06 CPU<br/>publish direct review evidence"]','    D --> F["Prospective independent tyres<br/>user-assisted capture / labels / fixed protocol"]','```','',
 '## Declared selected and final development-test endpoints','',
 'These are **103 existing photographs / two physical tyres**, low/high only. Twenty-five images have human point labels. The mid class has zero target support, although predictions may choose mid. All endpoints were already part of the frozen protocol; the test did not choose checkpoints.','',
 '| Condition | Endpoint | Epoch | Accuracy | Balanced accuracy, observed classes | Mean Dice | Human-point MAE, % width | NLL |','|---|---|---:|---:|---:|---:|---:|---:|']
 for r in endpoints:
  lines.append(f"| {r['condition']} | {r['endpoint']} | {r['epoch']} | {100*float(r['accuracy']):.3f}% | {100*float(r['balanced_accuracy_observed']):.3f}% | {100*float(r['mean_two_region_dice']):.3f}% | {100*float(r['human_point_mae']):.3f}% | {float(r['nll']):.4f} |")
 lines += ['', '![Declared endpoint comparison](revision2_validation/nb05_publication/analysis/figures/NB05_endpoint_comparison.png)','',
 'Selected balanced accuracy drops **32.927 percentage points**, from 50% to 17.073%. Mean Dice changes by only −0.070 percentage points, while human-point error rises from 1.067% to 1.885% width, about **76.6% higher**. At epoch 60 both classifiers reach the same 50% balanced accuracy, and projection still has higher point error and NLL. The final endpoint is not substituted for the selected one. Neither endpoint resolves the low-tyre failure.','',
 '## The validation gain did not transfer','',
 '| Condition | Selected epoch | Validation BA | Validation Dice | Validation human-point MAE | Frozen joint utility |','|---|---:|---:|---:|---:|---:|']
 for condition in ('baseline','shared_projection'):
  e=result['study']['selected'][condition]['epoch'];r=next(x for x in validation if x['condition']==condition and int(x['epoch'])==e)
  lines.append(f"| {condition} | {e} | {100*float(r['balanced_accuracy_observed']):.3f}% | {100*float(r['mean_two_region_dice']):.3f}% | {100*float(r['human_point_mae']):.3f}% width | {float(r['joint_validation_utility']):.9f} |")
 lines += ['', 'The original utility selected epoch 23 for baseline and epoch 17 for projection. Projection’s 83.75% validation BA looks better than baseline’s 55%, but its selected test BA is much worse. Its validation NLL/Brier also do not improve. The full 60-epoch curve shows unstable class transfer alongside consistently strong segmentation. This supports caution about validation representativeness and selection stability in a very small tyre cohort; it does **not** isolate a causal shortcut or prove that all gradient-projection methods fail.','',
 '![Full validation trajectories](revision2_validation/nb05_publication/analysis/figures/NB05_validation_trajectories.png)','',
 '## Class and tyre failures are explicit','',
 '| Selected endpoint | True class | Support | Predicted as this class | Correct | Recall |','|---|---|---:|---:|---:|---:|']
 for r in classes:
  if r['endpoint']!='selected':continue
  lines.append(f"| {r['condition']} | {['low','mid','high'][int(r['class_index'])]} | {r['support']} | {r['predicted']} | {r['tp']} | {100*float(r['recall']):.3f}% |" if r['recall'] else f"| {r['condition']} | mid | {r['support']} | {r['predicted']} | {r['tp']} | unavailable: no target support |")
 lines += ['', 'Both selected models predict **zero low** cases and miss all 62 low-class photos. Baseline correctly recognizes all 41 high-class photos. Projection recognizes 14/41 high photos and predicts mid for 89/103 images overall. Mid predictions on this cohort are errors; absence of mid ground truth is not a reason to remove them from the confusion matrix.','',
 'Per-tyre records and paired-image deltas are retained directly: [tyre metrics](revision2_validation/nb05_publication/analysis/endpoint_tyre_metrics.csv), [class metrics](revision2_validation/nb05_publication/analysis/endpoint_class_metrics.csv), [paired image deltas](revision2_validation/nb05_publication/analysis/paired_image_deltas.csv). No photo-level confidence interval or p-value is used to pretend that 103 correlated photos are 103 independent tyres.','',
 '## What the mechanism actually did','',
 '| Condition | Task-gradient observations | Negative shared dot products | Projection applied | Head-gradient change at routing |','|---|---:|---:|---:|---:|']
 for c in science['cases']:lines.append(f"| {c['condition']} | 2,940 | {c['gradient_conflicts']} ({100*c['gradient_conflicts']/2940:.3f}%) | {c['projections']} | zero |")
 lines += ['', 'The two trajectories diverge after intervention. Their later conflict counts therefore measure different evolving models; they are not paired samples of the same gradient vector. Projection acted on 91.84% of candidate updates, so this was a frequent change to the optimization path. A zero head-gradient change at routing does not preserve head weight updates: the subsequent **global clipping coefficient** can differ, as can the shared representation. Dense quality is not guaranteed by routing only shared gradients.','',
 '![Training and gradient observations](revision2_validation/nb05_publication/analysis/figures/NB05_gradient_and_training.png)','',
 '## Exact provenance and audit boundaries','',
 f"Executed NB05: **{inspection['bytes']:,} bytes**, SHA-256 `{inspection['sha256']}`. The original clean delivery and earlier failed execution remain separately preserved. [Original repair history](phase3NB05Repair.md). No NB05 rebuild or repeat training is required.",'',
 '| Condition | Final pinned HF revision | Runtime contract |','|---|---|---|']
 for receipt,c in zip(result['publications'],science['cases']):lines.append(f"| {c['condition']} | `{receipt['revision']}` | `{c['contract']}` |")
 lines += ['', f"Actual Kaggle paired initialization hash: `{science['paired_initialization_sha256']}`. Both conditions match each other in initialization, every scheduled sample/augmentation identity, and captured native input/target arrays. This actual Tensor hash differs from the earlier local GTX1650 fixture hash; identical seeds across different software environments are not a guarantee of identical initialized tensors. The actual T4 baseline original-math preflight is independently exact.",'',
 'All 27 embedded experiment files equal the readable frozen sources. Both final native states are at phase completed / epoch 60 / update 2,940. The 3,628-unit ledger per condition has matching native tail/cursor anchors. CSV schemas/counts match native row cursors. All 124 cohort summaries across validation/endpoints are independently recalculated. Every selected/final native mask map agrees with saved region counts and prepared source masks; signed evidence reconstructs logits with maximum residual below 4.8×10⁻⁷. Loss sufficient-statistic reconstruction differs by less than 3.6×10⁻⁷.','',
 'A numerical audit initially used an overly strict scalar point tolerance. The frozen NumPy-2 per-point subtraction rounds in FP32, while annotation MAE uses an explicit FP64 target array; observed maximum scalar discrepancy is 2.95×10⁻⁸ width. The audit now checks the appropriate precision and recomputes human MAE from saved prediction/target coordinates. No training data, stored metric, selection rule or checkpoint was changed. The separate frozen Torch selection field remains preserved.','',
 'The independent live preflight check downloads both pinned update-1 sources, original/reference and restored update-4 native states, and compares 20 scientific CSV families. The original baseline update matches frozen math. It does not prove a forced whole-Kaggle-container restart during the 60-epoch run. Raw SQLite is intentionally local-only; its complete SQL hash chain is not independently replayed from CSV strings. Published hashes, native tail anchors and all row cursors are checked.','',
 '## Retained evidence and measured cost','',
 '| Scope | Both conditions combined |','|---|---:|',
 '| Completed optimizer updates | 5,880 |','| Augmented training occurrences | 47,040 |','| Validation-image observations | 9,720 |','| Selected/final development-test predictions | 412 |','| Core CSV records, 23 families | 1,561,734 |','| Declared selected/final native image maps checked | 412 |','| Independent additional tyres | 0 |','',
 'There are 1,325 CSV files per condition including auxiliary files; 23 denotes schema families, not files. The compatibility validation_predictions family stays header-only; real validation is in eval_predictions and companions. Full continuous power/energy, every validation pixel map at every epoch and every historical gradient tensor were not collected.','',
 '| Condition | Reported worker-active minutes | Retained process intervals | Sampled CUDA peak | Minimum sampled free disk |','|---|---:|---:|---:|---:|']
 for r in resource:lines.append(f"| {r['condition']} | {float(r['reported_worker_active_seconds'])/60:.2f} | {r['worker_processes_started']} | {float(r['peak_sampled_cuda_bytes'])/1024**3:.3f} GiB | {float(r['minimum_sampled_free_disk_bytes'])/1024**3:.2f} GiB |")
 lines += ['', 'These are retained main-process intervals and resource samples, not complete notebook wall time, continuous peak process usage, energy or workstation latency. The conditions ran concurrently; summing their active minutes does not give elapsed notebook time. Publication waiting is separate.','',
 '## Next step already carried out locally','',
 'The [interactive comparison explorer](revision2_validation/nb05_publication/analysis/TREAD_ONE_NB05_Comparison_Explorer.html) contains **all 103 test source images and 412 endpoint records**, with selectable condition/endpoint, wrong-case filters, predicted/labelled overlapping masks, six predicted/human points and signed evidence for all three classes. It is self-contained, about 19.2 MB. Heat opacity is scaled per image/class and numerical maxima are shown. Intrinsic additive evidence does not establish physical or input-level causality.','',
 'Ten directly readable derived CSV tables record endpoint/class/tyre metrics, validation trajectories and stability, actual training gradient/clipping history, calibration bins, descriptive confidence-risk curves and paired image deltas. Confidence thresholds are descriptive post-hoc diagnostics, not a calibrated abstention/deployment policy.','',
 'The [NB06 CPU notebook](notebooks/phase3_NB06_TREAD_ONE_Paired_Review_And_Failure_Atlas.ipynb) reproduces this audit and publishes the review directly to a separate HF namespace using the user’s Kaggle secret. It trains no model and runs no new inference. Its local/fresh-CPU checks and publication boundary are described in the [run guide](phase3NB06Review.md). New review publication is prepared, not yet live-published by this audit.','',
 'The next **data** action is independent tyre acquisition with capture/label provenance, including the missing held-out mid-class/domain coverage if those claims are intended. The [intake pack](data_intake/README.md) contains empty templates only. Do not replace the installed models, pick another checkpoint using these test scores, or add more seeds just to seek a favorable score. A later architecture/training change needs a declared hypothesis and evaluation protocol.']
 (ROOT/'phase3NB05Completion.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 delivery=read(ROOT/'revision2_validation/nb06_delivery/DELIVERY.json');bundle=read(PUB/'REVIEW_DELIVERY.json')
 guide=f'''# NB06 — CPU review, failure atlas and direct HF publication

**29 October 2026.** This is the next step within TREAD-ONE after the
[completed NB05 audit](phase3NB05Completion.md). No further model fit or forward
inference is needed: the source training predictions, validation trajectories,
task gradients, native endpoint maps and recovery state were already retained.

## State and purpose

| Item | Current evidence |
|---|---|
| NB05 | Complete on both T4s; 3,515 pinned files / 1,561,734 core rows independently checked |
| Candidate | Selected test BA 17.073% vs baseline 50%; point error 1.885% vs 1.067% width |
| New local review | Completed; ten derived CSV tables, three static figures, complete 103-image/four-endpoint explorer |
| Prepared HF review | {bundle['files']} direct files / 10 CSV tables; **not yet published live** |
| NB06 | {delivery['notebook_bytes']:,} bytes; 36 embedded files; CPU-only reproducible review/publication |
| NB06 SHA-256 | `{delivery['notebook_sha256']}` |
| New training/inference | Zero optimizer updates / zero model forward passes |
| New independent tyres | Zero; user-assisted collection remains later |

Notebook: [phase3_NB06_TREAD_ONE_Paired_Review_And_Failure_Atlas.ipynb](notebooks/phase3_NB06_TREAD_ONE_Paired_Review_And_Failure_Atlas.ipynb).
Open the already generated [interactive explorer](revision2_validation/nb05_publication/analysis/TREAD_ONE_NB05_Comparison_Explorer.html)
or [endpoint table](revision2_validation/nb05_publication/analysis/endpoint_metrics.csv)
locally now. NB06 provides the portable Kaggle reproduction and authorized HF write.

## Kaggle instructions

1. Accelerator **None / CPU**; Internet **ON**. No T4 allocation is required.
2. Attach exactly one expanded `shanmuk4622/tire-dataset-prepared-phase2` dataset.
3. Keep at least **6 GiB free**. Final source evidence is about 2.05 GB; cached
   hard-link appearances count more logically. Smaller preflight fixtures and
   the approximately 19.2 MB self-contained review add overhead. Input images
   remain read-only. Do not assume a 1 TB scratch volume exists.
4. Enable **HF_TOKEN** from Kaggle Secrets with write access to
   `Shanmuk4622/tyre-wear-study`. It is read only by the publication cell.
5. **Run All**. No package install, upgrade/downgrade or model training.

| Stage | What happens | Restart behavior |
|---|---|---|
| Setup | Extract exact source payload and preserved executed NB05 | Local session lock prevents duplicate setup/steps |
| Retrieve | Pinned manifests, all published CSV/native objects, SHA-256 and Git/LFS checks | Reuse complete verified objects; an incomplete object is fetched again |
| Recovery/science | Both native preflights, full CSV recalculation and raw mask/evidence comparison | Read-only recomputation; no optimizer or forward inference |
| Review | Ten derived CSVs, three figures and all-image HTML explorer | Derived files are recomputed from fixed sources |
| Publish | Additive direct-file immutable snapshot, ownership/CAS, verify then select manifest | Retry same cell, deduplicate unchanged objects; catchable Stop/error attempts immediate flush |

The read stages explicitly access public revisions without alternate credential
files. Publication uses the explicit Kaggle secret, the frozen bounded publisher
and a separate review namespace:
`phase3/tread-one/r2/review/paired-seed1/<analysis-contract>`.
The contract pins source publications, analysis source hashes and the actual CPU
software environment; its hash can differ between the local and Kaggle runtime.

The local prepared contract is `{bundle['contract']}`. It is an analysis identity,
not a new training run or replacement for the original measured study identity.
Training files remain in their original namespaces. CSV, JSON, PNG, HTML and
readable analysis sources are published individually; no ZIP/NPZ/database or
model-weight upload is performed. The frozen training implementations remain
available at their original pinned publications.

## Evidence dictionary for the ten post-hoc tables

| CSV | Rows | Meaning / units |
|---|---:|---|
| endpoint_metrics | 4 | Declared selected/final metrics; accuracy/Dice and error/width are fractions |
| validation_trajectory | 120 | All 60 cohort validation records per condition; selection utility is unchanged |
| epoch_training_gradient | 120 | Augmented training accuracy, conflict/projection counts, cosine, clipping and measured seconds |
| endpoint_class_metrics | 12 | Support/predicted/correct counts and recall/precision; absent support is blank |
| endpoint_tyre_metrics | 8 | One row per condition/endpoint/test tyre; actual subject grouping |
| calibration_reliability | 40 | Ten confidence bins × four endpoints; counts/correct/confidence sums |
| selective_risk | 84 | Fixed 0–1 confidence thresholds; coverage/error on this inspected cohort only |
| validation_prediction_stability | 162 | Sixty-epoch correct counts and consecutive prediction flips per validation image |
| paired_image_deltas | 206 | Per-image condition deltas for selected/final endpoints; NLL/Dice/point error |
| resource_timing_summary | 2 | Retained worker-active seconds, sampled memory bytes and free disk; incomplete-runtime limits explicit |

Repeated image/epoch/point rows are not independent tyres. Blank means
unavailable/not applicable. No missing mid target examples, continuous energy,
new human labels or historical unlogged measurements are invented.

## Validation and limits

Local actual pinned-file/native/CSV audits passed. The fresh-workspace CPU test
uses the **actual embedded NB06 payload**, disables CUDA/network, supplies the
existing prepared dataset explicitly in its harness and reuses verified pinned
objects. It recomputes the audit, all ten CSVs, figures and explorer; resulting
review artifacts match the local originals. A fake HF service exercises the
35-file publication/ambiguous-response path with no real write. See
[NB06 validation](revision2_validation/nb06_validation/VALIDATION.json).
These checks do not claim a completed Kaggle NB06 execution or live publication.
The real preflight read audit is a separate passed receipt.

Static scientific figures were visually inspected. Explorer JavaScript syntax,
all 1,854 embedded image blocks and expected shapes are checked. Browser layout
and interaction are not claimed as visually verified. Download/open the HTML
for local offline review; it needs no external scripts or image server.

On Stop, keep the session open for safe cleanup/backoff. Forced kernel death
cannot execute a flush. An unexpired owner remains protected; stop any older
unused writer cleanly or wait for its lease. Do not delete OWNER.json. Preserve
the executed NB06, printed receipt and logs after completion.

## After the review

Use the [empty intake pack](data_intake/README.md) with the user's later help to
register genuinely independent tyres, sessions and annotation provenance before
a prospective tyre-disjoint protocol. The current projected candidate is not
installed or promoted. A later model design is a separate motivated decision;
this notebook does not launch another seed or search for a favorable test result.
'''
 (ROOT/'phase3NB06Review.md').write_text(guide,encoding='utf-8')
 print('Wrote structured NB05 completion and NB06 CPU run guide')

if __name__=='__main__':main()
