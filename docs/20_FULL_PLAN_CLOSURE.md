# Full experimental-plan reconciliation — 2026-09-09

**The full plan is not complete.** Notebook numbers are not experimental stage
numbers. NB10 finishing does not complete S5, S9, or every proposed experiment.
All 27 Markdown files present before this reconciliation were read in full.
Historical logs retain their dated claims; this ledger and PROGRESS are the
current completion record.

## Evidence and corrections

Latest recovery verification: HF revision `bf62f9e9cbedacc580aa42542da14a068b8f9215`.
NB01A, NB01B, NB03A and NB10R are executed and verified; see
`21_RECOVERY_COMPLETION_AUDIT.md`. Earlier baseline reconciliation used
`7c5b6461815a78aae589085d0152eaa6ae9995e1`.
[Public study](https://huggingface.co/datasets/Shanmuk4622/tyre-wear-study).
No remote records were modified in this repair.

### S1 was underreported, not unexecuted

`tables/baselines.csv` contains all five legacy rows. All three
`s1-resnet18-randinit-f{0,1,2}-s1` statuses are completed at epoch 15.

| Legacy result | Fold 0 | Fold 1 | Fold 2 | Mean |
|---|---:|---:|---:|---:|
| Colour macro-F1, current HF | .886959 | .464052 | .123216 | .491409 |
| Structure macro-F1 | .354447 | .119102 | .976375 | .483308 |
| HOG + SVM macro-F1 | .796021 | .181818 | .984118 | .653986 |
| Majority accuracy | not saved by fold | not saved by fold | not saved by fold | .409735 |
| Legacy random-init **final** macro-F1 | 1.000000 | .464052 | 1.000000 | .821351 |
| Legacy random-init selected macro-F1 | 1.000000 | .705742 | 1.000000 | .901914 |

Earlier colour per-fold numbers (.952/.399/.123) describe the earlier local
probe, not this HF table. Majority accuracy is not macro-F1. HOG uses the full
frame, not the related paper's segmented-tread pipeline. None of these scores
establishes leakage-free generalisation; folds 0 and 2 remain flagged.

The legacy random-init job used **ResNet-18, 224px, 15 epochs, one seed**.
The plan specifies a matched ResNet-50 transfer comparison. NB01B supplies nine
new **ResNet-50, 384px, 60-epoch** jobs under new IDs; it does not pretend the
old experiment satisfied that design.

### S2 cannot be fixed by rerunning an invalid weight identifier

153 valid runs cover 17 architectures. The nine `convnextv2_s` records remain
quarantined as mislabeled ResNet-18 executions. The current
[official timm registry](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/convnext.py)
defines `convnextv2_small.untrained`, not the requested pretrained tag.

No replacement or reduced model has been silently selected. Options are to
retain the documented 17-architecture pretrained sweep, add real Small from
scratch as a **separate** arm, or explicitly choose another pretrained model.
The last two change the experimental comparison and need new run IDs.
They do not retroactively change the locked NB07/NB06 architecture selection.
NB03A is an audit notebook, **not** nine repaired training jobs.

**Subsequent user decision (2026-09-09): retain the 17 valid architectures and
close this implemented S2 sweep as complete (153/153).** The nine mislabeled
runs stay quarantined and are not treated as valid Small results. The broader
Tier 5/6 plan is still separate; no replacement training is now required for S2.

### S10: computation bugs versus absent experiments

NB10R generates Figure 3 from fixed fold-1 images and the r3 surviving CAM
methods. The uploaded panel contains saliency and mask contours, not source
RGB/vehicle identifiers. Every panel has an image/run/method manifest. Failed
methods are not presented as faithful explanations; zero maps are labelled.

H2's missing correlation is now `inconclusive_undefined`, not False. A complete
inspection of the local 418 clean masks found:

| Fold | Images with marking | Images with damage |
|---|---:|---:|
| 0 | 67 | 58 |
| 1 | **0** | 5 |
| 2 | 0 | 0 |

NB08 only tests fold 1. Its stripe intervention has no labelled pixels to
remove. The five damage-positive images did not change aggregate recall;
that leaves no variation for the saved H2 correlation. NB10R repeats the mask
coverage audit on the actual Kaggle input. It does **not** retune the
interventions after observing the result or invent a positive/negative test.

Figure 10 now uses only eligible Stage-A run IDs and final-epoch session
metrics. Descriptive Figures 1/2 use final-epoch F1. The old selected-epoch H1
calculation is retained and labelled, not silently re-registered. H3 remains
untestable without the preregistered fine-grained model arms.

Even ten generated figures would complete only the **implemented** figure
package. The original plan's Figures 8/9/10 were mask quality, video temporal
consistency, and resolution × ROI interaction. NB10 instead implemented best
epoch, energy and per-session performance. Those are useful additional
figures, not evidence that the original three experiments were performed.

## Entire stage map

| Stage | Verified position | Still needed |
|---|---|---|
| S0 infrastructure | Built; real cross-session resumes exercised | Validate each new training harness on Kaggle |
| SA annotation | 418 manual masks and 4,180 repaired derivatives | Blind self-consistency is unmeasured and deferred by user; NBT1 is not that test |
| S1 baselines | NB01A reporting verified; NB01B 9/9 × 60 epochs verified | Interpretation under the existing fold-leakage limitation |
| S2 architectures | ✅ Retained 17-architecture sweep complete: 153/153; 9 excluded | Tier 6 FGVC and broader Tier 5 modes/models remain separate scope |
| S3 masks | Existing manual-supervised route ready; NB11 optional/deferred | User declines more annotation: SAM2 comparison and blind consistency stay unmeasured; not prerequisites for manual-supervised S5 (`docs/22`) |
| S4 techniques | **108/108 complete** for implemented fold-1 OFAT | Do not equate 12 implemented arms with every level in the original factor table |
| S4b / Stage C | NB12/NB12R ready: 18 new confirmation runs, six reused baselines | Execute and verify; two additional three-seed-confirmed architectures, frozen top-ranked factors; `23_S4B_CONFIRMATION.md` |
| S5 dense tasks | **Not started** | Detector/segmenter comparison and predicted-ROI downstream evaluation |
| S6 XAI | NB07 r3 Stage-B gate complete | Full all-run/all-fold, native-method/ROAD/model-agnostic/stability programme not completed by this gate |
| S7 stress | 63 implemented rows and 3 controls complete | H2 availability limitation; proposed dirt/session-holdout/video extensions not verified |
| S8 ensemble/calibration | Four tables and 27 prediction files verified | Coverage target not met; empty sets omitted from old abstention; image-split calibration not new-tyre assurance |
| S9 integrated pipeline | **Not started** | S5 winner, landmark/healthy-pool inputs, fusion and component ablations |
| S10 reporting | NB10R verified: 10/10 implemented figures, 30 saliency panels | Original-plan missing figures, statistical limitations, report/manuscript |

Do not give an overall percentage: these stages have very different scope and
cost. Classification execution is far ahead of dense tasks and integration.

## What exactly is S5?

S5 trains models to **predict** the tyre/tread box or mask from a new image:
YOLO26 detection/segmentation, SegFormer-B0/B2, U-Net, DeepLabV3+, and the
planned RT-DETRv2 comparison. The documented `rtdetr-l.pt` name must not be
assumed to be RT-DETRv2-S; resolve architecture identity before implementation.

NBT1's small U-Net validates annotation propagation. NB06's ROI arm crops using
provided masks. Neither measures a learned detector/segmenter or its predicted
ROI on held-out images. Existing masks make **manual-supervised S5 feasible**;
they do not make it completed. Fine groove/sipe/TWI masks and calibrated
alignment labels still do not exist.

S5 needs a dedicated tested dense-task training/resume harness, explicit
manual/SAM2 label-source arms, per-fold/per-seed metrics (mAP or IoU/Dice/boundary
scores), saved predictions, and comparison of classifier performance using
predicted rather than oracle crops. No S5 training notebook is claimed ready
by this delivery. No S5 scope reduction has been made without approval.

S9 must follow that work. Manual tyre/tread masks do not supply HRNet landmark
ground truth, curated healthy examples for PatchCore, or alignment calibration.

## Recovery notebook run order

**Execution complete:** all four notebooks below are now HF-verified. This order
is retained as reproduction guidance, not a request to repeat completed work.

1. **NB01A_Baseline_Recovery.ipynb** — one session, CPU sufficient. Reproduce
   CPU baselines, recover final-epoch legacy scores, save per-fold metrics and
   predictions. No neural retraining.
   Read fix r2 repairs the submitted HTTP 429 failure using authenticated,
   delay-aware retries without a repository inventory. Kaggle reporting outputs
   are now public and verified.
2. **NB01B_Matched_RandomInit.ipynb** — dual T4, nine new full-budget jobs.
   May use four correctly labelled copies; default is one. It resumes from
   the last completed epoch on HF and skips completed IDs. Other old training
   copies should be stopped before changing worker ownership.
3. **NB03A_Architecture_Audit.ipynb** — one session, CPU sufficient. Optional
   rerunnable 162-record coverage audit; does not train unsupported Small.
4. **NB10R_Analysis_Recovery.ipynb** — one T4 session, not four copies. It can
   run independently of NB01B because the original Stage-A set remains fixed.

All need Internet and `HF_TOKEN`; NB01A/B and NB10R need Tire Dataset Prepared.
Staging is `/kaggle/temp`, not the 20-GB output directory. New tables live in
`tables/closure_2026-09-09/`; new figures in `analysis/closure_2026-09-09/`.
Old runs/tables/figures and executed notebook outputs remain intact.

Normal commits batch every 30 minutes; major completion and catchable Stop
flush. **A forced kernel/OS kill cannot execute a handler**: unpublished work
since the last successful push may be lost. NB01B resumes at epoch boundaries,
not mid-batch. A notebook being ready is not an HF-completed experiment.

## Validation

Generator: `tyrelib/build_closure_notebooks.py`; test script:
`scripts/verify_closure_notebooks.py`. Generated cells parse and embedded
library bytes round-trip. Regression tests check constant H2 effects produce
an inconclusive outcome and Figure 10 excludes the quarantined architecture
while reading the last epoch. The real baseline recovery is tested locally
against the dataset/public HF with uploads disabled. Production nine-run
training and multi-architecture CAM outputs are now verified on public HF
(nine complete histories and ten readable figures). No expensive training was launched locally.
