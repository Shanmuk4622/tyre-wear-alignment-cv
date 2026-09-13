# Recovery completion audit — 2026-09-09

All four recovery notebooks have public outputs. Verified read-only against
[HF snapshot bf62f9e9](https://huggingface.co/datasets/Shanmuk4622/tyre-wear-study/tree/bf62f9e9cbedacc580aa42542da14a068b8f9215).
No remote files were changed and no training was launched during verification.
Reproducible check: `scripts/audit_recovery_completion.py`.

## What was verified

| Notebook | Evidence | Verdict |
|---|---|---|
| NB01A | 15 baseline/fold rows, 5 summary rows, 418 feature rows, 1,672 predictions; recomputed macro-F1 agrees for all 12 CPU baseline/fold groups | Reporting recovery complete |
| NB01B | Nine unique ResNet-50 random-init runs; every STATUS completed; histories contain epochs 1–60 exactly; final macro-F1 agrees with final.csv; best/last checkpoint paths present | Training execution complete |
| NB03A | 162 coverage rows with completed budgets and artifact flags; 153 eligible / 9 quarantined | Audit complete, not replacement training |
| NB10R | Ten PNGs decoded/verified, including Figure 3; 30 saliency records across 10 runs; 418 region-coverage rows; 17-architecture master and three hypotheses | Implemented reporting recovery complete |

Checkpoint presence was verified via the pinned file inventory, not by downloading
or loading the nine large checkpoint pairs. This audit does not certify every
checkpoint tensor or rerun training. Account summaries 1/2 contain nine rows;
3/4 contain eight-row earlier snapshots. They are not missing training: all nine
authoritative per-run records are completed. No duplicate/replacement training is needed.

## Baseline results

| Baseline | Mean macro-F1 |
|---|---:|
| Colour | 0.491409 |
| Structure | 0.483308 |
| Full-frame HOG + SVM | 0.653986 |
| Training-selected majority | 0.193092 |
| Legacy random ResNet-18, final epoch | 0.821351 |
| New matched random ResNet-50, final epoch | 0.823322 |

The new random-init fold-1 seeds score 0.464052, 0.481792 and 0.464052;
folds 0 and 2 score 1.0 for all three seeds. All nine report zero NaN/Inf batches.
The pretrained ResNet-50 final-epoch mean is 0.829217; its fold-1 mean is
0.615895 versus random-init 0.469966. These are descriptive comparisons, not
significance tests or proof of generalisation. Fold-identity/leakage flags remain.
Do not compare selected-epoch pretrained scores against final-epoch random scores.

## Reporting caveats that remain

- H1 is not supported in the retained legacy selected-epoch calculation
  (n=10; primary 0.427095 versus reference 0.907435). This is not a new
  fixed-budget hypothesis test.
- H2 is explicitly `inconclusive_undefined`, with missing supported value.
  Kaggle coverage confirms marking-positive images 67/0/0 and damage-positive
  images 58/5/0 across folds. NB08's fold-1 marking intervention is a no-op.
- H3 is still untestable: the registered fine-grained arms have not run.
- Two of the 30 saliency records have `zero_map=True`; retain that disclosure,
  not a claim that every displayed explanation is informative.
- The nine quarantine-table rows retain a legacy `scientific_complete=True`
  field. That old operational field does not override quarantine: the master
  excludes these runs and NB03A explicitly marks them quarantined.
- Ten implemented figures are not every figure in the original proposal.
  Original mask-quality/video/resolution-by-ROI experiments and manuscript
  remain unfinished. Calibration/coverage limitations remain unchanged.

NB10R records source revision `542b53bc8bd84c679ce1a447ea797fcdc713fe78`;
NB03A records `e0264f58a245e7902c84c8760b8d13b155faa584`;
NB01A records `7c5b6461815a78aae589085d0152eaa6ae9995e1`.
Different pinned input snapshots are expected, not a failed upload.
The old top-level NB10 output still has nine figures; use the newer
`analysis/closure_2026-09-09/` and `tables/closure_2026-09-09/` results.

## How far is the whole experiment?

**Subsequent2026-09-12 update:** declared S4b confirmation18/18 and manual-supervised
S5 dense tasks81/81 plus NB17 report are now verified. See `docs/23`, `docs/24`.

The implemented classification track and its targeted recovery are executed:
S1 baseline reporting/training, the 17-architecture eligible sweep, 108 OFAT
runs, NB07's XAI gate, 63 stress rows, ensemble/calibration outputs and ten
implemented figures. Earlier NB06–NB09 counts are carried from their prior
audits; this audit specifically rechecks the four recovery notebooks.

The full proposal is **not complete**. S3's manual-versus-SAM2 comparison,
the wider Tier-5/6 and XAI programme, **S9 integration**, original-plan extensions and
write-up remain. NBT1's propagation U-Net is not S5.

Next: resolve the tyre/fold-integrity decision before another large compute
commitment, interpret the completed S5 predicted-ROI results and establish S9
inputs/scope. No scope reduction or new model has been silently chosen.
See `20_FULL_PLAN_CLOSURE.md` for the complete stage-by-stage map. A single
percentage would misrepresent the unequal size and cost of these stages.
