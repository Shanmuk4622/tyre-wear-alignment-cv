# 07 — Roadmap and Team Plan

**Capstone Fall-Sem 2026–27 · Review-1 complete · dataset in hand.**

Companion to `13_EXPERIMENT_PLAN.md`. That document says what the study *is*; this one says who does what, in what order.

---

## Position as of 2026-09-12

**Full-plan status:** see `20_FULL_PLAN_CLOSURE.md`. All five legacy baselines
are public, and the matched random-init ResNet-50 arm is verified (9/9 × 60 epochs).
S9, Tier-6 FGVC and broader Tier-5 modes are not completed. The declared
S4b/Stage-C extension is complete:18/18 and NB12R HF-verified. Manual-supervised
S5 is now complete:81/81 ×60epochs and NB17 report HF-verified (`docs/24`).
NB01A/B and NB10R recovery is HF-verified; NB03A is only
an audit because the pretrained Small arm is unsupported. The user has now
accepted closing the retained 17-architecture sweep (153/153), with nine
substitutions permanently excluded. NB11/S3 independent-input comparison is
deferred by user; no more annotation is required for S5.

| | |
|---|---|
| Dataset | ✅ `final_v1` delivered, verified PASS, analysed |
| Hardware | ❌ **None needed.** No rig, no camera build |
| Approach | ✅ Broad comparative study; Stage A 153 valid + 9 quarantined mislabeled runs |
| Focus | Tyre **wear**. Alignment deferred (`13 §3`) |
| Compute | 30 GPU-h/week **per Kaggle account** |
| Compute so far | 162 executions (153 valid) · 163.4 recorded GPU-h · 11.51 kWh |
| Execution status | **NB06 complete: 108/108. All four recovery notebooks verified.** NB10R has 10/10 implemented figures; broader stages and write-up remain (`docs/21`) |
| Annotation | ✅ NBT1 real PASS; clean IoU 0.9780, propagated IoU 0.9747 |

**Current execution gate:** NB07 is complete and its public selection has been
audited. NB06–NB10 and the four recovery notebooks have executed. Next is
**S5 result interpretation and S9 input/scope decisions**, with H2/calibration limitations
and remaining scope recorded in `docs/21`.
NB06 gave each model a clean child process, and any RAM pause
automatically resumes the same HF checkpoint in another child. NB06 accepts only the
faithfulness-tested, three-seed public top three—RegNetY-16GF, DenseNet-121 and
ResNet-50—and runs fold 1 only while folds 0 and 2 remain leak-flagged.

### ⚑ Original compute planning (kept for provenance)

Annotation is now solo and that is fine — it is four hours of your time.

**Compute is different.** The run budget assumed four Kaggle accounts sharing the
work. On one account:

| | 4 accounts | 1 account |
|---|---:|---:|
| Stage A (S2) | ~6 h wall-clock per notebook | **~98 h ≈ 3.5 weeks** |
| Whole study | ~4 weeks | **~15 weeks** |

Fifteen weeks does not fit a semester alongside everything else. Three options,
in order of preference:

1. **Get three more Kaggle accounts.** They are free, they need no coordination
   (ownership is arithmetic — `05 §3`), and each one runs the identical notebook
   with a different `WORKER_ID`. Teammates, a second email, anyone. This is by
   far the cheapest fix.
2. **Cut the model zoo.** Tiers 0–2 alone (baselines + classical + modern CNNs)
   is ~47 GPU-h and still a complete, honest, reportable architecture result.
   Add transformers and foundation models only if time allows.
3. **Cut seeds from 3 to 2.** Saves a third. Do this *last* — cross-seed variance
   is large at this data scale and two seeds barely measures it.

Stage A did keep all three folds. Its result then showed that folds 0 and 2 are
leak-flagged and nearly saturated. The current plan therefore does **not** spend
another two-thirds of the Stage-B budget on them: NB06 runs fold 1 only, while
all three Stage-A folds remain reported. This evidence-based change supersedes
the pre-Stage-A “do not cut folds” instruction.

---

## Stages

| Stage | Theme | Deliverable | Gate |
|---|---|---|---|
| **S0** | Infrastructure | Preflight passes on all 4 accounts | Kill-and-resume test passes |
| **SA** | **Annotation** (solo) | 418 images hand-corrected, propagated to 4,598 | **Self-consistency** tread IoU > 0.90 |
| **S1** | Baselines | Tier 0 results, all folds | Beat majority class |
| **S2** | Architecture sweep | 17 valid architectures × 3 folds × 3 seeds | ⚠ **153 valid; 9 `convnextv2_s`/ResNet-18 substitutions quarantined** |
| **S3** | Masks | Manual + SAM2 mask sets, audit report | Agreement reported |
| **S4** | Technique OFAT | **RegNetY-16GF + DenseNet-121 + ResNet-50** × 12 factors | ✅ **108/108 completed and verified on public HF** |
| **S5** | Detection + segmentation | YOLO26, SegFormer, U-Net, DeepLabV3+ | ROI-crop Δ measured |
| **S6** | XAI | TER/BAR/SAR + faithfulness screen | ✅ **NB07 r3 gate complete; 1,208 evidence rows, top three locked** |
| **S7** | Stress tests | Shortcut intervention matrix | ✅ 63 rows public; control mean 0.375184 below gate 0.45 |
| **S8** | Ensembles + calibration | Seed/arch ensembles, conformal | ✅ Outputs public; observed90% coverage .86885/.97561/.93939 across folds0/1/2. Only fold0 below nominal; dependence/empty-set caveats remain |
| **S9** | **Tier 8 integrated pipeline** | SegFormer → best classifier → HRNet → PatchCore | Does it beat the best single model? |
| **S10** | Write-up | Report, Review-3, paper draft | Full illustrated report and reproducibility handoff prepared; author/guide review and venue format pending. Original-plan gaps remain |

Stages overlap. **SA runs in parallel with S0** — it is people-time, not compute. S6 consumes S2's checkpoints.

The original full-study budget was ~800 runs and ~440 GPU-h. Stage A used
163.4 recorded hours. The corrected Stage-B budget is selection-dependent and
tops out at about 108 fold-1 runs rather than the original three-fold sweep.

---

## S0 · Infrastructure — do this first, properly

This gate is complete. The checklist is retained as the record of what was
required before the public Stage-A run.

- [x] HF repo created (one account, `Shanmuk4622`); rate-limit budget recorded in `PROGRESS.md`
- [x] Token write scope verified; public repo re-listed anonymously
- [x] `final_v1` uploaded as a Kaggle Dataset and used by all Stage-A notebooks
- [x] Four Kaggle workers used for Stage A
- [x] Library self-test passes offline (56 checks)
- [x] Uploader: batching, dedup, 429 parsing, shared per-token rate limiter
- [x] Registry: per-writer shards, sticky terminal states, ownership-aware claims
- [x] Sharding: LPT bin packing on a static cost table
- [x] Lifecycle: SIGTERM + atexit + KeyboardInterrupt + 8.5 h watchdog
- [x] Telemetry schema exercised by 162 Stage-A runs
- [x] Kill-and-resume proved across fresh Kaggle sessions/accounts
- [x] Every valid Stage-A model built and completed at its configured resolution; invalid ConvNeXt-V2-S arm quarantined
- [x] Every Stage-A architecture screened for XAI; seven CAM-gate exclusions and one checkpoint mismatch explicitly recorded
- [ ] Work-split plan printed with estimated hours per worker

**Gate:** the resume test passes for real, not as a shorter-run extension (⚠ Bug 6).

---

## S1 · Baselines — a complete result in week 1

Tier 0 (`04 §3`). Minutes on CPU. Produces a full, reportable, honest result before anything risky starts.

All five legacy rows already exist on HF: colour .491409, structure .483308,
HOG .653986, majority accuracy .409735, and three ResNet-18/15-epoch jobs.
NB01A reporting and NB01B matched ResNet-50 3×3, 60-epoch comparison are now
verified complete. This training was not "minutes on CPU".

---

## S2 · Architecture sweep — the bulk

~31 configs × 3 folds × 3 seeds ≈ 279 runs ≈ **141 GPU-h** (`13 §8`).

Fixed recipe (`04 §2`) throughout. **If the recipe changes mid-sweep, the comparison is void** — note any forced exception in the results table.

**Run cheap tiers first.** Tier 1–2 CNNs give a complete architecture story in days; transformers and foundation models follow.

**Gate:** every config has 9 completed runs, or a documented reason it does not.

---

## SA · Annotation — people-time, runs in parallel with S0

Full protocol in `15_ANNOTATION_GUIDE.md`. Summary:

- labelme with SAM2 proposes masks for the **418 clean images**; you verify and correct
- Classes: `tyre`, `tread` (required), `marking`, `damage` (when visible)
- **One batch**, worked through one session-group at a time, ~3.5–4 h over 3–4 sittings
- **Consistency pass at the end**: re-annotate 30 images blind, compare against the first pass
- Propagate to all 4,180 derivatives by replaying the recorded geometric transforms
- Saved to `annotations/`, parallel to `images/`, never inside `FINAL/`

**Gate: self-consistency `tread` IoU > 0.90.** Below that your boundary judgement drifted across the job — re-annotate the first ~40 images, which were done before the rule was internalised. Every TER number inherits this, so it is worth the twenty minutes.

---

## S3 · Masks — the instrument

**Superseding user decision:** no more manual annotation. Defer NB11, the blind
repeat pass and SAM2 comparison; do not label those tests passed. Use existing
manual masks for supervised S5 and derive its training boxes automatically.
The delivery instructions below are retained as optional future work, not a gate
on this manual-supervised route. Fold integrity still needs resolution.

**Delivery:** `NB11_S3_Manual_SAM2_Agreement.ipynb` is ready in two passes:
prepare independent box prompts/30 blind reannotations, then run SAM2.1 Small
inference and report agreement. This is human-box-prompted, unedited SAM2,
not fully automatic semantic labelling. Kaggle execution and quality review
remain pending. Run instructions and dependencies: `22_S3_MASK_COMPARISON.md`.

- Manual masks (from SA) as ground truth
- SAM2-only masks retained as the **pseudo-label ablation arm**
- Derived regions: `M_bg` = 1 − `M_tyre`; `M_dirt` by rule inside `M_tyre`
- Publish the agreement report before anything depends on it

**Gate:** mask quality reported. Without it every S6 number is unfalsifiable.

---

## S4 · Technique OFAT

Top 3 architectures from S2 × 12 factors, one at a time (`04 §9`). ~324 runs.

**Run factor 5 (ROI) first** — framing variance is the most obvious weakness in this dataset, and I expect it to matter more than architecture.

Confirm the top 3 findings on two further architectures (S4b). If a factor helps only one architecture, **say so** — that is the honest reading of an OFAT design, which cannot detect interactions.

**S4b complete:** NB12's 18 fold-1 runs on ConvNeXt-V2 Tiny and MobileNetV4,
plus NB12R's paired report, are verified on public HF. Random-init's negative
direction repeats in both models; the two sampling directions do not both replicate.
The following records the original selection rationale, not pending work.
Six Stage-A baselines are reused. Only class-weighted sampling was positive;
random initialisation and uniform sampling were the next ranked effects, not
proven gains. No new annotation. Protocol/run instructions: `23_S4B_CONFIRMATION.md`.

---

## S5 · Detection and segmentation

Complete:81 manual-supervised runs and NB17 report. YOLO26 (det + seg), genuine
RT-DETRv2-R18, SegFormer-B0/B2, U-Net and DeepLabV3+ used existing manual masks;
SAM2 pseudo-label comparison remains deferred, not passed.

**The metric that matters is downstream:** Δ classification macro-F1 when the ROI crop is used. Standalone mAP against manual labels is secondary. NB17 reports full/predicted/oracle ROI comparisons; this does not complete S9.

---

## S6 · XAI — the primary axis

Inference only, ~15 GPU-h. For every trained model:

1. Attribution with the architecture-appropriate method (`14 §2`)
2. **Weight-randomisation sanity check** per method
3. Faithfulness — insertion, deletion, ROAD — to **select** the method per architecture
4. TER_norm, BAR, SAR, DAR, EDI
5. Cross-seed and cross-fold saliency IoU
6. Occlusion/RISE audit on a subsample

✅ H1–H3 were frozen publicly at **2026-08-30T10:06:21Z**, before the first XAI
evidence row.

---

## S7 · Stress tests

Six interventions × top models (`06 §5`). **Run the shuffled-label control first** — if it scores above chance, stop everything and find the leak.

---

## S8 · Ensembles and calibration

Reuses checkpoints, nearly free. Seed ensembles, architecture ensembles, TTA, temperature scaling, conformal prediction sets with an explicit `uncertain` outcome.

---

## S9 · Exploratory fusion complete; full integration still blocked

NB18 completed81 saved-prediction analyses;405 metrics and135 summaries are
HF-verified. Tested fixed fusion does not improve macro-F1 on average. No rerun
needed. This is not the original54-run Tier8 integration/ablation experiment.

Requires the S5 winner and independently justified landmark/healthy-pool
inputs. NBT1 is not a trained S5 model comparison; provided-mask ROI is not
predicted-mask ROI. Full Tier-8 pipeline results remain unverified; NB18's
exploratory component results are verified separately (`docs/25`).

## S10 · Write-up

NB19/NB20 reporting execution is complete and HF-verified:38 bundled sources,
14 public figures, HTML report and Markdown results draft. The full manuscript
is now prepared at [report/REPORT.html](report/REPORT.html), with16 visuals,
source-generated tables, reviewed claims/references and reproducibility/submission
appendices. Next: author/guide review and the required template. No GPU rerun.
The original missing video/factorial/H3 work is not completed by these figures.
See `26_S10_REPORTING_AND_NEXT_STEPS.md`.

Use the populated, revision-pinned report package rather than restarting empty tables. Keep the original `13 §9` figure plan separate from the implemented14-figure package.

### Venue targets

| Venue | Fit |
|---|---|
| ***Measurement*** (Elsevier) | Strong — two key references already published there |
| **IEEE Access** | Strong, rolling |
| IEEE T-IM | Instrumentation and measurement |
| WACV applications track | Good fit for an empirical study |
| CVPR/ICCV workshops (XAI, industrial vision) | **Very good fit for the XAI framing** |
| arXiv preprint | Do this regardless |

An XAI-focused workshop is arguably the best fit — the contribution is explanation-grounded benchmarking, not a new architecture.

---

## Team split

Four members, four subsystems. **Split by ownership, not by "everyone does a bit"** — that produces four half-finished modules.

| Role | Owns | Deliverables |
|---|---|---|
| **Infrastructure** | Library, uploader, registry, sharding, lifecycle, telemetry, notebook generator, preflight | S0. Every other stage depends on this landing first |
| **Model zoo** | Architecture implementations, recipes, cost table, S2 sweep | S2, S4 |
| **Masks & dense tasks** | SAM2 pipeline, mask audit, detection, segmentation, ROI ablation | S3, S5 |
| **XAI & evaluation** | Attribution, faithfulness, TER/BAR/SAR, stress tests, statistics, figures | S6, S7, S8, and the figures in S9 |

Everyone: runs a worker notebook, commits weekly, reads `13` and `14`.

### Interfaces — agree these in week 1 and write them down

Four-person projects fail at the seams, not inside the modules.

```
Infrastructure → Model zoo   : run_id scheme, config schema, checkpoint contract
Model zoo → Masks            : CAM export format (.npz, normalised, per-image)
Masks → XAI                  : mask format + naming; region definitions
Model zoo → XAI              : checkpoint loading, target-layer registry
All → Evaluation             : per-run metrics CSV schema (frozen early)
```

The metrics CSV schema was frozen for S2. Changing it now means re-deriving
the **162 public Stage-A runs**, so later notebooks add derived tables rather
than renaming those recorded columns.

---

## Weekly discipline

Every Friday, 30 minutes as a group:

1. Update the status board in `PROGRESS.md`
2. Append a dated session entry — what worked, what broke, **what was decided and why**
3. Audit the HF repo: is every expected run present and complete?
4. Check `dataload_frac`, `nan_or_inf_batches`, `amp_scale_decreases`
5. Move anything unrealistic **now**, not later

---

## Critical path

```
S0 infrastructure ──▶ S2 architecture sweep ──▶ S6 XAI ──▶ S9 write-up
        │                     ▲                    ▲
        └──▶ S3 masks ────────┴────────────────────┘
```

**Everything depends on S0.** A broken resume or a lost-update registry silently corrupts hundreds of runs, and you find out late. Spend the time.

---

## What to cut if you fall behind

In order:

1. The optional app (`11_APP.md`)
2. RT-DETRv2 (keep YOLO26)
3. Tier 5 foundation models (keep DINOv2 ViT-S only)
4. OFAT factors 2, 9, 11 (small expected effects)
5. RISE and Integrated Gradients (keep Grad-CAM family + occlusion)
6. Tier 6 FGVC — **cut last**; it is the most novel model axis

**Never cut:** the trivial baselines · 3 folds × 3 seeds · the shuffled-label control · the mask audit · the XAI sanity checks · the pre-registered hypotheses. Those are what make the results believable, and they are cheap.
