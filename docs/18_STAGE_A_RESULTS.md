# 18 — Stage A Results

**Status:** **153 scientifically valid; 9 quarantined architecture substitutions** · public audit 2026-08-30
**Source:** `Shanmuk4622/tyre-wear-study` (public) — `runs/a-*/metrics/final.csv`
**Reproduce:** aggregate public rows, then exclude architectures whose
`tl.ZOO[arch]["stage_a_valid"]` is false before calling `sess.honest_table(df)`.

| notebook | architectures | runs | status |
|---|---|---:|---|
| NB02 classical CNN | resnet50, resnext50, densenet121, vgg16bn | 36 | ✅ done |
| NB03 modern CNN | convnextv2_t, effnetv2s, regnety016, mobilenetv4 | 36 valid | ✅ done |
| NB03 quarantined | `convnextv2_s` run ids containing ResNet-18 weights | 9 | ⛔ excluded |
| NB04 transformer | vit_s, deit3_s, swin_t, swin_s, coatnet0, maxvit_t | 54 | ✅ done |
| NB05 foundation | clip_b16, dinov2_s, dinov2_b | 27 | ✅ done |

The public repository contains all 162 Stage-A execution directories, all 162
final metrics files, and all 162 best checkpoints. A later checkpoint-integrity
audit found that the nine directories labelled `convnextv2_s` are not
ConvNeXt-V2-S runs: every status reports **11,177,538 parameters**, and the
sampled public checkpoint has a ResNet-18 tensor signature (`conv1`,
`layer1`…`layer4`). The retired/non-existent timm Small pretrained tag had
triggered the historical emergency ResNet-18 fallback. Those nine rows remain
as an immutable failure record but are excluded from every comparison and from
NB07 selection. This leaves **17 architectures × 3 folds × 3 seeds = 153 valid
runs**. NB03 no longer plans that unsupported arm, and `build_model` no longer
permits any architecture fallback.

The repository contains 161/162 per-sample
prediction files: `a-regnety016-base-f2-s2` is missing that derived parquet,
but its best checkpoint is public, so corrected NB09 reconstructs and publishes
the predictions if RegNetY is selected. **160** `STATUS.json` files say
`completed`. Two old VGG runs say `failed` even though they reached epoch 60
and contain their checkpoints and final metrics: the pre-fix telemetry thread
raised a `ValueError` while serialising after training. They are scientifically
complete and operationally mislabelled; no GPU rerun is needed.

3 folds × 3 seeds per architecture, 60 epochs each, identical recipe, no early
stopping, across 4 Kaggle accounts.

---

## 1. The headline

> **Two folds are leak-flagged and nearly saturated: among valid runs, fold 0
> has 50/51 perfect runs and fold 2 has 43/51. Fold 1 is the only unflagged comparison, but it
> still validates on only four sessions.**

Stage A found large accuracy differences, but it still cannot establish which
model reads wear rather than tyre identity, framing, colour, markings, or
damage. That is why NB07 now selects Stage B on verified evidence location and
faithfulness, not on the accuracy table.

---

## 2. Fold 1 — the only interpretable fold

Folds 0 and 2 both carry `cross_fold_tyre_flags = 1`: the suspect pair
`mileage_070000__session_001` ↔ `mileage_090000__session_001` (tread-pattern
ratio 0.90 against a same-tyre reference of 1.0) sits on both sides of their
splits. After quarantining the three mislabeled rows per fold, fold 0 produced
50/51 perfect best macro-F1 runs; fold 2 produced 43/51. Their mean best
macro-F1 values remain 0.996 and 0.971. Treat that as
leak-inflated, not as architecture performance.

Fold 1 has no flag. 128 validation images, 4 sessions.

| arch | median best epoch | best QWK | best macro-F1 | **final macro-F1** | Wh/run |
|---|---:|---:|---:|---:|---:|
| mobilenetv4 | 13 | 0.997 | **0.992** | 0.739 | 47 |
| vit_s | 4 | 0.977 | 0.953 | **0.855** | 57 |
| swin_t | 11 | 0.977 | 0.930 | 0.537 | 43 |
| effnetv2s | 6 | 0.976 | 0.928 | 0.568 | 59 |
| swin_s | 43 | 0.964 | 0.858 | 0.572 | 53 |
| clip_b16 | 20 | 0.952 | 0.852 | 0.498 | 114 |
| resnet50 | 11 | 0.945 | 0.810 | 0.616 | 56 |
| regnety016 | 10 | 0.946 | 0.805 | 0.596 | 59 |
| maxvit_t | 29 | 0.911 | 0.774 | 0.593 | 125 |
| resnext50 | 8 | 0.938 | 0.768 | 0.549 | 62 |
| dinov2_b | 51 | 0.930 | 0.738 | 0.651 | 158 |
| convnextv2_t | 1 | 0.934 | 0.727 | 0.707 | 113 |
| densenet121 | 22 | 0.922 | 0.690 | 0.516 | 64 |
| vgg16bn | 3 | 0.920 | 0.676 | 0.317 | 100 |
| dinov2_s | 2 | 0.912 | 0.620 | 0.616 | 69 |
| deit3_s | 1 | 0.912 | 0.620 | 0.620 | 62 |
| coatnet0 | 32 | 0.603 | 0.432 | 0.428 | 44 |

| | value |
|---|---:|
| mean best val QWK — *the number the training log prints* | **0.924** |
| mean best macro-F1 — *chosen by looking at the val fold* | **0.775** |
| **mean final macro-F1 — fixed 60 epochs, chosen by nobody** | **0.587** |
| strongest trivial baseline on fold 1 (frame occupancy) | **0.455** |
| runs whose final macro-F1 is at or below that baseline | **10 of 54** |

The best checkpoint table looks strong. The fixed-epoch table is much less so,
and the four-session test set is still too small to identify generalisation.

---

## 3. Three things the table says that a single accuracy number would not

### 3.1 QWK 0.93 and final macro-F1 0.59 are the same runs

QWK rewards ordinal proximity, and with three ordered classes a model that
confuses `low` with `mid` keeps most of its credit. It is the right headline
metric for an ordinal task and it is **also** the most flattering one available.

Report both, always. The roughly 0.34 gap between mean best QWK and mean final
macro-F1 is not noise; it is the combination of ordinal credit and validation
checkpoint selection.

### 3.2 The best epoch is usually not epoch 60

Median best epoch across the 51 valid fold-1 runs: **11**. `vgg16bn` has a median of
**3**; `convnextv2_t` and `deit3_s` have a median of **1**. DINOv2-B is the
exception, with a median best epoch of **51**.

After that, every model degrades. `vgg16bn` ends at macro-F1 0.317 — a model
predicting close to one class. It is not that the schedule is wrong; there are
232 training tyre-photographs of 8 tyres, and the models memorise them in
under ten epochs and then overfit for fifty more.

> **This is not an argument for early stopping.** Equal budget is what keeps an
> architecture comparison a comparison, and the fixed-budget number is the
> honest one precisely because nothing chose it. The point is that
> `ckpt_best` — selected on a 4-tyre validation fold — is optimistically
> biased, and the `best` column should never be quoted without the `final`
> column beside it.

### 3.3 Several architecture rankings are still seed-sensitive

Within-architecture spread of best macro-F1 across three seeds:

| arch | spread |
|---|---:|
| swin_s | 0.367 |
| convnextv2_t | 0.319 |
| resnet50 | 0.319 |
| regnety016 | 0.299 |
| maxvit_t | 0.243 |
| resnext50 | 0.233 |
| clip_b16 | **0.018** |
| mobilenetv4 | **0.023** |

The gross gap between MobileNetV4 and CoAtNet is real on this fold, but many
middle rankings move by more than 0.20 across seeds. More importantly, stable
accuracy can still be a stable shortcut. Stage-B selection therefore uses
NB07's faithfulness-gated TER_norm result; accuracy is reported beside it but
is excluded from the selection rule.

NB07 is now complete. Its public three-seed result selected **RegNetY-16GF,
DenseNet-121, and ResNet-50** with TER_norm 1.5785, 1.5513, and 1.5146. Their
BAR values are 0.0310, 0.0455, and 0.0512; raw valid-map coverage is 180/180,
178/180, and 180/180. These three—not the fold-1 accuracy leaders—are the only
architectures NB06 may use.

---

## 4. Cost

| | |
|---|---|
| Executed records | **162** (plus earlier baseline/resume-test directories) |
| Scientifically valid runs | **153** |
| Recorded run time | **163.4 h executed; 156.6 h valid** |
| Recorded energy | **11.51 kWh executed; 11.15 kWh valid** |
| Public repository audit | 2,412 files · 167 run directories · 162 Stage-A directories |
| `mean_dataload_frac` | resnet50 **0.38** (GPU starved), vgg16bn **0.009** (GPU bound) |

That dataloader fraction is worth acting on before Stage B: resnet50 spent 38%
of every epoch waiting for JPEGs. The fix is the loader, not the model.

---

## 5. What went wrong during the run

| | |
|---|---|
| **Bug 12** — telemetry race | `HardwareMonitor.dump()` built a DataFrame from a list the 10 Hz sampler was still appending to. Killed 2 runs after 43 and 66 min. Now snapshots under the lock, and never raises. |
| **Bug 13** — steal race | `a-vgg16bn-base-f1-s1` was trained to completion by **acct1 and acct2**, same `config_hash`, ~1.4 GPU-h burnt twice. `can_claim` read registry shards last downloaded hours earlier. Now re-pulls before stealing and flushes the claim immediately. |
| **Bug 14** — `summary.json` | Written locally, never enqueued, while `confirm_on_hf` used its presence as the completion test. All 36 finished runs were reported `RESUMABLE`. |
| **Bug 15** — dinov2 resolution | `build_model` never told timm what resolution it would be fed. `vit_*_patch14_dinov2` is created at `img_size=518` and asserts an exact match, so all **18** dinov2 runs died on their first batch. |
| **Bug 16** — silent architecture substitution | The invalid `convnextv2_small.fcmae_ft_in22k_in1k` tag triggered the old emergency ResNet-18 fallback. Nine completed run ids are mislabeled and quarantined. The fallback is deleted, checkpoint signatures are checked, and NB03 no longer schedules the unsupported arm. |
| **Derived upload gap** | `a-regnety016-base-f2-s2` has final metrics and both checkpoints but no predictions parquet. NB09 now reconstructs it from `ckpt_best` if needed and publishes the repaired derived artifact. |

Details in `05 §7`. Bugs 12–14 cost no scientific results. Bug 15 cost the
first DINOv2 attempt, but all 18 repaired jobs are now complete. Bug 16 cost
nine executions and reduces the valid architecture sweep from 18 to 17; those
rows are never relabelled or reused as ConvNeXt results.

### Bug 15 is the one worth learning from

**NB00 caught it on 26 August.** It printed:

```
  FAIL dinov2_s   AssertionError: Input height (392) doesn't match model (518).
  FAIL dinov2_b   AssertionError: Input height (392) doesn't match model (518).
17/19 architectures build, forward and backprop.
Exclude the failures from the sweep, or fix them, BEFORE Stage A.
```

…and then returned normally, reported `architectures OK : 17/19` in a summary
where every other line said PASS, and finished successfully. The same run also
printed `resume equivalence : FAIL` — which was Bug 8, likewise unactioned.

Two days later, four accounts spent a session rediscovering both lines.

> **A preflight that reports but does not block is not a preflight.** It is a
> log entry competing for attention with nineteen other log entries.

NB00 now **raises** on any architecture failure and on resume-equivalence
failure, and every Stage notebook calls `tl.assert_zoo_ok(ARCHS)` before
planning — fifteen seconds, no pretrained download, one forward pass per
architecture at the resolution it will actually receive.

`build_model` now passes `img_size` to timm, tries `dynamic_img_size` as a
fallback, and **verifies with a real forward pass before returning**. A model
that cannot forward at its own configured resolution is a build failure, and it
now says so at build time rather than 100 lines later as a training crash.

---

## 6. What this means for the rest of the study

1. **The corrected NB06 runs Stage B on fold 1 only.** Folds 0 and 2 are
   saturated and leak-flagged, so spending two-thirds of the technique budget
   there cannot measure an effect. Re-cut folds can be added later as a new,
   explicitly versioned experiment.
2. **Fixing the folds is the highest-value thing available.** Merging the
   suspect pair into one group leaves ~11 tyres and 3 folds that mean
   something. Until then, two thirds of every result is uninterpretable.
3. **NB07 is complete; continue the repaired NB06.** The public gate selected RegNetY-16GF,
   DenseNet-121 and ResNet-50 after faithfulness screening and three-seed
   confirmation. Seven architectures have explicit no-faithful-CAM exclusions
   and the mislabeled ConvNeXt-V2-S checkpoint has an architecture-mismatch
   exclusion. NB06 has no fallback, re-verifies the selection against raw
   public evidence, and runs fold-1 OFAT only. Public HF now contains **42/108
   completed runs, 34 checkpointed incomplete runs, and 32 not started; all 76
   status-bearing runs have both checkpoints.** The 2026-09-03 tyrelib v11
   notebook retains the conservative RegNet CUDA runtime and every earlier
   resume/memory/history repair. Each model now runs in a disposable child
   process, and a RAM-paused child automatically resumes the same public
   checkpoint in a clean process. No architecture or scientific setting
   changed.
4. **NB08's shuffled-label control is now mandatory before anything else.** If
   a shuffled-label model also reaches 1.000 on folds 0 and 2, those folds are
   finished as evidence.

---

## 7. Reproducing this table

```python
df = sess.aggregate_remote(run_ids)   # every account's final.csv, from HF
bad = {a for a, spec in tl.ZOO.items() if spec.get("stage_a_valid") is False}
df = df[~df.arch.isin(bad)]           # quarantine mislabeled checkpoints
g  = sess.honest_table(df)            # split by leak flag, best vs final
```

Or directly, without Kaggle:

```
https://huggingface.co/datasets/Shanmuk4622/tyre-wear-study/resolve/main/runs/<run_id>/metrics/final.csv
```
