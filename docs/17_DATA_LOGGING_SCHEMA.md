# 17 — Data Logging Schema

> **We train once.** The cost of an extra column is bytes. The cost of a missing one is a re-run we cannot afford.
>
> This is the complete list of what gets recorded. Freeze it **before Stage A** — changing the schema mid-sweep means re-deriving results from 279 runs.

---

## 1. Principles

| Rule | Why |
|---|---|
| **Every column always present** | Missing quantity → `NA`, never omitted, never `0`. "This term doesn't exist" and "this term was zero" are different facts |
| **Fill from the schema at the end** | `for c in SCHEMA: row.setdefault(c, NA)` — a forgotten key cannot produce a ragged CSV |
| **Per-device, never aggregated** | Train on one of two GPUs and an aggregate reports ~50% util, hiding that half the allocation is idle |
| **Test the schema programmatically** | Map every requirement to its column(s) and assert none missing. `tests/test_schema.py` |
| **Float epoch timestamps** | ISO strings at second granularity sort ambiguously across workers |
| **Downsample raw streams** | Full-resolution step traces over 800 runs are gigabytes; 2,000 points per epoch is a few MB and shows everything |

---

## 2. The six files per run

| File | Rows | Written | Purpose |
|---|---|---|---|
| `metrics/epochs.csv` | one per epoch | end of each epoch | **The main artefact.** ~185 columns |
| `metrics/final.csv` | one | at completion | Flat summary for the master table |
| `metrics/confusion_matrix.csv` | 3×3 per checkpoint | best + last | Per-class error structure |
| `per_sample/predictions.parquet` | one per val image per epoch | best epoch + last | Everything needed for post-hoc analysis |
| `telemetry/energy_samples.csv` | 10 Hz per GPU | continuous | Power, energy, CO₂ |
| `telemetry/system_samples.csv` | 1 Hz per GPU + host | continuous | Util, temp, clocks, RAM, disk |
| `telemetry/step_traces.jsonl` | ≤2000 per epoch | per step, downsampled | Within-epoch timing |
| `xai/evidence.csv` | one per image per method | Stage E | TER, BAR, SAR… |
| `xai/faithfulness.csv` | one per method | Stage E | Insertion, deletion, ROAD |

---

## 3. `metrics/epochs.csv` — the full schema

### 3.1 Identity (17)

```
run_id · stage · arch · technique · fold · seed
epoch · global_step · samples_seen
ts_start · ts_end · iso_start · iso_end
account · worker_id · session_id · host
config_hash · lib_version · git_sha
```

### 3.2 Learning — train and validation (44)

Prefixed `train_` and `val_`:

```
loss · loss_ema
acc · balanced_acc
f1_macro · f1_micro · f1_weighted
precision_macro · recall_macro
f1_low · f1_mid · f1_high
recall_low · recall_mid · recall_high
precision_low · precision_mid · precision_high
support_low · support_mid · support_high
qwk                      ← quadratic weighted kappa. ORDINAL — the headline metric
mae_class                ← mean absolute class error
cohen_kappa · mcc
```

> `qwk` and `mae_class` are the ordinal metrics. The classes are ordered, so confusing `low↔high` must cost more than `low↔mid`. Never report macro-F1 alone.

### 3.3 Per-session validation breakdown (variable, ~12)

```
val_acc_session_<session_group>        one column per session in this fold
val_n_session_<session_group>
```

**This is how single-tyre memorisation becomes visible.** A model at 0.95 overall but 0.30 on one session has memorised the other three.

### 3.4 Calibration (9)

```
ece · mce · ace                       expected / max / adaptive calibration error
nll · brier
mean_confidence · mean_confidence_correct · mean_confidence_incorrect
overconfidence_gap                    mean_conf − accuracy
```

### 3.5 Loss components (8)

One column per term, `NA` when the term is not in this run's objective:

```
loss_ce · loss_coral · loss_focal · loss_label_smooth
loss_rank · loss_l2 · loss_aux · loss_total
```

### 3.6 Optimisation health (22)

```
lr_group0 · lr_group1 · lr_backbone · lr_head
grad_norm_mean · grad_norm_max · grad_norm_p50 · grad_norm_p95 · grad_norm_p99
grad_clip_hit_rate
weight_norm_total · weight_norm_backbone · weight_norm_head
update_to_weight_ratio                ← ‖Δw‖/‖w‖ — healthy ≈ 1e-3
amp_scale · amp_scale_decreases       ← each decrease = a DISCARDED step
nan_or_inf_batches                    ← silent under AMP
zero_grad_batches
ema_decay · effective_batch_size
optimizer_state_bytes
```

**The four that catch invisible failures:**

| Column | What it catches |
|---|---|
| `update_to_weight_ratio` | 1e-1 ⇒ LR far too high. 1e-6 ⇒ nothing is moving. Tells you before the loss curve does |
| `nan_or_inf_batches` | Under AMP, non-finite losses are silent — the run continues and learns nothing from those batches |
| `amp_scale_decreases` | Each one is a step whose gradients overflowed and were **discarded**. Invisible by default |
| `grad_clip_hit_rate` | Clipping every batch means the LR or the loss scale is wrong |

### 3.7 Timing and throughput (16)

```
epoch_seconds · train_seconds · val_seconds
dataload_seconds · compute_seconds · backward_seconds · optimizer_seconds
dataload_frac                         ← waiting-for-data ÷ total. HIGH = GPU starving
step_time_mean · step_time_p50 · step_time_p90 · step_time_p99
images_per_second · steps_per_second
checkpoint_save_seconds · hf_push_seconds
```

`dataload_frac` is the one people forget and cannot recover afterwards. If it is high, the fix is the dataloader (workers, caching, decode), **not** the model.

### 3.8 GPU — per device, suffixed `_gpu0`, `_gpu1` (17 × N)

```
gpu{i}_util_mean · gpu{i}_util_max · gpu{i}_util_p50
gpu{i}_mem_used_mb_mean · gpu{i}_mem_used_mb_peak · gpu{i}_mem_total_mb
gpu{i}_mem_reserved_mb_peak · gpu{i}_mem_allocated_mb_peak
gpu{i}_temp_c_mean · gpu{i}_temp_c_max
gpu{i}_power_w_mean · gpu{i}_power_w_max · gpu{i}_power_limit_w
gpu{i}_sm_clock_mhz_mean · gpu{i}_mem_clock_mhz_mean
gpu{i}_energy_joules_epoch · gpu{i}_energy_joules_cumulative
gpu{i}_throttle_reasons                ← non-zero ⇒ the card clocked down
gpu{i}_pcie_tx_mb · gpu{i}_pcie_rx_mb
gpu{i}_name · gpu{i}_driver · gpu{i}_uuid
```

> **`gpu{i}_throttle_reasons` is why a slow epoch is explainable instead of a permanent mystery.** Thermal or power throttling looks identical to "the model got slower".

### 3.9 Host / CPU / RAM (14)

```
cpu_percent_mean · cpu_percent_max · cpu_count
ram_used_gb_mean · ram_used_gb_peak · ram_total_gb · ram_percent_peak
proc_rss_gb_mean · proc_rss_gb_peak     ← PEAK RAM, as requested
proc_vms_gb_peak
swap_used_gb_peak
disk_free_gb_working · disk_free_gb_temp
disk_read_mb · disk_write_mb
n_dataloader_workers
runtime_loader_num_workers · runtime_loader_pin_memory
runtime_memory_safety_revision · runtime_host_ram_pause_percent
runtime_cuda_memory_format · runtime_cudnn_benchmark
runtime_cuda_safety_revision · runtime_scheduler_safety_revision
runtime_hf_commit_policy_revision
runtime_epoch_history_schema_revision
```

`proc_rss_gb_peak` is the number that tells you whether the run would fit on a smaller machine, and the first thing you need when a session dies with no error.

That is now a demonstrated use, not a hypothetical one. The first NB06 ROI
telemetry showed process RSS rising from ~3.3 GB to ~20.7 GB in one run and
31.1 GB total host use in the next sequential run, while GPU use stayed below
5 GB/card. The expanded public state then showed 53 host-RAM pauses across ROI
and standard full-frame arms, proving that the initial ROI-only explanation was
incomplete. Tyrelib v6 records the effective loader, memory, scheduler and HF
commit-policy revisions; `summary.json`/`STATUS.json` record
`pause_reason=host_ram_guard` when it checkpoints before the 88% threshold.

`runtime_memory_safety_revision=2026-08-31-r2` identifies the single full
serialization per epoch plus Linux arena trimming. The commit-policy revision
identifies runs where ordinary claim events ride the 30-minute batch rather
than consuming standalone commits.

`runtime_epoch_history_schema_revision=2026-09-01-r1` identifies the v6
column-name writer. Two public histories had 178-value v5 rows beneath a
177-column v4 header; the revision-aware reader inserts the known missing
heading, pads only the older rows, validates every width, and atomically rewrites
the table. Unknown width changes raise while preserving the source file.

The same fields separated a later RegNet CUDA fault from an OOM: both failed
epoch-0 attempts used only ~1.1 GB/card. RegNet now records
`runtime_cuda_memory_format=contiguous` and `runtime_cudnn_benchmark=false`;
other architectures record `channels_last`. A fatal launch error also records
`cuda_restart_required=true` in the run summary and error/status files so a
poisoned CUDA context is not reused.

### 3.10 Energy and carbon (8)

```
energy_joules_epoch · energy_joules_cumulative
energy_wh_epoch · energy_kwh_cumulative
co2_g_epoch · co2_g_cumulative
carbon_intensity_g_per_kwh            # region constant, recorded for reproducibility
power_sample_count
```

Energy is a genuine reportable result for a study of ~800 runs, and it costs nothing to record.

### 3.11 Config echo (24)

So the CSV is **self-describing** and analysis never has to join back to a YAML:

```
input_resolution · batch_size · batch_size_per_gpu · grad_accum_steps
optimizer_name · lr_initial · weight_decay · momentum · betas
scheduler_name · warmup_epochs · max_epochs · grad_clip
head_type · loss_name · label_smoothing · focal_gamma
sampler_name · augment_policy · roi_mode · preprocessing
pretrained_source · finetune_depth · ema_enabled
n_params_total · n_params_trainable · model_flops_g
precision · runtime_cuda_memory_format · runtime_cudnn_benchmark · compile_enabled
```

### 3.12 Bookkeeping (7)

```
is_best_epoch · best_metric_name · best_metric_value · epochs_since_improvement
checkpoint_path · epochs_planned · run_status
```

**Total ≈ 185 columns** (plus per-session and per-GPU expansion).

---

## 4. `per_sample/predictions.parquet`

One row per validation image, at the best epoch and the last epoch.

```
run_id · epoch · image_id · session_group · fold · proxy_label · ordinal_rank
logit_low · logit_mid · logit_high
prob_low · prob_mid · prob_high
pred_class · correct · confidence · entropy · margin
loss_per_sample
conformal_set          # after Stage G: e.g. "{low,mid}"
conformal_set_size
tta_prob_low/mid/high  # if TTA used, else NA
```

This is what makes post-hoc analysis possible without retraining: per-session error breakdown, calibration curves, ensemble construction, conformal calibration, hard-example mining, and the failure analysis in `06 §9`.

**Parquet, not CSV** — 800 runs × ~150 rows compresses well, and HF previews it.

---

## 5. Raw telemetry streams

### `telemetry/energy_samples.csv` — 10 Hz, per GPU

```
ts · gpu_index · power_w · energy_joules_cumulative · temp_c · sm_clock_mhz · util_pct
```

10 Hz because power fluctuates within a step; per-epoch means hide the peaks that trip a power limit.

### `telemetry/system_samples.csv` — 1 Hz

```
ts · gpu_index · gpu_util · gpu_mem_used_mb · gpu_temp_c · gpu_power_w
     gpu_sm_clock · gpu_mem_clock · gpu_throttle_reasons
     cpu_percent · ram_used_gb · proc_rss_gb · swap_gb · disk_free_temp_gb
```

### `telemetry/step_traces.jsonl` — downsampled to ≤2000/epoch

```json
{"epoch":12,"step":340,"t_data":0.031,"t_fwd":0.088,"t_bwd":0.142,
 "t_opt":0.011,"loss":0.412,"grad_norm":1.83,"lr":0.000217,"amp_scale":32768}
```

Enough to see a within-epoch slowdown; small enough to keep for every run.

---

## 6. `xai/evidence.csv` — Stage E

One row per (image, attribution method):

```
run_id · image_id · session_group · method · target_layer
ter · ter_norm · bar · sar · dar · edi
tread_area_frac · tyre_area_frac · marking_area_frac
saliency_peak_in_tread · saliency_peak_x · saliency_peak_y
mask_source                # manual | propagated | sam2_only
```

### `xai/faithfulness.csv`

```
run_id · method · target_layer
insertion_auc · deletion_auc · road_auc
pointing_game_acc
sanity_randomisation_delta         # must be large, else the method is an edge detector
cross_seed_saliency_iou
selected_for_reporting             # bool — the faithfulness-selected method
```

---

## 7. `metrics/final.csv` — one row, feeds the master table

Everything needed to build `tables/all_runs.csv` without opening `epochs.csv`:

```
run_id · stage · arch · technique · fold · seed · status
best_epoch · epochs_trained · epochs_planned · run_status
best_val_qwk · best_val_f1_macro · best_val_acc · best_val_mae_class
final_val_qwk · final_val_f1_macro
val_recall_low/mid/high
ece · nll · brier
ter_norm · bar · sar · road_auc              # joined from xai/ after Stage E
total_wall_seconds · total_gpu_seconds
total_energy_wh · total_co2_g
peak_ram_gb · peak_gpu_mem_mb
n_params_total · model_flops_g
n_sessions_train · n_sessions_val · n_images_train · n_images_val
nan_or_inf_batches_total · amp_scale_decreases_total
config_hash · lib_version · git_sha · account · finished_iso
```

---

## 8. The trained model — what "saving the model" means

A `state_dict` alone is not a saved model.

```python
{
  "epoch": int,                       # last COMPLETED epoch
  "model": state_dict,
  "optimizer": state_dict,
  "scheduler": state_dict,
  "scaler": state_dict,               # AMP — omit and loss scale resets
  "ema": state_dict,                  # if EMA enabled
  "rng": {"python":…, "numpy":…, "torch":…, "cuda":…},   # ALL FOUR
  "config": dict, "config_hash": str,
  "metrics_at_save": dict,
  "wall_seconds": float,              # cumulative, survives restarts
  "energy_joules": float,
  "arch": str, "n_params": int,
  "lib_version": str, "git_sha": str,
  "torch_version": str, "cuda_version": str,
  "dataset_version": "final_v1",
  "annotation_version": "v1",
  "class_names": ["low_mileage_proxy","mid_mileage_proxy","high_mileage_proxy"],
  "normalisation": {"mean":[...], "std":[...]},
  "input_resolution": int,
}
```

Each field prevents a specific silent corruption — see `05 §7`. The `rng` block matters most here **because this study compares seeds**: a resume that loses RNG state makes "same config, different seed" stop meaning what we think.

Write atomically: `torch.save(...".tmp")` then `os.replace(...)`.

---

## 9. Environment capture — once per run

`env/environment.json`:

```json
{"python":"3.11.15","torch":"2.5.1+cu121","torchvision":"...","timm":"...",
 "numpy":"...","pandas":"...","opencv":"...","ultralytics":"...",
 "cuda":"12.1","cudnn":"...","driver":"535.104.05",
 "gpus":[{"name":"Tesla T4","mem_mb":15360,"uuid":"GPU-..."}],
 "cpu":"Intel Xeon @2.00GHz","cpu_count":4,"ram_gb":31.4,
 "platform":"kaggle","kernel":"...","git_sha":"...","lib_version":"v3",
 "seed":1,"deterministic":false,"cudnn_benchmark":true}
```

Six months on, "why does this run differ?" is usually answered here.

---

## 10. Aggregated tables

Generated, never hand-edited.

| Table | Grain | Source |
|---|---|---|
| `tables/all_runs.csv` | one row per run | `runs/*/metrics/final.csv` |
| `tables/all_epochs.csv` | one row per run-epoch | `runs/*/metrics/epochs.csv` |
| `tables/xai_evidence_all.csv` | one row per image and screened/confirmation run | `runs/*/xai/evidence.csv` |
| `tables/xai_faithfulness.csv` | one row per screened run and candidate attribution method | `runs/*/xai/faithfulness.csv` |
| `tables/stage_b_selection.csv` | one row per screened architecture | NB07 r3 evidence gate |
| `tables/xai_summary.csv` | one row per run-method | `runs/*/xai/*` |
| `tables/stress_tests.csv` | run × intervention | Stage F |
| `tables/baselines.csv` | one row per baseline | Tier 0 |
| `tables/audit_report.json` | — | the aggregator's integrity check |

The public NB07 r3 snapshot has 1,208 evidence rows, 35 faithfulness rows and
18 selection rows. `stage_b_selection.csv` locks `regnety016`,
`densenet121`, and `resnet50`; NB06 also reconstructs the selected rows' seed
and valid-TER counts from `xai_evidence_all.csv` before scheduling any run.

---

## 11. Storage estimate

| Artefact | Per run | × 730 |
|---|---:|---:|
| `epochs.csv` (~185 cols × 45 rows) | ~120 KB | 88 MB |
| `final.csv` | 4 KB | 3 MB |
| `predictions.parquet` | ~40 KB | 29 MB |
| `energy_samples.csv` | ~3 MB | 2.2 GB |
| `system_samples.csv` | ~1 MB | 730 MB |
| `step_traces.jsonl` | ~2 MB | 1.5 GB |
| `xai/saliency/*.npz` (fp16) | ~15 MB | 4.4 GB (subset only) |
| Checkpoints (**with retention policy**) | — | ~30 GB |
| **Total** | | **~39 GB** |

Comfortable for HF. Without the checkpoint retention policy it would be ~150 GB — see `16 §5`.

**Compress the raw streams:** write `energy_samples.csv.gz` and `system_samples.csv.gz`. Roughly 5× smaller, and pandas reads them transparently.

---

## 12. Schema versioning

```json
{"schema_version": "1.0", "frozen": "2026-08-27", "n_columns": 185}
```

**Freeze before Stage A.** If a column must be added later:

1. Bump to `1.1`
2. **Add only** — never rename, never remove
3. Backfill older runs with `NA`
4. Record the change in `PROGRESS.md`

Renaming a column after the 162 public Stage-A runs means either re-deriving
everything or maintaining two analysis paths forever. Neither is acceptable.

---

## 13. Verification

`tests/test_schema.py` must assert:

- [ ] Every column in `SCHEMA` appears in a generated `epochs.csv`
- [ ] No column outside `SCHEMA` appears
- [ ] Missing values are `NA`, never blank, never `0`
- [ ] Every requirement in this document maps to ≥1 column
- [ ] `final.csv` columns ⊆ derivable from `epochs.csv` + config
- [ ] Column dtypes stable across runs
- [ ] Per-GPU columns expand correctly for 1 and 2 GPUs

Run it in the preflight notebook, on every account.
