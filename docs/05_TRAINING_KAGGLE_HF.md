# 05 — Training Infrastructure: Kaggle Dual-T4 + Hugging Face

> This is the **spec** for the training notebook. When you ask for the training code, I will deliver a `.ipynb` built exactly to this spec. Read it now so we agree on the contract before any code exists.

---

## 1. Constraints we are designing around

| Constraint | Value | Consequence |
|---|---|---|
| Kaggle persistent output | 20 GB | Never store raw frames here. Checkpoints + logs only. |
| Kaggle session scratch | ~1 TB (`/kaggle/temp`) | Extract the full dataset here. **Wiped on session end.** |
| Session limit | 12 h (GPU), 9 h if idle-killed | Must resume from mid-epoch, not just epoch boundary. |
| GPU | 2× Tesla T4 (16 GB each), no NVLink | fp16 not bf16. DDP over PCIe. |
| HF API | ~128 write ops/hour | Batch pushes. Never push per-step. |
| Weekly GPU quota | 30 h | ~3 sessions per week. Every session must count. |

**The design goal:** if the session dies at any moment — timeout, crash, or you hitting Stop — you lose at most 30 minutes of compute and **zero** metadata.

---

## 2. Storage layout

```
/kaggle/temp/grip/            ← 1 TB scratch, wiped each session
├── data/                     ← extracted dataset (large)
├── cache/                    ← preprocessed tensors, unrolled maps
└── ckpt_local/               ← every-N-step checkpoints (fast, local)

/kaggle/working/              ← 20 GB persistent, keep it lean
├── grip/                     ← code, cloned from HF or GitHub
├── state/                    ← run_state.json, RNG states, optimizer
└── logs/                     ← jsonl metrics, small

HF Hub                        ← the single source of truth
├── Shanmuk4622/grip-net              (model repo)
│   ├── checkpoints/latest/           ← rolling, overwritten
│   ├── checkpoints/epoch_{n}/        ← milestone, kept
│   ├── run_state.json                ← resume pointer
│   ├── metrics.jsonl                 ← append-only, every eval
│   ├── config.yaml
│   └── README.md                     ← model card
└── Shanmuk4622/grip-roll             (dataset repo)
```

**Rule: Hugging Face is the source of truth, not `/kaggle/working`.** A session that starts by pulling from HF works identically whether it's your 1st or 40th session, and works from any account or any machine.

---

## 3. The resume contract

`run_state.json` — pushed with every checkpoint, this is what makes restarts free:

```json
{
  "run_id": "grip-v3-2026-09-14",
  "stage": "finetune_real",
  "global_step": 48213,
  "epoch": 27,
  "batch_in_epoch": 1104,
  "samples_seen": 1542816,
  "best_val_depth_mae": 0.284,
  "best_ckpt": "checkpoints/epoch_24",
  "lr_scheduler_state": {...},
  "rng": {"python": "...", "numpy": "...", "torch": "...", "cuda": "..."},
  "dataloader_seed": 1337,
  "sampler_offset": 1104,
  "wall_clock_trained_s": 71204,
  "last_push_utc": "2026-09-14T11:32:07Z",
  "push_count_this_hour": 3,
  "history": [ {"session": 12, "steps": 3200, "duration_s": 40100}, ... ]
}
```

### Mid-epoch resume (the part most people get wrong)

Resuming at an epoch boundary is easy and wastes up to 40 minutes. Do it properly:

1. Use a **deterministic sampler** seeded by `(epoch, dataloader_seed)`.
2. Persist `sampler_offset` = number of batches consumed this epoch.
3. On resume, regenerate the identical permutation and **skip the first `sampler_offset` indices**.
4. Restore optimizer, scheduler, EMA, GradScaler, and all four RNG states.

Verification test (run it once, keep it in the repo):

```
Run A: train 200 steps straight through.
Run B: train 100 steps, kill, resume, train 100 more.
assert allclose(A.weights, B.weights, atol=1e-5)
```

If that fails, your resume is broken and you will not notice until you have wasted a week.

---

## 4. Hugging Face push policy — rate-limit safe

128 write operations per hour. A naive per-step push burns that in seconds. Policy:

| Trigger | Action | Budget |
|---|---|---|
| Every 30 min (wall clock) | Push `latest/` + `run_state.json` + `metrics.jsonl` | ~2 ops × 2/hr = 4 |
| Every N steps (N ≈ 500) | Save **locally only** to `/kaggle/temp/ckpt_local/` | 0 |
| Stage completion | Push milestone `epoch_{n}/`, tag it | ~2 |
| New best val | Push `best/` | ~2 |
| **On interrupt (SIGINT / Stop button)** | **Immediate flush of everything** | ~4 |
| Session end (`atexit`) | Immediate flush | ~4 |

Worst case ≈ 16 ops/hour. Comfortable margin.

### Implementation notes

```python
# Interrupt handling — this is the piece you asked for specifically
import signal, atexit, threading

_flush_lock = threading.Lock()

def emergency_flush(signum=None, frame=None):
    with _flush_lock:                 # never double-push
        save_full_state(LOCAL_CKPT)
        push_to_hf(LOCAL_CKPT, "checkpoints/latest", msg=f"interrupt @ step {state.global_step}")
        push_file(RUN_STATE, "run_state.json")
        push_file(METRICS,   "metrics.jsonl")
    if signum: raise KeyboardInterrupt

signal.signal(signal.SIGINT,  emergency_flush)
signal.signal(signal.SIGTERM, emergency_flush)
atexit.register(emergency_flush)
```

Also wrap the training loop in `try/except/finally` so the `finally` block flushes even on an unexpected exception. Belt and braces — the whole point is that *nothing* loses the run.

### Robustness of the push itself

- **Exponential backoff with jitter** on 429/5xx: 5 s → 10 s → 20 s → 40 s → 80 s, max 5 attempts.
- **Never let a failed push kill training.** Catch, log, mark `push_pending = True`, retry at the next trigger.
- Track `push_count_this_hour` in `run_state.json`; if it exceeds 100, skip non-critical pushes (but **never** skip an interrupt flush).
- Use `huggingface_hub.HfApi.upload_folder` with `delete_patterns` for `latest/` so stale shards don't accumulate.
- Prefer `create_commit` with multiple `CommitOperationAdd`s — **one commit for many files is one operation**, not N. This is the single biggest rate-limit saver.

---

## 5. Dual-T4 usage

```python
torchrun --nproc_per_node=2 train.py     # DDP, one process per GPU
```

T4 specifics that matter:
- **fp16, not bf16** — Turing has no bf16 tensor cores. Use `torch.cuda.amp` with `GradScaler`, and persist the scaler state in checkpoints.
- No NVLink → gradient all-reduce goes over PCIe. Keep the model small enough that compute dominates communication, or use `gradient_as_bucket_view=True` and `static_graph=True`.
- 16 GB each. At 512×512×4ch with ConvNeXt-Tiny, batch 16 per GPU fits comfortably. Use gradient accumulation for larger effective batches.
- `channels_last` memory format gives a real speedup on Turing for convnets. Free win, one line.
- Set `torch.backends.cudnn.benchmark = True` — inputs are fixed-size, so autotuning pays off.

**Only rank 0 writes.** Checkpointing, HF pushes, and logging must be rank-0-only, with `dist.barrier()` around them so rank 1 doesn't race ahead into a torn state.

---

## 6. Dataset download — Kaggle-native, always

Direct from Kaggle datasets is dramatically faster than HF or Drive inside a Kaggle session (same datacentre).

```python
# Preferred: attach as a Kaggle Dataset in the notebook sidebar → /kaggle/input/... (instant, no download)
# For public Kaggle datasets not attached:
!pip install -q kaggle
!mkdir -p ~/.kaggle && cp /kaggle/input/kaggle-creds/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d warcoder/tyre-quality-classification -p /kaggle/temp/grip/data --unzip
```

**Upload `GRIP-Roll` as a private Kaggle Dataset** as well as to HF. Then it mounts instantly at `/kaggle/input/grip-roll` with zero download time, every session. Push new versions with `kaggle datasets version`. HF stays the archival/public copy; Kaggle is the fast working copy.

Extract to `/kaggle/temp` (1 TB), never `/kaggle/working` (20 GB).

---

## 7. Notebook structure

The `.ipynb` I deliver will have these cells, in order:

| # | Cell | Idempotent? | Notes |
|---|---|---|---|
| 0 | Preflight | ✓ | GPU check, disk check, HF token from Kaggle Secrets (`HF_TOKEN`), print resume status **loudly** |
| 1 | Install & imports | ✓ | Pinned versions |
| 2 | Config | ✓ | Single dataclass, hashed into `run_id` |
| 3 | **Resume manager** | ✓ | Pull `run_state.json` from HF, decide fresh vs resume, restore everything |
| 4 | Data mount / download | ✓ | Kaggle-native |
| 5 | Preprocess & cache | ✓ | Skips if cache exists in `/kaggle/temp` |
| 6 | Datasets & samplers | ✓ | Deterministic, offset-aware |
| 7 | Model build | ✓ | Loads weights if resuming |
| 8 | **Signal handlers + flusher** | ✓ | Install *before* the training loop |
| 9 | Train loop | ✓ | Local ckpt every 500 steps, HF push every 30 min |
| 10 | Eval + conformal calibration | ✓ | Appends to `metrics.jsonl` |
| 11 | Final push + model card update | ✓ | |
| 12 | Sanity report | ✓ | Sample predictions, loss curves, resume-integrity check |

**Every cell must be safe to re-run.** You will re-run cells at 1 a.m. Design for that.

Cell 0 should print something impossible to miss:

```
╔══════════════════════════════════════════════════════════╗
║  RESUMING run grip-v3-2026-09-14                         ║
║  stage      : finetune_real                              ║
║  epoch      : 27 / 40      batch 1104 / 1832             ║
║  step       : 48,213                                     ║
║  best MAE   : 0.284 mm  (epoch 24)                       ║
║  trained    : 19h 47m across 12 sessions                 ║
║  last push  : 2026-09-14 11:32 UTC  (18 min ago)         ║
╚══════════════════════════════════════════════════════════╝
```

---

## 8. What gets pushed to HF (everything — you train once)

```
Shanmuk4622/grip-net/
├── README.md                   # model card: architecture, data, metrics, limits, intended use
├── config.yaml                 # exact config hash
├── requirements.lock.txt       # pinned env
├── git_sha.txt                 # code version
├── run_state.json
├── metrics.jsonl               # append-only: step, all losses, all val metrics, lr, grad-norm, time
├── checkpoints/latest/         # rolling
├── checkpoints/best/
├── checkpoints/epoch_{n}/      # milestones
├── conformal/                  # calibration quantiles per output, per stratum
├── artifacts/
│   ├── loss_curves.png
│   ├── confusion_pattern.png
│   ├── depth_scatter.png       # pred vs gauge
│   ├── bland_altman.png
│   └── sample_reports/         # rendered report cards
├── onnx/ , tensorrt/ , tflite/
└── ablations/{1..8}/           # each with its own metrics.jsonl
```

Log **every** eval to `metrics.jsonl`, never overwrite. It is a few hundred KB and it is your entire experimental record — the thing you will regret not having when writing the paper. Include: all loss components separately, per-class metrics, per-stratum metrics, learning rate, grad norm, GPU memory, wall time, and the git SHA.

---

## 9. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `429 Too Many Requests` | Rate limit | Backoff is automatic; check `push_count_this_hour`; batch files into one `create_commit` |
| Resume restarts from epoch 0 | `run_state.json` not found or `run_id` mismatch | Config hash changed → new `run_id`. Pass `--force-run-id` to continue an existing run after a config tweak |
| Loss spikes right after resume | RNG or optimizer state not restored | Run the 200-step resume-equivalence test |
| OOM on resume but not fresh | Optimizer state loaded to GPU before model | Load to `cpu` then `optimizer.load_state_dict`, move after |
| `/kaggle/working` full | Checkpoints written to persistent dir | Local ckpts belong in `/kaggle/temp/ckpt_local` |
| DDP hangs at checkpoint | Rank 1 waiting at a barrier rank 0 never reaches | Wrap rank-0 IO in try/finally with the barrier in `finally` |
| fp16 loss → NaN | Turing fp16 overflow | Check `GradScaler` state restored; lower initial scale; ensure no `bf16` autocast |
| Slow first epoch only | Preprocessing cache cold | Expected. Cache lives in `/kaggle/temp` and dies with the session — consider caching preprocessed tensors as a Kaggle Dataset |
| Interrupt didn't push | Handler installed after the loop started, or `KeyboardInterrupt` swallowed | Install handlers in cell 8, before cell 9; never bare-`except` in the loop |

---

## 10. Local development

Everything also runs locally for debugging on small subsets:

```bash
conda activate cv_conda
python train.py --config configs/debug.yaml --limit-batches 20 --no-hf-push
```

Debug locally, scale on Kaggle. Never debug on Kaggle — you will burn your 30 h weekly quota on typos.
