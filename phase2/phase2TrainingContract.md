# Phase 2 training and persistence contract

## Runnable training revision — 20 September 2026

**NB00 has passed on Kaggle and its HF report is independently verified. NB01–NB06 are implemented; see [phase2RunNotebooks.md](phase2RunNotebooks.md) for the current run order and settings.**

The user delegated the remaining decisions. The runnable revision keeps all three original mid tyres and all video frames in training, avoiding overlap under every possible unknown mapping. The fixed split is **386 train / 81 validation / 103 test images (8/2/2 tyres)**. Held-out evaluation covers old low/high tyres only. Exact video mapping is no longer a blocker for this revision and is not fabricated.

SegFormer ignores conflicts; YOLO uses a documented training-only tyre/tread union and disables the incompatible exclusive-class auxiliary semantic branch. HRNet uses new polygon-derived points with weight 0.25 as weak training labels; only existing human points serve as validation/test truth. No further manual labeling is required to run this revision. The original ZIP and raw annotations remain unchanged.

Default: 15 combined-data runs (five models × three seeds), 60 epochs each. Optional old-only comparison adds 15 runs with matched update budgets. Two local GPU processes and optional static multi-session workers are implemented. Real-model GPU resume tests run automatically before long training. Actual Kaggle T4 results, completed training, reporting and workstation integration remain execution stages, not missing preparation.

The earlier 6/3/3 stratified split, mandatory exact identity matching, mandatory acceptance of training point proposals, and default 30-run study below are historical design proposals superseded by this section. Source-v1 still correctly records unassigned splits; the separate training overlay supplies the runnable revision.


20 September 2026. The notebooks now implement the runnable revision above. All labels are packaged, the conservative split is frozen, and training-specific conflict/weak-point handling is implemented. Local model/resume tests passed; execute the notebooks to obtain actual Kaggle T4 validation and training results.

## Separate artifacts and truthful status

All implementations/notebook generators go inside `phase2/`; deliver self-contained Kaggle `.ipynb` files with a clear CPU/GPU run order. Do not regenerate old notebooks. Delivered notebooks:

1. `phase2_NB00_Dataset_Preflight.ipynb` — source hashes, labels, split/parent leakage and package verification; CPU.
2. `phase2_NB01_GPU_Resume_Smoke.ipynb` — exact model identities, memory/timing and stop/resume equivalence; dual-T4 session.
3. `phase2_NB02_Classification.ipynb` — MobileNetV4 and ResNet50 jobs.
4. `phase2_NB03_SegFormer.ipynb` — overlapping tyre/tread masks.
5. `phase2_NB04_YOLO26M_Seg.ipynb` — larger segmentation candidate.
6. `phase2_NB05_HRNet.ipynb` — valid/invalid boundary supervision.
7. `phase2_NB06_Evaluate_Export.ipynb` — frozen reporting/export, no test tuning.

NB00 passed on Kaggle and its exact HF report was independently verified. NB01 through NB06 are delivered. NB00 describes immutable source-v1; the later training overlay supplies the safe split without rewriting that source release. Notebook implementation is complete; long training and dual-T4 execution are separate result states.

Freeze a canonical JSON protocol containing dataset manifest hash, physical identities, split hash, class definitions, label schema, annotation revision, source-only/combined sampling policies, guide coordinates, model/pretrained revision/tensor signature, preprocessing, optimizer/scheduler, loss, epoch step counts, effective batch/accumulation, precision, seeds, versions, source hashes and selection rule. Hash it into every run ID and checkpoint. A repair gets an explicit compatibility record; no opportunistic hash bypass.

## Kaggle operation and storage

- Attach the self-contained **Tire Dataset Prepared phase2** source release as a Kaggle input. It includes all 418 original clean photos and labels plus 152 new frames; no separate original input is required. Prefer Kaggle-native attached inputs over HF data transfers.
- The release excludes all 4,180 old augmented copies. Perform future augmentation online and keep the source dataset read-only. See `phase2KaggleUpload.md`.
- Enable Internet and Kaggle secret `HF_TOKEN` for account `Shanmuk4622`. No token in notebook cells, JSON, logs or packages.
- Select dual T4. Prefer one independent job per GPU after resource tests, with unique static ownership and a single parent uploader. Do not assume naive DataParallel is faster; this repo already encountered severe slowdown. One-GPU execution remains valid for a model if clearly logged; never pretend both are active.
- Measure `/kaggle/working` and scratch capacity, cgroup RAM and free space. Keep generated large checkpoints in verified scratch and only necessary deliverables in the 20 GB output area. Do not assume scratch is 1 TB or persistent.
- Estimate peak live model/optimizer/scaler size plus atomic-save and immutable-upload snapshot copies before accepting a job. Reserve disk/RAM headroom; record projected hours from measured steady steps. Reclaim completed local copies only after remote verification.
- Use bounded dataloaders/telemetry and separate per-job processes to release retained native memory. A capacity problem causes a logged pause or an explicit pre-run recipe amendment, never an unnoticed smaller model/batch/resolution during a frozen run.

## Resume granularity and state

Save atomically at every **completed optimizer update**, including model, optimizer, scheduler, scaler, Python/NumPy/CPU/CUDA RNG, epoch, next batch/update cursor, deterministic sampler order/state, augmentation randomness, partial loss/metric counts, exposure counts, elapsed budget, best checkpoint identity and protocol/runtime hashes. Gradient accumulation boundaries must be explicit; avoid claiming exact mid-accumulation recovery unless gradients and microbatch state are saved too.

If interrupted inside a batch, discard that incomplete update and replay it from the last durable checkpoint/RNG state. Resume at the next completed-update boundary. This is mid-epoch recovery, not an impossible promise to continue inside a GPU instruction.

Use atomic replace with metadata journals to prevent weights/status disagreement. The uploader reads immutable snapshots, never a file being rewritten. Local per-step saves and remote commits are different layers.

Before planning any new work, reconcile per-run remote status and hash-verified state. Download a resumable checkpoint into a fresh Kaggle session before invoking restore. `failed` with a valid state means resume. Completed runs are skipped only after required result/evaluation artifacts verify. Missing/incompatible state fails clearly; do not restart from epoch zero unnoticed.

Stock framework resume support is insufficient evidence. In particular, implement and verify YOLO optimizer-step/sampler/RNG recovery in the pinned trainer, or mark it blocked; an epoch-only `resume=True` implementation does not satisfy this user requirement.

## Hugging Face publication

Implemented new namespace in the existing dataset repository:

`phase2/training-r1/<protocol_sha256>/runs/<architecture>-<condition>-seed<N>/`

Never overwrite Phase 1 paths. Use a single parent publisher for multiple local GPU jobs. Publish checkpoints/state, detailed metrics/traces, configs, identities, errors, final/selected weights, evaluation predictions, environment/repair records, notebook/source hashes, status and a manifest. Full information means durable reproducibility records, not an unbounded dump of disposable intermediate tensors.

Normal timer: one batched commit approximately every 30 minutes. Also request a flush at major cell completion, run completion, watchdog pause and catchable Stop/SIGINT/SIGTERM/error. Save durable local state first. Deduplicate unchanged files and batch major-cell outputs to avoid one-commit-per-file behavior.

Use authenticated cached/pinned reads, request budgeting shared across workers, bounded retry with exponential backoff/jitter and server delay headers. Treat the user's historical ~128/hour limit conservatively; do not hard-code it as a guaranteed universal quota. Consult actual request-class limits and server headers. A conservative team write cap of 100/hour is a ceiling, not a target. Metadata/listing/read calls also need rate handling. No heartbeat/claim spam.

“Immediate push” means trigger and attempt the flush as soon as the interruption is catchable, while honoring a server-imposed delay. A network outage, forced process kill, GPU-driver reset or kernel OOM cannot guarantee remote publication. Report last successful remote revision/update, local pending snapshot and potential lost-work window explicitly. A forced kill of ephemeral storage can lose all work since the last verified upload.

See [Hugging Face's rate-limit documentation](https://huggingface.co/docs/hub/rate-limits), checked 19 September 2026. Do not replace the pinned working environment merely because a newer client offers retry behavior; first prove compatibility.

## Required preflight tests

1. Actual model parameter/tensor identity, strict loading and correct output heads; no architecture fallback.
2. Dataset split disjointness by physical tyre, original/derivative parent, source clip and exact image hash. No test examples in pretrained tyre checkpoints for the primary experiment.
3. Image/mask/ignore/point alignment through resize, letterbox and augmentation, with coordinate round trips and horizontal left/right swap tests.
4. Real T4 uninterrupted versus interrupted/restored training seam, including the next batches, losses, weights, optimizer/scaler and scheduler state. Use the actual model/trainer/precision, not a toy model alone.
5. Fresh-process and fresh-directory recovery from the published snapshot, including interrupted local-save journal recovery.
6. Upload 429/transient-error/timeout faults, deduplication, immutable snapshot consistency, Stop during backoff and exhausted retry reporting.
7. Two workers cannot train the same run or clobber registry/status files; changing ownership requires stopping old workers.
8. Timed steady-state steps, cgroup RAM growth, dual-job VRAM/CPU contention and disk peak estimate. Report expected whole-study cost before long training.
9. Nonfinite update retries replay the same batch/RNG with bounded recovery; no silently skipped samples or lowered precision assertions.
10. A partial run is restored without deleting the successful logs; a completed run is not retrained on a routine restart.

Until these pass, label notebooks and docs “implemented/local-tested” or “GPU validation pending” accurately. Do not claim fail-proof recovery based on code inspection.

## Evaluation/export boundary

Freeze all selection, thresholds, tolerances and A/B budgets before measured runs. Keep validation-selected and fixed-final outputs separately named. Never pick the best test seed. Aggregate by tyre/domain and show rejection coverage alongside conditional point errors.

Export compact inference weights with SHA-256, preprocessing/coordinate contract, dataset/protocol revision, seed/epoch and selection provenance. Build a separate Phase 2 workstation copy and compare against the unchanged original. Only the separately evaluated candidate can become its default; an all-data deployment refit needs a different provenance and cannot inherit an unseen-test claim.
