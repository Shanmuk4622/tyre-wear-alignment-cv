# Phase 2 — notebooks to run

> **Final status — 21 September 2026:** [All 15 runs and NB06 are verified complete](phase2Seed3Completion.md). Installed selections still match the final registry. Earlier partial/deferred/run-next statements below describe historical snapshots. No retraining or extra labeling is required.


## NB00 is finished

The executed notebook shows PASS for 2,882 checksummed files and all 570 image/mask sets. Its report was independently downloaded and checked at HF commit `3da9c33dc5d1ca9e57d2c85ec87bdcd4f6fcf876`. The executed notebook is preserved, including an archive copy. **Do not rerun NB00.**

Dataset: [shanmuk4622/tire-dataset-prepared-phase2](https://www.kaggle.com/datasets/shanmuk4622/tire-dataset-prepared-phase2).

## Run order

| Notebook | Accelerator | What it does |
|---|---|---|
| `phase2_NB01_GPU_Resume_Smoke.ipynb` | T4 ×2 | Full-size pretrained GPU and independent-process checkpoint-resume tests for all five models |
| `phase2_NB02_Classification.ipynb` | T4 ×2 | MobileNetV4 Medium and ResNet50; three seeds each, six training runs |
| `phase2_NB03_SegFormer.ipynb` | T4 ×2 | SegFormer-B0; three training runs |
| `phase2_NB04_YOLO26M_Seg.ipynb` | T4 ×2 | YOLO26 Medium segmentation; three training runs |
| `phase2_NB05_HRNet.ipynb` | T4 ×2 | HRNet-W18 geometry; three training runs |
| `phase2_NB06_Evaluate_Export.ipynb` | CPU | Audit completed runs, report results and create an inference-weight registry |

For each notebook, import its `.ipynb` into Kaggle, attach the Phase 2 dataset, enable Internet, enable `HF_TOKEN` in Secrets and Run All. NB01 does short tests only. Each training notebook repeats its own actual-model test before long training, so a later environment change is checked automatically.

Default configuration: one Kaggle session, two independent GPU processes, 60 epochs per run, three seeds. This is **15 combined-data training runs** across NB02–NB05. Run those notebooks sequentially if you have one available session. Different model notebooks can run in separate sessions because their job IDs differ. Do not run NB01 concurrently with the training notebooks: its smoke jobs reserve the same run ownership.

When a session reaches its time guard or you stop it, wait for the HF commit message. Start the same notebook again with the same settings to continue. Completed jobs are verified and skipped. Do not delete checkpoints to resolve an error; retain the output and inspect the reported failure.

## Parallel workers

There are two levels:

1. Within one notebook, the coordinator launches at most two independent model processes, one per T4. These are the GPU workers. Data preparation is synchronous and deterministic; data-loader worker count is deliberately zero to avoid hidden prefetch state during exact mid-epoch recovery.
2. Across multiple Kaggle sessions running the same notebook, use the same `NUM_WORKERS=N` and a different `WORKER_ID` in each: `0` through `N-1`. The notebook supports up to four. Jobs are assigned by a fixed full-study index; a worker with no matching jobs exits cleanly.

Keep the default `NUM_WORKERS=1`, `WORKER_ID=0` unless deliberately distributing the same workload across sessions. Use only one live copy of each worker. Keep worker count fixed while runs are active. After all old sessions stop, you can change the worker count: checkpoints keep their job IDs and new owners resume existing work.

Remote leases and compare-and-swap commits prevent stale writers from overwriting progress. A crashed session may leave a lease for up to 90 minutes. Set `TAKE_OVER=True` only after confirming the former notebook stopped; then rerun. Do not use it to run duplicate writers.

## Dataset decisions already handled

| Partition | Images | Physical tyres |
|---|---:|---:|
| Training | **386** | 8 |
| Validation | **81** | 2 |
| Test | **103** | 2 |
| Total | **570** | 12 |

Training contains 234 old photos plus all 152 new frames. Since the exact video-to-original identity mapping is unknown, **all three original mid-mileage tyres and all videos stay in training**. Every possible mapping therefore stays within training. Validation/test use different original low/high tyres; scores cannot establish unseen-mid, unseen-video or full three-class generalization.

The ZIP remains the immutable source release. A separate versioned training overlay assigns roles; it does not rewrite the ZIP or claim its old `training_allowed=false` source metadata changed. The earlier proposed 6/3/3 class-stratified study required exact video mapping; this 8/2/2 design supersedes that proposal for the runnable training release.

- **SegFormer:** retains raw masks and excludes contradiction pixels from loss/metrics.
- **YOLO:** consumes bitmap instance masks directly, avoiding polygon round-trip loss. For its training targets only, tyre is the union of raw tyre and tread; tread remains unchanged. This explicit automatic consistency policy is not a human correction. The two nested instances remain overlapping. YOLO26’s auxiliary exclusive-class semantic branch is disabled; the main instance/detection losses remain active. Evaluation uses the raw targets with ignored conflict pixels.
- **HRNet:** 70 original human-labeled training images plus 152 polygon-derived weakly supervised images = **222 training images with geometry targets**. Weak new targets receive loss weight 0.25. Validation and test each use 25 existing human-labeled images. No derived points become human ground truth, and no new labeling is required for this protocol.
- **Sampling:** balance class, then old/video domain, then original tyre or video clip. Video2 does not dominate merely because it has more frames. Horizontal flips preserve mask/point alignment and swap left/right point labels. Mild brightness/contrast/blur augmentation runs online.

The five model adapters use pinned general-pretrained weights, not Phase 1 tyre checkpoints that may have seen the holdout tyres.

## Storage and Hugging Face

The dataset stays in Kaggle’s read-only input directory; no image dataset is copied to working storage. Live checkpoints use measured scratch storage. Small reports/runtime and pretrained weights use `/kaggle/working/phase2_training`. Available space is checked, and snapshots have an 18 GiB managed-output budget to leave headroom under the 20 GB output allowance; 1 TB scratch is never assumed.

Each completed optimizer update saves model, optimizer, scheduler, AMP scaler, random states, sampler position, sampled IDs/loss history, validation records and best weights. A batch interrupted before completion is replayed from the last durable checkpoint. A change in the saved runtime/GPU is rejected for compatibility review instead of silently starting again.

One parent uploader per notebook publishes an immutable snapshot about every 30 minutes and on completed runs/catchable Stop/error. There is no commit per image, batch or epoch. Workers have no HF token and cannot upload independently. API retries respect server delays and a conservative per-session share of the request budget. Commits preserve other workers’ paths, verify manifests at the exact revision and check large-file SHA-256 metadata.

HF destination: `Shanmuk4622/tyre-wear-study`, under `phase2/training-r1/<protocol>/`. Old experiment namespaces are untouched. The printed `RUN_PREFIX` identifies your exact protocol; NB06 auto-detects one protocol or accepts this value when there are several.

A catchable Stop triggers a final publication attempt after worker checkpointing. A forced kernel kill, lost storage or network outage cannot guarantee that final upload. A fresh session resumes the last verified remote snapshot, so up to the work since that upload can be lost.

## Optional comparison study

The default produces the requested combined-data models. To also measure the effect of adding the videos, set `RUN_CONDITIONS='combined,old_only'` consistently. This expands to **30 runs**. Old-only and combined runs have equal optimizer-update budgets for each architecture. NB06 pairs seeds and reports the difference. Completed combined runs retain the same IDs and are skipped when adding the old-only runs later.

Model selection uses validation scores only. Test results are generated after training, not used to choose an epoch or seed. NB06 records a per-model validation-selected seed and each seed’s evaluation. Integrating those weights into the workstation and checking its actual latency/video behavior comes after training; neither is falsely marked complete now.

## Verification already performed

All five pinned pretrained models passed full-size independent-process GPU resume checks on the local GTX 1650, with zero differences in losses, model/optimizer state, RNG, scaler and scheduler. CPU architecture checks, upload/checkpoint fault injections and pause/resume/validation-selection/export lifecycle checks also passed. These are smoke/control-flow results, not 60-epoch training or accuracy results. Kaggle dual-T4 validation runs in NB01 and automatically inside each training notebook.
