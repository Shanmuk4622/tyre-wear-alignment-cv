# Environment Setup — `cv_conda`

> **Every Python command in this project runs inside `cv_conda`.** No exceptions.
> ```bash
> conda activate cv_conda
> ```

---

## 1. Create or update the environment

```bash
# create from scratch
conda env create -f environment.yml

# or update an existing cv_conda
conda env update -n cv_conda -f environment.yml --prune

conda activate cv_conda
```

## 2. Verify

```bash
conda activate cv_conda
python scripts/verify_env.py
```

Expected output:

```
python        3.11.x
torch         2.x.x   cuda available: True   device: NVIDIA ...
opencv        4.10.x
ultralytics   8.x.x
huggingface   0.2x.x
kaggle        1.6.x
sam2          (optional, for annotation)
--- ALL CHECKS PASSED ---
```

If CUDA is unavailable locally that's fine — heavy training happens on Kaggle. You need local GPU only for fast iteration.

## 3. Secrets

Never commit these.

| Secret | Local | Kaggle |
|---|---|---|
| `HF_TOKEN` | `~/.cache/huggingface/token` via `huggingface-cli login` | Kaggle Secrets, name `HF_TOKEN` |
| Kaggle API | `~/.kaggle/kaggle.json`, chmod 600 | Built in |

```bash
conda activate cv_conda
huggingface-cli login          # paste your write token
huggingface-cli whoami         # should print Shanmuk4622
```

`.gitignore` already excludes `*.json` credentials, `data/`, `runs/`, `*.pt`, `*.onnx`.

## 4. SAM2 checkpoints (annotation workflow)

SAM2 weights are not on PyPI. Download once:

```bash
conda activate cv_conda
mkdir -p checkpoints && cd checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

Used by `annotate/sam2_propagate.py` for video mask propagation (`03_DATA.md §5`). Not needed at inference time.

## 5. Common problems

| Problem | Fix |
|---|---|
| `conda activate` fails in a script | Add `source "$(conda info --base)/etc/profile.d/conda.sh"` first |
| OpenCV `ImportError: libGL.so.1` | `conda install -c conda-forge libgl` or use `opencv-python-headless` on servers |
| Torch installed CPU-only | Install torch from the pytorch channel with the right `pytorch-cuda` version, before pip packages |
| pip/conda fighting | Install all conda packages first, pip packages last. Never mix in one command. |
| Slow solve | `conda install -n base conda-libmamba-solver && conda config --set solver libmamba` |
| Kaggle CLI 403 | `chmod 600 ~/.kaggle/kaggle.json` |

## 6. Repo layout (to be created as you build)

```
Tyre/
├── README.md   PROGRESS.md   ENVIRONMENT.md   environment.yml
├── docs/                  00..14 + LOGBOOK.md + GITHUB_SETUP.md
├── tyrelib/               ← the library; notebooks are GENERATED from it
│   ├── infra/             uploader.py  registry.py  sharding.py  lifecycle.py
│   │                      rate_limit.py  atomic_io.py  config_hash.py
│   ├── telemetry/         schema.py  gpu_monitor.py  epoch_accumulator.py
│   ├── data/              datasets.py  samplers.py  augment.py  splits.py
│   ├── models/            zoo.py  heads.py  fgvc.py  target_layers.py
│   ├── train/             loop.py  resume.py  ema.py
│   ├── xai/               attribution.py  masks.py  ter_bar_sar.py
│   │                      faithfulness.py  stress_tests.py
│   ├── filters/           clahe.py  scharr.py  gabor.py  structure_tensor.py
│   └── eval/              metrics.py  conformal.py  stats.py  plots.py
├── notebooks/             00_preflight  01_train_worker  02_xai  03_analysis
├── build_notebooks.py     ← regenerates notebooks from tyrelib (base64 bootstrap)
├── scripts/               verify_env.py  dataset_shortcut_probe.py
├── configs/               recipe_base.yaml  zoo.yaml  ofat.yaml
└── tests/                 test_resume.py  test_registry_shards.py
                           test_sharding_determinism.py  test_augment_signs.py
                           test_schema.py  test_xai_sanity.py
```

**The `.py` library is the source of truth; notebooks are generated.** Edit Python, run `build_notebooks.py`, re-upload. Never hand-edit the base64 blob (`05 §11`).

Six tests matter more than the rest:

| Test | Catches |
|---|---|
| **`test_resume.py`** | Real kill mid-run → resume → **compare post-seam per-epoch losses**, not final accuracy (⚠ Bug 6) |
| **`test_registry_shards.py`** | Two writers must not lose each other's entries; terminal states sticky (⚠ Bug 2) |
| **`test_sharding_determinism.py`** | Assignment invariant to measured costs (⚠ Bug 7) |
| `test_schema.py` | Every telemetry column always present; `NA` where undefined |
| `test_xai_sanity.py` | Weight randomisation degrades every saliency map (`14 §4`) |
| `test_augment_signs.py` | Horizontal flip negates signed lateral labels; flipping twice is identity |

Write them early. Each catches a bug class that is otherwise invisible until it has cost a week — and in the sharded multi-account setup, until it has silently corrupted a few hundred runs.
