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
blender       (optional, check separately)
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

## 4. Blender (separate install, not conda)

Blender 4.2 LTS, installed system-wide. Headless rendering is driven from Python scripts:

```bash
blender --background scenes/tyre.blend --python render/generate.py -- --n 1000 --out /data/synth
```

Blender ships its own Python; don't try to install it into `cv_conda`. The `cv_conda` side only reads Blender's outputs.

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
├── README.md              ENVIRONMENT.md      environment.yml
├── docs/                  01..08 + LOGBOOK.md
├── capture/               trigger.py  illum_sync.py  record.py  qc.py
├── calib/                 intrinsics.py  plate.py  laser_plane.py
├── grip/
│   ├── data/              datasets, samplers, augment
│   ├── recon/             unroll.py  rolling_speed.py  stitch.py
│   ├── geometry/          ellipse.py  estimators.py  fusion.py
│   ├── models/            backbone, heads, losses
│   ├── train/             loop, resume, hf_sync
│   └── eval/              metrics, conformal, plots
├── render/                Blender generation scripts
├── notebooks/             00_feasibility  01_calib  02_explore  03_train(kaggle)
├── scripts/               verify_env.py, cli entry points
├── configs/               debug.yaml  synth.yaml  jig.yaml  real.yaml
└── tests/                 test_resume.py  test_augment_signs.py  test_calib.py
```

Three tests matter more than the rest:

- `test_resume.py` — 200 steps straight vs 100+resume+100 must match (`05_TRAINING_KAGGLE_HF.md §3`)
- `test_augment_signs.py` — lateral flip must negate camber and toe; flipping twice is identity
- `test_calib.py` — known-height step must reconstruct within 0.05 mm

Write them early. Each one catches a bug class that is otherwise invisible until it has cost you a week.
