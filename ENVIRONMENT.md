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
├── README.md              ENVIRONMENT.md      environment.yml
├── docs/                  01..11 + LOGBOOK.md + GITHUB_SETUP.md
├── capture/               record.py  illum.py  flatfield.py  qc.py
├── calib/                 intrinsics.py  extrinsics.py  lights.py
├── tyrevision/
│   ├── data/              datasets, samplers, augment, splits
│   ├── photometric/       normals.py  albedo.py  integrate.py
│   ├── filters/           clahe.py  scharr.py  gabor.py  structure_tensor.py
│   ├── recon/             register.py  unroll.py  coverage.py
│   ├── geometry/          landmarks.py  estimators.py  residual.py  fusion.py
│   ├── models/            segformer.py  convnext_heads.py  hrnet.py  patchcore.py
│   ├── losses/            focal.py  dice.py  boundary.py  cldice.py  coral.py
│   ├── train/             loop, resume, hf_sync
│   └── eval/              metrics, conformal, plots, stratify
├── annotate/              sam2_propagate.py  ranking_ui/  guidelines.md
├── app/                   fastapi backend, gradio demo   (optional)
├── notebooks/             00_feasibility  01_calib  02_explore  03_train(kaggle)
├── scripts/               verify_env.py, cli entry points
├── configs/               debug.yaml  seg.yaml  wear.yaml  align.yaml
└── tests/                 test_resume.py  test_augment_signs.py  test_calib.py
                           test_angle_conventions.py
```

Four tests matter more than the rest:

- `test_augment_signs.py` — **horizontal flip must negate camber and toe; flipping twice is the identity.** This bug class is invisible until it has cost you a month
- `test_angle_conventions.py` — sign convention fixed for left and right wheels; known synthetic normals produce known angles
- `test_resume.py` — 200 steps straight vs 100 + resume + 100 must match (`05_TRAINING_KAGGLE_HF.md §3`)
- `test_calib.py` — a known ground distance reconstructs within 0.5 mm; light directions within 5°

Write them early. Each catches a bug class that is otherwise invisible until it has cost you a week.
