"""Verify the cv_conda environment is correctly set up.

Usage:
    conda activate cv_conda
    python scripts/verify_env.py
"""

import importlib
import os
import shutil
import sys

CHECKS = [
    # (import name, friendly name, version attr, required)
    ("numpy", "numpy", "__version__", True),
    ("scipy", "scipy", "__version__", True),
    ("pandas", "pandas", "__version__", True),
    ("sklearn", "scikit-learn", "__version__", True),
    ("cv2", "opencv", "__version__", True),
    ("skimage", "scikit-image", "__version__", True),
    ("torch", "torch", "__version__", True),
    ("torchvision", "torchvision", "__version__", True),
    ("timm", "timm", "__version__", True),
    ("ultralytics", "ultralytics", "__version__", True),
    ("transformers", "transformers", "__version__", False),
    ("huggingface_hub", "huggingface_hub", "__version__", True),
    ("kaggle", "kaggle", "__version__", False),
    ("torchmetrics", "torchmetrics", "__version__", False),
    ("onnx", "onnx", "__version__", False),
    ("gradio", "gradio", "__version__", False),
    ("zarr", "zarr", "__version__", False),
]

GREEN, RED, YELLOW, RESET = "\033[92m", "\033[91m", "\033[93m", "\033[0m"


def main() -> int:
    failures, warnings = [], []

    print(f"python        {sys.version.split()[0]}")
    print(f"executable    {sys.executable}")

    env = os.environ.get("CONDA_DEFAULT_ENV", "<none>")
    if env != "cv_conda":
        warnings.append(f"active conda env is '{env}', expected 'cv_conda'")
    print(f"conda env     {env}")
    print("-" * 60)

    for mod, name, attr, required in CHECKS:
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, attr, "?")
            print(f"{GREEN}  ok  {RESET}{name:<22} {ver}")
        except Exception as exc:  # noqa: BLE001
            if required:
                failures.append(f"{name}: {exc}")
                print(f"{RED} FAIL {RESET}{name:<22} {exc}")
            else:
                warnings.append(f"{name} missing (optional)")
                print(f"{YELLOW} warn {RESET}{name:<22} missing (optional)")

    print("-" * 60)

    # CUDA
    try:
        import torch

        avail = torch.cuda.is_available()
        dev = torch.cuda.get_device_name(0) if avail else "cpu only"
        print(f"cuda available: {avail}   device: {dev}")
        if not avail:
            warnings.append("CUDA unavailable locally (fine — training runs on Kaggle)")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"torch cuda check: {exc}")

    # Credentials
    hf_token = os.path.expanduser("~/.cache/huggingface/token")
    print(f"HF token      {'found' if os.path.exists(hf_token) else 'NOT FOUND -> huggingface-cli login'}")
    if not os.path.exists(hf_token):
        warnings.append("HF token missing; run: huggingface-cli login")

    kg = os.path.expanduser("~/.kaggle/kaggle.json")
    print(f"Kaggle creds  {'found' if os.path.exists(kg) else 'NOT FOUND -> place kaggle.json in ~/.kaggle/'}")
    if not os.path.exists(kg):
        warnings.append("kaggle.json missing")

    # Blender
    blender = shutil.which("blender")
    print(f"blender       {blender or 'not on PATH (optional, needed for synthetic data)'}")

    print("-" * 60)
    for w in warnings:
        print(f"{YELLOW}WARN{RESET} {w}")
    for f in failures:
        print(f"{RED}FAIL{RESET} {f}")

    if failures:
        print(f"\n{RED}--- {len(failures)} REQUIRED CHECK(S) FAILED ---{RESET}")
        print("Fix with:  conda env update -n cv_conda -f environment.yml --prune")
        return 1

    print(f"\n{GREEN}--- ALL CHECKS PASSED ---{RESET}"
          + (f"  ({len(warnings)} warning(s))" if warnings else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
