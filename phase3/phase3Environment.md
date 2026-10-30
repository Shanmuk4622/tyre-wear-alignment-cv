# Phase 3 — environment and troubleshooting

Recorded: 27 October 2026. The project instruction is explicit in [ENVIRONMENT.md](../ENVIRONMENT.md): **every Python command runs inside `cv_conda`**.

**Later in the same session: a working activation route was verified.** `run_cv.cmd` points the existing Conda wrapper at `C:\Users\shanm\dev\python.exe -m conda`, activates `cv_conda`, and executes Python there. Local model tests confirmed `C:\Users\shanm\dev\envs\cv_conda\python.exe`, Torch 2.5.1+cu121 and GTX1650 CUDA. NumPy 2.4.6/Pillow 12.1.1 were observed in this environment; older prototype tables are historical. No package replacement was performed. The activation hook emits OpenCL access/missing-temp warnings, but the intended interpreter and tests complete successfully.

## Normal local use

From a Conda-enabled terminal:

```powershell
conda activate cv_conda
Set-Location 'D:\Documents\norse\web Applicarion\Tyre'
```

For the app, the existing `prototype/Launch Tread Station.cmd` already activates this environment. The prototype README documents a PowerShell hook when only `conda.bat` is found:

```powershell
(& conda 'shell.powershell' 'hook') | Out-String | Invoke-Expression
conda activate cv_conda
```

These are existing project instructions; successful execution in the current agent shell has not been established.

## Observed in this orientation

| Check | Observation |
|---|---|
| Command discovery | `conda` resolves to `C:\Users\shanm\dev\condabin\conda.bat` |
| Filesystem check | The wrapper, `C:\Users\shanm\dev\Scripts\conda.exe`, and `C:\Users\shanm\dev\envs\cv_conda\python.exe` exist |
| Read-only `conda env list` | Returned **“The system cannot execute the specified program.”** |
| Python/GPU/package validation | Not run; this documentation task used PowerShell only |

The original failure did not establish its cause or prove that the workstation launcher was broken. The later workaround above avoids the failing executable wrapper while retaining `conda activate cv_conda`. Continue verifying the interpreter, and do not reinstall the environment or bypass it with system Python based only on the earlier message.

## Preserve existing runtime choices

The root environment file describes development; exact Kaggle runtime/protocol records control replay. Prototype dependencies are isolated under `.vendor/`; existing checks record CUDA PyTorch, a GTX 1650 4 GB, Qt isolation and a 32 MB inference-worker stack. The recorded versions/timings are historical evidence, not packages or performance newly verified today.

Keep original CUDA PyTorch, dependency isolation and model adapters intact. Heavy future training runs on Kaggle under the notebook's tested environment, with actual GPU/storage/RAM preflight results and the persistence requirements in [AGENTS.md](AGENTS.md).
