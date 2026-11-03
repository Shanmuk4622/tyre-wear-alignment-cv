# NB16 — Kaggle runbook

**Use the new preparation notebook; do not rerun NB08–NB15.**

| Setting | Required value |
|---|---|
| Notebook | [phase3_NB16_TREAD_TWO_GEO_Preparation_And_Contract_Gate.ipynb](../../notebooks/phase3_NB16_TREAD_TWO_GEO_Preparation_And_Contract_Gate.ipynb) |
| Accelerator | GPU **T4 ×2** |
| Internet | ON |
| Data | Add expanded `shanmuk4622/tire-dataset-prepared-phase2` |
| Secrets | Enable `HF_TOKEN`, authorized to write `Shanmuk4622/tyre-wear-study` |
| Output space | At least 4 GB free initially; a 2 GB free reserve is checked between updates |
| Start | Run All in order, in a fresh kernel if another model runtime/HF SDK is already imported |

The images are read directly from Kaggle input. Only the pinned 5.5 MB selected
anchor is downloaded from HF. Source extraction and the private HF SDK live in
working output; the notebook does not replace Kaggle Torch/CUDA or install a new
backbone. The optional 1 TB scratch assumption is not used.

## Three cells and expected progress

| Code cell | Work | Expected evidence |
|---|---|---|
| 1 | Private `huggingface_hub==0.36.2` with no dependency replacement; verify/extract every embedded file | 27 matching source/config files |
| 2 | Full dataset SHA/overlay checks; exact anchor warm start; math/isolation/crash tests; batch8 update1→4 and local restored comparison; publish unit1 plus latest completed evidence; actually download and continue unit1→4 in a fresh worker; final publication/release | `ENGINEERING_PROOF.json`, `HF_CONTINUATION_PROOF.json`, `FINAL_GATE.json`, verified HF revision |
| 3 | Print result, show the interactive 32-case gallery and download links | `RUN_RESULT.json`, `Engineering_Passport.html` |

GPU1 owns immutable anchor features, one source image at a time. GPU0 owns all
355,879 trainable parameters. There is one intended model; direct/local/HF paths
are disposable equivalence checks, not three measured candidate fits. No held-out
image enters the optimizer or engineering fixture selection.

The coordinator prints the last committed stage/update every 30 seconds while
waiting. Worker output identifies checksum verification, inference, backward,
CSV/native commit and actual downloaded-state continuation. HF quota/backoff
messages are visible. Long checksum/upload waits are not silently labelled a
numerical failure. Actual speed/memory is recorded; no T4 wall-time promise.

## Stop and restart

```mermaid
sequenceDiagram
  participant U as Kaggle Stop
  participant C as Coordinator
  participant W as Secret-free worker
  participant H as Hugging Face
  U->>C: Catchable interrupt
  C->>W: STOP_REQUESTED marker
  W->>W: Finish/commit current unit; stop before next
  C->>H: Immediate direct-file flush attempt
  H-->>C: Verified revision or visible backoff/failure
  C->>C: Keep last verified/pending receipt
```

Keep the existing Kaggle files. Rerun the unchanged notebook. Partial local
fixtures restore the complete pointer and continue; scientific evidence must
match. A compatible completed run displays its existing receipt with no new
optimizer step or HF write. A fresh empty session can repeat these disposable
engineering checks, then prove a true published-checkpoint continuation. That is
distinct from the required fresh-session auto-resume of the future measured fit.

Forced kernel/process/session death cannot execute cleanup. Preserve saved
Kaggle output where available; bytes never uploaded cannot be guaranteed. Never
reset a protocol/hash, delete a live owner or create a second writer to bypass
an incompatible checkpoint or quota wait.

| Failure | What to do |
|---|---|
| Wrong dataset/GPU/secret | Correct the setting and rerun unchanged |
| Different source/environment | Preserve output and the exact notebook; report the mismatch |
| Worker stops with traceback | Keep `WORKER_STOP.json`, output and notebook transcript |
| HF publication pending | Keep output; inspect `HF_PENDING.json` and `LAST_HF_PUBLICATION.json` |
| Live owner conflict | Stop the other copy cleanly or wait for its 45-minute lease |
| HTTP429 / local budget | Allow the server-directed wait; never reset request history |
| Invalid support / determinism check | Preserve evidence; do not disable the check or change loss/data |

## Finish and return

Success prints `engineering_gate_passed`, with actual dual-T4 and actual
HF-restored continuation true. `measured_fit_ready=false` is intentional: this
notebook proves preparation, while the complete measured telemetry/shard/staging
integration still requires acceptance. Save the executed notebook with outputs.
The next step is independent HF verification, documentation update and final
NB17 integration/freeze. Do not rerun old training or start another classifier.
