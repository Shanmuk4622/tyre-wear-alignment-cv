# ATLAS-LC r2 — research model card

**Status: measured fit completed; research-only and unpromoted.**

Paired local-contrast reference evidence.

| Item | Recorded value |
|---|---|
| Seed / fit | 1 / 60 epochs / 2,940 updates |
| Validation-selected epoch | 7; original selection retained |
| Frozen source inventory | 30 entries / `fd9c1b2940d79b2dcb65f712366c859832a876f070f69c2a696a7e9cd9f22bc3` |
| Selected native checkpoint SHA-256 | `f6156aa560aeb880317c6acd745ec1b287aa615fd76656411059033b6a5959fd` |
| Final native checkpoint SHA-256 | `a64be8a17bba4c8d630aaa8efe7f5dc6b5771272452425a853b1670c7e3cb9af` |

| Endpoint | Observed-class BA | Low / high recall | NLL | Brier | ECE |
|---|---:|---|---:|---:|---:|
| final | 50.00% | 0/62 / 41/41 | 1.369872 | 0.854731 | 0.538923 |
| selected | 50.00% | 0/62 / 41/41 | 0.961273 | 0.568355 | 0.425969 |

[Selected native artifact](https://huggingface.co/datasets/Shanmuk4622/tyre-wear-study/resolve/3d5ae39602c3b25caa6af65c1e8d45f53a08e1c8/phase3/tread-two/atlas-lc-r2/fit/ae918ce787775192ce38031936cbfd8285b51b72fbbedb1da570c9271018eb28/evidence/artifacts/epochs/epoch_0007-f6156aa560aeb880317c.pt) · [Final artifact](https://huggingface.co/datasets/Shanmuk4622/tyre-wear-study/resolve/3d5ae39602c3b25caa6af65c1e8d45f53a08e1c8/phase3/tread-two/atlas-lc-r2/fit/ae918ce787775192ce38031936cbfd8285b51b72fbbedb1da570c9271018eb28/evidence/artifacts/epochs/epoch_0060-a64be8a17bba4c8d630a.pt) · [Fit receipt](../../tread_two/nb11_completion/16fa0443f902/VERIFICATION.json) · [Architecture / findings](../../phase3TreadTwoLCArchitecture.md).

Classes are mileage proxies, not physical wear, tread depth or safety. The 103-image historical endpoint is already inspected and has low/high targets only. Scores do not establish unseen-tyre reliability. Training repeats correlated views of the same existing tyres. Retained dense geometry comes from the frozen TREAD-ONE branch; no new dense improvement is attributed to this classifier. Signed evidence/attention records internal calculations and is not a causal physical explanation.

[Source and capability limits](README.md) · [Complete artifact registry](artifacts.csv). Weights are referenced by immutable HF revision/object SHA; this handoff does not deserialize or republish them.
