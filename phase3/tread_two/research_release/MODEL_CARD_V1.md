# ATLAS v1 — research model card

**Status: measured fit completed; research-only and unpromoted.**

Shared local reference evidence on RGB.

| Item | Recorded value |
|---|---|
| Seed / fit | 1 / 60 epochs / 2,940 updates |
| Validation-selected epoch | 18; original selection retained |
| Frozen source inventory | 24 entries / `5f29f62840452059ae0c3acdfc7703a4022d9b011ce0b66b0b558cce8511d493` |
| Selected native checkpoint SHA-256 | `0fa02aa3b1e1c9a08245324a0257bd1918a52a1b490f790b01f7abef186eb634` |
| Final native checkpoint SHA-256 | `1c869b114d9f02d743df0f7a1e30069ee4c181eb7c9b740621890de02b87754f` |

| Endpoint | Observed-class BA | Low / high recall | NLL | Brier | ECE |
|---|---:|---|---:|---:|---:|
| final | 50.00% | 0/62 / 41/41 | 1.428298 | 0.881229 | 0.602820 |
| selected | 50.00% | 0/62 / 41/41 | 1.178296 | 0.711430 | 0.162204 |

[Selected native artifact](https://huggingface.co/datasets/Shanmuk4622/tyre-wear-study/resolve/82e7e1c5774f54c786571cdef7b3b60c1e96630f/phase3/tread-two/atlas-v1/fit/fe7dc763fe6d4b64ce8ccf2f22c4222b6da28ca7975c925d56ba70c3e09d19dd/evidence/artifacts/epochs/epoch_0018-0fa02aa3b1e1c9a08245.pt) · [Final artifact](https://huggingface.co/datasets/Shanmuk4622/tyre-wear-study/resolve/82e7e1c5774f54c786571cdef7b3b60c1e96630f/phase3/tread-two/atlas-v1/fit/fe7dc763fe6d4b64ce8ccf2f22c4222b6da28ca7975c925d56ba70c3e09d19dd/evidence/artifacts/epochs/epoch_0060-1c869b114d9f02d743df.pt) · [Fit receipt](../../tread_two/pilot_validation/nb08_followup/82e7e1c5774f/VERIFICATION.json) · [Architecture / findings](../../phase3TreadTwoFindings.md).

Classes are mileage proxies, not physical wear, tread depth or safety. The 103-image historical endpoint is already inspected and has low/high targets only. Scores do not establish unseen-tyre reliability. Training repeats correlated views of the same existing tyres. Retained dense geometry comes from the frozen TREAD-ONE branch; no new dense improvement is attributed to this classifier. Signed evidence/attention records internal calculations and is not a causal physical explanation.

[Source and capability limits](README.md) · [Complete artifact registry](artifacts.csv). Weights are referenced by immutable HF revision/object SHA; this handoff does not deserialize or republish them.
