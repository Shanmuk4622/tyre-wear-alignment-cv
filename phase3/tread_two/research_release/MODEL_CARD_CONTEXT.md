# ATLAS-CONTEXT r3 — research model card

**Status: measured fit completed; research-only and unpromoted.**

Frozen selected LC prior plus bounded positional correction.

| Item | Recorded value |
|---|---|
| Seed / fit | 1 / 60 epochs / 2,940 updates |
| Validation-selected epoch | 7; original selection retained |
| Frozen source inventory | 32 entries / `f63d309ebed9dec516dba02ef8ed082b65b1cbbbfe2132368f47cc136f4193b8` |
| Selected native checkpoint SHA-256 | `913b6a667a9d5c5d5572e804a5cdd930949901654fe8d5f4d08bbb3e8b8f8ab6` |
| Final native checkpoint SHA-256 | `3ed868c05ba228cb0c874226abea2bac1be66e010d16039b99dba8ed34f5d9ee` |

| Endpoint | Observed-class BA | Low / high recall | NLL | Brier | ECE |
|---|---:|---|---:|---:|---:|
| final | 50.00% | 0/62 / 41/41 | 2.358475 | 1.129997 | 0.561834 |
| selected | 50.00% | 0/62 / 41/41 | 2.174813 | 1.105581 | 0.552019 |

[Selected native artifact](https://huggingface.co/datasets/Shanmuk4622/tyre-wear-study/resolve/a44418844ba3f1c98adee2cae76a8bbf9f5f4e9e/phase3/tread-two/atlas-context-r3/fit/36f473b0886d6c2f49c416436d270198af3982230e3045da9969b0bf42169625/evidence/artifacts/epochs/epoch_0007-913b6a667a9d5c5d5572.pt) · [Final artifact](https://huggingface.co/datasets/Shanmuk4622/tyre-wear-study/resolve/a44418844ba3f1c98adee2cae76a8bbf9f5f4e9e/phase3/tread-two/atlas-context-r3/fit/36f473b0886d6c2f49c416436d270198af3982230e3045da9969b0bf42169625/evidence/artifacts/epochs/epoch_0060-3ed868c05ba228cb0c87.pt) · [Fit receipt](../../tread_two/nb13_completion/8bdf46ec2040/VERIFICATION.json) · [Architecture / findings](../../phase3TreadTwoContextArchitecture.md).

Classes are mileage proxies, not physical wear, tread depth or safety. The 103-image historical endpoint is already inspected and has low/high targets only. Scores do not establish unseen-tyre reliability. Training repeats correlated views of the same existing tyres. Retained dense geometry comes from the frozen TREAD-ONE branch; no new dense improvement is attributed to this classifier. Signed evidence/attention records internal calculations and is not a causal physical explanation.

[Source and capability limits](README.md) · [Complete artifact registry](artifacts.csv). Weights are referenced by immutable HF revision/object SHA; this handoff does not deserialize or republish them.
