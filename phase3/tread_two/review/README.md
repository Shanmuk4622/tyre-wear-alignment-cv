# ATLAS completed-run review — code and execution

<!-- NB09_GATE_UPDATE_20261030 -->
**Completion update, 30 October:** all four NB09 cells and all 51 HF files are
independently verified. [Completion audit](../../phase3NB09Completion.md).
No rerun is needed. Next is the separate [NB10 gate](../appearance_gate/README.md).
Earlier local-only validation descriptions below retain their original scopes.

**30 October 2026.** New post-hoc analysis of immutable NB08 evidence. This
directory does not change `../pilot`, the original notebook, source data or
either workstation.

| Module | Responsibility |
|---|---|
| [review_atlas.py](review_atlas.py) | Pinned public reads; hash/cache/atomic download checks; 29,410 predictions, sampling/gradient/selection audit, exact reference and patch arithmetic; 30 derived CSVs |
| [render_atlas_review.py](render_atlas_review.py) | Three saved-data scientific figures, complete self-contained interactive explorer, methods, README and table inventory |
| [publish_atlas_review.py](publish_atlas_review.py) | Immutable separate review bundle and optional HF publication; reuses audited direct-file publisher plus transport repair; no fit launch |
| [Notebook builder](../../analysis/build_nb09_atlas_review.py) | Self-contained four-code-cell NB09; 30 embedded files, frozen original source and pinned manifest |
| [Independent checks](../../analysis/validate_atlas_review.py) | Source-result crosschecks, cache faults, actual JavaScript controls, extracted CPU reproduction and offline HF wrapper exercise |

## Kaggle run

Open [NB09](../../notebooks/phase3_NB09_TREAD_TWO_ATLAS_Completed_Review.ipynb).
Select **None / CPU**, enable **Internet** and the Kaggle **HF_TOKEN** secret.
No dataset attachment is required: the original report contains the images,
and the scientific input is already saved on HF. Keep **2 GiB free** and choose
**Run All**. No T4 or new model forward pass is used. The tested Hub client is
pinned; Torch/CUDA are not changed. Secret access occurs only in the final
publication cell, with no token-file or `.env` fallback.

| Cell | Result |
|---:|---|
| 1 | Verify/extract sources and define the stop-aware launcher |
| 2 | Verify about 147 MB / 522 source files, audit and render; prepare a complete immutable local bundle |
| 3 | Show endpoint/probe metrics, figure, explorer link and prepared destination |
| 4 | Publish direct CSV/JSON/Python/PNG/HTML to `phase3/tread-two/atlas-v1/review/<contract>`, verify and release ownership |

All source inputs come from pinned revision
`82e7e1c5774f54c786571cdef7b3b60c1e96630f`. The review contract includes code,
environment, original manifest, transport identity and output hashes. Every
local CSV is validated before preparing a bundle. Original scientific files
remain under the existing `fit/fe7dc763…` namespace.

## Recovery and publication

Verified cached downloads are reused; mismatched or incomplete files are
downloaded and hash-checked before atomic replacement. Direct derived CSVs
are written atomically. The finite analysis is repeatable from saved rows;
it does not need model/optimizer epoch recovery. Different code/environment/
artifact identities receive a distinct review contract.

The complete bundle is local before publication. There is one major-completion
flush, using the existing immutable/CAS/one-owner publisher and the audited
five-minute HTTP / separate hourly commit policies. An identical final manifest
is acknowledged without another upload. Catchable Stop/error during publication
allows a flush attempt or same-bundle retry; server backoff can delay it. Forced
kernel death cannot publish unsent files. Rerun the same cell; do not delete or
steal live leases. No ZIP, NPZ or database upload is introduced.

## Read the results

[Findings and next gate](../../phase3TreadTwoFindings.md),
[local interactive report](../analysis_report/ATLAS_Failure_Explorer.html),
[direct table inventory](../analysis_report/CSV_INDEX.md),
[local verification receipt](../review_validation/VALIDATION.json).
Local checks and the prepared receipt do not establish actual NB09 Kaggle/HF
completion. The user-run notebook is retained for that later independent audit.
