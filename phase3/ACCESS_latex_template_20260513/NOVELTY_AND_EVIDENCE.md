# Novelty and evidence for manuscript revision 3

Revision 3 adds principal-angle and decoder Jacobian measurements and a train-only affine readout on frozen tokens. At five modes, the local advantage of the fitted bases persists under the affine readout, while masked retention drops from 83.96% / 82.51% to 18.07% / 10.62%. The analysis concerns one checkpoint. [Reviewer response](REVIEW_RESPONSE.md)

**6 November 2026.** The paper now centers an equal-budget spatial audit and a
decoder-order diagnostic. The previous manuscript is recorded in repository
revision `7f82528`. The new draft retains the IEEE Access format.

| Contribution | What is established here | Closest published ideas |
|---|---|---|
| Matched spatial budget for aligned texture tokens | Same shared position template, constant mode, spatial mass, frozen decoder, and (48k) input scalars for every method | FcaNet relates the mean to a DCT coefficient. Token Pooling uses clustering to approximate token sets. |
| Three exact fitting objectives under one constrained subspace | A 31-dimensional spatial eigensystem minimizes each squared surrogate. The omitted eigenvalue sum equals its training distortion. | PCA and gradient-based subspace methods supply the mathematical ingredients. The new comparison applies them to the same fixed decoder and canonical atlas. |
| Decoder-order accounting | Changing the order separates output-space projection loss from a signed order effect. A weighted absolute-output bound exposes changes that cancel in aggregate task loss. | Task-driven compression often trains a bottleneck. Here the decoder is fixed and the output-first path is a diagnostic with a different scalar budget. |
| Retrospective empirical result | Train-fitted bases outperform the tested fixed bases at five modes. Decoder-aware surrogates yield small, inconsistent gains over token PCA at that budget. | The comparison adds a fixed-budget result and component sensitivity to the original nine aggregation interventions. |

## Findings at five modes

| Result | Validation | Historical test |
|---|---:|---:|
| Local retention, ordinary token PCA | 89.47% | 89.01% |
| Local retention, fixed spatial moments | 81.71% | 82.17% |
| Local retention, token PCA with equal component weight | 89.68% | 87.92% |
| Local retention, six separate component omissions | 88.46–89.47% | 86.45–89.01% |
| Masked retention, ordinary token PCA | 83.96% | 82.51% |
| Masked retention, fixed spatial moments | 65.81% | 65.29% |

At five modes, the local decoder receives a reconstruction from 240
image-dependent coefficients instead of 1,536 token entries. Shared basis and
template values are counted separately. The comparison measures coefficients.
Quantization, bit rate, throughput, and energy would need their own evaluation.

The matched-budget spatial moments use a fixed mass. The earlier intervention
uses each image's own mass. Their rounded local validation values of 81.71%
and 81.76% therefore refer to different protocols.

## Review boundaries

- The four additional primary sources establish the antecedents. The claim is
  the specific audit, its controlled comparisons, and its observed result.
  See the [citation audit](CITATION_AUDIT.md).
- Train-derived bases use clean view 0 only. The retained TRACE model and
  historical cohorts predate this revision. Both held-out roles were
  previously inspected and each has two tyre components.
- A squared surrogate optimum applies to the fitting objective. It gives no
  guarantee of nonlinear descriptor-loss optimality, decoder refit performance,
  or cross-tyre transfer.
- Component omission changes composition and sample count. The fixed training
  mass stays common, and the template and bases are rebuilt for each omission.
- TRACE remains unselected under its original gate. The measured targets are
  image descriptors. A physical inspection claim requires calibrated references.

## Reviewable evidence

| Record | Contents |
|---|---|
| [Frozen protocol](evidence/spatial_budget_protocol.json) | Fit cohorts, matched methods, budgets, primary comparison and limits |
| [Summary CSV](evidence/spatial_budget_summary.csv) | 1,488 role/view/basis/budget summaries with feature error, signed loss and output drift |
| [Component CSV](evidence/spatial_budget_components.csv) | Component-specific five-mode scores |
| [Descriptor CSV](evidence/spatial_budget_descriptors.csv) | Absolute loss by descriptor, role and view |
| [Fit cohorts](evidence/spatial_budget_fits.csv) | Counts and excluded training-component identifiers |
| [Computation receipt](audit/SPATIAL_BUDGET.json) | Source and output hashes, numerical checks and readout count |
| [Validation receipt](audit/VALIDATION.json) | Independent reduction of 282,720 paired image rows |

The detailed image and fitted-vector CSVs are local under
`.build/spatial_budget/`. Their hashes are in the computation receipt.
