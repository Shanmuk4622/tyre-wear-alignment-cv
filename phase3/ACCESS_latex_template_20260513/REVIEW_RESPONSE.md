# Reviewer response and revision record

This revision addresses the supplied major-revision review in the requested IEEE Access template. The revisions strengthen the methods and evidence for the paper's central contribution, a matched-budget audit of local texture evidence under a frozen decoder.

## Scientific revisions

| Review point | Change and evidence |
|---|---|
| Make TRACE independently implementable | Section IV and Table III give the layer widths, kernels, strides, padding, normalization, tensor shapes, and parameter counts. Figure 1 traces image and support tensors, metadata, mask placement, shared modules, and decoder outputs. An execution check on the actual source verifies 71,022 parameters and shared-module calls. Changing hidden RGB leaves masked predictions bit-identical when geometry and support are fixed. [Architecture receipt](audit/TRACE_ARCHITECTURE.json) |
| Explain the similar fitted curves | The four nonconstant modes of Jacobian and decoded PCA overlap token PCA by 0.9773–0.9896. Fixed moments and DCT overlap by 0.4581–0.4859. Median relative Jacobian variation on saved full-support train tokens reaches 31.65% locally and 69.22% in masked context. Position-specific linearization reduces error, with appreciable residuals in both branches. [Angles](evidence/review_subspace_angles.csv), [Jacobian variation](evidence/review_jacobian_variation.csv), [linearization error](evidence/review_linearization_error.csv) |
| Check whether the result survives a different readout | Two cell-specific affine ridge readouts were solved on frozen, clean-view train tokens. The five-mode fitted bases still beat fixed bases and all 16 random controls locally. Masked PCA retention falls from 83.96% / 82.51% with the original decoder to 18.07% / 10.62% with the affine readout, relative to each readout's own full-budget advantage. This addresses the lightweight-readout option. A comparison across epochs and seeds remains open. [Affine results](evidence/review_affine_readout.csv), [computation receipt](audit/REVIEWER_ANALYSIS.json) |
| Correct the stencil, covariance notation, and comparison details | Section IV defines the Laplacian and its padding. Sections V–VI identify position-wise covariance indices. The stated 7.76 and 6.85 percentage-point improvements use unrounded inputs. The original intervention uses image-specific coverage mass, whereas the matched-budget study uses fixed training mass. |
| Reduce operational detail in the paper | Section IX summarizes the verification scope and links the pinned fit and review. Exact identities, retention counts, audit boundaries, and regeneration commands are in the companion notes. [Reproducibility](REPRODUCIBILITY.md) |
| Qualify the evidence and novelty | The current evaluation covers two held-out components per role, with no mid-class examples in those roles. The historical test was reused during development. The manuscript scopes its novelty to a geometry-anchored, task-specific representation audit with matched budgets, separated order effects, and a readout sensitivity analysis. [Claim ledger](CLAIMS_AND_LIMITATIONS.md) |
| Clarify the implication for depth | The atlas, descriptors, and PCA modes preserve evidence about visible tread texture. The study contains no calibrated groove-depth targets or millimeter error estimates. A physical depth claim requires calibrated capture, matched reference measurements, and validation on independent tyres. [Monocular depth and scale](https://arxiv.org/abs/1406.2283), [TireEye optical cross-section study](https://papers.phmsociety.org/index.php/phmconf/article/view/3242) |

## Figure revisions

| Reviewed figure | Revision |
|---|---|
| TRACE architecture | Added exact shapes, explicit routing, and labels for the reused modules. The layer table follows the computation graph. |
| Classifier endpoints | Added the uniform three-output NLL reference and direct numeric bar labels. |
| TRACE training | Added an inset at the observed rank range while retaining the original rank floor on the main axes. |
| Spatial budget | Added a companion plot of signed percentage-point differences from token PCA, split by branch and held-out role. |
| Aggregation interventions | Added a zero reference and spatial mode counts. |
| Descriptor losses | Offset the descriptor markers vertically to reveal overlaps without changing the measured losses. |
| Qualitative readout | Added the five-mode PCA reconstruction of the same two deterministic validation examples on the same descriptor scale. [Cell values](evidence/review_qualitative_pca.csv) |
| Representation diagnostics | Shortened the covariance panel title and retained the explanation of the fixed training-mass partition in its caption. |

## Template and author details

The supplied biographies and `a1.jpg` and `a2.jpg` are used in author order. The byline follows the supplied biography names, Edara Sreenivasa Reddy and Shanmukesh Bonala. Correspondence uses `sreenivasareddy.e@vitap.ac.in`. The manuscript uses one acknowledgment with a direct link identifying the AI system and specifying its roles in drafting, editing, LaTeX, and analysis code. The references are reserved for research works. At submission, review the venue's requirements for the placement and form of the AI citation. [IEEE Access submission guidelines](https://ieeeaccess.ieee.org/authors/submission-guidelines/)

The previous Phase 3 paper folder was copied and hash-checked before revision and removed after the new PDF passed verification. Its final PDF is retained in the local build history. The editable manuscript lives in this supplied template. [Migration inventory](audit/MIGRATION_INPUTS.json) · [Completion receipt](audit/MIGRATION_COMPLETION.json)
