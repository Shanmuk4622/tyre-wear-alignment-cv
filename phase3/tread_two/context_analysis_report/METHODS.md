# ATLAS-CONTEXT completed pilot — post-hoc methods

All new CSVs are derived from immutable saved NB13 records. No Torch/model
loading, optimizer, new inference, new checkpoint selection or new tyre is used.
Source revision, input hashes and exact row counts are retained in ANALYSIS.json
and input_inventory.csv. Missing values mean unavailable, including absent
mid-class targets. Counts of repeated training draws are not unique image counts.

## Calculations and units

| Quantity | Definition / units |
|---|---|
| BA observed | Mean recall over target classes actually present; fraction |
| NLL | Negative natural log of saved true-class probability, three classes |
| Brier | Sum of squared differences from one-hot target, three classes |
| ECE | Ten equal-width maximum-confidence bins, weighted absolute accuracy gap |
| True margin | True-class logit minus largest other-class logit |
| Pooled reference affinity | Sum over 32 patches of saved geometric mass × exp((cosine−1)/temperature) |
| Reference contribution | Pooled affinity × the saved signed class/reference coefficient |
| Reconstructed logit | Sum of 18 frozen reference contributions plus weighted context correction and combined class bias |
| Patch concentration | Largest absolute patch contribution / total absolute mass, for predicted class |
| Probe delta | Probe minus identity, same image and checkpoint; never a different epoch/model |
| Training accuracy | Online pre-update draws with replacement and augmentation, mixing model states; not a fixed final training-set test |
| Dense error | Fraction of image width; only comparable guide rows and actual target kind |

The grayscale probe modifies both queries and bank, brightness modifies query
only, and the occlusion probe replaces a single top-signed patch with its saved
per-channel mean. None uniquely isolates paint, identity, light or capture
devices. Exact additive accounting is not causal explanation. No confidence
threshold is selected for deployment. No image-wise confidence interval is
reported as independent-tyre evidence. Existing low/high validation and test
roles are preserved; test is an already-inspected development endpoint.

The figure/model-selection and probability checks are independent calculations
from CSV. The original frozen anchor remains the control on original-frame RGB,
not the transformed query. No comparisons suggest transformed-anchor forwards.
All original native states and CSVs remain in their original HF namespace.

## Context and attention accounting

Every observation retains a frozen LC logit, 32 weighted spatial correction vectors and three bounded global bias corrections. Their sum must equal the combined saved logits; corrections are bounded by 2×pooling mass per tile and 0.5 globally. The paired objective additionally includes 0.02 times the mean two-view squared total correction. Per-pair CE/JS/contextual cosine loss and per-update means are independently reconstructed.

All 53,754 correction records and 1,720,128 source-patch mappings/bounds are checked. Endpoint/probe reference affinities, weights and combined patch values are audited for all 1,854 cases. Full training affinities are retained in raw HF CSV; this review does not independently recalculate every training affinity. The separate full publication audit verifies all raw bytes/schemas/row counts.

Full attention matrices exist only for 206 selected/final identity endpoints: two layers, four heads, 32 query rows and 32 keys. Every value and masked key is checked; rows sum to one. Normalized entropy divides by log(max(2, valid keys)); query rows are averaged with geometric pooling mass. Saved per-head maxima use the same weights. The explorer displays seven-decimal copies for interaction; raw CSV retains original precision. No full training/probe matrices are invented from summary statistics.

The frozen selected LC epoch7 prior is checked against pinned peer logits on all206 identity endpoint images. It is not LC's final checkpoint. LC/context/v1 comparisons describe changed models on an already inspected cohort, not an independent single-factor causal experiment or a deployment gate.
