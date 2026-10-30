# ATLAS-LC completed pilot — post-hoc methods

All new CSVs are derived from immutable saved NB11 records. No Torch/model
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
| Reconstructed logit | Sum of 18 reference contributions plus class bias |
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

## R2 paired training and source permissions

Training observations are two views per draw. The combined online epoch metric
has784view observations per epoch; view-specific metrics have392. All retain
pre-update state ownership. CE/JS/weighted embedding losses are independently
reconstructed from both saved predictions and patch distances; joint update loss
is the mean of eight draw totals. Seeds regenerate brightness/contrast/gamma
factors; no fresh tensor or model inference is used.

All386train photos/frames are eligible queries. Full source SHA suppresses a
matching literal reference and scales its reference class by6/n_allowed,using
no query target. The held-out endpoint has no identical bank source and all18
references allowed. Thus endpoint reference sums use base signed coefficients;
training SHA permissions/scales are audited separately for every observation.
Greyscale invariance follows LC's RGB mean; it is not learned improvement.

The v1 comparison uses pinned NB09 cohort metrics on the same historical
endpoint; r2 additionally has two gamma probes unavailable in v1. Absent v1 rows
remain absent. Changed bank/query/sampler/objective and representation prevent
single-factor causal attribution. Confidence risk curves are descriptive; no
threshold is selected from this endpoint for deployment.
