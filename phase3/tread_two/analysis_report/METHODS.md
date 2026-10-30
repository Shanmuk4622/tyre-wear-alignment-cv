# ATLAS completed pilot — post-hoc methods

All new CSVs are derived from immutable saved NB08 records. No Torch/model
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
