# NB24/NB25 — 120-image submission verified

<!-- current-status:start -->
> **Current status (15 September 2026):** [Completed work and remaining validation](CURRENT_STATUS.md). The report is refreshed; the app, learned-geometry integration and target-assisted alignment software exist. Dated plans below retain their original context.
<!-- current-status:end -->

**Subsequent identity confirmation:** user states the 12 sessions represent 12
different physical tyres. Recorded as user-confirmed, not independently verified.
NB26–NB29 are now implemented; 72/24/24 image split. See [current run guide](32_HRNET_NOTEBOOK_RUN_GUIDE.md).
The unanswered-identity and unimplemented-training notes below are historical.

15 September 2026. **Both notebooks executed successfully; no rerun required.**

- NB24 publication: `78d90571a46adab0cb41ed40d5dcd1d63628e19c`.
- NB25 publication: `2b4773914e6eefe421d8ddd74e79c5b84606c9aa`.
- Package: `100dac5b12e1192ede59f9d1231c5aeeed80428850661a9add1cba147fe22f92`.
- Annotation SHA-256: `8e8fd0734b9f4fe4236e699b804c2f8afb85ae4bb847f8d3c83b0abff5a8769e`.
- [Public package and reviews](https://huggingface.co/datasets/Shanmuk4622/tyre-wear-study/tree/2b4773914e6eefe421d8ddd74e79c5b84606c9aa/s9/s9-geometry-labels-r2-all120/packages/100dac5b12e1192ede59f9d1231c5aeeed80428850661a9add1cba147fe22f92).

## Verification performed

Neither saved notebook contains an exception output. NB25 prints 120/120 complete.
The public manifest matches local bytes and is identical at both publication
revisions. The public annotations exactly match the local export in ALL_120_IMAGES.
Independent validation matches every published review record; plan/package hashes
also match. All 120 extracted original JPEGs match their manifest hashes.
Downloaded only 250,884 bytes of public JSON; no remote image/ZIP/weight download,
HF write, training or modification of user annotations.

[Machine audit](../outputs/s9_120_completion/AUDIT.json) and
[validation records](../outputs/s9_120_completion/REVIEW.json) preserve the checks.

## What the labels currently say

120 complete images, 720 visible points, zero uncertain/occluded/outside-frame
states. All 120 observations say no visible issue; all independent records are
unknown. This is completion of the required fields, not certification of visibility,
anatomical meaning, surface condition or healthy-reference eligibility.

Two visible points are within 10 pixels of a frame edge: G006 left-middle and
G018 left-middle. This is a review flag, not an automatic error. Native images
were inspected for these two cases; near-edge location alone does not justify
changing the labels. Fine crack-like surface lines are visible in these photographs,
so the all-no-visible-issue observations also need clarification before anomaly use.
No correction request for all 120 is made. Full point-overlay visual quality review
is still pending; this audit does not pretend all 720 points were visually approved.

## Next: review, split lock, HRNet smoke test then training

1. Review the 120 point overlays and boundary convention, returning only specific
   affected IDs if corrections are necessary. The annotation collection is complete;
   no additional image batch is requested now.
2. Confirm physical-tyre identity across the 12 capture sessions. The local identity
   ledger still has all 12 IDs blank. Ask the user which sessions share a tyre;
   do not infer independent tyres from session names or mileage folders.
3. Lock train/development/evaluation groups and audit the segmentation baseline's
   training overlap. If identities cannot be established, explicitly agree a
   session-held-out exploratory study, not independent new-tyre validation.
4. Build/run the HRNet resume smoke test and then the planned training/comparison
   notebook using that approved label/split contract. No HRNet training notebook
   has been implemented or run by this verification turn.

The detailed design remains in [docs/30](30_S9_HRNET_AND_COMPLETION_PLAN.md).
PatchCore's verified-reference requirement and physical-angle validation are
separate and still open. Full S9 is not complete merely because labels are uploaded.
