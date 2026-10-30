# Phase 3 working instructions

<!-- NB11_COMPLETE_UPDATE_20261030 -->
## Current — NB11 complete; NB12 review ready

**30 October 2026. NB11 is complete and independently HF-verified:** 60 epochs / 2,940 updates, 1,351 direct files and 54,336,368 raw scientific rows. The actual T4×2/live-HF preflight passed; ownership is released. No NB11 rerun is needed. [Full completion and findings](phase3NB11Completion.md).

- Preserve executed NB11 (`16fa0443…`), all 30 frozen r2 source/assets, all epoch weights and the completed fit namespace. Do not regenerate executed notebooks or change an active/completed protocol.
- The candidate is not promoted: endpoint BA stays 50%, low recall is 0/62. Improved stability/NLL are narrower findings; ECE worsened.
- Existing tyre data only; no new acquisition. Both workstations remain deferred and unchanged.
- Next is the separate CPU saved-evidence review/publication. Distinguish exhaustive raw publication verification from the 696-input scientific review and pending actual NB12 publication.
- Keep actual draws separate from two correlated view observations. Preserve all direct CSV/native evidence and exact declared gaps; never ZIP.

Next: [NB12 saved-evidence review](notebooks/phase3_NB12_TREAD_TWO_ATLAS_LC_Completed_Review.ipynb), **CPU / Internet / HF_TOKEN / no dataset attachment**, then Run All. It adds 40 derived CSVs / 380,595 rows, four figures and a 1,442-case explorer with zero fitting or model forwards. Local recovery/cache/publication/package checks pass; actual NB12/HF review publication is pending. [Run guide](tread_two/lc_review/README.md).

<!-- END_NB11_COMPLETE_UPDATE -->

## Git content policy — 30 October 2026

- Keep the root and Phase 3 ignore rules current as new notebooks add local HF
  mirrors, resume state, publication bundles or acceptance fixtures. Full model
  artifacts and telemetry belong locally/on HF; do not force-add them to Git.
- Keep current runnable notebooks, code, small provenance receipts, compact
  summaries and explicitly curated documentation figures visible. Six small
  frozen reference-bank NPY inputs are deliberately allowed and must remain
  available for `FILES.json` verification and notebook generation.
- Before committing, run `phase3/run_cv.cmd -B scripts/check_git_content.py --history`
  from the repository root and review `git add --dry-run .`. The check is read-only.
- The October cleanup removed one model binary and six archived notebook copies
  from tracking only; local files and Git history remain intact.
- See [push/content guide](../docs/GIT_PUSH_CHECK.md) and
  [policy validation](audit/GIT_IGNORE_POLICY_CHECK.json).

<!-- NB10_LC_UPDATE_20261030 -->
## Latest status — NB10 closed; NB11 ready, 30 October2026

NB10 is completed and independently HF-verified:495 direct files,1,918,052 scientific rows,386native training records,T4×2,zero fitting/held-out forwards and released ownership. No NB10 rerun is needed. [Completion proof](phase3NB10Completion.md).

- Preserve the executed NB10 (`499c84cf…`) and all earlier frozen studies. NB10 is a zero-update training-only gate; its bank proposal is not installed in v1.
- ATLAS-LC r2 is separately implemented/frozen under `tread_two/lc_pilot/`,new HF namespace `phase3/tread-two/atlas-lc-r2`. Use NB11; do not rewrite active source/assets/contracts.
- Existing data only; all386training queries remain eligible with mandatory full-source SHA self exclusion and6/n reference scaling. Never substitute label-conditioned references.
- Log both appearance views and all24CSV families/285columns,complete recovery state and declared gaps. Keep all60epoch weights; direct CSV/native publication,never ZIP.
- Local fixture/package checks passed. Distinguish those from pending actual T4×2/live-HF/fit and unmeasured accuracy. No workstations changed.

Next is [NB11 / ATLAS-LC r2](notebooks/phase3_NB11_TREAD_TWO_ATLAS_LC_Resumable_Pilot.ipynb): one separately versioned seed,paired LC views,18-photo source-SHA exclusions,all386queries and sqrt-size group sampling. Local batch8 recovery/package checks pass; actual Kaggle/live-HF preflight and measured accuracy remain pending. Use T4×2,Internet,the existing expanded Kaggle dataset and HF_TOKEN. [Architecture](phase3TreadTwoLCArchitecture.md), [protocol](phase3TreadTwoLCProtocol.md).

<!-- END_NB10_LC_UPDATE -->
<!-- NB09_GATE_UPDATE_20261030 -->
## Latest superseding status — 30 October 2026

- NB09 completed on Kaggle and is independently HF-verified at `93d7ea34…`: 51 direct files, 30 CSVs / 246,157 rows, released owner. No rerun is needed; [completion](phase3NB09Completion.md).
- Preserve the supplied executed NB09 and updated executed NB08. Do not force old clean-delivery hashes onto executed notebooks. NB08's original 25-file payload and 1,177 published scientific objects remain unchanged; distinguish refreshed administrative/history paths from scientific rewrites.
- The training-only appearance/reference gate is implemented and fully executed locally: 386 train images, 1,918,052 scientific rows, zero fitting or held-out forwards. The 18-photo bank is UNTRAINED and must not be installed in v1; two high groups retain only one query photo. Preserve the repetition/artifact caveats.
- Next is NB10: T4×2, existing expanded Kaggle input, Internet/HF_TOKEN. Distinguish the one-GPU local proof from pending actual dual-T4/live-HF gate execution. A future model needs a separately frozen representation, support/query and repetition policy; do not edit the completed fit or workstations. [Gate findings](phase3TreadTwoAppearanceGate.md).

Recorded from the user's instructions and the existing project documentation on 27 October 2026. These instructions apply to work in this folder. Later explicit user instructions take precedence.

## Start and maintain context

- **Latest user scope, 29 October 2026:** use only the existing tyre photos/data. The user will not collect additional tyre photos. This supersedes the earlier assumed user-assisted acquisition plan. Do not make new acquisition a prerequisite for the current requested work. Preserve the documented cohort/generalization limits and historical prospective-readiness result.
- **Completed workstation delivery:** TREAD-ONE has a separate, visibly distinct desktop workstation under `phase3/`, with its own launcher, model/runtime, settings, cache and evidence outputs. Keep the original `prototype/` workstation and its selected models untouched. Use the declared validation-selected export; do not choose another checkpoint by test score or retrain without a separate reason.
- **Latest direction, 29 October 2026:** develop a creative, project-specific TREAD-TWO concept using existing data and the observed TREAD-ONE evidence. Workstation changes are deferred at the user's request. Keep both workstation source trees and selected checkpoints untouched. Distinguish design/prototype checks from a measured trained model; do not present an untrained branch's scores as results.

- **Latest implementation authorization, 29 October 2026:** the user approved implementing TREAD-TWO / ATLAS. NB08 and the isolated pilot runtime are now delivered with local engineering acceptance; actual dual-T4/HF gates and the measured fit remain pending Kaggle execution. Preserve the frozen source/bank/recipe and namespace once that fit starts. Keep both workstations unchanged. The run guide and pilot validation notes distinguish fixtures from study outcomes.
- **NB08 repair, 30 October 2026:** the supplied run passed the actual T4/HF gate, then a local hourly HTTP cap timed out. The audited external launcher in `analysis/nb08_transport_launcher.py` repairs transport without editing the original 24 hashed source/assets, experiment identity or checkpoints. Preserve its separate snapshot provenance and compatibility checks; do not regenerate FILES/change the namespace to bypass the saved run. Actual repaired execution and measured completion remain pending; [repair record](phase3NB08Repair.md).

- **Superseding NB08 status, 30 October 2026:** the repaired measured fit is now published complete at 60 epochs / 2,940 updates, including selected/final evaluation and diagnostics; the owner is released. The user clarified that the latest stop had no displayed error. Do not restart fitting or apply a speculative repair. Preserve the unchanged notebook/runtime and use retained evidence for further review. [Completion/status and audit scope](phase3NB08CompletionStatus.md).

- **Completed ATLAS review / NB09, 30 October 2026:** full saved-result review is locally complete. Correct confusion is 62 low→mid / 41 high→high, correcting an earlier prose error; BA50% remains unchanged. Saved grayscale/dimming probes expose appearance sensitivity. Preserve 30 derived CSVs / 246,157 rows and complete explorer. Next is **CPU NB09** for additive HF review publication, not another fit. No data attachment is needed for this saved-evidence analysis. Local extraction/publication-fixture checks pass; actual NB09/HF completion stays pending until independently verified. Any new bank/input/model/sampler/objective requires a separate frozen protocol; do not edit the completed fit. [Findings and next gate](phase3TreadTwoFindings.md).

- Read [README.md](README.md), [phase3Understanding.md](phase3Understanding.md), [phase3Progress.md](phase3Progress.md) and relevant source documents before changing anything.
- Keep new Phase 3 deliverables in `phase3/`. Preserve earlier source data, annotations, executed notebooks, checkpoints, evidence and manuscript history.
- Maintain the progress, session, decision and evidence records as work happens. Keep proposed, implemented, locally checked, Kaggle-completed and independently validated states distinct.
- Resolve documentation conflicts by experiment scope, actual artifact identity, dated evidence and later user decisions. An old run instruction does not authorize repeating completed work.
- User reaffirmation, 27 October 2026: design models specifically around this tyre application's observed successes, failures, data and deployment constraints. Document the connection from each observation to the chosen component; distinguish a motivated design choice from an experimentally demonstrated improvement.
- Give technical documentation special care: record architecture/math, module and data flow, code ownership, dependencies/resources, execution/recovery behavior, metrics, limitations, failures, fixes and exact artifact provenance. Maintain these records as work progresses.
- User reaffirmation, 28 October 2026: write structured Markdown with useful diagrams, metric/inventory/comparison tables and clear evidence-linked next steps. Use diagrams to explain actual flows and decision gates; keep completed observations separate from proposed work.
- Historical user reaffirmation, 28 October 2026 (**acquisition assumption superseded by the 29 October existing-data-only decision above**): the diagnostic and later controlled revisions remain part of **TREAD-ONE**. Continually improve data coverage and evidence retention; the user plans to help collect more tyre data later. Prioritize new independent tyres and documented capture/annotation provenance, save authorized data/evidence in HF, and do not invent new subjects or scrape unrelated images as a substitute. Keep new-data acquisition separate from richer telemetry derived from existing data. Provide useful interactive image exploration where feasible, alongside durable CSV and static/Markdown evidence.
- Retaining training evidence is a high-priority standing requirement. For future runs, freeze an extensive telemetry schema before training; preserve every meaningful, feasible metric at the appropriate step/epoch/image/class/tyre/seed level, detailed predictions, errors and complete recovery state. Log explicit gaps and costs rather than claiming that uncomputed or unsaved metrics exist. Prefer retaining sufficient statistics/raw predictions for later analysis without retraining.
- Preserve an active experiment's runtime, generated notebooks, protocol, data and HF namespace. A casual reminder or progress report is not a request to modify or restart training. Plan nonessential improvements for a later revision.

## Local environment

- **Every local Python command must run in `cv_conda`: `conda activate cv_conda`.** See [phase3Environment.md](phase3Environment.md) for the current shell issue.
- Preserve the existing CUDA PyTorch installation and the prototype's isolated dependencies. Do not replace packages merely because a newer version exists.
- Training/runtime Python is the editable source of truth. Regenerate the relevant self-contained notebook after changing its source; do not hand-edit an embedded base64 payload or regenerate unrelated executed notebooks.

## Any future training deliverable

**Superseding user requirement, 28 October 2026: publish HF records directly, never as ZIP bundles.** Tabular training/validation/test/telemetry data must be directly browsable `.csv` files with a documented schema. Publish manifests/configuration as JSON and tensor recovery/weights in appropriate native binary files directly. Batch files in a single commit for consistency/rate control; do not hide metrics/history in ZIP, NPZ or checkpoint-only containers. This supersedes r1's archive-based publication. Preserve historical r1 artifacts and experiment identity; do not rewrite or delete completed evidence to disguise the old format.

- More evidence is a standing priority: freeze a substantially richer logging schema before the next measured run, retain training and validation per-image predictions, class/task/group metrics, scalar sufficient statistics, numerical events, and bounded resource/timing/gradient diagnostics. Do not call a small scalar summary complete telemetry. Audit which requested fields are actually stored before delivery.
- The prepared CSV publication uses the authorized Kaggle `HF_TOKEN` secret. Do not add project `.env` credential fallbacks or probe alternative credential files without explicit user authorization.
- Existing saved records may be exported/reanalysed without retraining. Label newly derived metrics as post-hoc calculations and mark unsaved measurements unavailable. More logged views/rows do not increase the independent tyre count.

- Deliver a Kaggle-compatible **`.ipynb`**, with exact input dataset, accelerator, Internet/Secrets settings, cell/run order, resume procedure, outputs and troubleshooting instructions.
- Design for a **dual T4 session** and describe actual GPU use honestly. Prefer the dataset attached directly from **Kaggle** for fast access; keep it read-only instead of duplicating it into output storage.
- Budget the **20 GB output area** carefully. The user notes that some sessions offer roughly 1 TB scratch; measure actual free space, filesystem location and memory limits before relying on that capacity. Scratch is temporary.
- Use Hugging Face account **`Shanmuk4622`**, secret **`HF_TOKEN`** from Kaggle Secrets, and the established dataset repository **`Shanmuk4622/tyre-wear-study`** where appropriate. Never print, copy or commit token values or `.env` contents.
- Persist complete reproducibility records: model, optimizer, scheduler, scaler, Python/NumPy/CPU/CUDA RNG, deterministic sample order, next completed-update cursor, partial epoch statistics, metrics, detailed logs, configuration, data/split/model/source hashes, environment, errors, repair history, selected/final checkpoints and predictions.
- Save atomically at **completed optimizer updates within an epoch**, with an explicit accumulation policy. Restore the remote checkpoint in a fresh session before deciding resume/skip. Replay an interrupted incomplete update; do not quietly restart the epoch or training run.
- Save locally first, publish consistent immutable snapshots, and skip verified completed work. Test real-model interruption/resume equivalence and published-state recovery before long training.
- Batch normal HF pushes **every 30 minutes**, and flush after a major cell/run completes or on catchable Stop/error/SIGINT/SIGTERM. Local commits/snapshots and remote HF publication are separate layers.
- Trigger an immediate flush attempt on a catchable stop while respecting server backoff. Record the last verified remote revision and pending local work. Forced kernel/process death cannot execute cleanup or guarantee publication of unsent work.
- Handle the user's historical **approximately 128 requests/hour** concern conservatively. Share a request budget across sessions using the account, batch/deduplicate unchanged files, authenticate/cache reads, respect server delay headers and retry transient/429 failures with bounded backoff. Verify actual current limits when implementing; do not treat 128/hour as a universal guaranteed quota.
- Give separate jobs unambiguous ownership; avoid concurrent writers for the same run. Preserve compatible protocol/runtime identities during repairs and verify checkpoint compatibility.

## Scientific and application contracts

- Class labels are **low/mid/high mileage proxies**; they are not measured wear, depth or safety labels.
- Keep all views/derivatives of a physical tyre together. Original-study folds and Phase 2 splits are different experiments.
- Keep original-paper models separate from Phase 2 workstation models. Preserve each checkpoint's actual preprocessing, prediction decoder, selection endpoint and provenance.
- Tyre/tread masks overlap. Weak polygon-derived point targets are not independent human evaluation truth.
- Choose deployable checkpoints by the declared validation rule, never by the best observed test score. Never silently substitute an architecture or hide failed predictions.
- Image-relative tread tilt is not physical camber/toe. Physical alignment requires the calibrated-target workflow and separate real-reference validation.

## NB07 data-retention reaffirmation — 29 October 2026

The user authorized implementation and explicitly requested more CSV collection.
Keep source/intake registries, descriptor definitions, labels/identity review,
annotation provenance, per-image exposure, coverage, validation issues and gaps
as direct CSV with hashes/schema. Richer derived rows do not add independent
tyres. The shipped empty intake must never manufacture training-ready status.
Keep future training's complete-state recovery and HF cadence requirements;
document the finite CPU audit's narrower publication/recovery guarantees.
