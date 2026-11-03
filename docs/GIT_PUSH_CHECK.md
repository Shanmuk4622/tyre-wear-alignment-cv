# Git content and push preparation

Updated 30 October 2026. Git contains source and reproducibility inputs;
Hugging Face remains the store for full experiment records and model artifacts.
The current root and Phase 3 ignore rules follow this division without deleting
local evidence or changing a completed experiment.

## What will be committed

| Keep in Git | Keep locally / on Hugging Face |
|---|---|
| Application, training, review and notebook-builder source | Datasets, raw photos/video, annotation working directories |
| Current runnable `.ipynb` files, including their existing execution evidence | Duplicate archived executions and downloaded notebook copies |
| Environment definitions, protocols, source inventories and small audit receipts | Checkpoints, model binaries, native recovery states and downloaded dependencies |
| Compact result summaries, documentation and curated figures | Full HF mirrors, raw per-update/per-image telemetry and test/publication fixtures |
| Ten explicitly allowed small reference-bank `.npy` inputs | Other NumPy datasets/maps/arrays and generated snapshot bundles |

The ten arrays under `phase3/tread_two/pilot/assets/` and
`phase3/tread_two/lc_pilot/assets/` and
`phase3/tread_two/context_pilot/assets/` are deliberate frozen model inputs. They are
required by their `FILES.json` inventories and notebook builders. They contain
reference RGB, validity, labels and source identifiers, not trained weights.
The narrow exceptions keep these builders reproducible without allowing an
entire data or checkpoint directory into Git.

Large row-level review CSVs and the two full failure explorers are generated
outputs. Their local copies remain intact; NB09/NB12 reproduce and publish them
as direct files on HF. Compact summary tables, methods, CSV indexes and figures
remain visible in Git. Some older evidence links intentionally refer to local/HF
artifacts rather than files distributed in a fresh checkout.

Four published figures were copied byte-for-byte out of ignored HF mirrors
into curated figure folders, and three Phase 3 reports now link those copies.
This preserves their actual published visuals without committing the mirrors.

## Existing tracked files

Ignore rules do not affect a file already in Git's index. This maintenance
stopped tracking **one downloaded model binary and six duplicate Phase 2 archived
notebooks** with `git rm --cached`. All seven local files were checked by size
and SHA-256 before and afterward and remain byte-identical.

Their staged `D` entries mean “remove from the next repository snapshot”, not
“delete from my computer”. Current runnable notebooks, trained local models and
the original experiment folders were not rewritten. No commit, push or history
rewrite was performed.

The working files eligible for a whole-repository add fell from approximately
**4.21 GB to 116 MB**. This is not the size of Git history or a predicted network
push. Existing older commits still contain their historical objects. The history
size check found no blobs at least 100 MiB and no secret filenames; historical
file contents were not exhaustively credential-scanned.

## Check before committing

From the repository root in PowerShell:

```powershell
.\phase3\run_cv.cmd -B scripts/check_git_content.py --history
git add --dry-run .
git diff --cached --stat
```

The Python check is read-only unless `--report` is supplied to save a small JSON
receipt. It checks Git-visible files for model binaries, the project's 100 MiB
file guard, known credential patterns and missing/ignored frozen source inputs.
It skips recognized secret filenames, never prints credential values, and does
not stage changes or modify history. The content scan covers Git-visible UTF-8
text and the configured token formats; history inspection covers blob sizes and
filenames.

After reviewing the preview, use your normal `git add`, commit and push workflow.
Do not force-add ignored datasets, weights, secret files or generated HF mirrors.
The `.env.example` / `.env.template` exceptions are for placeholders only.

## Verification records

| Record | Evidence |
|---|---|
| [Content check](../phase3/audit/GIT_CONTENT_CHECK.json) | Git-visible file sizes, credential-pattern scan and history size/filename check |
| [Ignore policy check](../phase3/audit/GIT_IGNORE_POLICY_CHECK.json) | 22 ignored / 27 retained path cases, ten frozen arrays and four published figures |
| [Index-only cleanup](../phase3/audit/git-index-cleanup.json) | Original local size/SHA for the seven files removed from tracking |

The earlier [content policy](GIT_CONTENT_POLICY.md) remains the historical record
of the September cleanup. This note documents the additional Phase 3 safeguards.
