# What belongs in Git

Keep application/library/script source, current runnable notebooks, environment
definitions, documentation and deliberately curated report figures/tables.
Keep top-level JSON audit evidence in `outputs/<audit-name>/`, including the
contracts and seed-1 statuses required by `prototype/prepare_learned.py`.

Do not commit datasets, recordings, image annotation packages, trained weights,
downloaded dependencies, caches, generated experiment trees, training histories,
archived notebook executions, workstation captures or personal camera profiles.
Store the experimental artifacts on Hugging Face as the existing workflows do.

The root `.gitignore` applies these defaults. `prototype/.gitignore` permits only
the `data/incoming/.gitkeep` placeholder inside prototype data. Printable targets
are generated again by the app. Photos are excluded by default; intentional
documentation images belong under `docs/`, and UI images under `prototype/assets/`.

## Cleanup performed — 16 September 2026

Removed 2,547 generated files (about 387.66 MiB of working-tree contents) from
Git's index using `git rm --cached`. Every local file was retained, with its size
checked afterward. The staged deletions mean “stop versioning”, not “delete my
local datasets”. No commit, push or history rewrite was performed.

Verified that the app's required contracts/statuses, report assets and runnable
notebooks remain tracked, and `git add --dry-run .` does not re-add the removed
artifacts. Source changes made by other work remain available for your normal
add/commit/push workflow.

Previously committed artifacts remain in older commits. This cleanup changes
future repository snapshots; it does not shrink historical Git objects or remove
them from an initial push of old commits. History cleanup is a separate operation.

`.gitignore` matches paths, not file size or notebook cell contents. Keep embedded
notebook outputs modest. Before committing, use `git add --dry-run .` and inspect
the staged summary. Do not force-add ignored datasets or credentials.
