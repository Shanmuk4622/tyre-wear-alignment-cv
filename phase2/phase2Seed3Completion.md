# Final Phase 2 training verification — 21 September 2026

**All 15 combined-data runs completed 60 epochs. No training runs remain pending.** The latest NB06 report also says `completed`, contains all three seeds for all five models, and has an empty pending list.

HF dataset: `Shanmuk4622/tyre-wear-study`
Verified immutable revision: `714a8daa65305f450e4e72ca82dd36b2922e7151`
Protocol: `661e97642e10bf8cfcbb8f581142069733789eb8f413dc70654aa00dd7854af3`
Latest NB06 report: `reports/1789972637/` within that protocol.

YOLO seed 3 finished 60 epochs and 11,580 optimizer updates. Its validation-selected checkpoint is epoch 8, validation Dice **99.1866%**, test Dice **98.6091%** over 103 test photos. All three YOLO seeds average **98.0640%** test Dice (population standard deviation 0.9018 percentage points).

Verified every run's JSON/log hashes, binary LFS SHA256 references, T4 resume smoke, finite losses, update counts, 60 validation records, test membership/labels, recalculated test score, validation-best score and latest report hashes. Evidence is in `completion_seed3/verification.json`, `audit.json` and `reports/1789972637/`.

All five installed export hashes and selected job IDs still exactly match the completed NB06 registry. YOLO seed 1 has a higher validation Dice (99.2162%) than seed 3 (99.1866%), so the installed weights correctly remain unchanged. Selection was not changed using test scores. No further model download or training rerun is needed.

The preceding incomplete/deferred seed-3 notes are historical and superseded by this verification. Optional old-only controls, all-data refit and a fresh independent video cohort are separate experiments, not pending work in the completed 15-run schedule. Known classifier/test limitations remain unchanged.
