"""Synchronize active Phase 3 notes without modifying original study documents."""
from pathlib import Path
import sys
assert Path(sys.prefix).name == 'cv_conda'
ROOT = Path(__file__).resolve().parents[1]


def update(name, old, new):
    path = ROOT / name
    text = path.read_text(encoding='utf-8')
    assert old in text, name
    path.write_bytes(text.replace(old, new, 1).encode('utf-8'))


def section(name, start, end, replacement):
    path = ROOT / name
    text = path.read_text(encoding='utf-8')
    left, right = text.index(start), text.index(end, text.index(start))
    path.write_bytes((text[:left] + replacement + '\n\n' + text[right:]).encode('utf-8'))


def append(name, text):
    path = ROOT / name
    value = path.read_text(encoding='utf-8')
    assert text.splitlines()[0] not in value, name
    path.write_bytes((value.rstrip() + '\n\n' + text.rstrip() + '\n').encode('utf-8'))


active = '''**29 October 2026 — TREAD-TWO / ATLAS pilot implemented and locally checked.**
The [self-contained NB08 notebook](notebooks/phase3_NB08_TREAD_TWO_ATLAS_Resumable_Pilot.ipynb)
and [run guide](tread_two/PILOT_RUN_GUIDE.md) are ready. One seed-1 classifier
fit is gated behind actual Kaggle dual-T4/pinned-HF recovery and resource checks.
Local batch-eight fixtures pass pause/abrupt/fresh-process equivalence;
**no measured TREAD-TWO result exists yet**.

The isolated runtime retains 21 transactional CSV families, 247 column
definitions, complete native recovery, all epoch checkpoints, frozen-control
scores and full selected/final evidence maps. [Implementation](tread_two/PILOT_IMPLEMENTATION.md),
[schema](tread_two/PILOT_EVIDENCE_SCHEMA.md), [validation](tread_two/PILOT_VALIDATION.md).
Use existing data only. Both workstations and selected prior models are
preserved; the user deferred UI changes. The earlier prototype/app chronology
below remains history, not a request to rerun completed TREAD-ONE work.'''

section('README.md', '<!-- TREAD TWO CURRENT -->', '<!-- STATION CURRENT -->',
        '<!-- TREAD TWO CURRENT -->\n' + active + '\n<!-- END TREAD TWO CURRENT -->')
section('phase3Progress.md', '## Latest milestone — TREAD-TWO', '<!-- STATION CURRENT -->',
        '## Latest milestone — TREAD-TWO implementation, 29 October 2026\n\n' + active)
section('phase3Understanding.md', '## Latest model direction — TREAD-TWO', '<!-- STATION CURRENT -->',
        '## Latest model direction — TREAD-TWO implemented pilot\n\n' + active)
section('phase3OpenItems.md', '## Active TREAD-TWO work', '<!-- STATION CURRENT -->', '''## Active TREAD-TWO execution gates — 29 October 2026

| Item | Actual state / next gate |
|---|---|
| Architecture/runtime/notebook | Implemented; [NB08](notebooks/phase3_NB08_TREAD_TWO_ATLAS_Resumable_Pilot.ipynb), six code cells and exact source/bank |
| Bank review and recipe | Fixed training-only bank retained with explicit paint/glare/detail risks; [review](tread_two/BANK_REVIEW.md), [recipe](tread_two/pilot/RECIPE.json) |
| Rich evidence and complete recovery | Implemented; 21 transactional CSV families / 247 definitions; local actual-model acceptance passes |
| Actual Kaggle/T4/HF acceptance | Pending user Run All; automatic live gate precedes long fit |
| One measured seed-1 fit | Not started locally; complete native/CSV resume, selected/final review and diagnostics are implemented |
| Post-run independent audit | Pending actual executed notebook and pinned remote evidence |
| Workstation changes | Deferred by user; both applications/selected models remain preserved |
| Additional tyre photos | Not requested or required; existing data only |

The earlier workstation and research backlog follow. Acquisition-dependent
historical entries are not the current execution queue.''')
section('phase3NextSteps.md', '# Next steps', '## Completed separate app', '''# Next steps — run NB08 TREAD-TWO / ATLAS on Kaggle

**Use existing data only. Workstation changes stay deferred.**

The isolated training implementation is ready with local engineering acceptance.
Open [NB08](notebooks/phase3_NB08_TREAD_TWO_ATLAS_Resumable_Pilot.ipynb), select
**T4×2**, attach expanded **shanmuk4622/tire-dataset-prepared-phase2**, enable
Internet and **HF_TOKEN**, then **Run All**. [Complete run/restart guide](tread_two/PILOT_RUN_GUIDE.md).

| Next execution | Meaning |
|---|---|
| Notebook preparation | Verify attached release, frozen anchor, environment and actual free space |
| Automatic live engineering gate | Actual dual-T4 batch eight, abrupt/pause/pinned-HF restore and output-budget acceptance |
| One measured fit | Seed 1, 60 epochs / 2,940 updates; 378 queries plus fixed support; full direct CSV/native evidence |
| Selected/final review | All endpoint images, frozen baseline scores, signed patch/reference evidence, full projections and fixed diagnostics |
| Independent completion audit | Preserve the executed notebook; verify pinned remote files, selections, counts, raw metrics and failure cases |

Local checks establish engineering mechanics, not better classification. The
old low/high endpoint remains inspected development evidence, with no held-out
mid subjects. No NB02–NB07 rerun or additional data campaign is required.
See [implemented module/state flow](tread_two/PILOT_IMPLEMENTATION.md) and
[evidence schema](tread_two/PILOT_EVIDENCE_SCHEMA.md).

''')
update('AGENTS.md', '- Read [README.md]', '''- **Latest implementation authorization, 29 October 2026:** the user approved implementing TREAD-TWO / ATLAS. NB08 and the isolated pilot runtime are now delivered with local engineering acceptance; actual dual-T4/HF gates and the measured fit remain pending Kaggle execution. Preserve the frozen source/bank/recipe and namespace once that fit starts. Keep both workstations unchanged. The run guide and pilot validation notes distinguish fixtures from study outcomes.

- Read [README.md]''')

path = ROOT / 'phase3EvidenceRegister.md'
text = path.read_text(encoding='utf-8')
insert = '''## TREAD-TWO / ATLAS implemented NB08 — 29 October 2026

| Evidence | What it supports |
|---|---|
| [NB08 notebook](notebooks/phase3_NB08_TREAD_TWO_ATLAS_Resumable_Pilot.ipynb), [packaging receipt](tread_two/pilot_validation/NOTEBOOK_DELIVERY.json) | Six unexecuted code cells, 24 exact source/assets plus FILES inventory, self-contained native bank |
| [Local acceptance](tread_two/pilot_validation/LOCAL_VALIDATION.json) | Full batch eight, four-update fixture, exact model/optimizer/scaler/scheduler/RNG and all 21 scientific CSV families after pause/abrupt/restore; 17 offline faults |
| [Stop-path acceptance](tread_two/pilot_validation/STOP_VALIDATION.json) | Same marker used by catchable notebook signal; complete-update pause and exact fresh-process continuation |
| [Extracted notebook check](tread_two/pilot_validation/NOTEBOOK_VALIDATION.json) | Exact packaged bytes and an actual isolated extracted worker; source-model/native/RNG/scientific rows match |
| [Frozen source inventory](tread_two/pilot/FILES.json), [recipe](tread_two/pilot/RECIPE.json) | Current pilot identity; original dependency copies versioned separately |
| [Column dictionary](tread_two/pilot/assets/COLUMN_DICTIONARY.csv) | 247 explicit table-column definitions, units/types/NA rules; direct CSV publication |
| [Implementation](tread_two/PILOT_IMPLEMENTATION.md), [validation](tread_two/PILOT_VALIDATION.md) | Architecture/process/state flow, actual fixes, preservation and explicit local/live boundaries |

No live HF request or measured 60-epoch fit was made by local acceptance. Actual
Kaggle dual-T4/pinned-HF recovery remains an automatic execution gate. The
original prototype below is a separate earlier zero-update artifact.

'''
path.write_bytes(text.replace('# Phase 3 — evidence register\n\n', '# Phase 3 — evidence register\n\n' + insert, 1).encode('utf-8'))

update('phase3Decisions.md', 'Append a new decision when its rationale changes.', '''| P3-D064 | 2026-10-29 | Implement ATLAS as isolated NB08 runtime/source/native-bank payload; keep both workstations and completed studies unchanged | User approved the proposed TREAD-TWO implementation; measured fit remains a Kaggle action |
| P3-D065 | 2026-10-29 | Freeze FP32, dense microbatch one, class/domain/group sampling, query-only appearance changes and the existing deterministic bank | Small head fits local GPU; stable per-image dense geometry enables faithful deduplication. Bank artifacts remain explicit risks |
| P3-D066 | 2026-10-29 | Use append-only direct CSV with hash-anchored byte cursors and immutable native/pointer units | Avoid large local database duplication, rollback incomplete observations, keep every committed scientific row and all epoch checkpoints |
| P3-D067 | 2026-10-29 | Keep GPU 0 optimizer and GPU 1 immutable evaluation copy behind barriers; one HF publisher | No concurrent weight mutation, DDP claim or second measured fit; full topology is live-tested automatically |
| P3-D068 | 2026-10-29 | Retain all patch/reference comparisons, frozen baseline scores and original-image evidence conservation | Literal support/opposition is inspectable; full maps and raw rows are preserved, not only a curated gallery |
| P3-D069 | 2026-10-29 | Require actual Kaggle pinned-state restore and measured output projection before the long run | Local batch-eight/17-offline-check success is not proof of network/T4 execution or accuracy gain |

Append a new decision when its rationale changes.''')
append('phase3DocumentationInventory.md', '''## ATLAS implemented pilot documentation — 29 October 2026

| File | Purpose |
|---|---|
| [PILOT_RUN_GUIDE.md](tread_two/PILOT_RUN_GUIDE.md) | Exact NB08 input/settings/cell order, stop/resume, output/storage/HF and troubleshooting |
| [PILOT_IMPLEMENTATION.md](tread_two/PILOT_IMPLEMENTATION.md) | Module ownership, copies/provenance, precision, two-GPU flow, transactional state machine and evidence splatting |
| [PILOT_EVIDENCE_SCHEMA.md](tread_two/PILOT_EVIDENCE_SCHEMA.md) | 21 scientific CSV families, 247 definitions, auxiliary/native files, row estimates and honest retention gaps |
| [PILOT_VALIDATION.md](tread_two/PILOT_VALIDATION.md) | Actual local checks, faults/fixes, source packaging, isolation and pending live gates |
| [BANK_REVIEW.md](tread_two/BANK_REVIEW.md) | Training-only appearance review, fixed-bank choice and shortcut risks |

The earlier design/experiment/evidence documents now point to implemented v1
and NB08. Active status/next-step/decision/evidence/session notes are synchronized.
The original 103-document baseline and both workstation source sets remain
preserved; no measured result or live Kaggle success is invented.''')
append('phase3SessionLog.md', '''## 29 October 2026 — TREAD-TWO / ATLAS implemented NB08

The user approved implementing the proposed model. Built an isolated pilot
under `tread_two/pilot/`, with exact copied anchor/geometry/role dependencies,
fixed native case bank and a self-contained six-code-cell Kaggle notebook.
No original runtime, executed notebook, selected checkpoint or workstation
source was edited. Existing data only; application changes remain deferred.

Froze one seed-1 head fit: 60 epochs, 49 batch-eight updates per epoch, 378 query
images, fixed 18-case support, hierarchical class/domain/group sampling,
query-only [0.9,1.1] appearance factors, AdamW/cosine/clip-one and FP32.
Fixed microbatch-one dense inference preserves the application-style anchor
path and supports exact per-image dense retention. GPU 0 trains; GPU 1 uses an
immutable evaluation copy at barriers. It is one measured candidate, not DDP.

Implemented 21 transactional CSV families and a direct 247-definition column
dictionary, complete update/RNG/sampler/partial-epoch/evaluation recovery,
all epoch checkpoints, detailed gradients/resources/commit timings, raw
candidate/frozen-control scores and every patch/reference comparison. Added
mass-conserving native evidence projection with direct residual checks,
complete selected/final cases and four fixed sensitivity probes. The explorer
shows actual image/atlas/supporting/opposing cases from saved evidence.

The new append-only CSV/native pointer store survives abrupt process exits,
verifies committed chunk hashes and rolls back only uncommitted suffixes.
Immutable publication copies CSV and consolidates metadata without ZIP/NPZ/
database uploads. A single publisher handles leases, CAS, ambiguous replies,
deduplication, header-aware delays and a shared local HTTP budget. Catchable
Stop uses a marker, completes the current unit and attempts immediate HF flush.
Forced death and other-host account traffic remain explicit boundaries.

Local actual-model fixtures pass uninterrupted/pause/abrupt/fresh-process
equivalence, including all scientific CSV families and native model/optimizer/
scheduler/scaler/RNG/evidence values. Seventeen offline faults cover storage
and publication. A separate stop-marker path and extracted-notebook worker
are checked. Local device is GTX 1650; no token, live upload, actual dual-T4
execution or measured classifier fit is claimed.

Development checks caught mutable pre-update bias logging, a Windows optional
resource handle issue, shared working/committed state and stale-source guards.
Fixed them before delivery; failed attempts remain separate fixture history.
Added stable prior-session log names and direct dictionary/state-version/
commit-timing metadata. Five new structured guides, diagrams and schema/review
tables document implementation and limits. Next: user runs NB08 on Kaggle,
where actual dual-T4/pinned-HF/storage gates precede the one measured fit; then
independently audit the executed notebook and remote evidence.''')
append('phase3DataRetentionPolicy.md', '''## ATLAS v1 implementation — 29 October 2026

The [NB08 schema](tread_two/PILOT_EVIDENCE_SCHEMA.md) implements 21 scientific
CSV families and 247 explicit column definitions, plus direct dictionary,
state-version and commit-timing CSV and full native selected/final maps.
Every occurrence retains complete patch/reference comparisons and candidate/
frozen-control scores. Complete update/evaluation recovery and exact source/
bank/role identities are coupled to the scientific rows. Larger telemetry
does not add independent tyres. Live HF acceptance and measured results remain
pending Kaggle execution. [Run/recovery guide](tread_two/PILOT_RUN_GUIDE.md).''')
print('ATLAS implementation status and project trackers synchronized.')
