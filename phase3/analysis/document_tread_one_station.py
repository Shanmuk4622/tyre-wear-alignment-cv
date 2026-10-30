"""Record the user's existing-data-only scope and independent desktop delivery."""
import json,re,sys
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'workstation_validation'
def write(name,text):(ROOT/name).write_text(text.rstrip()+'\n',encoding='utf-8')
def append(name,text):
 p=ROOT/name;s=p.read_text(encoding='utf-8')
 if text.strip() not in s:write(name,s.rstrip()+'\n\n'+text)
def current(name,text):
 p=ROOT/name;s=p.read_text(encoding='utf-8');heading,body=s.split('\n',1)
 for marker in ('NB07','STATION'):
  body=re.sub(r'<!-- '+marker+r' CURRENT -->.*?<!-- END '+marker+r' CURRENT -->\s*','',body,flags=re.S)
 write(name,heading+'\n\n<!-- STATION CURRENT -->\n'+text+'\n<!-- END STATION CURRENT -->\n\n'+body.lstrip())
def main():
 engine=json.loads((OUT/'ENGINE_VALIDATION.json').read_text());ui=json.loads((OUT/'UI_VALIDATION.json').read_text());assert engine['status']==ui['status']=='passed'
 status='''**Current scope, 29 October 2026: existing tyre data only. The user will not
collect more tyre photos.** The earlier assumed acquisition campaign is
superseded. The current deliverable is **[TREAD-ONE Studio](phase3TreadOneWorkstation.md)**,
a separate native workstation in [workstation/](workstation/README.md).
It uses the declared r1 seed-3/epoch-20 export, has its own launcher/model/Qt/
cache/evidence, and preserves the original workstation. Real model and UI
checks pass. [User guide](workstation/README.md).

Training and NB02–NB07 audits remain complete. Historical classification and
cohort limits remain; the prospective-readiness=false result is preserved as
research evidence and does not block this authorized desktop implementation.
**Next: use the separate app with existing media. No new photos, NB07 rerun or
retraining is required.**'''
 for name in ('README.md','phase3Progress.md','phase3Understanding.md','phase3OpenItems.md','phase3DataGrowth.md','phase3DataRetentionPolicy.md'):current(name,status)
 # Put the current deliverable first; preserve earlier study/selection history.
 p=ROOT/'phase3NextSteps.md';s=p.read_text(encoding='utf-8');history=s[s.index('The remaining sections preserve'):]
 write(p.name,'''# Next steps — use TREAD-ONE Studio with existing data

**User decision, 29 October 2026: no additional tyre-photo collection. Use what
already exists.** Earlier acquisition assumptions are superseded.

| Current action | Ready deliverable |
|---|---|
| Open the special separate model window | [TREAD-ONE Studio user guide](workstation/README.md); launch the independent app in phase3/workstation |
| Inspect existing photos or videos | Single-model classification, overlapping masks/boxes, six boundaries and signed evidence; native playback/freeze/seek |
| Retain results | Automatic session CSV; explicit full capture with ten CSV tables / 245,784 rows, native arrays/PNGs and hashes |
| Restore/compare captures | Verified saved outputs; class/texture/layer/wipe controls render without retraining |
| Refine the application if needed | Keep edits in the separate app; retain the original workstation/model registry and frozen study sources |

[Implementation, isolation and tests](phase3TreadOneWorkstation.md).
No new data campaign, NB07 rerun or model fit is scheduled. NB07's false
prospective-readiness result describes the unfulfilled new-cohort protocol;
the desktop faithfully exposes the already completed model on existing media.
The model's classification failure remains in its record; app integration is
not a new accuracy or deployment-superiority claim.

The [old identity/capture pack](phase3CollectionReview.md) is optional historical
support, not a required collection step. The original application remains
available independently; no source/model/launcher was replaced.

'''+history)
 # Clearly supersede acquisition-dependent fronts without altering audit JSON.
 for name in ('phase3ProspectiveDataGate.md','phase3CollectionReview.md'):
  current(name,'''**Superseded acquisition plan — 29 October 2026.** The user has explicitly
decided to use only existing tyre data and will not collect more tyre photos.
The current authorized work is the [separate TREAD-ONE workstation](phase3TreadOneWorkstation.md).
The identity review and prospective gate below remain historical/optional
evidence. Their unmet prospective criteria do not block current desktop use.
No additional collection, protocol-freeze exercise or NB07 rerun is requested now.''')
 p=ROOT/'AGENTS.md';s=p.read_text(encoding='utf-8');s=s.replace('- User reaffirmation, 28 October 2026: the diagnostic and later controlled revisions remain part of **TREAD-ONE**.','- Historical user reaffirmation, 28 October 2026 (**acquisition assumption superseded by the 29 October existing-data-only decision above**): the diagnostic and later controlled revisions remain part of **TREAD-ONE**.');write(p.name,s)
 p=ROOT/'phase3Progress.md';s=p.read_text(encoding='utf-8');s=s.replace('Next is the prepared identity/capture review, actual independent tyre coverage and a frozen prospective protocol.','The user subsequently declined additional collection and authorized the separate TREAD-ONE Studio application; that is now the active deliverable.');write(p.name,s)
 p=ROOT/'phase3OpenItems.md';s=p.read_text(encoding='utf-8');s=s.replace('Identity review and independent\nuser-assisted tyre collection remain open; the local review/export tool is ready.','The user has superseded the acquisition plan with existing-data-only desktop integration; the old review tool remains optional.')
 s=s.replace('| Acquire more independent tyre data | Standing user priority; user will help later. Record identities/capture/annotations and reserve an independent cohort; [data plan](phase3DataGrowth.md) |','| Acquire more independent tyre data | Not planned: user explicitly declines additional tyre photos; use existing data. Generalization limits remain documented |')
 s=s.replace('| Next classification milestone | [NB07](phase3NB07Completion.md) complete/HF-verified; use [identity/capture review](phase3CollectionReview.md), then actual intake/frozen protocol; no new subjects or r3 fit |','| Current integration milestone | [Separate TREAD-ONE Studio](phase3TreadOneWorkstation.md) implemented; use existing media, retain CSV and preserve original workstation |');write(p.name,s)
 append('phase3DataRetentionPolicy.md','''## Independent desktop retention — 29 October 2026

The user is not collecting additional tyre photos. Existing-data inference now
has its own application-managed sessions/captures under `workstation/data/`.
Each completed inference retains 24 scalar/class/point/region/latent rows, plus
events. Explicit full capture retains every quarter-grid mask/logit/evidence/
texture/gate value: ten CSVs / 245,784 rows plus native arrays, PNGs, schema and
hashes. Unknown reference metrics remain blank. No CSV is hidden in an archive.

Desktop outputs stay local and are distinct from completed HF training/audit
records. The app performs no credential lookup or automatic media upload. Full
video-frame arrays are saved only on explicit capture; session scalars remain
automatic. [Exact definitions](workstation/EVIDENCE_SCHEMA.md).''')
 append('phase3RepositoryMap.md','''## TREAD-ONE Studio — separate app, 29 October 2026

| Path | Role |
|---|---|
| `phase3/workstation/` | New native app, private model/Qt, launch script, own data/cache; [guide](workstation/README.md) |
| `phase3/workstation/models/` | Exact copied declared seed-3/epoch-20 export and frozen architecture identity |
| `phase3/workstation/data/` | This app's session CSV, completed captures and settings |
| `phase3/workstation_validation/` | 47-file original baseline, model/UI/native-window proof, previews and isolated engineering fixtures |
| `prototype/`, `phase2/workstation/` | Original application/selected registry retained byte-for-byte in the recorded baseline |

[Integration and ownership](phase3TreadOneWorkstation.md). Acquisition-dependent
future plans are superseded for current work by the user's existing-data-only scope.''')
 append('phase3EvidenceRegister.md','''## TREAD-ONE Studio implementation — 29 October 2026

| Evidence | Verified state |
|---|---|
| [Preparation](workstation_validation/PREPARATION.json) | Local exact export/model copy; private Qt; cv_conda packages preserved |
| [Original baseline](workstation_validation/ORIGINAL_BASELINE.json) | 47 original source/launcher/registry files, unchanged after tests |
| [Engine validation](workstation_validation/ENGINE_VALIDATION.json) | 12 checks; ten CPU arrays match frozen reference exactly; GPU comparison and full dense CSV fidelity |
| [UI validation](workstation_validation/UI_VALIDATION.json) | 14 checks; actual child inference, stale-frame protection, compact-layout scrolling, all views, capture restore, video and clean shutdown |
| [Model record](workstation/models/MODEL.json) | Declared seed 3 / epoch 20; 1,372,159 parameters; original limitations retained |
| [Evidence schema](workstation/EVIDENCE_SCHEMA.md) | Ten direct CSVs / 245,784 rows per full capture; unknown ground-truth metrics blank |

Software fixtures are not new study data. Desktop outputs are local; completed
training/audit HF evidence is unchanged. [Delivery](phase3TreadOneWorkstation.md).''')
 append('phase3DocumentationInventory.md','''## Separate desktop / changed user scope — 29 October 2026

[phase3TreadOneWorkstation.md](phase3TreadOneWorkstation.md) records the independent
app, declared checkpoint, isolation, native UI, CSV retention and tests.
[workstation/README.md](workstation/README.md), [ARCHITECTURE.md](workstation/ARCHITECTURE.md)
and [EVIDENCE_SCHEMA.md](workstation/EVIDENCE_SCHEMA.md) cover operation, module/
process/data flow, coordinate rules and every retained evidence family.
AGENTS, README, Progress, Understanding, NextSteps, OpenItems, data-growth/
retention notes and the old prospective/collection fronts now explicitly state
the user's **no additional tyre photos; existing data only** decision.
Original Markdown and study notebook/source snapshots remain preserved.''')
 p=ROOT/'phase3Decisions.md';s=p.read_text(encoding='utf-8');rows='''| P3-D053 | 2026-10-29 | Use only existing tyre data; supersede the assumed additional-photo campaign | Explicit user instruction: no more tyre photos will be collected; keep limits factual without requiring acquisition for current work |
| P3-D054 | 2026-10-29 | Implement a visibly distinct independent TREAD-ONE desktop app under phase3/workstation | Own launcher, checkpoint, Qt, worker, cache and evidence; original prototype/registry left unchanged |
| P3-D055 | 2026-10-29 | Install the declared original r1 seed-3/epoch-20 export in the separate research app | Preserve validation selection/provenance; no best-test substitution or failed projected-candidate promotion |
| P3-D056 | 2026-10-29 | Use owned-process inference, exact-frame identity and rich direct capture CSV | Keep UI responsive, prevent stale overlays and retain every quarter-grid output on explicit capture; local inference scope |
| P3-D057 | 2026-10-29 | Register existing OS fonts per process for reliable Windows/offscreen Qt rendering | Visual QA exposed missing Latin glyphs; fix without changing/installing system fonts or modifying the original workstation |
'''
 if 'P3-D053' not in s:write(p.name,s.replace('\n\nAppend a new decision','\n'+rows+'\nAppend a new decision'))
 append('phase3SessionLog.md','''## 29 October 2026 — user chooses existing data; independent TREAD-ONE Studio

The user explicitly declined further tyre-photo collection and asked for
TREAD-ONE in another distinctive window or separate folder without disturbing
the original workstation. Recorded that scope in AGENTS and active notes;
acquisition-dependent plans remain historical and do not block this work.

Implemented a separate native PySide6 application under `workstation/`, with
private copied Qt bindings and the exact declared r1 seed-3/epoch-20 model.
Added own taskbar identity/icon, photo/example/video/optional-camera controls,
seven views, compare wipe/zoom, three-class probabilities, overlapping masks/
boxes, six native-pixel boundary proposals and exact signed evidence/bias/
texture readouts. The UI supervises one owned Torch child, verifies request/
input hashes and discards stale display results after source changes.

Automatic sessions retain per-frame/class/point/region/latent scalar CSV.
Explicit captures preserve ten CSV tables / 245,784 rows, ten native NPYs,
five PNGs, complete schema/identity and hashes. Save/restore is asynchronous
and checked; reference metrics remain blank and no HF credential is read.

Twelve engine checks pass: frozen CPU-reference differences are zero for all
ten arrays; GPU differences are recorded; every dense CSV value reconstructs
its array exactly. Fourteen UI checks cover actual inference-process exchange,
stale frame rejection, all controls, native widget screenshots, full captures,
restore, video seek/freeze/live and owned shutdown. The generated test movie
is engineering-only derived input, not another tyre/data release. Camera
hardware remains untested. Visual QA caught missing offscreen Latin glyphs;
per-process system-font registration fixed it without a package/environment
upgrade. All 47 original source/launcher/registry hashes still match.

The original selected models and workstation stay available. TREAD-ONE's known
classification limits remain in the app's model record. This work adds an
inference application and new inference records, not training, independent
subjects or a claim of better model accuracy. [Delivery](phase3TreadOneWorkstation.md).''')
 print('Existing-data-only scope and separate workstation delivery synchronized across Phase 3 notes.')

if __name__=='__main__':main()
