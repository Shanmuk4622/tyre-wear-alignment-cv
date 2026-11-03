"""Record the observed missing-workdir error and standalone recovery revision."""
import datetime,json,re,sys,textwrap
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'tread_two/context_validation/nb13_status'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
error=read(OUT/'NB13A_ERROR_LATEST.json');inspection=read(Path(error['path'])/'INSPECTION.json')
delivery=read(OUT/'recovery_validation/NOTEBOOK.json');test=read(OUT/'recovery_validation/VALIDATION.json');package=read(OUT/'recovery_validation/PACKAGE_VALIDATION.json')
assert delivery['revision']=='r2' and test['status']==package['status']=='passed'
assert test['launcher_sha256']==delivery['launcher_sha256'] and package['notebook_sha256']==delivery['sha256']
latest=read(ROOT/'tread_two/context_validation/NB13_STATUS_LATEST.json');head=read(Path(latest['path'])/'LATEST_HEAD_CHECK.json')
checked=datetime.datetime.fromisoformat(head['checked_UTC']).astimezone(datetime.timezone(datetime.timedelta(hours=5,minutes=30))).strftime('%d October %Y, %H:%M IST')
def write(name,text):
    (ROOT/name).write_text(textwrap.dedent(text).strip()+'\n',encoding='utf-8')
write('phase3NB13ARepair.md',f'''
# NB13A repair — recover when local working files are unavailable

**31 October 2026. The supplied error is identified and fixed in the same NB13A filename, revision 2.** Run the revised notebook directly with **Run All** on Kaggle. The original 32-file model/runtime, measured protocol, scientific CSV schema and HF namespace remain unchanged.

## Actual failure and cause

| Observed item | Evidence |
|---|---|
| Supplied executed NB13A | SHA `b5a4dd68b4c1882ac5723c4ce0e0bdba5d1733b71ff1ac72fd599c14fdb0bdda` |
| First cell error | `AssertionError: Existing NB13 working files are required. Keep the old session; do not reset it.` |
| Second cell error | `AssertionError: Do not start a new empty session.` |
| Missing path | `/kaggle/working/tread_two_atlas_context_r3/run/POINTER.json` |
| Cause | Revision 1 required the original session's local checkpoint; that checkpoint is not accessible at the expected path in this execution |
| Execution boundary | Both assertions happened before recovery, fitting, credential retrieval or HF publication |
| What this establishes | A local-file availability error in NB13A, not a model/optimizer failure |
| What remains unresolved | The precise cause of the earlier NB13 freeze; its saved output has no traceback |

The evidence does not establish whether the older working files still exist in another session or saved output. They must not be assumed lost or remotely saved. [Archived inspection](tread_two/context_validation/nb13_status/nb13a_errors/b5a4dd68b4c1/INSPECTION.json) preserves both errors; the supplied notebook, old launcher, old generator and prior delivery/check receipts are archived alongside it. The requested notebook was then replaced by a clean repaired delivery. Original executed NB13 remains preserved.

## Verified recovery position

The latest bounded public HF check at **{checked}** still reports revision `2d148d17346f7bf0e2404127a5f88b212bfcb52a`, with the same partial fit manifest. The [verified native/control checkpoint](tread_two/context_validation/nb13_status/e0c714be2aba/CHECKPOINT_VERIFICATION.json) is at **1,294 / 2,940 updates**, epoch 27, cursor 20, phase train. Model/optimizer tensors are finite; the actual dual-T4/live-HF engineering preflight passed. The earlier notebook output printed local update 2,467, which is not the remotely verified recovery point.

The local audit verified 15 diagnostic/native/control objects, not all 798 snapshot objects. During actual remote recovery, the original downloader/Store verify the complete selected snapshot, native state, raw CSV chunks, artifacts, ledger and row cursors before continuing. Unsent updates after 1,294 can only be recovered from a surviving local or saved output tree. No missing progress or metrics are fabricated.

## Revision 2 behavior

```mermaid
flowchart TD
    A[Self-contained NB13A\n35 embedded files] --> B[Verify original source / data / environment]
    B --> C{{Compatible current local checkpoint?}}
    C -->|yes| F[Verify complete local science\nand remote history prefix]
    C -->|no| D{{Compatible saved output attached?}}
    D -->|yes| E[Copy input read-only\nverify copied run]
    E --> F
    D -->|no| G[Restore full verified HF snapshot\nlatest committed recovery point]
    F --> H[Publish ahead local progress first]
    G --> I[Reuse original passed live gate]
    H --> I
    I --> J[Resume same measured protocol\ncompleted-update boundary]
```

| Area | Revision 1 | Revision 2 |
|---|---|---|
| Startup without local pointer | Assertion stops execution | Rebuild frozen source/bootstrap, then recover from a saved output or HF |
| Distribution | Small companion dependent on old workspace | Self-contained 35-file payload: 33 original runtime/inventory files, launcher and original contract |
| Current local state | Required | Preferred and verified; ahead state published before more fitting |
| Attached saved notebook output | Not discovered | Compatible complete `run/` tree is found by pointer/contract/native identity and copied into writable output |
| No accessible local/saved run | Refused | Full verified remote snapshot restored; absence of a matching remote checkpoint still refuses a fresh fit |
| Existing unanchored partial directory | Not handled by bootstrap | Preserved under a uniquely named `NB13_RECOVERY/PRESERVED_PARTIAL_RUN_*` before restore |
| Environment / model changes | Rejected | Still rejected; environment mismatch JSON explains differing fields |
| Preflight | Expected already-local receipt | Original hash-verified passed HF gate receipt reused, without another fit fixture |

The external launcher is `nb13-recovery-20261031-r2`, stored separately from frozen model source. It keeps the prior 15-second phase heartbeat, three-minute local thread stacks, bounded otherwise-unconfigured HTTP timeouts, existing shared rate/commit budgets, server backoff, ownership fencing and direct CSV/native publication. No ZIP/NPZ/database payload is introduced. Native recovery still restores model, optimizer, scheduler, scaler, RNG, deterministic sample order, partial statistics and completed-update cursors.

## Run the repaired file

1. Open/upload the updated [NB13A](notebooks/phase3_NB13A_TREAD_TWO_CONTEXT_Existing_Session_Recovery.ipynb). Its historical filename is retained; revision 2 can run as a standalone recovery.
2. Select **GPU T4 ×2**, **Internet ON**, and enable Kaggle Secret **HF_TOKEN** with write access to `Shanmuk4622/tyre-wear-study`.
3. Attach the same **expanded** `shanmuk4622/tire-dataset-prepared-phase2` input. The images stay read-only on Kaggle. No additional photos are needed.
4. If the original session is available, use it to retain newer local progress. Alternatively attach a saved original notebook output containing its complete `run/` directory; the recovery can discover that input. A lone weights file or screenshot is not a full recovery state.
5. Ensure at least **12 GiB free** working output, then choose **Run All**. Do not run a concurrent writer for the same NB13 experiment.
6. Wait through full HF restore/verification. Messages identify the current operation, saved cursor and free disk. Keep this session alive during publication; catchable Stop attempts an immediate flush subject to server backoff.

Cell 1 extracts/verifies frozen runtime and reports whether local state exists. Cell 2 verifies exact original environment/data/upstream weights, selects a compatible recovery path, reuses the passed gate and resumes the original experiment. It does not select a new checkpoint by test score or create a new model/namespace to evade compatibility checks.

## Checks and limits

| Check | Actual result |
|---|---|
| [Recovery checks](tread_two/context_validation/nb13_status/recovery_validation/VALIDATION.json) | **18 offline checks passed**, including empty-session native/CSV restore, missing-remote refusal, saved-output import, partial-tree preservation, gate integrity and environment drift |
| [Actual package setup](tread_two/context_validation/nb13_status/recovery_validation/PACKAGE_VALIDATION.json) | Executed the actual extracted setup cell in an empty fixture directory; succeeds without a pointer; idempotent rerun and retained-pointer handling pass |
| Frozen sources | All original 32 entries/33 payload files byte-identical; source drift rejected without overwrite |
| Model/CSV preservation | Native values and scientific CSV bytes unchanged by local-prefix recovery; corruption/divergence/failed-first-flush checks pass |
| Notebook | Two code cells; 35 embedded files; SHA `{delivery['sha256']}` |
| Launcher | SHA `{delivery['launcher_sha256']}` |
| Local fitting/network in recovery tests | Zero model forwards, zero optimizer updates, zero network requests |
| Actual revised Kaggle recovery | Pending user execution; not claimed from local fixtures |

Save the executed repaired NB13A. If it stops, keep `run/console.log`, `NB13_RECOVERY/STATUS.json`, `EVENTS.jsonl`, `THREAD_STACKS.txt`, and any `ENVIRONMENT_MISMATCH.json`. A forced kernel/session loss cannot publish unsent state. Full completion still requires 2,940 updates, declared endpoint/diagnostic evidence, verified final publication and owner release.

[Original NB13 status and audit limits](phase3NB13FreezeAndRecovery.md) · [Frozen protocol](phase3TreadTwoContextProtocol.md) · [27-family CSV dictionary](phase3TreadTwoContextEvidenceSchema.md).
''')
block='''
<!-- NB13A_REPAIR_UPDATE_20261031 -->
## Current — NB13A missing-file error fixed; standalone recovery ready

**31 October 2026:** supplied NB13A failed before recovery because its original local `run/POINTER.json` was unavailable. The executed error is archived (`b5a4dd68…`). Revision 2 now embeds the unchanged runtime and original contract, prefers compatible local/attached saved state, and otherwise restores the full verified HF checkpoint. Latest independently verified HF recovery remains update **1,294**; earlier local notebook output showed **2,467**. Missing unsent progress is recoverable only from surviving files. [Repair, evidence and instructions](phase3NB13ARepair.md).

Next: run the updated [NB13A](notebooks/phase3_NB13A_TREAD_TWO_CONTEXT_Existing_Session_Recovery.ipynb) directly with **Run All**: T4×2, Internet, existing expanded Kaggle input and HF_TOKEN. It can run without the old workspace. Preserve/use the older session or attach its saved output if available. Do not run concurrent writers. The original passed live gate is reused, compatibility is verified and the same experiment resumes; no fresh fit is allowed without a matching recovery snapshot.

Eighteen offline recovery checks and the actual notebook setup in an empty fixture directory pass. Actual revised Kaggle recovery remains pending. All 32 frozen model/runtime entries, original NB13 protocol/source/notebooks and both workstations are preserved. Earlier existing-session-only instructions below describe superseded revision 1.
<!-- END_NB13A_REPAIR_UPDATE -->
'''
for name in ['README.md','AGENTS.md','phase3Progress.md','phase3NextSteps.md','phase3OpenItems.md','phase3Understanding.md','phase3EvidenceRegister.md','phase3DocumentationInventory.md','phase3RepositoryMap.md','phase3DataRetentionPolicy.md']:
    path=ROOT/name;s=path.read_text(encoding='utf-8');s=re.sub(r'\n<!-- NB13A_REPAIR_UPDATE_20261031 -->.*?<!-- END_NB13A_REPAIR_UPDATE -->\n','\n',s,flags=re.S)
    first,rest=s.split('\n',1);path.write_text(first+'\n\n'+block.strip()+'\n\n'+rest.lstrip(),encoding='utf-8')
for name,target in [('phase3NB13FreezeAndRecovery.md','phase3NB13ARepair.md'),('tread_two/context_pilot/README.md','../../phase3NB13ARepair.md'),('phase3TreadTwoContextValidation.md','phase3NB13ARepair.md'),('phase3TreadTwoContextProtocol.md','phase3NB13ARepair.md')]:
    path=ROOT/name;s=path.read_text(encoding='utf-8');first,rest=s.split('\n',1)
    note=f'> Superseding recovery update, 31 October 2026: revision 1 failed because local working files were unavailable. [NB13A revision 2]({target}) is now self-contained, prefers retained/attached saved state and otherwise restores the verified HF checkpoint. Use the updated notebook with Run All. Earlier existing-session-only instructions below are historical.\n'
    if note not in s:path.write_text(first+'\n\n'+note+'\n'+rest.lstrip(),encoding='utf-8')
for name,text in [
('phase3SessionLog.md','''## 31 October 2026 — NB13A unavailable-workdir error repaired

User supplied executed NB13A with two AssertionErrors: expected local run pointer is unavailable in that execution. Archived the complete notebook b5a4dd68…, old launcher/generator and original delivery/check receipts. Both guards fired before recovery, Secret retrieval, fitting or publication. The error does not identify a defect in the deep-learning model or prove the older files no longer exist elsewhere.

Built standalone revision 2 under the same filename: embeds 33 unchanged runtime/inventory payload files, external launcher and original measured contract. Reconstructs Kaggle paths and exact upstream weights, enforces the saved environment/data/source protocol, discovers compatible attached output, preserves unanchored partial files, reuses the verified HF gate and performs full remote restore when local state is absent. Missing remote state still refuses a fresh fit; no new namespace, model, seed or scientific schema is introduced. Fresh public head check confirms the original partial HF checkpoint remains at update 1,294; local output 2,467 is only recoverable from retained/saved files.

Eighteen offline checks pass, including real native/CSV transfer through the in-memory immutable service and negative integrity/environment cases. The actual notebook setup cell executes successfully in an empty fixture directory, repeats idempotently, recognizes retained pointer metadata and rejects altered source without overwrite. No local model forwards, optimizer updates or remote writes. Updated current documentation with tables, flow diagram, precise run settings, recovery boundaries and actual-execution caveats. Next is revised NB13A Run All on Kaggle; actual recovery and final fit completion remain pending.'''),
('phase3Decisions.md','''## NB13A revision 2 — 31 October 2026

| ID | Decision | Evidence / boundary |
|---|---|---|
| P3-D106 | Fix missing working-state handling through a standalone bootstrap | Both revision-1 guards failed before any recovery/fitting; cannot assume old workspace exists |
| P3-D107 | Prefer compatible current/saved state, otherwise full verified HF restore | HF update 1,294 is verified; unuploaded later progress cannot be fabricated |
| P3-D108 | Keep exact measured identity and original gate; reject drift or missing remote state | No fresh fit/model/namespace created to evade compatibility checks |
| P3-D109 | Deliver revision 2 after empty-directory package and 18 recovery checks | Actual Kaggle recovery remains pending; earlier execution/source history archived |
''')]:
    path=ROOT/name;s=path.read_text(encoding='utf-8');marker='NB13A_R2_RECORD'
    s=re.sub(r'\n<!-- '+marker+r' -->.*?<!-- END_'+marker+r' -->\n','\n',s,flags=re.S)
    path.write_text(s.rstrip()+'\n\n<!-- '+marker+' -->\n'+text+'\n<!-- END_'+marker+' -->\n',encoding='utf-8')
print('Documented NB13A missing-file cause, standalone repair, recovery evidence and current run instructions.')
