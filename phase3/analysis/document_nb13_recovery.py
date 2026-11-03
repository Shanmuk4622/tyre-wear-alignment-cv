"""Record verified partial NB13 status and the existing-session recovery handoff."""
import ast,base64,hashlib,json,re,sys,textwrap
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'tread_two/context_validation/nb13_status'
done=Path(json.loads((ROOT/'tread_two/context_validation/NB13_STATUS_LATEST.json').read_text())['path'])
v=json.loads((done/'CHECKPOINT_VERIFICATION.json').read_text());p=json.loads((OUT/'recovery_validation/NOTEBOOK.json').read_text());test=json.loads((OUT/'recovery_validation/VALIDATION.json').read_text())
assert v['status']=='verified_partial_checkpoint' and test['status']=='passed'
def write(name,s):
    path=ROOT/name;path.write_text(textwrap.dedent(s).strip()+'\n',encoding='utf-8')
write('phase3NB13FreezeAndRecovery.md',f'''
# NB13 — interrupted run, HF verification and retained-session recovery

**31 October 2026. NB13 is partially trained, not complete.** The user reports that it becomes unresponsive after some time and confirms the execution is now stopped while the Kaggle session/files remain. Keep those files. The precise cause of the stop is not established by the saved output; no notebook traceback was saved.

## What was actually verified

| Evidence | Observed result | Boundary |
|---|---|---|
| Supplied NB13 | SHA `e0c714be2ababb69f965572e9698d1e2145a90340ab2aa5ad3f7614a786c2f3e` | Preserved byte-identically; no regeneration |
| Frozen payload | All 33 embedded files match the original source inventory | All 32 source/asset entries unchanged |
| Setup / prepare / preflight | First three code cells completed; preflight reused a verified completed proof | Setup pip warnings did not prevent later training |
| Last saved training-cell message | **2,467 / 2,940 updates**, stage train | Notebook output, not direct verification of the current Kaggle native state |
| Last display cell | Not executed | No completion receipt or final endpoint report |
| HF pinned repository revision | `2d148d17346f7bf0e2404127a5f88b212bfcb52a` | Read-only public inspection |
| Measured protocol | `36f473b0886d6c2f49c416436d270198af3982230e3045da9969b0bf42169625` | Same source/data/recipe and active experiment |
| Selected remote recovery snapshot | **1,294 / 2,940 updates**, epoch 27 / within-epoch cursor 20, phase train | Verified native checkpoint and control records |
| HF snapshot manifest | 798 scientific/control/source objects | Full 798-object snapshot was not downloaded or exhaustively audited |
| Diagnostic download | 15 objects / 11,973,539 bytes, every size/SHA checked | Includes native model/optimizer state, source contract, cursors, preflight and recent update/log records |
| Native recovery state | Model and optimizer tensors finite; 27 CSV row-cursor families; 22,810 observations | Does not establish final accuracy or later local tensor state |
| Actual Kaggle preflight | Passed on **Tesla T4 ×2**; native/optimizer/scheduler/scaler/RNG and all 27 CSV families agree; live pinned-HF restore passed | Real engineering proof, not measured fit completion |
| Latest heartbeat | 31 October **10:48:19 IST** | No later commit in the pinned inspected head |
| Recorded owner lease expiry | 31 October **11:33:19 IST**; expired at inspection | Not a clean owner release or proof of a live process; no lease was deleted |

The public inspection was around **14:06 IST**. [Inspection](tread_two/context_validation/nb13_status/e0c714be2aba/INSPECTION.json), [public status](tread_two/context_validation/nb13_status/e0c714be2aba/PUBLIC_STATUS.json), [checkpoint verification](tread_two/context_validation/nb13_status/e0c714be2aba/CHECKPOINT_VERIFICATION.json), [pinned HF run](https://huggingface.co/datasets/Shanmuk4622/tyre-wear-study/tree/2d148d17346f7bf0e2404127a5f88b212bfcb52a/phase3/tread-two/atlas-context-r3/fit/36f473b0886d6c2f49c416436d270198af3982230e3045da9969b0bf42169625).

**The distinction matters:** 1,173 additional update numbers appear in local progress after the remote checkpoint. They must not be treated as remotely protected. A current local native checkpoint may be at 2,467 or later; the first recovery cell reads the actual retained cursor. A new empty Kaggle session would only have the older verified HF recovery point.

```mermaid
flowchart LR
    A[HF selected snapshot\nverified update 1294] --> B[Remote prefix identity]
    C[Retained Kaggle files\nnotebook reports update 2467] --> D[Verify local native / CSV / history]
    B --> E[Require identical history prefix\nreject divergence]
    D --> E
    E --> F[Publish ahead local progress first]
    F --> G[Resume original protocol\nno repeated completed updates]
    G --> H[Phase heartbeat / stack diagnostics\ncomplete remaining fit and review]
```

## Diagnosis: confirmed gaps and unresolved cause

The original coordinator reports saved training progress while its worker runs. It does not continuously report progress while constructing/verifying a large snapshot, exporting tables/native artifacts, restoring an entire remote snapshot, or waiting inside an HF commit. Those operations can therefore look frozen. Restarting the original launcher also downloads the entire older remote snapshot even when a verified newer local prefix is available.

The saved output contains upload progress and continued training through update 2,467. The pinned HF state remains at 1,294, and no final clean release is present. These observations establish incomplete publication and poor visibility; they do **not** prove a particular HTTP timeout, Xet fault, GPU deadlock, out-of-memory event or numerical failure. The retained local logs/stacks are needed to identify the exact waiting stage if it recurs. The recovery does not claim to have repaired an unobserved model defect.

The actual preflight capacity estimate was 14,372,539,305 bytes below the 16 GB guard. The published update-1,294 snapshot recorded about 1.97 GB logical output and 18.96 GB free at that earlier time. Neither value describes disk/memory at the later stop. The recovery cell reports current free disk and local log tails.

## Recovery delivery and exact instructions

Open [NB13A — existing-session recovery](notebooks/phase3_NB13A_TREAD_TWO_CONTEXT_Existing_Session_Recovery.ipynb). **Copy its two code cells into the bottom of the existing stopped Kaggle NB13 and run them in order.** Keep the same T4×2, expanded Kaggle input, Internet and HF_TOKEN. Do not reset the session or run the original training cell concurrently. Opening NB13A as a new separate Kaggle session will not preserve the old working files; this recovery deliberately refuses an empty run.

| Cell | Behavior |
|---|---|
| 1 — lightweight diagnostic | Read current local pointer/unit, last remote receipt, free disk and bounded safe log tails; no source/data/training mutation |
| 2 — recovery / resume | Verify the exact frozen source and audited protocol; request safe exit of leftover owned run scripts; acquire existing controller/owner locks; verify local science and remote history prefix; flush ahead local progress before fitting; resume original experiment |

Expected directory is `/kaggle/working/tread_two_atlas_context_r3`. If local files are missing or the contract/source/history differs, stop and preserve the diagnostic output. Do not erase state or force a new namespace to bypass the check. An expired remote lease is handled by the inherited CAS claim; live owners are never stolen.

The external [recovery launcher](analysis/nb13_recovery_launcher.py) does not edit any of the 32 frozen runtime/assets, native model values, recipe, sampler, objective, scientific schemas or measured identity. It adds:

- A verified local-history-prefix path that avoids downloading the complete old remote snapshot when compatible local data are already present. Each local committed CSV chunk/native artifact is still verified by the original Store. Remote history/cursor object hashes and exact prefix equality are required; divergent branches are rejected.
- An initial HF publication of ahead local state before the worker resumes. Failure at that step stops further optimizer work; current local evidence remains.
- Phase/status messages every 15 seconds through local verification/export and HF operations, with saved update/stage, elapsed operation time and free disk.
- Read/connect timeouts of 20/150 seconds only for HTTP sends that otherwise had no timeout; existing explicit timeouts, request/commit budgets, server backoff, 80-object batches and ownership fencing remain.
- Local thread stacks every three minutes, without local variables; repair identity/logs included as direct snapshot provenance and retained across verified restores.
- Catchable Stop forwarding and the inherited immediate flush attempt. Forced kernel/session death still cannot publish unsent updates.

This is visibility/recovery orchestration. It does not impose a blanket timeout on the whole multi-file publication, disable safety checks or guarantee HF availability.

## Validation and remaining work

[Offline recovery checks](tread_two/context_validation/nb13_status/recovery_validation/VALIDATION.json) passed: verified local-ahead/equal-prefix paths, publication before fitting, unchanged native/CSV bytes, divergence rejection, failed-first-flush preservation, provenance retention, heartbeat reporting, bounded-default/unchanged-explicit HTTP timeouts, and run-specific stop requests without process killing. No network requests, model forwards or optimizer updates were made by those tests.

[NB13A package](tread_two/context_validation/nb13_status/recovery_validation/NOTEBOOK.json): two code cells; launcher SHA `{p['launcher_sha256']}`. Actual Kaggle recovery/resume remains pending. Save the executed NB13A and original NB13. If waiting recurs, preserve `NB13_RECOVERY/STATUS.json`, `EVENTS.jsonl`, `THREAD_STACKS.txt`, and `run/console.log`.

After recovery, verify an HF receipt at least as new as the retained local cursor, then allow the remaining original fit/validation/selected/final/diagnostic stages to finish. Only a final completed manifest, 2,940 updates, endpoint evidence and owner release can close NB13. No new model, additional seed, photo acquisition or workstation change is authorized by this status check.
''')

block='''
<!-- NB13_RECOVERY_UPDATE_20261031 -->
## Current — NB13 incomplete; preserve the stopped Kaggle session

**31 October 2026:** the user reports an apparent freeze and confirms stopped execution with local files retained. Saved NB13 output reaches **2,467 / 2,940 updates**; the independently verified HF checkpoint is only at **1,294**, phase train. All 33 embedded files match the unchanged 32-file runtime. The actual T4×2/live-HF preflight passed. No final completion or clean owner release is verified. [Status, proof and diagnosis](phase3NB13FreezeAndRecovery.md).

Next: copy the **two code cells** from [NB13A existing-session recovery](notebooks/phase3_NB13A_TREAD_TWO_CONTEXT_Existing_Session_Recovery.ipynb) into the existing stopped Kaggle NB13 and run them in order. Keep its files, same settings/input and HF_TOKEN. Do not reset into a new empty session or run another original training cell concurrently. Recovery verifies local/remote history, publishes ahead local progress before fitting, and resumes the original protocol with phase/status/stack diagnostics. [Original run guide](tread_two/context_pilot/README.md).

Exact stop cause is not established: no traceback was saved. Long snapshot/verification/upload paths lacked continuous status; do not claim a proven numerical/GPU/HF failure. The external orchestration repair and notebook pass local offline checks; actual Kaggle recovery remains pending. Preserve executed NB13 (`e0c714be…`), all frozen sources, protocol `36f473b0…`, native/CSV progress and earlier completed experiments. No model/recipe/data/workstation change. Earlier pending-NB13 delivery text below is historical.
<!-- END_NB13_RECOVERY_UPDATE -->
'''
names=['README.md','AGENTS.md','phase3Progress.md','phase3NextSteps.md','phase3OpenItems.md','phase3Understanding.md','phase3EvidenceRegister.md','phase3DocumentationInventory.md','phase3RepositoryMap.md','phase3ModelRationaleAndLogging.md','phase3DataRetentionPolicy.md']
for name in names:
    path=ROOT/name;s=path.read_text(encoding='utf-8')
    s=re.sub(r'\n<!-- NB13_RECOVERY_UPDATE_20261031 -->.*?<!-- END_NB13_RECOVERY_UPDATE -->\n','\n',s,flags=re.S)
    first,rest=s.split('\n',1)
    if name=='phase3NextSteps.md':first='# Next steps — recover and resume the existing ATLAS-CONTEXT run'
    path.write_text(first+'\n\n'+block.strip()+'\n\n'+rest.lstrip(),encoding='utf-8')
for name,link in [('tread_two/context_pilot/README.md','../../phase3NB13FreezeAndRecovery.md'),('phase3TreadTwoContextProtocol.md','phase3NB13FreezeAndRecovery.md'),('phase3TreadTwoContextValidation.md','phase3NB13FreezeAndRecovery.md'),('phase3TreadTwoContextArchitecture.md','phase3NB13FreezeAndRecovery.md')]:
    path=ROOT/name;s=path.read_text(encoding='utf-8');first,rest=s.split('\n',1)
    note=f'> Current, 31 October 2026: actual dual-T4/live-HF preflight passed; NB13 is partially trained and stopped. HF update 1,294 is verified; notebook output reaches local update 2,467. Preserve the existing session and use [the NB13A recovery instructions]({link}). Earlier implementation/pending-execution statements below record the original delivery.\n'
    if note not in s:path.write_text(first+'\n\n'+note+'\n'+rest.lstrip(),encoding='utf-8')
for name,title,body in [
('phase3SessionLog.md','31 October 2026 — NB13 partial/HF check and recovery',
'''User reported apparent freezing; confirmed execution stopped with session/files retained. Archived supplied NB13 e0c714be…; all33payloadfiles match unchanged32source entries. No notebook exception, last displayed local update2467/2940, last displaycellunexecuted. Public pinnedHF head2d148d… holds798object partialmanifest. Downloaded and size/SHA-verified15diagnostic/native/control objects (11,973,539bytes); native checkpoint1294/2940, epoch27cursor20, model/optimizerfinite, sourceprotocol36f473b0… matches. Actual T4×2/live-HF preflightpassed; owner expired11:33IST, no clean release. This is not a full798object/rawrow audit. No local credentials or remote writes.

Confirmed orchestration visibility gaps; precise stop causeunknown without retained Kagglelogs. Implemented externalNB13recovery with exactsource/contract guards, verifiedlocal-prefix restore avoiding redundantold snapshotdownload, aheadlocalflushbeforefit,15secondphaseheartbeat,180secondstackcapture and timeoutsonlyotherwiseunboundedHTTP sends. Ratebudgets/backoff/CASscientificstateunchanged. PackagedNB13A two-code-cell existing-session companion; do not reset or replace original execution. Offline prefix/flush/divergence/byte-preservation/provenance/status/timeout/owned-process checks pass; actualKagglerecoverypending. UpdatedcurrentMarkdownstatus/evidence/inventory/retention/protocol guides, with diagramandtables. No new fitting, photos or workstation changes.'''),
('phase3Decisions.md','NB13 incomplete-run recovery — 31 October 2026',
'''| ID | Decision | Evidence / constraint |
|---|---|---|
| P3-D102 | Preserve stopped session and treat NB13 as incomplete | Notebook local2467 vs verifiedHF1294; no final completion/cleanrelease |
| P3-D103 | Do not identify an unobserved GPU/HTTP/numerical cause | No savedtraceback; exact waitingstage needs localdiagnostics |
| P3-D104 | Use an external orchestration repair, preserve all32frozen entries and actualprotocol | Initialaheadlocalpublication; verifiedhistoryprefix; same model/optimizer/schema |
| P3-D105 | Deliver existing-session NB13A with status/stackdiagnostics | Offlinechecks pass; actualKagglerecoveryandfitcompletionpending |
''')]:
    path=ROOT/name;s=path.read_text(encoding='utf-8');marker='NB13_RECOVERY_LOG'
    s=re.sub(r'\n<!-- '+marker+r' -->.*?<!-- END_'+marker+r' -->\n','\n',s,flags=re.S)
    path.write_text(s.rstrip()+'\n\n<!-- '+marker+' -->\n## '+title+'\n\n'+body+'\n<!-- END_'+marker+' -->\n',encoding='utf-8')
print('Updated NB13 status, recovery findings/run guide, current trackers, decisions and session record.')
