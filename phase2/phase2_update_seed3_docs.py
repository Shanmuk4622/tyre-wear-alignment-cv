from pathlib import Path
import json
root=Path('phase2');audit=json.loads((root/'completion_seed3/verification.json').read_text());row=next(r for r in audit['verified_runs'] if r['run']=='yolo26m-combined-seed3')
(root/'phase2Seed3Completion.md').write_text(f'''# Final Phase 2 training verification — 21 September 2026

**All 15 combined-data runs completed 60 epochs. No training runs remain pending.** The latest NB06 report also says `completed`, contains all three seeds for all five models, and has an empty pending list.

HF dataset: `Shanmuk4622/tyre-wear-study`\nVerified immutable revision: `{audit['revision']}`\nProtocol: `661e97642e10bf8cfcbb8f581142069733789eb8f413dc70654aa00dd7854af3`\nLatest NB06 report: `reports/1789972637/` within that protocol.

YOLO seed 3 finished 60 epochs and 11,580 optimizer updates. Its validation-selected checkpoint is epoch {row['best_epoch']}, validation Dice **{row['validation']*100:.4f}%**, test Dice **{row['test']*100:.4f}%** over 103 test photos. All three YOLO seeds average **98.0640%** test Dice (population standard deviation 0.9018 percentage points).

Verified every run's JSON/log hashes, binary LFS SHA256 references, T4 resume smoke, finite losses, update counts, 60 validation records, test membership/labels, recalculated test score, validation-best score and latest report hashes. Evidence is in `completion_seed3/verification.json`, `audit.json` and `reports/1789972637/`.

All five installed export hashes and selected job IDs still exactly match the completed NB06 registry. YOLO seed 1 has a higher validation Dice (99.2162%) than seed 3 (99.1866%), so the installed weights correctly remain unchanged. Selection was not changed using test scores. No further model download or training rerun is needed.

The preceding incomplete/deferred seed-3 notes are historical and superseded by this verification. Optional old-only controls, all-data refit and a fresh independent video cohort are separate experiments, not pending work in the completed 15-run schedule. Known classifier/test limitations remain unchanged.
''',encoding='utf-8')
# Derive actual update count from verified manifest, not a guessed epoch budget.
p=root/'phase2Seed3Completion.md';s=p.read_text(encoding='utf-8');m=json.loads((root/'completion_seed3/runs/yolo26m-combined-seed3/MANIFEST.json').read_text());s=s.replace('11,580 optimizer updates',f"{m['updates']:,} optimizer updates");p.write_text(s,encoding='utf-8')
p=root/'pahse2Progress.md';s=p.read_text(encoding='utf-8');s=s.replace('**Fourteen runs completed and independently HF-verified; five selected exports downloaded and GPU-tested. YOLO seed 3 is explicitly deferred by the user, who requested using completed models now.**','**All 15 runs completed 60 epochs and are independently HF-verified. Final NB06 is complete. All five installed exports remain the validation-selected winners.**')
s=s.replace('| YOLO training NB04 | Two complete; third deferred by user | Seed 3 resumable during epoch 52; use completed seed 1 now |','| YOLO training NB04 | Complete | All three seeds ×60 epochs; seed 1 remains validation-selected |')
s=s.replace('| NB06 audit and export registry | Verified for all available results | Correctly partial: 14/15; refresh after deferred seed 3 finishes |','| NB06 audit and export registry | Complete / HF-verified | Final report 1789972637; 15/15 runs; pending list empty |')
s=s.replace('No required local preparation or annotation remains. Deferred seed 3 does not block the user-requested installation of available models.','No required training, local preparation or annotation remains for the 15-run schedule. [Final seed-3 verification](phase2Seed3Completion.md) supersedes earlier incomplete/deferred entries.')
s+='\n## 2026-09-21 — seed 3 and final NB06 complete\n\nUser finished seed 3. Verified HF revision `714a8daa65305f450e4e72ca82dd36b2922e7151`, all 15 runs ×60 epochs and complete report 1789972637. Seed 3 best epoch 8: validation Dice 99.1866%, test Dice 98.6091%. All five installed export hashes still match the completed registry; YOLO seed 1 remains best by validation. No weights were replaced unnecessarily. See phase2Seed3Completion.md.\n'
p.write_text(s,encoding='utf-8')
(root/'phase2StartHere.md').write_text('''# Phase 2 — current entry point

**All 15 training runs and final NB06 are HF-verified complete.** No training notebook needs rerunning. [Final verification](phase2Seed3Completion.md).

Tread Station uses the selected Phase 2 models; the completed registry confirms the installed selections, including YOLO seed 1. [Workstation guide and rollback](workstation/README.md).

New video controls: choose export FPS from 1–60 (default 10), optionally include an information panel, and see signed tread tilt on learned overlays. Physical camber/toe remain available only through calibrated targets. See [video controls](phase2VideoControls.md).

Dataset: 570 images total; 386 train / 81 validation / 103 test. All supplied videos are training-cohort material. Optional old-only comparisons and independent new-video evaluation remain separate studies.
''',encoding='utf-8')
for name in ['phase2CompletionAudit.md','phase2Plan.md','phase2RunNotebooks.md','phase2Understanding.md','phase2DatasetAudit.md','phase2LabelingGuide.md','phase2ActiveRunAudit.md']:
 p=root/name;s=p.read_text(encoding='utf-8');lines=s.splitlines();lines=[line for line in lines if not line.startswith('> **Current status — 21 September 2026:')];s='\n'.join(lines)+'\n';i=s.find('\n')+1
 s=s[:i]+'\n> **Final status — 21 September 2026:** [All 15 runs and NB06 are verified complete](phase2Seed3Completion.md). Installed selections still match the final registry. Earlier partial/deferred/run-next statements below describe historical snapshots. No retraining or extra labeling is required.\n'+s[i:];p.write_text(s,encoding='utf-8')
for name in ['README.md','prototype/README.md']:
 p=Path(name);s=p.read_text(encoding='utf-8');s=s.replace('Fourteen training runs and the available NB06 report are HF-verified; YOLO seed 3 is deferred by the user.','All fifteen training runs and final NB06 are HF-verified complete. Installed model selections remain unchanged.');p.write_text(s,encoding='utf-8')
p=root/'workstation/README.md';s=p.read_text(encoding='utf-8');s=s.replace('YOLO seed 3 is deferred by the user. Finish that worker later and rerun NB06; the current YOLO choice is the best validation export among completed seeds. The app can be used now.','All 15 runs and final NB06 are now HF-verified complete at revision `714a8daa65305f450e4e72ca82dd36b2922e7151`. All five installed hashes match the final registry; YOLO seed 1 remains validation-best across all three seeds. No notebook rerun is required.');p.write_text(s,encoding='utf-8')
print('Final completion documentation updated; actual seed3 updates:',m['updates'])
