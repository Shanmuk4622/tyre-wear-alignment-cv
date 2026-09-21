import pathlib,json,shutil,hashlib,ast
root=pathlib.Path('phase2');work=root/'workstation'
for name in ['smoke-check.json','learned-ui-check.json']:shutil.copy2(pathlib.Path('prototype/results')/name,work/('phase2_'+name))
shutil.copy2('prototype/results/video1-learned-workstation.png',work/'phase2_workstation_preview.png')
(work/'README.md').write_text('''# Phase 2 workstation — installed 21 September 2026

The existing `prototype/Launch Tread Station.cmd` now loads Phase 2 exports. Close and reopen any running workstation window. No package reinstall or training is required. Internet is not needed for inference.

| Component | Installed export |
|---|---|
| MobileNet V4 Medium | Seed 1, validation-best epoch 4 |
| ResNet50 | Seed 2, validation-best epoch 1 |
| YOLO26 Medium segmentation | Seed 1, validation-best epoch 10; replaces Nano in the current app |
| SegFormer B0 | Seed 2, validation-best epoch 29; regions and learned-boundary comparison |
| HRNet W18 | Seed 3, validation-best epoch 2 |

The weights were selected by validation only using the NB06 first-seed tie rule, then downloaded at immutable HF revision `be0eadb2bbb698091b563d699e2519b1e262a31c` and SHA256-verified. Five files total 290,983,569 bytes (~277.5 MiB). `registry.json` retains the source revision, hash, contract and scores. Checkpoints remain under `phase2/workstation/models/`; original model files are preserved.

## What changed

The existing engine now loads strict Phase 2 state dictionaries and uses the trained input sizes: classifiers 384×384; YOLO, SegFormer and HRNet height 512 × width 384. Images use direct PIL bilinear resizing; YOLO uses RGB [0,1], other models ImageNet normalization. Classifiers use three-class softmax. The masks remain overlapping independent tyre/tread channels, resized back to the original frame. HRNet preserves the fixed guide rows and six horizontal positions. SegFormer boundary comparison uses the new segmentation export, not the earlier matched study weights.

The UI identifies YOLO as Medium. Its historical internal key `yolo26n_seg` and the boundary key `matched` are retained for recipe/evidence compatibility; titles and checkpoint provenance identify the actual Phase 2 models. The geometry explanation now shows the actual HRNet seed and validation-selected status.

## Verification

- All five strict GPU loads passed on NVIDIA GTX 1650.
- Repeated predictions were stable; segmentation adapter output matched the training inference implementation exactly on the checked photos.
- HRNet reproduced the saved exported-checkpoint test coordinates within 0.00002 normalized width.
- Nine sampled frames across all three supplied videos passed same-frame masks/points and evidence round trip.
- Desktop UI passed live video, portrait handling, overlays, seeking/reset, evidence JSON/PNG saving/restoration and releasing learned-model references when disabled.
- Five existing contract tests passed. Four-model station smoke passed.
- Geometry explanation was separately checked after fixing its old hard-coded seed/epoch caption.

Evidence: `validation.json`, `phase2_smoke-check.json`, `phase2_learned-ui-check.json`, and the preview PNGs in this folder. The nine-frame combined pipeline check had about 623 ms warm median across eight subsequent sampled frames; this is not a sustained 3-inspections/sec certification. Cold model loading takes longer. UI test allocations peaked around 311 MiB; the separate nine-frame check recorded about 327 MiB of PyTorch allocated memory, not total GPU or driver memory.

## Results and limits

Selected classifiers each achieved only 50% balanced accuracy on held-out low/high tyres. MobileNet seed 3 scored better on test, but it was not substituted after observing test scores. Selected SegFormer test Dice was 98.957%; YOLO 98.790%; HRNet boundary-position error 0.859% of image width. These are small internal test results, not a before/after improvement guarantee. The three videos are training-cohort data. Do not treat video checks as unseen generalization, tread-depth measurement or safety assessment.

YOLO seed 3 is deferred by the user. Finish that worker later and rerun NB06; the current YOLO choice is the best validation export among completed seeds. The app can be used now.

## Rollback

Close the app, run `phase2_Restore_Legacy.cmd`, and reopen the normal launcher to use the preserved original models. To return to Phase 2, close the app, run `phase2_Enable_Models.cmd`, then reopen. The Phase 2 switch verifies all five model hashes before enabling. No checkpoint is overwritten or deleted by switching.

`backup-20260921/` contains the pre-change app files and readmes with a hash manifest. The current integration was explicitly authorized by the user's model-replacement request; training source files, raw data, executed notebooks and old weights were not modified.
''',encoding='utf-8')
p=root/'pahse2Progress.md';s=p.read_text(encoding='utf-8').replace('| Desktop integration | Final UI verification in progress | Existing launcher; old weights retained; rollback switch provided |','| Desktop integration | Complete / GPU and UI verified | All three videos, seeking, overlays, save/restore and model release passed; existing launcher, rollback available |');s+='\nFinal verification: five existing contract tests passed; four-model GPU station smoke passed; paired desktop video tests passed for all three videos. Corrected the learned-diagram legacy seed/epoch caption and checked its rendered Phase 2 metadata. The existing launcher now selects Phase 2 via the ACTIVE marker; restart any existing process. Old checkpoint files are unchanged. Deferred and optional work remains accurately classified, not checked off as completed.\n';p.write_text(s,encoding='utf-8')
for f,link in [('README.md','phase2/phase2CompletionAudit.md'),('prototype/README.md','../phase2/workstation/README.md')]:
 p=pathlib.Path(f);s=p.read_text(encoding='utf-8');i=s.find('\n')+1
 banner='\n> **Current — 21 September 2026:** Phase 2 models are installed in Tread Station: MobileNet V4, ResNet50, YOLO26 Medium, SegFormer B0 and HRNet. Fourteen training runs and the available NB06 report are HF-verified; YOLO seed 3 is deferred by the user. GPU/video/UI checks passed. Restart the existing launcher. [Results, limitations and current instructions]('+link+'). Old checkpoints remain available for rollback. Earlier dated status sections below are historical.\n'
 p.write_text(s[:i]+banner+s[i:],encoding='utf-8')
p=root/'phase2Understanding.md';s=p.read_text(encoding='utf-8');s+='\n## Installed state — 21 September 2026\n\nUser authorized replacing current workstation models. Five exports installed with validation-only selection; existing app now uses Phase 2 while preserving old checkpoint files and a reversible mode switch. GPU and three-video UI checks passed. 14 runs completed; YOLO seed 3 explicitly deferred. See workstation/README.md and phase2CompletionAudit.md. Classifier weakness remains explicit; no new-data benefit or unseen-video claim.\n';p.write_text(s,encoding='utf-8')
p=root/'phase2_audit_completion.py';s=p.read_text(encoding='utf-8').replace("if name.endswith('.json'):","if name.endswith(('.json', '.log')):");p.write_text(s,encoding='utf-8')
for p in list(pathlib.Path('prototype').glob('*.py'))+list(root.glob('phase2_*completion*.py'))+[work/'phase2_select_models.py']:ast.parse(p.read_text(encoding='utf-8-sig'))
base=json.loads((root/'manifests/phase2_originals_baseline.json').read_text());changed=[]
for name,item in base.items():
 p=pathlib.Path(name)
 if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()!=item['sha256']:changed.append(name)
(root/'manifests/phase2_integration_original_changes.json').write_text(json.dumps(dict(changed=changed,note='Existing-app integration and documentation explicitly authorized on 2026-09-21; compare backup manifest for pre-change bytes.'),indent=2))
print('Syntax passed; tracked baseline changes:',changed)
