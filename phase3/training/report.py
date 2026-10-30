"""Small engineering report and direct metric derivatives; no study claims."""
from PIL import Image
import base64,csv,io,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from transactions import atomic,csv_write,read

def case_report(folder):
 folder=Path(folder);cursor=read(folder/'CURSOR.json');steps=[];samples=[]
 for table,target in [('train_steps',steps),('train_samples',samples)]:
  for p in sorted((folder/'csv').glob(f'epoch_*/{table}.csv')):
   with open(p,encoding='utf-8',newline='') as f:target.extend(csv.DictReader(f))
 classes=[]
 for epoch in sorted({r['epoch'] for r in samples}):
  group=[r for r in samples if r['epoch']==epoch]
  for cls in range(3):
   tp=sum(int(r['label'])==cls and int(r['prediction'])==cls for r in group)
   fp=sum(int(r['label'])!=cls and int(r['prediction'])==cls for r in group)
   fn=sum(int(r['label'])==cls and int(r['prediction'])!=cls for r in group)
   tn=len(group)-tp-fp-fn
   classes.append(dict(epoch=epoch,class_index=cls,support=tp+fn,tp=tp,fp=fp,fn=fn,tn=tn,
     precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None))
 csv_write(folder/'class_metrics.csv',classes,['epoch','class_index','support','tp','fp','fn','tn','precision','recall','f1'])
 atomic(folder/'COVERAGE.json',dict(mode='engineering_gate',role='instrumented' if samples else 'unlogged_frozen_math_reference',updates=cursor['updates'],sample_occurrences=len(samples),
   historical_training_accuracy_reconstructed=False,new_independent_tyres=0,
   available=['Actual pre-update augmented predictions','Transformed targets and augmentation parameters','Loss numerators/denominators',
     'Pixel sufficient statistics','Human/weak/unlabelled point status','Latent/evidence/activation/module-gradient statistics',
     'Complete optimizer/scheduler/AMP/RNG state','CSV row cursors','Native augmented inputs/targets/maps for each gate update','Stage timing and bounded resource samples'],
   unavailable={'continuous_power_energy':'Only scheduled instantaneous nvidia-smi samples; unsupported fields blank.',
     'input_pipeline_full_wall_breakdown':'Explicit measured stages plus remaining overhead; not all overhead separately attributed.',
     'new_study_results':'This is a three-update engineering test, not a measured 60-epoch run.',
     'future_training_map_retention':'All gate maps retained; a bounded capture/retention schedule must be frozen for a long run.',
     'validation_coverage':'Two human-annotated low/high fixture images at one surrogate boundary; not the full 81-image validation protocol.',
     'failed_attempt_tensors':'Retry events/scales retained; full failed-gradient tensors are not retained.',
     'post_pointer_commit_timing':'Per-commit timing journal is observational; a kill immediately after the pointer can omit that timing row.'}))
 gallery=[]
 for p in sorted((folder/'maps').glob('update_*/inputs.npy')):
  source=np.load(p,mmap_mode='r',allow_pickle=False)[0].transpose(1,2,0)
  image=np.clip((source*np.array([.229,.224,.225])+np.array([.485,.456,.406]))*255,0,255).astype(np.uint8)
  stream=io.BytesIO();Image.fromarray(image).save(stream,format='JPEG',quality=75)
  gallery.append(dict(update=int(p.parent.name.split('_')[1]),image='data:image/jpeg;base64,'+base64.b64encode(stream.getvalue()).decode()))
 payload=json.dumps(dict(cursor=cursor,steps=steps,gallery=gallery),separators=(',',':')).replace('<','\\u003c')
 html='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:; connect-src 'none'"><title>TREAD-ONE evidence gate</title><style>body{font:16px/1.5 system-ui;max-width:1000px;margin:auto;padding:24px;background:#14202d;color:#edf3fa}h1{font-size:28px}table{border-collapse:collapse;width:100%}th,td{padding:9px;border-bottom:1px solid #526779;text-align:left}select{padding:8px}img{max-width:100%;max-height:520px}aside{color:#b5c8d9}section{margin:25px 0}.pair{display:grid;grid-template-columns:1fr 1fr;gap:22px}@media(max-width:650px){.pair{grid-template-columns:1fr}}</style></head><body><h1>TREAD-ONE · evidence and recovery gate</h1><p>Engineering fixture only. These few updates validate logging and resumption; they are not model-quality or wear results.</p><p id="cursor"></p><section><table><thead><tr><th>Update</th><th>Epoch / cursor</th><th>Objective</th><th>Reconstruction error</th><th>AMP scale</th></tr></thead><tbody id="steps"></tbody></table></section><section class="pair"><div><label>Actual augmented input <select id="update"></select></label><p><img id="image" alt="First actual augmented training image at the selected engineering update"></p></div><div><h2>Transaction boundary</h2><p>Optimizer step → full native recovery state → CSV and native maps → manifest → atomic pointer.</p><p>Only pointer-committed updates are resumed. An incomplete generation is replayed, and every retained record belongs to a completed update.</p><aside>Native arrays and direct CSV retain the complete gate batch. The image preview is JPEG and shows only its first example. No new independent tyre was acquired.</aside></div></section><script>const DATA=__DATA__;const el=id=>document.getElementById(id);el('cursor').textContent=`Committed updates: ${DATA.cursor.updates}; next epoch ${DATA.cursor.next_epoch}, cursor ${DATA.cursor.next_cursor}`;for(const r of DATA.steps){const tr=document.createElement('tr');for(const v of [r.update,`${r.epoch} / ${r.cursor}`,Number(r.loss).toFixed(6),Number(r.loss_reconstruction_error).toExponential(2),r.amp_scale_after]){const td=document.createElement('td');td.textContent=v;tr.append(td)}el('steps').append(tr)}for(const g of DATA.gallery){const o=document.createElement('option');o.value=g.update;o.textContent='Update '+g.update;el('update').append(o)}function draw(){const g=DATA.gallery.find(x=>String(x.update)===el('update').value);if(g)el('image').src=g.image}el('update').addEventListener('change',draw);draw();</script></body></html>'''.replace('__DATA__',payload)
 (folder/'EVIDENCE_GATE.html').write_text(html,encoding='utf-8')
