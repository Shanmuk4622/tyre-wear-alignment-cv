"""Self-contained scientific progress viewer; CSV remains the complete evidence."""
from PIL import Image
import base64,csv,io,json
from pathlib import Path
import numpy as np
from transactions import atomic,read

def rows(folder,table):
 for p in sorted((Path(folder)/'csv').glob(f'epoch_*/{table}.csv')):
  with open(p,encoding='utf-8',newline='') as f:yield from csv.DictReader(f)

def report(folder):
 folder=Path(folder);cfg=read(folder/'CONTRACT.json');cursor=read(folder/'CURSOR.json')
 training=[{k:r[k] for k in ('update','epoch','loss','classification','segmentation','geometry')} for r in rows(folder,'train_steps')]
 stride=max(1,len(training)//300);training=training[::stride]
 validation=[r for r in rows(folder,'eval_summaries') if r['level']=='cohort'];grad=list(rows(folder,'task_gradients'))
 finite=[r for r in grad if r['cosine_before']!=''];conflicts=sum(float(r['cosine_before'])<0 for r in finite);applied=sum(int(r['projection_applied']) for r in grad)
 gallery=[]
 for p in sorted((folder/'artifacts/validation_maps').glob('e*/i*/inputs.npy'))[:12]:
  image=np.load(p,allow_pickle=False).transpose(1,2,0);image=np.clip((image*np.array([.229,.224,.225])+np.array([.485,.456,.406]))*255,0,255).astype(np.uint8)
  stream=io.BytesIO();Image.fromarray(image).save(stream,format='JPEG',quality=72)
  ident=read(p.parent/'IDENTITY.json');gallery.append(dict(**ident,image='data:image/jpeg;base64,'+base64.b64encode(stream.getvalue()).decode()))
 summary=dict(condition=cfg['condition'],mode=cfg['mode'],completed_updates=cursor['updates'],phase=cursor['phase'],best=cursor['best'],
  task_gradient_observations=len(grad),finite_task_observations=len(finite),observed_conflicts=conflicts,applied_projections=applied,
  quality_improvement_established=False,independent_tyres_added=0,architecture_changed=False,scope=cfg['test_scope'])
 atomic(folder/'PROGRESS.json',summary)
 payload=json.dumps(dict(summary=summary,training=training,validation=validation,gallery=gallery),separators=(',',':')).replace('<','\\u003c')
 html='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="Content-Security-Policy" content="default-src 'none';script-src 'unsafe-inline';style-src 'unsafe-inline';img-src data:;connect-src 'none'"><title>TREAD-ONE paired study</title><style>body{font:16px/1.5 system-ui;background:#101d29;color:#edf5fc;margin:24px auto;padding:0 22px;max-width:1100px}h1{font-size:28px}canvas{width:100%;background:#192d3d;border-radius:12px}table{width:100%;border-collapse:collapse}td,th{padding:8px;border-bottom:1px solid #36516a;text-align:left}img{max-height:480px;max-width:100%}select{padding:8px}aside{color:#b8cada}</style></head><body><h1>TREAD-ONE · shared-task gradient comparison</h1><p id="state"></p><aside>One paired seed; unchanged architecture. Training and validation observations are development evidence. The already inspected test cohort is not a fresh blind generalization test.</aside><h2>Sampled training objective</h2><canvas id="curve" width="1000" height="260"></canvas><h2>Validation / endpoint records</h2><table><thead><tr><th>Epoch</th><th>Endpoint</th><th>Balanced accuracy</th><th>Dice</th><th>Point MAE (% width)</th></tr></thead><tbody id="scores"></tbody></table><h2>Fixed validation images</h2><select id="imageChoice"></select><p><img id="image" alt="Fixed validation gallery input"></p><p id="caption"></p><script>const DATA=__DATA__;const el=id=>document.getElementById(id);el('state').textContent=`${DATA.summary.condition} · ${DATA.summary.mode} · ${DATA.summary.phase} · ${DATA.summary.completed_updates} updates · conflicts ${DATA.summary.observed_conflicts}/${DATA.summary.finite_task_observations} · projections ${DATA.summary.applied_projections}`;const c=el('curve').getContext('2d');c.strokeStyle='#48d8b1';c.lineWidth=2;const rs=DATA.training,max=Math.max(1,...rs.map(r=>Number(r.loss)));c.beginPath();rs.forEach((r,i)=>{const x=20+i/Math.max(1,rs.length-1)*960,y=240-Number(r.loss)/max*220;if(i)c.lineTo(x,y);else c.moveTo(x,y)});c.stroke();for(const r of DATA.validation){const tr=document.createElement('tr');for(const v of [r.epoch,r.endpoint,(100*Number(r.balanced_accuracy_observed)).toFixed(2)+'%',(100*Number(r.mean_two_region_dice)).toFixed(2)+'%',r.human_point_mae===''?'unavailable':(100*Number(r.human_point_mae)).toFixed(3)]){const td=document.createElement('td');td.textContent=v;tr.append(td)}el('scores').append(tr)}for(let i=0;i<DATA.gallery.length;i++){const o=document.createElement('option');o.value=i;o.textContent=`Epoch ${DATA.gallery[i].epoch} · ${DATA.gallery[i].image_id}`;el('imageChoice').append(o)}function draw(){const g=DATA.gallery[Number(el('imageChoice').value)];if(g){el('image').src=g.image;el('caption').textContent=g.image_id+' · source '+g.image_sha256}}el('imageChoice').addEventListener('change',draw);draw();</script></body></html>'''.replace('__DATA__',payload)
 (folder/'TREAD_ONE_R2_Progress.html').write_text(html,encoding='utf-8')
 atomic(folder/'COVERAGE.json',dict(all_augmented_training_scalar_observations=True,all_validation_images_each_epoch=True,
  retained_native_training_maps='First batch at declared capture epochs only',retained_native_validation_maps='Fixed 6/class low/high gallery at declared capture epochs; smoke 1/class',
  test_native_maps='All selected/final endpoint output maps in chunks',per_task_gradients='Every optimizer update on declared shared modules',
  continuous_energy_measured=False,independent_new_tyres=0,unavailable=['Every validation pixel map at every epoch','Full per-parameter gradient tensors/history','Continuous power/energy','External physical wear/depth/safety ground truth']))
