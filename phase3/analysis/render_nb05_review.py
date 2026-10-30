"""Durable figures and a self-contained image review from saved NB05 outputs."""
from PIL import Image,ImageDraw
import base64,csv,io,json,sys
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda' or Path('/kaggle/input').exists()
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'revision2'))
import spec
from data import plan,legacy,HW
from transactions import atomic,read
from audit_nb05_results import rows
PUB=ROOT/'revision2_validation/nb05_publication';OUT=PUB/'analysis';COLORS={'baseline':'#2586ad','shared_projection':'#cf6943'}

def table(name):
 with open(OUT/(name+'.csv'),encoding='utf-8',newline='') as f:return list(csv.DictReader(f))

def encode(array,jpeg=False):
 stream=io.BytesIO();Image.fromarray(array).save(stream,format='JPEG' if jpeg else 'PNG',**(dict(quality=78) if jpeg else {}))
 return 'data:image/'+('jpeg' if jpeg else 'png')+';base64,'+base64.b64encode(stream.getvalue()).decode()

def mask_rgba(masks):
 rgba=np.zeros((512,384,4),dtype=np.uint8)
 rgba[masks[0]>=.5]=[64,157,240,75];rgba[masks[1]>=.5]=[65,230,160,130]
 return rgba

def main():
 assert read(PUB/'SCIENTIFIC_VERIFICATION.json')['status']=='passed'
 figures=OUT/'figures';figures.mkdir(exist_ok=True);endpoints=table('endpoint_metrics');validation=table('validation_trajectory');training=table('epoch_training_gradient')
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
 fig,axes=plt.subplots(1,3,figsize=(12,4.5),layout='constrained')
 for ax,key,title in zip(axes,('balanced_accuracy_observed','mean_two_region_dice','human_point_mae'),('Observed-class balanced accuracy (%) ↑','Mean tyre/tread Dice (%) ↑','Human-point MAE (% width) ↓')):
  for offset,condition in enumerate(spec.CONDITIONS):
   rr=[next(r for r in endpoints if r['condition']==condition and r['endpoint']==endpoint) for endpoint in ('selected','final')]
   values=[100*float(r[key]) for r in rr];x=np.arange(2)+(offset-.5)*.34
   ax.bar(x,values,width=.32,color=COLORS[condition],label=condition)
   for xx,value in zip(x,values):ax.text(xx,value+.02*max(values),f'{value:.2f}',ha='center',fontsize=9)
  ax.set_xticks([0,1],['Validation-selected','Epoch 60']);ax.set_title(title,fontsize=10);ax.set_ylim(0,ax.get_ylim()[1]*1.14);ax.grid(axis='y',alpha=.16)
 axes[0].legend(fontsize=8);fig.suptitle('NB05 · one paired seed · inspected development test cohort (103 images / 2 tyres)',fontsize=12)
 fig.savefig(figures/'NB05_endpoint_comparison.png',dpi=165);plt.close(fig)
 fig,axes=plt.subplots(3,1,figsize=(10,9),layout='constrained',sharex=True)
 for ax,key,title in zip(axes,('balanced_accuracy_observed','mean_two_region_dice','human_point_mae'),('Validation balanced accuracy (%)','Validation mean Dice (%)','Validation human-point MAE (% width)')):
  for condition in spec.CONDITIONS:
   rr=sorted([r for r in validation if r['condition']==condition],key=lambda r:int(r['epoch']));ax.plot([int(r['epoch']) for r in rr],[100*float(r[key]) for r in rr],color=COLORS[condition],label=condition)
   selected=next(r for r in endpoints if r['condition']==condition and r['endpoint']=='selected');v=rr[int(selected['epoch'])-1]
   ax.scatter([int(v['epoch'])],[100*float(v[key])],color=COLORS[condition],s=50,marker='D',zorder=3)
  ax.set_ylabel(title);ax.grid(alpha=.2)
 axes[0].legend();axes[-1].set_xlabel('Epoch');fig.suptitle('Full validation trajectory · diamonds mark the declared joint-utility selection')
 fig.savefig(figures/'NB05_validation_trajectories.png',dpi=155);plt.close(fig)
 fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
 keys=('training_augmented_accuracy','observed_conflicts','mean_cosine_before','mean_clip_coefficient')
 titles=('Augmented training accuracy (%)','Shared-gradient conflict observations (%)','Mean class/dense gradient cosine','Mean global clipping coefficient')
 for ax,key,title in zip(axes.flat,keys,titles):
  for condition in spec.CONDITIONS:
   rr=[r for r in training if r['condition']==condition];scale=100 if key=='training_augmented_accuracy' else 100/49 if key=='observed_conflicts' else 1
   ax.plot([int(r['epoch']) for r in rr],[scale*float(r[key]) for r in rr],color=COLORS[condition],label=condition)
  ax.set(title=title,xlabel='Epoch');ax.grid(alpha=.2)
 axes[0,0].legend();fig.suptitle('Observed training mechanism · head gradients are unchanged before global clipping')
 fig.savefig(figures/'NB05_gradient_and_training.png',dpi=155);plt.close(fig)
 prepared=spec.locate();p=plan(prepared);sources={r['image_id']:r for r in p['rows'] if r['role']=='test'}
 images={};records=[];cache={}
 for image_id,source in sorted(sources.items()):
  image,masks,points=legacy.sample(prepared,source,HW,None,'segformer');cache[image_id]=(image,masks,points)
  images[image_id]=dict(image=encode(np.clip(image.transpose(1,2,0)*255,0,255).astype(np.uint8),True),truth_masks=encode(mask_rgba(masks)),
    label=source['class_index'],tyre=source['physical_tyre_id'],point_kind=source['point_kind'],targets=source['points'],guides=source['guide_fractions'],image_sha256=source['image_sha256'])
 for condition in spec.CONDITIONS:
  folder=PUB/'remote'/condition;points={}
  for r in rows(folder,'eval_points'):
   if r['endpoint'] in ('selected','final'):points.setdefault(r['unit'],[]).append(r)
  for endpoint in ('selected','final'):
   rr=sorted([r for r in rows(folder,'eval_predictions') if r['endpoint']==endpoint],key=lambda r:r['image_id'])
   for path in sorted((folder/'artifacts/test_maps'/endpoint).glob('*.npy')):
    start,end=map(int,path.stem[1:].split('_'));maps=np.load(path,allow_pickle=False)
    for index,a in enumerate(maps):
     r=rr[start+index];binary=np.repeat(np.repeat(a[:2]>=.5,4,1),4,2);heats=[];scales=[]
     for c in range(3):
      field=a[14+c];scale=max(float(np.max(np.abs(field))),1e-12);norm=np.clip(field/scale,-1,1);rgba=np.zeros((128,96,4),dtype=np.uint8)
      rgba[...,0]=np.where(norm>=0,245,55);rgba[...,1]=np.where(norm>=0,95,140);rgba[...,2]=np.where(norm>=0,60,245);rgba[...,3]=np.clip(np.abs(norm)*205,0,205).astype(np.uint8)
      heats.append(encode(rgba));scales.append(scale)
     pts=sorted(points[r['unit']],key=lambda x:int(x['point_index']))
     records.append(dict(condition=condition,endpoint=endpoint,image_id=r['image_id'],epoch=int(r['epoch']),prediction=int(r['prediction']),correct=int(r['correct']),
       confidence=float(r['confidence']),probabilities=[float(r['probability_'+str(c)]) for c in range(3)],nll=float(r['nll']),dice=float(r['mean_two_region_dice']),
       point_mae=float(r['human_point_mae']) if r['human_point_mae'] else None,evidence_residual=float(r['evidence_residual']),
       points=[float(v['prediction']) for v in pts],masks=encode(mask_rgba(binary)),heats=heats,heat_scale=scales))
 assert len(records)==412 and len(images)==103
 data=dict(endpoints=endpoints,validation=validation,training=training,images=images,records=records)
 payload=json.dumps(data,separators=(',',':'),allow_nan=False).replace('<','\\u003c')
 html=r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="Content-Security-Policy" content="default-src 'none';script-src 'unsafe-inline';style-src 'unsafe-inline';img-src data:;connect-src 'none'"><title>TREAD-ONE NB05 · completed comparison</title><style>
body{font:16px/1.5 system-ui;margin:24px auto;padding:0 24px;max-width:1180px;background:#10202c;color:#edf5fa}h1{font-size:30px}aside,.muted{color:#b9cbd9}table{border-collapse:collapse;width:100%}td,th{padding:9px;border-bottom:1px solid #355264;text-align:left}select,input{font:inherit;padding:8px;background:#f3f8fb;color:#163344;border:0;border-radius:5px}label{display:inline-block;margin:6px}.controls{padding:12px;background:#203746;border-radius:10px}.views{display:flex;gap:18px;flex-wrap:wrap}canvas{max-width:100%;background:#172d3e;border-radius:9px}#photo{width:384px;height:512px}#curve{width:100%;height:230px}.detail{flex:1;min-width:280px}pre{white-space:pre-wrap}.pill{padding:10px;background:#314959;border-radius:8px}</style></head><body>
<h1>TREAD-ONE · NB05 completed evidence review</h1><p class="pill">Both conditions completed 60 epochs. The projected candidate is not supported for promotion by this paired development result.</p><aside>One paired seed, two existing test tyres. These are mileage-proxy labels and inspected development observations. Repeated photos, epochs and point rows are not independent tyres. No additional model training is performed by this review.</aside>
<h2>Declared endpoints</h2><table><thead><tr><th>Condition</th><th>Endpoint / epoch</th><th>Balanced accuracy</th><th>Dice</th><th>Human point error</th></tr></thead><tbody id="metrics"></tbody></table>
<h2>All 60 validation epochs</h2><label>Metric <select id="curveMetric"><option value="balanced_accuracy_observed">Balanced accuracy</option><option value="mean_two_region_dice">Mean Dice</option><option value="human_point_mae">Human point MAE</option><option value="joint_validation_utility">Joint utility</option></select></label><canvas id="curve" width="1100" height="230"></canvas><p class="muted">Blue: baseline. Orange: shared projection. Diamonds are the predeclared validation selections. Test results did not select these checkpoints.</p>
<h2>All 103 development test images</h2><div class="controls"><label>Condition <select id="condition"><option>baseline</option><option>shared_projection</option></select></label><label>Endpoint <select id="endpoint"><option>selected</option><option>final</option></select></label><label>Filter <select id="filter"><option value="all">All images</option><option value="wrong">Current prediction wrong</option><option value="projection_worse">Baseline correct / projection wrong</option><option value="human">Human point labels available</option></select></label><label>Find image <input id="search" type="search" placeholder="Source image ID"></label><br><label>Image <select id="imageChoice"></select></label><label>Overlay <select id="mode"><option value="masks">Predicted tyre / tread</option><option value="heat">Signed additive class evidence</option><option value="points">Predicted point geometry</option><option value="truth">Labelled tyre / tread</option><option value="input">Input photo</option></select></label><label>Evidence class <select id="class"><option value="0">low</option><option value="1">mid</option><option value="2">high</option></select></label><label><input id="showTargets" type="checkbox" checked> Human point targets</label></div>
<div class="views"><canvas id="photo" width="384" height="512"></canvas><div class="detail"><h3 id="caption"></h3><pre id="details"></pre><p class="muted">Masks: blue tyre, green tread; regions overlap. Points: yellow predictions, magenta human targets. Heat: red positive / blue negative signed contribution to the chosen class logit. Heat opacity is normalized within each image/class; the displayed maximum gives its numerical scale.</p><p class="muted">Additive accounting is an intrinsic model explanation. It does not establish causal input reasoning, physical wear, depth, alignment or safety. Confidence is not a validated abstention rule.</p></div></div><script>
const DATA=__DATA__;const $=id=>document.getElementById(id);const names=['low','mid','high'];let version=0;
for(const r of DATA.endpoints){const tr=document.createElement('tr');for(const value of [r.condition,r.endpoint+' / '+r.epoch,(100*Number(r.balanced_accuracy_observed)).toFixed(2)+'%',(100*Number(r.mean_two_region_dice)).toFixed(3)+'%',(100*Number(r.human_point_mae)).toFixed(3)+'% width']){const td=document.createElement('td');td.textContent=value;tr.append(td)}$('metrics').append(tr)}
function curve(){const c=$('curve').getContext('2d'),key=$('curveMetric').value;c.clearRect(0,0,1100,230);const values=DATA.validation.map(r=>Number(r[key])),lo=0,hi=Math.max(...values,0.01)*1.08;for(const [condition,color] of [['baseline','#4db8e2'],['shared_projection','#f29870']]){const rows=DATA.validation.filter(r=>r.condition===condition).sort((a,b)=>Number(a.epoch)-Number(b.epoch));c.strokeStyle=color;c.lineWidth=2;c.beginPath();for(const [i,r] of rows.entries()){const x=50+(Number(r.epoch)-1)/59*1020,y=200-(Number(r[key])-lo)/(hi-lo)*175;if(i)c.lineTo(x,y);else c.moveTo(x,y)}c.stroke();const best=DATA.endpoints.find(r=>r.condition===condition&&r.endpoint==='selected'),r=rows.find(r=>r.epoch===best.epoch),x=50+(Number(r.epoch)-1)/59*1020,y=200-Number(r[key])/hi*175;c.fillStyle=color;c.beginPath();c.moveTo(x,y-6);c.lineTo(x+6,y);c.lineTo(x,y+6);c.lineTo(x-6,y);c.closePath();c.fill()}c.fillStyle='#d9e8f0';c.font='13px system-ui';c.fillText('Epoch 1',50,224);c.fillText('Epoch 60',1010,224);c.fillText(hi.toFixed(3),3,25);c.fillText('0',20,200)}
function lookup(condition,endpoint,id){return DATA.records.find(r=>r.condition===condition&&r.endpoint===endpoint&&r.image_id===id)}
function choices(){const prior=$('imageChoice').value,condition=$('condition').value,endpoint=$('endpoint').value,filter=$('filter').value,search=$('search').value.toLowerCase();$('imageChoice').replaceChildren();for(const [id,image] of Object.entries(DATA.images)){const r=lookup(condition,endpoint,id),a=lookup('baseline',endpoint,id),b=lookup('shared_projection',endpoint,id);if(!id.toLowerCase().includes(search)||filter==='wrong'&&r.correct||filter==='human'&&image.point_kind!=='human'||filter==='projection_worse'&&!(a.correct&&!b.correct))continue;const o=document.createElement('option');o.value=id;o.textContent=id+' · '+names[image.label]+(r.correct?' · correct':' · wrong');$('imageChoice').append(o)}if([...$('imageChoice').options].some(o=>o.value===prior))$('imageChoice').value=prior;draw()}
function load(src){return new Promise((resolve,reject)=>{const i=new Image();i.onload=()=>resolve(i);i.onerror=reject;i.src=src})}
async function draw(){const mine=++version,id=$('imageChoice').value,c=$('photo').getContext('2d');c.clearRect(0,0,384,512);if(!id){$('caption').textContent='No images match';$('details').textContent='';return}const image=DATA.images[id],r=lookup($('condition').value,$('endpoint').value,id),mode=$('mode').value,cls=Number($('class').value);const photo=await load(image.image);if(mine!==version)return;c.drawImage(photo,0,0,384,512);let overlay=mode==='masks'?r.masks:mode==='truth'?image.truth_masks:mode==='heat'?r.heats[cls]:null;if(overlay){const o=await load(overlay);if(mine!==version)return;c.imageSmoothingEnabled=mode==='heat';c.drawImage(o,0,0,384,512)}function points(xs,color){c.strokeStyle=color;c.fillStyle=color;c.lineWidth=2;for(let side=0;side<2;side++){c.beginPath();for(let j=0;j<3;j++){const x=xs[2*j+side]*383,y=image.guides[j]*511;if(j)c.lineTo(x,y);else c.moveTo(x,y)}c.stroke()}for(let j=0;j<6;j++){c.beginPath();c.arc(xs[j]*383,image.guides[Math.floor(j/2)]*511,3,0,2*Math.PI);c.fill()}}if(mode==='points')points(r.points,'#ffe675');if($('showTargets').checked&&image.point_kind==='human')points(image.targets,'#ff79e1');$('caption').textContent=id+' · '+r.condition+' / '+r.endpoint+' / epoch '+r.epoch;const paired=lookup(r.condition==='baseline'?'shared_projection':'baseline',r.endpoint,id);$('details').textContent='Tyre: '+image.tyre+'\nLabel: '+names[image.label]+'\nPrediction: '+names[r.prediction]+'\nProbabilities low / mid / high: '+r.probabilities.map(v=>(100*v).toFixed(2)+'%').join(' / ')+'\nNLL: '+r.nll.toFixed(4)+'\nMean Dice: '+(100*r.dice).toFixed(3)+'%\nHuman-point error: '+(r.point_mae===null?'unavailable':(100*r.point_mae).toFixed(3)+'% width')+'\nOther condition prediction: '+names[paired.prediction]+'\nAdditive logit residual: '+r.evidence_residual.toExponential(3)+'\nHeat maximum absolute contribution: '+r.heat_scale[cls].toExponential(3)+'\nSource SHA-256: '+image.image_sha256}
for(const id of ['condition','endpoint','filter'])$(id).addEventListener('change',choices);$('search').addEventListener('input',choices);for(const id of ['imageChoice','mode','class','showTargets'])$(id).addEventListener('change',draw);$('curveMetric').addEventListener('change',curve);curve();choices();
</script></body></html>'''.replace('__DATA__',payload)
 target=OUT/'TREAD_ONE_NB05_Comparison_Explorer.html';target.write_text(html,encoding='utf-8')
 atomic(OUT/'VISUALS.json',dict(status='generated',figures=[p.name for p in sorted(figures.glob('*.png'))],interactive_records=412,source_test_images=103,
  html_bytes=target.stat().st_size,new_inference=False,new_training=False,heat_scale='per image and class; actual maximum retained',browser_render_verified=False))
 print('Rendered three scientific figures and complete 103-image / four-endpoint explorer',target.stat().st_size,'bytes',flush=True)

if __name__=='__main__':main()
