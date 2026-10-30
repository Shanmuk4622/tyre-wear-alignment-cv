"""Independently audit retained real-update evidence and prepare durable figures."""
from PIL import Image
import ast,base64,csv,gzip,hashlib,json,re,shutil,subprocess,sys
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'training'))
import numpy as np,torch
from transactions import Store,atomic,read,sha
from telemetry import SCHEMAS
import gate
from data import plan,batch

def rows(snapshot,table):
 result=[]
 for p in sorted((snapshot/'csv').glob(f'epoch_*/{table}.csv')):
  with open(p,encoding='utf-8',newline='') as f:
   reader=csv.DictReader(f);assert reader.fieldnames==SCHEMAS[table];result.extend(reader)
 return result

def main():
 completion=read(ROOT/'training_gates/GATE_COMPLETION.json');assert completion['status']=='passed' and not completion['new_study_started']
 device=completion['devices'][0];path=Path(device['path']);cfg=read(path/'CONFIG.json')
 assert all(sha(ROOT/n)==h for n,h in cfg['source_hashes'].items()),'Stale local gate sources'
 assert device['local']['status']=='passed'
 current=gate.store(path/'uninterrupted',cfg);snap=Path(read(path/'LATEST_ENVELOPE.json')['path'])/'cases/uninterrupted'
 manifest=read(snap/'MANIFEST.json')
 for n,m in manifest['files'].items():assert sha(snap/n)==m['sha256'] and (snap/n).stat().st_size==m['bytes']
 tables={t:rows(snap,t) for t in SCHEMAS};counts={t:len(v) for t,v in tables.items()}
 assert counts==current.state()['row_cursors']==read(snap/'CURSOR.json')['row_cursors']
 assert counts['train_steps']==3 and counts['train_samples']==24 and counts['geometry_points']==144
 scientific=('train_steps','train_samples','region_stats','geometry_points','latent_metrics','evidence_metrics','class_confusion','gradient_modules','activation_modules','validation_predictions','epoch_summaries')
 omit={t:{'checkpoint_before_sha256','checkpoint_after_sha256'} for t in scientific}
 omit['train_steps'].update(k for k in SCHEMAS['train_steps'] if k.endswith('_seconds'))
 same={}
 for name in ('paused','abrupt','installed'):
  other=Path(read(path/'LATEST_ENVELOPE.json')['path'])/'cases'/name
  for t in scientific:
   a=[{k:v for k,v in r.items() if k not in omit[t]} for r in tables[t]]
   b=[{k:v for k,v in r.items() if k not in omit[t]} for r in rows(other,t)]
   assert a==b,(name,t)
  for file in (snap/'maps').glob('update_*/*.npy'):assert sha(file)==sha(other/'maps'/file.relative_to(snap/'maps'))
  same[name]=dict(scientific_tables_equal=len(scientific),native_batch_arrays_equal=9)
 source={r['image_id']:r for r in plan(Path(cfg['data_root']))['rows']}
 max_reconstruction=0.;samples=tables['train_samples']
 for step in tables['train_steps']:
  u=int(step['update']);group=[r for r in samples if int(r['update'])==u];assert len(group)==8
  def total(k):return sum(float(r[k]) for r in group)
  rebuilt=total('label_smoothed_ce')/8+.2*total('ordinal_loss')/8+total('seg_bce_numerator')/max(total('seg_bce_denominator'),1)+total('seg_dice_loss')/8
  rebuilt+=2*total('geometry_numerator')/max(total('geometry_weight'),1)+.1*total('containment_numerator')/max(total('containment_denominator'),1)
  rebuilt+=.1*total('alignment_numerator')/max(total('alignment_denominator'),1)+.2*total('background_kl')/8
  err=abs(rebuilt-float(step['loss']));max_reconstruction=max(max_reconstruction,err);assert err<2e-5
  assert abs(total('objective_contribution')-float(step['sample_contribution_sum']))<1e-10
  assert float(step['grad_norm_after_clip'])<=1.0001
  folder=snap/'maps'/f'update_{u:08d}'
  inputs=np.load(folder/'inputs.npy',allow_pickle=False);target=np.load(folder/'targets.npy',allow_pickle=False);out=np.load(folder/'outputs.npy',allow_pickle=False)
  assert inputs.shape==(8,3,512,384) and target.shape==(8,3,512,384) and out.shape==(8,18,128,96)
  assert inputs.dtype==out.dtype==np.float32 and target.dtype==np.uint8
  assert np.isfinite(inputs).all() and np.isfinite(out).all() and set(np.unique(target))<={0,1}
  batch_rows=[source[r['image_id']] for r in group]
  expected=batch(Path(cfg['data_root']),batch_rows,'cpu',cfg['seed'],int(step['epoch']),int(step['cursor']))
  assert np.array_equal(inputs,expected['img'].numpy()) and np.array_equal(target,expected['masks'].numpy())
  for r in group:
   p=np.array([float(r[f'probability_{c}']) for c in range(3)])
   assert abs(p.sum()-1)<.002 and int(r['prediction'])==int(p.argmax())
   s=source[r['image_id']];assert s['role']=='train' and s['image_sha256']==r['image_sha256'] and int(r['label'])==s['class_index']
  for r in [r for r in tables['region_stats'] if int(r['update'])==u]:
   i=int(r['slot']);c=0 if r['region']=='tyre' else 1;valid=target[i,2]<.5
   pred=np.repeat(np.repeat(out[i,c]>=.5,4,0),4,1)&valid;truth=(target[i,c]>.5)&valid
   vals=[int(v.sum()) for v in (pred&truth,pred&~truth&valid,~pred&truth&valid,~pred&~truth&valid)]
   assert vals==[int(r[k]) for k in ('tp','fp','fn','tn')]
  for r in [r for r in tables['geometry_points'] if int(r['update'])==u]:
   i,k=int(r['slot']),int(r['point_index']);s=batch_rows[i]
   assert float(r['optimization_target'])==float(expected['points'][i,k])
   if s['points'] is None:assert r['target']=='' and float(r['point_weight'])==0
   else:
    assert abs(float(r['target'])-float(expected['points'][i,k]))<1e-12
    e=float(r['prediction'])-float(r['target']);assert abs(e-float(r['signed_error_width_fraction']))<1e-12
    assert abs(abs(e)*(s['width']-1)-float(r['absolute_error_native_pixels']))<1e-9
   assert int(r['human_eligible'])==int(s['point_kind']=='human')
  for r in [r for r in tables['evidence_metrics'] if int(r['update'])==u]:
   assert abs(float(r['sum_contribution'])+float(r['bias'])-float(r['logit']))<.002
 assert [int(r['update']) for r in tables['validation_predictions']]==[2,2]
 assert all(source[r['image_id']]['role']=='validation' for r in tables['validation_predictions'])
 # Payload, delivery and static JS syntax; no browser workaround is attempted.
 import nbformat
 notebook=ROOT/'notebooks/phase3_NB04_TREAD_ONE_Evidence_And_Recovery_Gate.ipynb';nb=nbformat.read(notebook,as_version=4);nbformat.validate(nb)
 delivery=read(ROOT/'training_gates/DELIVERY.json')
 publication_path=ROOT/'training_gates/nb04_publication/VERIFICATION.json'
 publication=read(publication_path) if publication_path.exists() else None
 executed=bool(publication and sha(notebook)==publication['executed_notebook_sha256'])
 assert executed or sha(notebook)==delivery['notebook_sha256']
 embedded={}
 for cell in nb.cells:
  if cell.cell_type!='code':continue
  assert not any(o.get('output_type')=='error' for o in cell.outputs)
  if not executed:assert not cell.outputs and cell.execution_count is None
  for call in ast.walk(ast.parse(cell.source)):
   if isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute) and call.func.attr=='b64decode' and call.args and isinstance(call.args[0],ast.Constant):
    payload=json.loads(gzip.decompress(base64.b64decode(call.args[0].value)))
    for name,value in payload.items():
     data=base64.b64decode(value);assert data==(ROOT/name).read_bytes();embedded[name]=hashlib.sha256(data).hexdigest()
 assert embedded==delivery['embedded_sources'] and len(embedded)==17
 outdir=ROOT/'training_gates/summary_exports';outdir.mkdir(exist_ok=True)
 shutil.copy2(snap/'EVIDENCE_GATE.html',outdir/'TREAD_ONE_Evidence_Gate.html')
 html=(snap/'EVIDENCE_GATE.html').read_text();script=re.search(r'<script>(.*?)</script>',html,re.S).group(1)
 testjs=outdir/'evidence_gate_syntax.js';testjs.write_text(script,encoding='utf-8')
 node=shutil.which('node');assert node,'Node syntax checker required'
 r=subprocess.run([node,'--check',str(testjs)],capture_output=True,text=True);assert r.returncode==0,r.stderr
 data=json.JSONDecoder().raw_decode(script.split('const DATA=',1)[1])[0]
 assert len(data['gallery'])==3 and len(data['steps'])==3
 for g in data['gallery']:
  import io
  with Image.open(io.BytesIO(base64.b64decode(g['image'].split(',',1)[1]))) as im:assert im.size==(384,512);im.verify()
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(2,3,figsize=(10,8),layout='constrained')
 for col,u in enumerate((1,2,3)):
  p=snap/'maps'/f'update_{u:08d}';image=np.load(p/'inputs.npy')[0].transpose(1,2,0)
  image=np.clip(image*np.array([.229,.224,.225])+np.array([.485,.456,.406]),0,1)
  output=np.load(p/'outputs.npy')[0];axes[0,col].imshow(image);axes[0,col].set_title(f'Update {u} · actual augmented input')
  axes[1,col].imshow(image);mask=np.repeat(np.repeat(output[1]>=.5,4,0),4,1)
  overlay=np.zeros((512,384,4));overlay[mask]=[0,.8,.7,.48];axes[1,col].imshow(overlay);axes[1,col].set_title('Pre-update predicted tread overlay')
  axes[0,col].axis('off');axes[1,col].axis('off')
 fig.suptitle('TREAD-ONE engineering fixture · 3 updates · logging/recovery evidence only',fontsize=12)
 fig.savefig(outdir/'engineering_inputs_and_masks.png',dpi=160);plt.close(fig)
 result=dict(status='passed',protocol=device['protocol'],local_device=device['device'],counts=counts,total_core_rows=sum(counts.values()),
  scalar_columns=sum(len(v) for v in SCHEMAS.values()),source_files=len(embedded),loss_reconstruction_max_abs_error=max_reconstruction,
  real_augmented_sample_occurrences_verified=24,scientific_recovery_comparisons=same,
  original_r1_math_exact=device['local']['original_r1_update_math'],native_maps_per_instrumented_case=9,
  notebook_sha256=sha(notebook),historical_clean_notebook_sha256=delivery['notebook_sha256'],interactive_html_sha256=sha(outdir/'TREAD_ONE_Evidence_Gate.html'),
  browser_render_tested=False,JS_syntax_checked=True,kaggle_execution_tested=executed,live_hf_gate_tested=executed,
  engineering_only=True,new_study_started=False,local_run_path=str(path),snapshot=str(snap))
 atomic(ROOT/'training_gates/EVIDENCE_VALIDATION.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
