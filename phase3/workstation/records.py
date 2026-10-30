"""Direct CSV inference ledger and independently restorable native evidence captures."""
import csv,json,os,uuid
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from PIL import Image
from paths import ROOT,PHASE3,DATA,sha,read,write_json
from imaging import working_image,masks_at,mask_boxes,fingerprint,render
from schema import describe
def now():return datetime.now(timezone.utc).isoformat()
def safe_cell(value):
 if isinstance(value,str) and value.startswith(('=','+','-','@','\t','\r')):return "'"+value
 return value
def csv_file(path,rows,fields):
 with Path(path).open('w',encoding='utf-8',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader()
  for r in rows:writer.writerow({k:safe_cell(r.get(k)) for k in fields})
def tables(rgb,result,source,note=''):
 a=result['arrays'];s=result['summary'];h,w=rgb.shape[:2];labels=result['identity']['labels'];classes=[];points=[];regions=[]
 for i,label in enumerate(labels):
  e=a['contributions'][i];classes.append(dict(class_index=i,label=label,probability=float(a['probabilities'][i]),logit=float(a['logits'][i]),bias=float(a['bias'][i]),positive_evidence=float(e[e>0].sum()),negative_evidence=float(e[e<0].sum()),spatial_evidence_sum=float(e.sum()),reconstruction_error=float(e.sum()+a['bias'][i]-a['logits'][i])))
 for i,x in enumerate(a['points']):
  y=float(a['guides'][i//2]);points.append(dict(point_index=i,guide=i//2+1,side='left' if i%2==0 else 'right',x_width_fraction=float(x),y_height_fraction=y,x_native_pixels=float(x)*(w-1),y_native_pixels=y*(h-1),native_width=w,native_height=h,scope='image-space model proposal; no physical angle'))
 masks=masks_at(a,(w,h))
 for name,mask,box in zip(('tyre','tread'),masks,mask_boxes(masks)):
  regions.append(dict(region=name,positive_native_pixels=int(mask.sum()),native_area_fraction=float(mask.mean()),x0=box[0] if box else None,y0=box[1] if box else None,x1_exclusive=box[2] if box else None,y1_exclusive=box[3] if box else None,threshold=.5,native_width=w,native_height=h,scope='thresholded model mask; nearest resize for display coordinates'))
 record=dict(utc=now(),frame_sha256=fingerprint(rgb),source=source.get('path',''),media_kind=source.get('media_kind','photo'),frame_index=source.get('frame_index'),timestamp_seconds=source.get('timestamp_seconds'),native_width=w,native_height=h,
  model=result['identity']['model'],seed=result['identity']['seed'],epoch=result['identity']['epoch'],weights_sha256=result['identity']['weights_sha256'],predicted_class=s['prediction'],predicted_label=labels[s['prediction']],
  probability_low=classes[0]['probability'],probability_mid=classes[1]['probability'],probability_high=classes[2]['probability'],evidence_residual=s['evidence_residual'],model_ms=s['model_ms'],preprocess_ms=s['preprocess_ms'],device=s['device'],precision=s['precision'],review_flags='; '.join(s['review_flags']),reference_proxy_class=source.get('reference_proxy_class'),reference_scope=source.get('reference_scope','unlabelled input'),accuracy=None,dice=None,point_error=None,metrics_missing_reason='No reference masks/points are used by workstation inference; probabilities are not accuracy',note=note)
 return record,classes,points,regions

def concept_rows(result):
 a=result['arrays'];mass=(a['concepts']*a['gate']).sum((-2,-1));return [dict(channel=i,mean_activation=float(a['concepts'][i].mean()),gated_mass=float(mass[i]),scope='latent channel, not a human-labelled damage concept') for i in range(12)]

class Session:
 def __init__(self,identity,root=None):
  base=Path(root or DATA/'sessions').resolve();assert base.is_relative_to(PHASE3.resolve())
  self.root=base/(datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]);self.root.mkdir(parents=True)
  write_json(self.root/'SESSION.json',dict(schema='tread-one-station-session-v1',created=now(),identity=identity,scope='Desktop inference, not training or a new held-out benchmark'))
 def append(self,name,rows):
  rows=list(rows)
  if not rows:return
  path=self.root/(name+'.csv');header=not path.exists()
  with path.open('a',encoding='utf-8',newline='') as f:
   writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
   if header:writer.writeheader()
   for r in rows:writer.writerow({k:safe_cell(v) for k,v in r.items()})
   f.flush();os.fsync(f.fileno())
 def event(self,event,detail=''):self.append('events',[dict(utc=now(),event=event,detail=str(detail))])
 def prediction(self,rgb,result,source):
  record,classes,points,regions=tables(rgb,result,source);key=record['frame_sha256']
  self.append('predictions',[record])
  for name,rows in [('class_scores',classes),('points',points),('regions',regions)]:self.append(name,[dict(utc=record['utc'],frame_sha256=key,**r) for r in rows])
  self.append('latent_mass',[dict(utc=record['utc'],frame_sha256=key,**r) for r in concept_rows(result)])

def save_capture(rgb,result,source,note='',captures=None):
 root=Path(captures or DATA/'captures').resolve();assert root.is_relative_to(PHASE3.resolve());root.mkdir(parents=True,exist_ok=True)
 key=datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'_'+uuid.uuid4().hex[:8];stage=(root/(key+'.partial')).resolve();final=(root/key).resolve()
 assert stage.is_relative_to(root) and final.is_relative_to(root) and not stage.exists() and not final.exists();stage.mkdir()
 record,classes,points,regions=tables(rgb,result,source,note)
 catalog={}
 for name,rows in [('prediction',[record]),('class_scores',classes),('points',points),('regions',regions)]:csv_file(stage/(name+'.csv'),rows,list(rows[0]));catalog[name+'.csv']=rows
 concepts=concept_rows(result);csv_file(stage/'latent_mass.csv',concepts,list(concepts[0]))
 catalog['latent_mass.csv']=concepts
 Image.fromarray(rgb).save(stage/'input.png');Image.fromarray(working_image(rgb)).save(stage/'model_input.png');Image.fromarray(render(rgb,result)).save(stage/'overlay.png')
 for i,mask in enumerate(masks_at(result['arrays'],(rgb.shape[1],rgb.shape[0]))):Image.fromarray(mask.astype(np.uint8)*255).save(stage/(('tyre','tread')[i]+'_mask.png'))
 for name,array in result['arrays'].items():np.save(stage/(name+'.npy'),array,allow_pickle=False)
 grid_names={'masks':'mask_probabilities','seg_logits':'mask_logits','contributions':'signed_evidence','concepts':'latent_textures','gate':'tread_gate'};grid_rows=0
 for name,filename in grid_names.items():
  array=result['arrays'][name];count=int(array.size);grid_rows+=count
  records=(dict(channel=c,row=y,column=x,value=float(array[c,y,x])) for c in range(array.shape[0]) for y in range(array.shape[1]) for x in range(array.shape[2]))
  csv_file(stage/(filename+'.csv'),records,['channel','row','column','value'])
  catalog[filename+'.csv']=dict(rows=count,fields=['channel','row','column','value'])
 write_json(stage/'CSV_SCHEMA.json',describe(catalog,result['arrays']))
 metadata={k:v for k,v in result.items() if k!='arrays'};metadata.update(source=source,note=note,record=record,schema='tread-one-station-capture-v1',coordinates='Native oriented pixels; points use width-1/height-1. Mask boxes use exclusive upper bounds.',grid_CSV_rows=grid_rows,scalar_CSV_rows=24,raw_arrays=list(result['arrays']),scope='Inference evidence only. Unknown ground-truth metrics stay blank. CSV text cells starting formula characters receive an apostrophe; the original note is retained here.')
 write_json(stage/'CAPTURE.json',metadata)
 write_json(stage/'MANIFEST.json',dict(schema='tread-one-station-capture-v1',files={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in stage.iterdir() if p.is_file()}))
 stage.rename(final);return final

def load_capture(folder):
 folder=Path(folder).resolve();manifest=read(folder/'MANIFEST.json');assert manifest['schema']=='tread-one-station-capture-v1'
 assert sum(v['bytes'] for v in manifest['files'].values())<300*1024**2
 for name,info in manifest['files'].items():
  p=(folder/name).resolve();assert p.is_relative_to(folder) and p.parent==folder and sha(p)==info['sha256'],'Capture checksum failed: '+name
 meta=read(folder/'CAPTURE.json');assert meta['schema']==manifest['schema']
 with Image.open(folder/'input.png') as image:rgb=np.array(image.convert('RGB'))
 assert fingerprint(rgb)==meta['record']['frame_sha256'];arrays={name:np.load(folder/(name+'.npy'),allow_pickle=False) for name in meta['raw_arrays']}
 assert all(v.dtype.kind=='f' and np.isfinite(v).all() for v in arrays.values())
 assert arrays['masks'].shape==(2,128,96) and arrays['points'].shape==(6,) and arrays['contributions'].shape==(3,128,96)
 result={k:meta[k] for k in ('summary','identity','environment','app_source_hashes')};result['arrays']=arrays
 return rgb,result,meta['source'],meta.get('note','')
