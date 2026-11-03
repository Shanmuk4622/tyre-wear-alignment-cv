"""Versioned training overlay; source-v1 is always read-only."""
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

RELEASE_SHA = '06c6c9d003aff808babf7ebb5786f441e95064eb000f5e5ed5b155d0ee9e7c6a'
REVISION = 'phase2-training-r1'
MODELS = ['mobilenetv4', 'resnet50', 'segformer', 'yolo26m', 'hrnet']
SPECS = {
 'mobilenetv4': dict(model='mobilenetv4_conv_medium.e500_r256_in1k', repo='timm/mobilenetv4_conv_medium.e500_r256_in1k', revision='ad66898c045c1b5223ea3f2c0830b74cf2e75bac', file='model.safetensors', hw=[384,384], batch=8),
 'resnet50': dict(model='resnet50.a1_in1k', repo='timm/resnet50.a1_in1k', revision='767268603ca0cb0bfe326fa87277f19c419566ef', file='model.safetensors', hw=[384,384], batch=8),
 'segformer': dict(model='nvidia/mit-b0', repo='nvidia/mit-b0', revision='80983a413c30d36a39c20203974ae7807835e2b4', file='pytorch_model.bin', hw=[512,384], batch=2),
 'hrnet': dict(model='hrnet_w18.ms_aug_in1k', repo='timm/hrnet_w18.ms_aug_in1k', revision='7e2c5583769f54514fd87e3ba9de408e33eaba0f', file='model.safetensors', hw=[512,384], batch=2),
 'yolo26m': dict(model='yolo26m-seg.yaml', url='https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m-seg.pt', sha256='16b636f04e8fb6a325b3370f22dc5e5535ff473e384f4d041fd28d788f6ee9f5', hw=[512,384], batch=2),
}

def canonical(x): return json.dumps(x, sort_keys=True, separators=(',', ':')).encode()
def digest(x): return hashlib.sha256(canonical(x)).hexdigest()
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,data):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
 tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_bytes(canonical(data));tmp.replace(p)

def locate(root=''):
 if root: paths=[Path(root)]
 else: paths=[p.parent for p in Path('/kaggle/input').rglob('VERSION.json') if read(p).get('dataset_title')=='Tire Dataset Prepared phase2']
 if len(paths)!=1:raise ValueError('Attach exactly one expanded tire-dataset-prepared-phase2 input, or set DATA_ROOT.')
 root=paths[0].resolve()
 if sha(root/'SHA256SUMS.txt')!=RELEASE_SHA:raise ValueError('Wrong dataset release; do not bypass the checksum.')
 return root

def overlay(root):
 rows=read(Path(root)/'manifests/images.json')
 # All possible matches of every unresolved video are restricted to TRAIN.
 # Holdouts are whole known low/high tyres, selected by counts, never by model results.
 old=[r for r in rows if r['domain']=='original_photo']
 counts=Counter(r['physical_tyre_id'] for r in old)
 classes={r['physical_tyre_id']:r['class_index'] for r in old}
 groups=dict(train=[],validation=[],test=[])
 for cls in (0,2):
  candidates=sorted((g for g in counts if classes[g]==cls and counts[g]>=20),key=lambda g:(-counts[g],g))
  assert len(candidates)>=3
  groups['validation'].append(candidates[2]);groups['test'].append(candidates[1])
 groups['train']=sorted(set(counts)-set(groups['validation'])-set(groups['test']))
 assert all(g in groups['train'] for g in counts if classes[g]==1)
 human={r['image_id']:r for r in read(Path(root)/'geometry/original_human_points.json')}
 proposed={r['image_id']:r for r in read(Path(root)/'geometry/new_point_proposals.json')}
 for r in rows:
  r['role']='train' if r['domain']=='new_video_frame' else next(k for k,v in groups.items() if r['physical_tyre_id'] in v)
  p=human.get(r['image_id']) or proposed.get(r['image_id'])
  r['point_weight']=1. if r['image_id'] in human else .25 if p else 0.
  r['points']=[q['x']/(r['width']-1) for q in p['points']] if p else None
  r['point_kind']='human' if r['image_id'] in human else 'polygon_weak' if p else 'none'
  if p:assert all(q['x'] is not None for q in p['points'])
 assert len(rows)==570 and len({r['image_sha256'] for r in rows})==570
 assert not any(r['domain']=='new_video_frame' and r['role']!='train' for r in rows)
 return dict(revision=REVISION,source_checksum=RELEASE_SHA,groups=groups,rows=rows,
  counts=dict(Counter(r['role'] for r in rows)),
  geometry_counts=dict(Counter(r['role'] for r in rows if r['points'] is not None)),
  decisions=dict(identity='All three original mid tyres and all new frames are training-only; unknown exact mapping is not invented.',
   masks='SegFormer ignores contradictory pixels. YOLO training tyre = raw tyre OR raw tread; tread unchanged. Derived policy, not human correction.',
   geometry='New polygon-derived points are weak training targets with weight 0.25; validation/test use only existing human points.',
   evaluation='Held-out old low/high tyres only. No unseen-mid, unseen-video, three-class-generalisation or safety claim.'))

def train_rows(plan,job):
 return [r for r in plan['rows'] if r['role']=='train' and (job['condition']=='combined' or r['domain']=='original_photo') and (job['model']!='hrnet' or r['points'] is not None)]

def schedule(plan,job,epoch):
 """Stateless class-balanced sampler, then domain-balanced within each class."""
 rows=train_rows(plan,job);rs=np.random.default_rng(job['seed']*1000003+epoch)
 pools={}
 for i,r in enumerate(rows):
  group=r['physical_tyre_id'] or r['source_video']
  pools.setdefault(r['class_index'],{}).setdefault(r['domain'],{}).setdefault(group,[]).append(i)
 # Same optimizer-update budget for old-only and combined within each architecture.
 reference=train_rows(plan,dict(job,condition='combined'))
 batch=SPECS[job['model']]['batch'];steps=math.ceil(len(reference)/batch)
 order=[]
 for _ in range(steps*batch):
  cls=int(rs.choice(sorted(pools)));domains=pools[cls];domain=str(rs.choice(sorted(domains)))
  bags=domains[domain];group=str(rs.choice(sorted(bags)));order.append(int(rs.choice(bags[group])))
 return rows,[order[i:i+batch] for i in range(0,len(order),batch)]

def sample(root,row,hw,seed=None,model='segformer'):
 """Synchronous deterministic CPU transforms; no hidden prefetch/RNG state."""
 with Image.open(Path(root)/row['image_path']) as im:im=im.convert('RGB').resize(tuple(hw[::-1]),Image.Resampling.BILINEAR)
 masks=[]
 for key in ('tyre_mask','tread_mask','ignore_mask'):
  with Image.open(Path(root)/row[key]) as m:masks.append(np.asarray(m.resize(tuple(hw[::-1]),Image.Resampling.NEAREST))>0)
 masks=np.stack(masks);points=np.array(row['points'] or [0.]*6,dtype=np.float32)
 if seed is not None:
  rs=np.random.default_rng(seed)
  if rs.random()<.5:
   im=im.transpose(Image.Transpose.FLIP_LEFT_RIGHT);masks=masks[:,:,::-1].copy();points=1-points[[1,0,3,2,5,4]]
  im=ImageEnhance.Brightness(im).enhance(float(rs.uniform(.8,1.2)))
  im=ImageEnhance.Contrast(im).enhance(float(rs.uniform(.85,1.15)))
  if rs.random()<.2:im=im.filter(ImageFilter.GaussianBlur(float(rs.uniform(.2,.8))))
 if model=='yolo26m':masks[0]|=masks[1]
 return np.asarray(im,dtype=np.float32).transpose(2,0,1).copy()/255.,masks.copy(),points

def jobs(conditions=('combined',),seeds=(1,2,3)):
 return [dict(model=m,condition=c,seed=s,id=f'{m}-{c}-seed{s}') for m in MODELS for c in conditions for s in seeds]
