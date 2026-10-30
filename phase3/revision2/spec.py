"""Frozen paired pilot: one architecture, one seed, one shared-gradient intervention."""
from PIL import Image
import json,sys,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for name in ('runtime','training'):sys.path.append(str(ROOT/name))
import torch
from common import environment
from data import plan,schedule
from transactions import digest,read,sha
CHECKSUM='06c6c9d003aff808babf7ebb5786f441e95064eb000f5e5ed5b155d0ee9e7c6a'
OVERLAY='f81a80d20b60ba505d94e31c26615e7e7171a91a8506948be6c392b68107a039'
CONDITIONS=('baseline','shared_projection')
SHARED=('stem','s1','s2','s3','s4','position','context','d3','d2','d1')
CAPTURE_EPOCHS=(1,10,20,30,40,50,60)

def locate(value=''):
 if value:return Path(value).resolve()
 if Path('/kaggle/input').exists():
  roots=[p.parent for p in Path('/kaggle/input').rglob('VERSION.json') if read(p).get('dataset_title')=='Tire Dataset Prepared phase2']
  assert len(roots)==1,'Attach exactly one expanded prepared Phase 2 input';return roots[0]
 return ROOT.parent/'phase2/data/phase2_dataset_v1'

def configuration(root,device,condition,smoke=False):
 assert condition in CONDITIONS and sha(root/'SHA256SUMS.txt')==CHECKSUM
 base=read(ROOT/'revision2/BASE_SOURCES.json')
 assert all(sha(ROOT/n)==h for n,h in base.items()),'Frozen r1/NB04 dependency changed'
 p=plan(root);assert digest(p)==OVERLAY
 hashes=dict(line.split('  ',1)[::-1] for line in (root/'SHA256SUMS.txt').read_text().splitlines())
 metadata={n:h for n,h in hashes.items() if n.endswith('.json')}
 for n,h in metadata.items():assert sha(root/n)==h
 sources=dict(base);sources.update({v.relative_to(ROOT).as_posix():sha(v) for v in (ROOT/'revision2').glob('*') if v.suffix in ('.py','.json')})
 recipe=dict(family='TREAD-ONE',revision='r2-shared-projection-paired-pilot',mode='engineering_smoke' if smoke else 'measured_paired_pilot',
  conditions=list(CONDITIONS),seed=1,epochs=2 if smoke else 60,batches_per_epoch=2 if smoke else len(schedule(p,1,0)[1]),batch=8,input_hw=[512,384],
  objective='Unchanged r1 weighted objective; symmetric shared-only two-objective conflict projection versus summed gradient',
  shared_modules=list(SHARED),projection_epsilon=1e-20,validation_chunk=8,
  capture_epochs=[1,2] if smoke else list(CAPTURE_EPOCHS),validation_gallery_per_class=1 if smoke else 6,
  validation_scope='two human validation fixtures' if smoke else 'all 81 fixed validation images',
  test_scope='validation fixtures only; no test image accessed' if smoke else '103 previously inspected development test images, selected and final endpoints',
  selection='.5*BA_observed+.35*mean_two_region_Dice+.15*max(0,1-frozen_r1_human_point_MAE/.05); strict improvement, earliest tie',
  source_checksum=CHECKSUM,overlay_sha256=OVERLAY,metadata_hashes=metadata,sources=sources,
  full_native_local_keep=2,publish_seconds=1800,account='Shanmuk4622/tyre-wear-study')
 recipe['working_logical_budget_bytes']=16_000_000_000
 intent=digest(recipe)
 cfg=dict(recipe,study_id=intent,condition=condition,device=str(device),environment=environment(torch.device(device)),data_root=str(root))
 return cfg

def contract(cfg):return {k:v for k,v in cfg.items() if k!='data_root'}
def protocol(cfg):return digest(contract(cfg))

def capacity(enforce=True):
 scope=Path('/kaggle/working') if Path('/kaggle/working').exists() else ROOT/'revision2_runs'
 size=sum(p.stat().st_size for p in scope.rglob('*') if p.is_file()) if scope.exists() else 0
 result=dict(scope=str(scope),logical_file_bytes=size,free_filesystem_bytes=shutil.disk_usage(ROOT).free,budget_bytes=16_000_000_000,
   accounting='Logical file lengths including hard-link appearances; conservative for saved output')
 if enforce:assert size<result['budget_bytes'],'Working output above 16 GB logical guard; safely flush before growing further'
 return result

def eval_rows(p,cfg,split):
 rows=sorted([r for r in p['rows'] if r['role']==split],key=lambda r:r['image_id'])
 if cfg['mode']=='engineering_smoke':
  rows=[sorted([r for r in p['rows'] if r['role']=='validation' and r['class_index']==c and r['point_kind']=='human'],key=lambda r:r['image_id'])[0] for c in (0,2)]
 return rows
