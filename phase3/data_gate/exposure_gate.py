"""Aggregate saved training observations, never run the trained models."""
import csv,sys,threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import requests
from common_gate import ROOT,read,sha
sys.path.insert(0,str(ROOT/'analysis'))
from nb05_public_reads import retry
LOCAL=threading.local()

def aggregate(cache,rows):
 pins=read(ROOT/'data_gate/exposure_sources.json');source={r['image_id']:r for r in rows};cache=Path(cache);cache.mkdir(exist_ok=True,parents=True)
 def fetch(job):
  condition,entry,pin=job;old=ROOT/'revision2_validation/nb05_publication/remote'/condition/entry['name'];target=cache/condition/Path(entry['name']).parent.name/'train_samples.csv'
  if old.exists():assert sha(old)==entry['sha256'];return condition,old
  target.parent.mkdir(parents=True,exist_ok=True)
  if not target.exists() or sha(target)!=entry['sha256']:
   if not hasattr(LOCAL,'session'):LOCAL.session=requests.Session();LOCAL.session.trust_env=False
   def download():
    with LOCAL.session.get(f'https://huggingface.co/datasets/{pins["repository"]}/resolve/{pin["revision"]}/{pin["destination"]}/{entry["object"]}',timeout=(20,120),stream=True) as r:
     r.raise_for_status()
     with open(target.with_suffix('.part'),'wb') as f:
      for block in r.iter_content(1024**2):f.write(block)
    target.with_suffix('.part').replace(target)
   retry(download)
  assert sha(target)==entry['sha256'] and target.stat().st_size==entry['bytes'];return condition,target
 jobs=[(c,e,p) for c,p in pins['conditions'].items() for e in p['files']];paths=[]
 with ThreadPoolExecutor(max_workers=4) as pool:
  for future in as_completed([pool.submit(fetch,j) for j in jobs]):paths.append(future.result())
 statistics=defaultdict(lambda:dict(occurrences=0,correct=0,nll_sum=0.,confidence_sum=0.,first_update=10**9,last_update=0));totals=defaultdict(int)
 for condition,path in sorted(paths):
  with open(path,encoding='utf-8',newline='') as f:
   for r in csv.DictReader(f):
    assert r['condition']==condition and source[r['image_id']]['role']=='train' and source[r['image_id']]['image_sha256']==r['image_sha256']
    value=statistics[(condition,r['image_id'])];value['occurrences']+=1;value['correct']+=int(r['correct']);value['nll_sum']+=float(r['nll']);value['confidence_sum']+=float(r['confidence'])
    value['first_update']=min(value['first_update'],int(r['update']));value['last_update']=max(value['last_update'],int(r['update']));totals[condition]+=1
 assert dict(totals)==dict(baseline=23520,shared_projection=23520)
 result=[]
 for condition in ('baseline','shared_projection'):
  for image_id,row in sorted(source.items()):
   value=statistics[(condition,image_id)];n=value['occurrences'];result.append(dict(condition=condition,image_id=image_id,class_index=row['class_index'],domain=row['domain'],historical_split=row['role'],
    sampled_augmented_occurrences=n,correct_augmented_occurrences=value['correct'],augmented_accuracy=value['correct']/n if n else None,
    mean_sampled_nll=value['nll_sum']/n if n else None,mean_sampled_confidence=value['confidence_sum']/n if n else None,
    first_update=value['first_update'] if n else None,last_update=value['last_update'] if n else None,
    scope='aggregate of saved pre-update augmented observations, not nominal-image or independent-subject accuracy'))
 return result,dict(source_files=len(paths),saved_occurrences=dict(totals),new_training=False,new_inference=False)
