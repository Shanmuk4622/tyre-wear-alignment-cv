"""Counts, earlier-publication preservation and bounded public-audit receipt."""
import collections,csv,json,subprocess
import numpy as np
import pandas as pd
import torch
from nb24_public import *
from review import Saved
science=read(OUT/'SCIENTIFIC_VERIFICATION.json');assert science['status']=='passed'
objects=read(OUT/'OBJECT_VERIFICATION.json');saved=Saved(OUT/'remote/fit')
cache=read(OUT/'remote/fit/inputs/cache/CACHE.json')
families=[]
for family,rows in saved.state['rows'].items():
    nested=[n for n in MANIFESTS['fit']['files'] if n.startswith('inputs/cache/csv/'+family+'/')]
    cache_rows=0
    for n in nested:
        if family=='target_cells':
            chunk=cache['chunks'][int(Path(n).stem)];cache_rows+=(chunk['last']-chunk['first']+1)*512
        elif family=='geometry_sentinels':
            chunk=cache['chunks'][int(Path(n).stem)];cache_rows+=chunk['last']-chunk['first']+1
        else:
            with (OUT/'remote/fit'/n).open(encoding='utf-8',newline='') as stream:cache_rows+=sum(1 for _ in csv.DictReader(stream))
    filenames=[n for n in MANIFESTS['fit']['files'] if n.startswith('csv/'+family+'/')]+nested
    families.append(dict(family=family,committed_fit_rows=rows,inherited_preparation_rows=cache_rows,total_fit_publication_rows=rows+cache_rows,
        files=len(filenames),bytes=sum(MANIFESTS['fit']['files'][n]['bytes'] for n in filenames),populated=rows+cache_rows>0,
        count_basis='complete-state counters + preparation chunk/source counts; not independent full dense CSV reparse'))
pd.DataFrame(families).to_csv(OUT/'CSV_FAMILY_INVENTORY.csv',index=False)
probes=saved.frame('intervention_metrics').groupby(['variant','space'],as_index=False).agg(images=('image_id','nunique'),cosine_change=('cosine_change','mean'),l2_change=('l2_change','mean'),local_loss=('local_loss','mean'),masked_loss=('masked_loss','mean'))
probes.to_csv(OUT/'INTERVENTION_SUMMARY.csv',index=False)
events=saved.frame('numerical_events');assert events.finite.eq(1).all() and len(saved.frame('errors'))==0
old=[]
for n in (13,14,15,16,17):
    earlier=Path(read(ROOT/f'tread_two/nb{n}_completion/LATEST.json')['path']);v=read(earlier/'VERIFICATION.json');m=read(earlier/'REMOTE_MANIFEST.json')
    current=public.request(f'https://huggingface.co/datasets/{REPO}/resolve/{META["latest_revision"]}/{v["prefix"]}/MANIFEST.json').json();assert current==m
    old.append(dict(notebook=n,entries=len(m['files']),unchanged=True))
earlier=Path(read(ROOT/'tread_two/nb20_completion/LATEST.json')['path']);v=read(earlier/'VERIFICATION.json')
for stage,s in v['stages'].items():
    m=read(earlier/(stage.upper()+'_MANIFEST.json'));current=public.request(f'https://huggingface.co/datasets/{REPO}/resolve/{META["latest_revision"]}/{s["prefix"]}/MANIFEST.json').json();assert current==m
    old.append(dict(notebook=20,stage=stage,entries=len(m['files']),unchanged=True))
for n,m in MANIFESTS['review']['files'].items():assert file_meta(OUT/'remote/review'/n)=={k:m[k] for k in ('bytes','sha256')}
viewer=subprocess.run(['node',str(ROOT/'analysis/test_nb24_passport.cjs'),str(OUT/'remote/review/TRACE_Evidence_Passport.html')],capture_output=True,text=True)
assert viewer.returncode==0,viewer.stderr
stages={}
for name,s in META['stages'].items():
    m=MANIFESTS[name]['files'];stages[name]=dict(**s,logical_files_including_manifest=len(m)+1,CSV_files=sum(n.endswith('.csv') for n in m),NPY_files=sum(n.endswith('.npy') for n in m),PT_files=sum(n.endswith('.pt') for n in m))
report=dict(status='passed',executed_notebook_sha256=read(OUT/'INSPECTION.json')['executed_notebook_sha256'],protocol=META['protocol'],review_protocol=META['review_protocol'],latest_revision=META['latest_revision'],
    payload_files=33,code_cells_completed=3,notebook_errors=0,stages=stages,objects=objects['objects'],review_all_objects_full_body_SHA256=True,
    verification_scope=objects['scope'],independent_science_inputs=dict(native_files=science['full_body_native_inputs'],native_bytes=science['full_body_native_bytes'],objectives=2940,validation_epochs=60),
    committed_fit_rows=sum(saved.state['rows'].values()),inherited_preparation_CSV_rows=sum(x['inherited_preparation_rows'] for x in families),
    full_fit_publication_CSV_rows=sum(x['total_fit_publication_rows'] for x in families),schema_families=48,schema_fields=671,fit_populated_families=sum(x['populated'] for x in families),
    original_publications_unchanged=old,viewer=json.loads(viewer.stdout),selected=None,scientific_status='research_only_unpromoted',local_credentials_read=False,remote_writes=0,new_optimizer_updates=0,new_photos=0)
atomic(OUT/'VERIFICATION.json',report);print(json.dumps(report,indent=2))
