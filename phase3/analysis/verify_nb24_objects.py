"""Verify every selected HF object against pinned Git/LFS content metadata.

Large LFS bodies use the server's SHA256+size identity. Small or already cached
Git bodies receive both full SHA256 and Git SHA1 checks. Remaining large Git
CSVs/HTML receive existence/size/Git identity checks only. Follow-on science
checks download their own bounded native/CSV input set in full.
"""
import hashlib,json,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from urllib.parse import quote
import pandas as pd
from nb24_public import *
counts={};all_rows=[]
for stage,s in META['stages'].items():
    prefix=s['prefix'];cache=OUT/(stage.upper()+'_TREE.json')
    if cache.exists():tree=read(cache)
    else:
        link=f'https://huggingface.co/api/datasets/{REPO}/tree/{s["revision"]}/{quote(prefix,safe="/")}?recursive=true&limit=1000'
        tree=[]
        while link:
            response=public.request(link);tree.extend(response.json());link=response.links.get('next',{}).get('url')
        atomic(cache,tree)
    indexed={r['path'][len(prefix)+1:]:r for r in tree if r['type']=='file'}
    rows=[];git=[]
    for name,m in MANIFESTS[stage]['files'].items():
        assert not name.endswith(('.zip','.npz','.sqlite','.raw')),(stage,name)
        assert m['object'] in indexed,(stage,name)
        obj=indexed[m['object']];assert obj['size']==m['bytes'],(stage,name,'size')
        if obj.get('lfs'):
            lfs=obj['lfs'];assert lfs['oid']==m['sha256'] and lfs['size']==m['bytes'],(stage,name,'LFS hash')
            method='pinned_server_LFS_SHA256_and_size'
        elif m['bytes']<=750000 or (OUT/'remote'/stage/name).exists():
            git.append((name,obj));method='downloaded_body_SHA256_and_Git_blob_SHA1'
        else:method='pinned_server_Git_identity_and_size_only'
        rows.append(dict(stage=stage,file=name,bytes=m['bytes'],sha256=m['sha256'],object=m['object'],method=method,verified=True))
    def check_git(item):
        name,obj=item;p=fetch(stage,name);body=p.read_bytes()
        assert hashlib.sha1(b'blob '+str(len(body)).encode()+b'\0'+body).hexdigest()==obj['oid'],(stage,name,'git')
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i,_ in enumerate(pool.map(check_git,git)):
            if (i+1)%100==0:print(stage,'Git bodies',i+1,'/',len(git),flush=True)
    df=pd.DataFrame(rows);df.to_csv(OUT/(stage.upper()+'_OBJECT_VERIFICATION.csv'),index=False)
    counts[stage]=dict(objects=len(rows),bytes=int(df.bytes.sum()),LFS_SHA256_metadata=int(sum(df.method=='pinned_server_LFS_SHA256_and_size')),Git_bodies_downloaded=len(git),large_Git_metadata_only=int(sum(df.method=='pinned_server_Git_identity_and_size_only')))
    all_rows.extend(rows);print(stage,json.dumps(counts[stage]),flush=True)
# Download small/high-value science and all review content in full.
tasks=[]
for stage in META['stages']:
    for name,m in MANIFESTS[stage]['files'].items():
        take=stage in ('review','transport_tail') and not name.endswith('.html') or name.endswith('.json') or name.startswith(('source/','epochs/','exports/','controls/')) or name==read(OUT/'FIT_POINTER.json')['state']
        if name.startswith('csv/') and not name.split('/')[1] in ('reconstruction_cells','spatial_moment_vectors','mask_plans','staging_inventory','representation_spectra'):
            take=True
        if take:tasks.append((stage,name))
with ThreadPoolExecutor(max_workers=8) as pool:
    for i,_ in enumerate(pool.map(lambda item:fetch(*item),tasks)):
        if (i+1)%100==0:print('Full audit bodies',i+1,'/',len(tasks),flush=True)
report=dict(status='passed',stages=counts,objects=len(all_rows),logical_content_bytes=sum(r['bytes'] for r in all_rows),
    full_selected_LFS_redownload=False,scope='All selected object existence/size; all LFS pinned SHA256 identities; bounded Git bodies full SHA256+Git SHA1; remaining large Git metadata only. Not an independent full-body audit of every file.',
    additional_full_body_inputs=len(tasks),public_reads=public.CALLS,local_credentials_read=False,remote_writes=0)
atomic(OUT/'OBJECT_VERIFICATION.json',report);print(json.dumps(report,indent=2))
