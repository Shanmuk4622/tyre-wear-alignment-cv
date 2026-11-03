"""Bounded streaming public download of the actual TRACE interactive passport."""
import hashlib,time
import requests
from nb24_public import *
name='TRACE_Evidence_Passport.html';stage='review';m=MANIFESTS[stage]['files'][name]
target=OUT/'remote'/stage/name;expected={k:m[k] for k in ('bytes','sha256')}
if target.exists() and file_meta(target)==expected:print('Passport already checksum-verified');raise SystemExit(0)
session=requests.Session();session.trust_env=False
for attempt in range(4):
    try:
        public.reserve('resolver');began=time.monotonic();size=0;h=hashlib.sha256();part=target.with_name(target.name+'.part')
        with session.get(url(stage,m['object'])+'?download=true',stream=True,timeout=(20,45)) as response:
            print('Passport public response',response.status_code,'expected bytes',m['bytes'],flush=True);response.raise_for_status()
            with part.open('wb') as stream:
                last=0
                for block in response.iter_content(1024**2):
                    stream.write(block);h.update(block);size+=len(block)
                    if size-last>=4*1024**2:print('Passport bytes',size,'/',m['bytes'],flush=True);last=size
                    assert size<=m['bytes'] and time.monotonic()-began<180,'Bounded public download exceeded size/time'
        assert size==m['bytes'] and h.hexdigest()==m['sha256']
        part.replace(target);print('Actual passport checksum verified.',flush=True);break
    except Exception as exc:
        print('Passport retry',attempt+1,type(exc).__name__,flush=True)
        if attempt==3:raise
        time.sleep(2**attempt)
