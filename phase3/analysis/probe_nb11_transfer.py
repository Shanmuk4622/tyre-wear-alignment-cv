"""Bounded public transfer probe; never logs signed URLs or credentials."""
import json,sys,time
from pathlib import Path
from urllib.parse import urlparse
import requests
assert Path(sys.prefix).name=='cv_conda'
import nb11_public as pub
m=json.loads((pub.OUT/'REMOTE_MANIFEST.json').read_text())
names=['csv/epoch_0007/affinities.csv','artifacts/endpoint_maps/selected/chunk_00000.pt']
s=requests.Session();s.trust_env=False
for name in names:
    begun=time.monotonic();url=pub.BASE+m['files'][name]['object']+'?download=true'
    try:
        with s.get(url,headers={'Range':'bytes=0-1023'},timeout=(15,25),stream=True) as r:
            block=next(r.iter_content(1024),b'')
            print(json.dumps(dict(name=name,status=r.status_code,host=urlparse(r.url).hostname,first_bytes=len(block),seconds=time.monotonic()-begun,content_length=r.headers.get('Content-Length'),content_range=r.headers.get('Content-Range'))),flush=True)
    except requests.RequestException as e:print(json.dumps(dict(name=name,error=type(e).__name__,seconds=time.monotonic()-begun)),flush=True)
