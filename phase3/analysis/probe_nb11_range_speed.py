"""Measure one small public byte range without exposing signed CDN URLs."""
import json,sys,time
from pathlib import Path
import requests
import nb11_public as pub
assert Path(sys.prefix).name=='cv_conda'
m=json.loads((pub.OUT/'REMOTE_MANIFEST.json').read_text());name='csv/epoch_0007/affinities.csv'
s=requests.Session();s.trust_env=False;start=time.monotonic();count=0
with s.get(pub.BASE+m['files'][name]['object']+'?download=true&probe=one_mib',headers={'Range':'bytes=0-1048575'},timeout=(15,25),stream=True) as r:
    for block in r.iter_content(64*1024):
        count+=len(block)
        if time.monotonic()-start>30:break
    print(json.dumps(dict(status=r.status_code,bytes=count,seconds=time.monotonic()-start,content_range=r.headers.get('Content-Range'))))
