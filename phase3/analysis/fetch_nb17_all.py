"""Resumable public mirror: every selected file gets independent size/SHA verification."""
import json,sys,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
import nb17_public as client
m=json.loads((client.OUT/'REMOTE_MANIFEST.json').read_text());items=list(m['files'].items())
# Scalar/CSV/epoch data arrive first so scientific review can proceed during the
# larger native-map transfer. Native bodies remain fully checked before closure.
items.sort(key=lambda v:(not v[0].endswith(('.json','.csv','.pt','.html')),v[0]))
done=0;total_bytes=0;start=time.monotonic();last=0.
with ThreadPoolExecutor(max_workers=32) as pool:
    futures={pool.submit(client.fetch,item):item for item in items}
    for future in as_completed(futures):
        name,meta=futures[future];future.result();done+=1;total_bytes+=meta['bytes']
        if time.monotonic()-last>=20 or done==len(items):
            status=dict(status='running' if done<len(items) else 'complete',files=done,total=len(items),bytes=total_bytes,requests=client.CALLS,seconds=time.monotonic()-start)
            (client.OUT/'DOWNLOAD_STATUS.json').write_text(json.dumps(status,indent=2)+'\n',encoding='utf-8');print(json.dumps(status),flush=True);last=time.monotonic()
print('All selected file bytes independently verified.',flush=True)
