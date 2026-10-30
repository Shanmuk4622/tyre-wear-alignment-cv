"""Prioritize the two review affinity tables; same immutable cache, unique parts."""
import json,sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import nb11_public as pub
assert Path(sys.prefix).name=='cv_conda'
m=json.loads((pub.OUT/'REMOTE_MANIFEST.json').read_text())
names=[f'csv/epoch_{e:04d}/affinities.csv' for e in (7,60)]
with ThreadPoolExecutor(max_workers=2) as pool:
    for p in pool.map(pub.fetch,[(n,m['files'][n]) for n in names]):print('Endpoint table verified:',p.name,str(p.parent.name),flush=True)
