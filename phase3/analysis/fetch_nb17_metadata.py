import json,sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
import nb17_public as client
m=client.manifest();priority=['RESULT.json','CONTRACT.json','POINTER.json','controls/G0_validation.json','controls/quality_median.json','reports/final_metrics.json','source/FILES.json']
with ThreadPoolExecutor(max_workers=6) as pool:list(pool.map(client.fetch,[(n,m['files'][n]) for n in priority if n in m['files']]))
root=client.OUT/'remote'
result=json.loads((root/'RESULT.json').read_text());pointer=json.loads((root/'POINTER.json').read_text())
compact={k:v for k,v in result.items() if k!='history'};compact['history_epochs']=len(result['history']);compact['last_validation']=result['history'][-1]
print(json.dumps(dict(manifest_files=len(m['files']),bytes=sum(v['bytes'] for v in m['files'].values()),suffix_counts={s:sum(n.endswith(s) for n in m['files']) for s in ('.csv','.npy','.pt','.raw','.html')},result=compact,pointer=pointer,final_metrics=json.loads((root/'reports/final_metrics.json').read_text())),indent=2))
