"""Preserve the supplied NB09 execution and summarize safe saved outputs."""
import ast
import base64
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1]
path=ROOT/'notebooks/phase3_NB09_TREAD_TWO_ATLAS_Completed_Review.ipynb'
body=path.read_bytes();identity=hashlib.sha256(body).hexdigest()
out=ROOT/'tread_two/nb09_completion'/identity[:12];out.mkdir(parents=True,exist_ok=True)
archive=out/'executed.ipynb'
if archive.exists(): assert archive.read_bytes()==body
else: archive.write_bytes(body)
for name in ['NOTEBOOK.json','VALIDATION.json','HANDOFF.json','REVIEW_DELIVERY.json']:
    target=out/('DELIVERY_'+name)
    if not target.exists(): shutil.copy2(ROOT/'tread_two/review_validation'/name,target)
def safe(text):
    text=''.join(text) if isinstance(text,list) else str(text)
    text=re.sub(r'\x1b\[[0-9;]*m','',text)
    text=re.sub(r'hf_[A-Za-z0-9_]+','[HF_TOKEN_REDACTED]',text)
    text=re.sub(r'(?i)Bearer\s+[^\s\"\']+','Bearer [REDACTED]',text)
    return text
nb=json.loads(body);cells=[];payload=None
for i,cell in enumerate(nb['cells']):
    if cell['cell_type']!='code': continue
    outputs=[]
    for o in cell.get('outputs',[]):
        if o['output_type']=='stream': outputs.append(dict(type='stream',text=safe(o.get('text',''))[-22000:]))
        elif o['output_type']=='error': outputs.append(dict(type='error',ename=o.get('ename'),evalue=safe(o.get('evalue','')),traceback=safe(o.get('traceback',[]))))
    cells.append(dict(index=i,execution_count=cell.get('execution_count'),outputs=outputs))
    for node in ast.walk(ast.parse(''.join(cell['source']))):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='b64decode' and node.args and isinstance(node.args[0],ast.Constant):
            payload=json.loads(base64.b64decode(node.args[0].value))
delivery=json.loads((out/'DELIVERY_NOTEBOOK.json').read_text())
assert payload and {n:hashlib.sha256(base64.b64decode(v)).hexdigest() for n,v in payload.items()}==delivery['source_hashes']
result=dict(executed_notebook_sha256=identity,bytes=len(body),embedded_source_files=len(payload),embedded_source_matches_delivery=True,cells=cells)
(out/'INSPECTION.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
(ROOT/'tread_two/nb09_completion/LATEST.json').write_text(json.dumps(dict(path=str(out.resolve()),notebook_sha256=identity),indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,indent=2))
