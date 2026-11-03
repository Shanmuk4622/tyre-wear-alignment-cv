"""Archive the user's executed NB16 unchanged; inspect bounded safe output and payloads."""
import ast,base64,hashlib,json,re,shutil,sys
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1]
path=ROOT/'notebooks/phase3_NB16_TREAD_TWO_GEO_Preparation_And_Contract_Gate.ipynb'
body=path.read_bytes();identity=hashlib.sha256(body).hexdigest()
out=ROOT/'tread_two/nb16_completion'/identity[:12];out.mkdir(parents=True,exist_ok=True)
archive=out/'executed.ipynb'
if archive.exists():assert archive.read_bytes()==body
else:archive.write_bytes(body)
if not (out/'DELIVERY_NOTEBOOK.json').exists():shutil.copy2(ROOT/'tread_two/geometry_first/validation/NOTEBOOK.json',out/'DELIVERY_NOTEBOOK.json')
def safe(value):
    text=''.join(value) if isinstance(value,list) else str(value)
    return re.sub(r'hf_[A-Za-z0-9_]+','[REDACTED]',re.sub(r'\x1b\[[0-9;]*m','',text))
nb=json.loads(body);cells=[];payload=None
for index,cell in enumerate(nb['cells']):
    if cell['cell_type']!='code':continue
    outputs=[]
    for item in cell.get('outputs',[]):
        if item['output_type']=='stream':outputs.append(dict(type='stream',text=safe(item.get('text',''))))
        elif item['output_type']=='error':outputs.append(dict(type='error',name=item.get('ename'),value=safe(item.get('evalue','')),traceback=safe(item.get('traceback',[]))))
    cells.append(dict(index=index,execution_count=cell.get('execution_count'),outputs=outputs))
    for node in ast.walk(ast.parse(''.join(cell['source']))):
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PAYLOAD' for t in node.targets):
            assert payload is None;payload=ast.literal_eval(node.value)
assert payload and len(payload)==27
source=ROOT/'tread_two/geometry_first/runtime'
assert all(hashlib.sha256(base64.b64decode(v['base64'],validate=True)).hexdigest()==v['sha256']==hashlib.sha256((source/n).read_bytes()).hexdigest() for n,v in payload.items())
result=dict(executed_notebook_sha256=identity,bytes=len(body),embedded_files=len(payload),all_payload_hashes_match=True,cells=cells)
(out/'INSPECTION.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
(ROOT/'tread_two/nb16_completion/LATEST.json').write_text(json.dumps(dict(path=str(out.resolve()),sha256=identity),indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(sha256=identity,archive=str(out),embedded_files=len(payload),cells=[dict(index=c['index'],execution_count=c['execution_count'],errors=[o for o in c['outputs'] if o['type']=='error'],last_stream='\n'.join(o['text'] for o in c['outputs'] if o['type']=='stream')[-6500:]) for c in cells]),indent=2))
