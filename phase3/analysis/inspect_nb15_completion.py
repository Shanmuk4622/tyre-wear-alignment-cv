"""Preserve executed NB15 and inspect its exact payload and safe outputs."""
import ast,base64,hashlib,json,re,shutil,sys
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];path=ROOT/'notebooks/phase3_NB15_TREAD_TWO_Existing_Data_Readiness_And_Evidence_Passport.ipynb'
body=path.read_bytes();identity=hashlib.sha256(body).hexdigest();out=ROOT/'tread_two/nb15_completion'/identity[:12];out.mkdir(parents=True,exist_ok=True)
archive=out/'executed.ipynb'
if archive.exists():assert archive.read_bytes()==body
else:archive.write_bytes(body)
if not (out/'DELIVERY_NOTEBOOK.json').exists():shutil.copy2(ROOT/'tread_two/readiness_validation/NOTEBOOK.json',out/'DELIVERY_NOTEBOOK.json')
def safe(value):
    text=''.join(value) if isinstance(value,list) else str(value)
    return re.sub(r'hf_[A-Za-z0-9_]+','[REDACTED]',re.sub(r'\x1b\[[0-9;]*m','',text))
cells=[];payload=None
for index,cell in enumerate(json.loads(body)['cells']):
    if cell['cell_type']!='code':continue
    outputs=[]
    for item in cell.get('outputs',[]):
        kind=item['output_type']
        if kind=='stream':outputs.append(dict(type=kind,text=safe(item.get('text',''))))
        elif kind=='error':outputs.append(dict(type=kind,name=item.get('ename'),value=safe(item.get('evalue','')),traceback=safe(item.get('traceback',[]))))
    cells.append(dict(index=index,execution_count=cell.get('execution_count'),outputs=outputs))
    for node in ast.walk(ast.parse(''.join(cell['source']))):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='b64decode' and node.args and isinstance(node.args[0],ast.Constant):
            assert payload is None;payload={n:base64.b64decode(b) for n,b in json.loads(base64.b64decode(node.args[0].value)).items()}
expected=json.loads((out/'DELIVERY_NOTEBOOK.json').read_text(encoding='utf-8'))
assert payload and len(payload)==40 and {n:hashlib.sha256(b).hexdigest() for n,b in payload.items()}==expected['sources']
result=dict(executed_notebook_sha256=identity,bytes=len(body),embedded_files=len(payload),all_payload_hashes_match=True,cells=cells)
(out/'INSPECTION.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
(ROOT/'tread_two/nb15_completion/LATEST.json').write_text(json.dumps(dict(path=str(out.resolve()),sha256=identity),indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(sha256=identity,archive=str(out),embedded_files=len(payload),cells=[dict(index=c['index'],execution_count=c['execution_count'],errors=[o for o in c['outputs'] if o['type']=='error'],last_stream='\n'.join(o['text'] for o in c['outputs'] if o['type']=='stream')[-4500:]) for c in cells]),indent=2))
