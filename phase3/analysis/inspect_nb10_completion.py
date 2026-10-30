"""Archive the user-executed NB10 and extract safe execution/publication facts."""
import ast,base64,hashlib,json,re,shutil,sys
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1]
path=ROOT/'notebooks/phase3_NB10_TREAD_TWO_Train_Only_Appearance_And_Reference_Gate.ipynb'
body=path.read_bytes();identity=hashlib.sha256(body).hexdigest()
out=ROOT/'tread_two/nb10_completion'/identity[:12];out.mkdir(parents=True,exist_ok=True)
archive=out/'executed.ipynb'
if archive.exists(): assert archive.read_bytes()==body
else: archive.write_bytes(body)
delivery=ROOT/'tread_two/gate_validation/NOTEBOOK.json'
if not (out/'DELIVERY_NOTEBOOK.json').exists(): shutil.copy2(delivery,out/'DELIVERY_NOTEBOOK.json')
def safe(value):
    text=''.join(value) if isinstance(value,list) else str(value)
    text=re.sub(r'\x1b\[[0-9;]*m','',text)
    return re.sub(r'hf_[A-Za-z0-9_]+','[HF_TOKEN_REDACTED]',text)
nb=json.loads(body);cells=[];payload=None
for i,cell in enumerate(nb['cells']):
    if cell['cell_type']!='code':continue
    outputs=[]
    for item in cell.get('outputs',[]):
        if item['output_type']=='stream':outputs.append(dict(type='stream',text=safe(item.get('text',''))[-9000:]))
        elif item['output_type']=='error':outputs.append(dict(type='error',name=item.get('ename'),value=safe(item.get('evalue','')),traceback=safe(item.get('traceback',[]))))
    cells.append(dict(index=i,execution_count=cell.get('execution_count'),outputs=outputs))
    for node in ast.walk(ast.parse(''.join(cell['source']))):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='b64decode' and node.args and isinstance(node.args[0],ast.Constant):payload=json.loads(base64.b64decode(node.args[0].value))
expected=json.loads((out/'DELIVERY_NOTEBOOK.json').read_text())
assert payload and len(payload)==36
assert {n:hashlib.sha256(base64.b64decode(v)).hexdigest() for n,v in payload.items()}==expected['sources']
result=dict(executed_notebook_sha256=identity,bytes=len(body),embedded_files=36,embedded_source_matches_delivery=True,cells=cells)
(out/'INSPECTION.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
(ROOT/'tread_two/nb10_completion/LATEST.json').write_text(json.dumps(dict(path=str(out.resolve()),sha256=identity),indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,indent=2))
