"""Preserve the executed standalone recovery and inspect its exact payload/output."""
import ast,base64,hashlib,json,re,sys
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];path=ROOT/'notebooks/phase3_NB13A_TREAD_TWO_CONTEXT_Existing_Session_Recovery.ipynb'
body=path.read_bytes();identity=hashlib.sha256(body).hexdigest();out=ROOT/'tread_two/nb13_completion'/identity[:12];out.mkdir(parents=True,exist_ok=True)
if (out/'executed.ipynb').exists():assert (out/'executed.ipynb').read_bytes()==body
else:(out/'executed.ipynb').write_bytes(body)
def safe(value):
    text=''.join(value) if isinstance(value,list) else str(value)
    text=re.sub(r'\x1b\[[0-9;]*m','',text);text=re.sub(r'hf_[A-Za-z0-9_]+','[REDACTED]',text)
    return re.sub(r"b64decode\('[A-Za-z0-9+/=]{200,}'\)","b64decode('[EMBEDDED PAYLOAD OMITTED]')",text)
cells=[];payload=None
for i,cell in enumerate(json.loads(body)['cells']):
    if cell['cell_type']!='code':continue
    outputs=[]
    for o in cell.get('outputs',[]):
        if o['output_type']=='stream':outputs.append(dict(type='stream',text=safe(o.get('text',''))))
        elif o['output_type']=='error':outputs.append(dict(type='error',name=o.get('ename'),value=safe(o.get('evalue','')),traceback=safe(o.get('traceback',[]))))
    cells.append(dict(index=i,execution_count=cell.get('execution_count'),outputs=outputs))
    for node in ast.walk(ast.parse(''.join(cell['source']))):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='b64decode' and node.args and isinstance(node.args[0],ast.Constant):
            assert payload is None;payload={n:base64.b64decode(b) for n,b in json.loads(base64.b64decode(node.args[0].value)).items()}
assert payload and len(payload)==35
expected={}
for name,b in payload.items():
    if name.startswith('source/'):p=ROOT/'tread_two/context_pilot'/name.removeprefix('source/')
    elif name.endswith('launcher_r2.py'):p=ROOT/'analysis/nb13_recovery_launcher.py'
    else:p=ROOT/'tread_two/context_validation/nb13_status/e0c714be2aba/remote/CONTRACT.json'
    assert b==p.read_bytes(),('Payload drift',name)
    expected[name]=hashlib.sha256(b).hexdigest()
result=dict(notebook_sha256=identity,bytes=len(body),embedded_files=len(payload),all_payloads_match=True,payload_SHA256=expected,cells=cells)
(out/'INSPECTION.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
(ROOT/'tread_two/nb13_completion/LATEST.json').write_text(json.dumps(dict(path=str(out.resolve()),sha256=identity),indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in ('payload_SHA256','cells')},indent=2))
print(json.dumps([dict(index=c['index'],execution_count=c['execution_count'],errors=[o for o in c['outputs'] if o['type']=='error'],last_stream='\n'.join(o['text'] for o in c['outputs'] if o['type']=='stream')[-11000:]) for c in cells],indent=2))
