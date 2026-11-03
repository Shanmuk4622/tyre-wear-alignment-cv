"""Preserve supplied NB13A and extract bounded, credential-redacted diagnostics."""
import ast,base64,hashlib,json,re,sys
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1]
path=ROOT/'notebooks/phase3_NB13A_TREAD_TWO_CONTEXT_Existing_Session_Recovery.ipynb'
body=path.read_bytes();identity=hashlib.sha256(body).hexdigest()
out=ROOT/'tread_two/context_validation/nb13_status/nb13a_errors'/identity[:12];out.mkdir(parents=True,exist_ok=True)
archive=out/'executed.ipynb'
if archive.exists():assert archive.read_bytes()==body
else:archive.write_bytes(body)
def safe(value):
    text=''.join(value) if isinstance(value,list) else str(value)
    text=re.sub(r'hf_[A-Za-z0-9_]+','[REDACTED]',re.sub(r'\x1b\[[0-9;]*m','',text))
    return re.sub(r"b64decode\('[A-Za-z0-9+/=]{200,}'\)","b64decode('[EMBEDDED PAYLOAD OMITTED]')",text)
cells=[];embedded=[]
for i,c in enumerate(json.loads(body)['cells']):
    if c['cell_type']!='code':continue
    outputs=[]
    for o in c.get('outputs',[]):
        if o['output_type']=='stream':outputs.append(dict(type='stream',text=safe(o.get('text',''))[-18000:]))
        elif o['output_type']=='error':outputs.append(dict(type='error',name=o.get('ename'),value=safe(o.get('evalue','')),traceback=safe(o.get('traceback',[]))))
    cells.append(dict(index=i,execution_count=c.get('execution_count'),outputs=outputs))
    for node in ast.walk(ast.parse(''.join(c['source']))):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='b64decode' and node.args and isinstance(node.args[0],ast.Constant):embedded.append(hashlib.sha256(base64.b64decode(node.args[0].value)).hexdigest())
result=dict(notebook_sha256=identity,bytes=len(body),archive=str(out),embedded_launcher_sha256=embedded,cells=cells)
(out/'INSPECTION.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
(ROOT/'tread_two/context_validation/nb13_status/NB13A_ERROR_LATEST.json').write_text(json.dumps(dict(path=str(out.resolve()),sha256=identity),indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,indent=2))
