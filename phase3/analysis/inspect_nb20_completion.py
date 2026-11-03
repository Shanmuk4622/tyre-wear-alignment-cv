"""Archive supplied NB20 unchanged; inspect data and source identity without execution."""
import ast,base64,hashlib,json,re,shutil,sys
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];path=ROOT/'notebooks/phase3_NB20_TREAD_TWO_GEO_All_In_One_Postrun.ipynb'
body=path.read_bytes();identity=hashlib.sha256(body).hexdigest();out=ROOT/'tread_two/nb20_completion'/identity[:12];out.mkdir(parents=True,exist_ok=True)
archive=out/'executed.ipynb'
if archive.exists():assert archive.read_bytes()==body
else:archive.write_bytes(body)
if not (out/'DELIVERY_NOTEBOOKS.json').exists():shutil.copyfile(ROOT/'tread_two/geo_postrun_validation/NOTEBOOKS.json',out/'DELIVERY_NOTEBOOKS.json')
def safe(value):
    value=''.join(value) if isinstance(value,list) else str(value)
    return re.sub(r'hf_[A-Za-z0-9_]+','[REDACTED]',re.sub(r'\x1b\[[0-9;]*m','',value))
cells=[];payload=None
for index,cell in enumerate(json.loads(body)['cells']):
    if cell['cell_type']!='code':continue
    outputs=[]
    for item in cell.get('outputs',[]):
        kind=item['output_type']
        if kind=='stream':outputs.append(dict(type=kind,text=safe(item.get('text',''))))
        elif kind=='error':outputs.append(dict(type=kind,name=item.get('ename'),value=safe(item.get('evalue','')),traceback=safe(item.get('traceback',[]))))
        else:outputs.append(dict(type=kind,mime_types=list(item.get('data',{}))))
    cells.append(dict(index=index,execution_count=cell.get('execution_count'),outputs=outputs))
    for node in ast.walk(ast.parse(''.join(cell['source']))):
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PAYLOAD' for t in node.targets):assert payload is None;payload=ast.literal_eval(node.value)
source=ROOT/'tread_two/geo_postrun';assert payload and len(payload)==19
assert all(hashlib.sha256(base64.b64decode(v['base64'],validate=True)).hexdigest()==v['sha256']==hashlib.sha256((source/n).read_bytes()).hexdigest() for n,v in payload.items())
result=dict(executed_notebook_sha256=identity,bytes=len(body),embedded_files=len(payload),all_payload_hashes_match=True,cells=cells)
(out/'INSPECTION.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
(ROOT/'tread_two/nb20_completion/LATEST.json').write_text(json.dumps(dict(path=str(out.resolve()),sha256=identity),indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(sha256=identity,bytes=len(body),archive=str(out),embedded_files=len(payload),cells=[dict(index=c['index'],execution_count=c['execution_count'],errors=[o for o in c['outputs'] if o['type']=='error'],last_stream='\n'.join(o['text'] for o in c['outputs'] if o['type']=='stream')[-6500:],rich_outputs=[o for o in c['outputs'] if o['type'] not in ('error','stream')]) for c in cells]),indent=2))
