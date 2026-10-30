"""Capture current user-executed notebooks; never replace them with clean deliveries."""
import ast,base64,hashlib,json,sys
from pathlib import Path
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'tread_two/gate_validation'
OUT.mkdir(exist_ok=True);saved=OUT/'user_notebooks';saved.mkdir(exist_ok=True)
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
notebooks={}
for label,name in [('NB08','phase3_NB08_TREAD_TWO_ATLAS_Resumable_Pilot.ipynb'),('NB09','phase3_NB09_TREAD_TWO_ATLAS_Completed_Review.ipynb')]:
    path=ROOT/'notebooks'/name;identity=sha(path)
    target=saved/(label+'_'+identity[:12]+'.ipynb')
    if target.exists(): assert target.read_bytes()==path.read_bytes()
    else: target.write_bytes(path.read_bytes())
    nb=json.loads(path.read_text(encoding='utf-8'));payload=None
    for cell in nb['cells']:
        if cell['cell_type']!='code': continue
        for node in ast.walk(ast.parse(''.join(cell['source']))):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='b64decode' and node.args and isinstance(node.args[0],ast.Constant):
                payload=json.loads(base64.b64decode(node.args[0].value))
    assert payload
    if label=='NB08':
        assert len(payload)==25 and all(base64.b64decode(body)==(ROOT/'tread_two/pilot'/n).read_bytes() for n,body in payload.items())
    notebooks[label]=dict(path=str(path.resolve()),sha256=identity,archived=str(target.resolve()),code_cells_executed=[c.get('execution_count') for c in nb['cells'] if c['cell_type']=='code'],frozen_payload_matches=True)
receipt=OUT/'PRESERVATION.json'
if receipt.exists(): assert json.loads(receipt.read_text())['notebooks']==notebooks,'Notebook changed since gate baseline; preserve and inspect the new execution'
else: receipt.write_text(json.dumps(dict(notebooks=notebooks),indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(notebooks=notebooks),indent=2))
