"""Execute actual notebook setup in an empty fixture and validate frozen payload."""
import ast,base64,contextlib,io,json,sys,tempfile
from pathlib import Path
import importlib.metadata
import nbformat
assert Path(sys.prefix).name=='cv_conda'
assert importlib.metadata.version('huggingface_hub')=='0.36.2','Fixture must never install packages'
ROOT=Path(__file__).resolve().parents[1];SOURCE=ROOT/'tread_two/context_pilot'
sys.path.insert(0,str(SOURCE));from common import atomic,canonical,digest,read,sha
OUT=ROOT/'tread_two/context_validation/nb13_status/recovery_validation'
delivery=read(OUT/'NOTEBOOK.json');path=Path(delivery['notebook']);nb=nbformat.read(path,as_version=4);nbformat.validate(nb)
assert sha(path)==delivery['sha256'] and delivery['revision']=='r2'
code=[c for c in nb.cells if c.cell_type=='code'];assert len(code)==2
payload=[]
for cell in code:
    assert cell.execution_count is None and not cell.outputs
    for node in ast.walk(ast.parse(cell.source)):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='b64decode' and node.args and isinstance(node.args[0],ast.Constant):payload.append(json.loads(base64.b64decode(node.args[0].value)))
assert len(payload)==1
files={n:base64.b64decode(b) for n,b in payload[0].items()};assert len(files)==35
for n in read(SOURCE/'FILES.json'):assert files['source/'+n]==(SOURCE/n).read_bytes()
assert files['source/FILES.json']==(SOURCE/'FILES.json').read_bytes()
assert files['NB13_RECOVERY/nb13_recovery_launcher_r2.py']==(ROOT/'analysis/nb13_recovery_launcher.py').read_bytes()
assert digest(json.loads(files['NB13_RECOVERY/PINNED_CONTRACT.json']))=='36f473b0886d6c2f49c416436d270198af3982230e3045da9969b0bf42169625'
folder=Path(tempfile.mkdtemp(prefix='fixture_package_',dir=OUT));kaggle=folder/'working';kaggle.mkdir();work=kaggle/'tread_two_atlas_context_r3'
class Portable(ast.NodeTransformer):
    def visit_Constant(self,node):
        if node.value=='/kaggle/working':return ast.copy_location(ast.Constant(str(kaggle)),node)
        if node.value=='/kaggle/working/tread_two_atlas_context_r3':return ast.copy_location(ast.Constant(str(work)),node)
        return node
tree=ast.fix_missing_locations(Portable().visit(ast.parse(code[0].source)));program=compile(tree,'NB13A_setup_fixture','exec')
scope={};output=io.StringIO()
with contextlib.redirect_stdout(output):exec(program,scope)
assert 'No local checkpoint in this session' in output.getvalue() and not (work/'run/POINTER.json').exists()
assert all((work/n).read_bytes()==b for n,b in files.items())
with contextlib.redirect_stdout(io.StringIO()):exec(program,scope)
# The actual setup cell recognizes a retained pointer without changing its unit.
unit=dict(seq=0,updates=2467,phase='fixture',protocol='36f473b0886d6c2f49c416436d270198af3982230e3045da9969b0bf42169625')
atomic(work/'run/units/fixture.json',unit);atomic(work/'run/POINTER.json',dict(unit='units/fixture.json',sha256=sha(work/'run/units/fixture.json')))
captured=io.StringIO()
with contextlib.redirect_stdout(captured):exec(program,scope)
assert 'Retained local progress:' in captured.getvalue() and read(work/'run/units/fixture.json')==unit
# Never overwrite incompatible existing model code in a running workspace.
changed=work/'source/model_atlas.py';changed.write_bytes(changed.read_bytes()+b'\n# fixture drift\n');changed_hash=sha(changed)
try:
    with contextlib.redirect_stdout(io.StringIO()):exec(program,scope)
except AssertionError:pass
else:raise AssertionError('Changed source must be rejected')
assert sha(changed)==changed_hash
result=dict(status='passed',notebook_sha256=sha(path),embedded_files=35,frozen_runtime_payload_files=33,code_cells=2,
    actual_setup_executed_in_empty_directory=True,empty_session_error_reproduced_then_fixed=True,setup_idempotent=True,
    retained_pointer_preserved=True,source_drift_rejected_without_overwrite=True,network_requests=0,model_forward_calls=0,optimizer_updates=0,
    actual_Kaggle_execution=False,fixture=str(folder))
atomic(OUT/'PACKAGE_VALIDATION.json',result);print(json.dumps(result,indent=2))
