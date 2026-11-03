"""Self-contained CPU NB15; exact existing-data inputs, isolated HF SDK for writes."""
import ast,base64,hashlib,json,sys
from pathlib import Path
import nbformat
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];GATE=ROOT/'tread_two/readiness_gate';SOURCE=ROOT/'tread_two/context_pilot'
done=Path(json.loads((ROOT/'tread_two/nb14_completion/LATEST.json').read_text())['path']);assert json.loads((done/'VERIFICATION.json').read_text())['status']=='passed'
assert json.loads((ROOT/'tread_two/readiness_report/ANALYSIS.json').read_text())['status']=='passed'
inventory=json.loads((SOURCE/'FILES.json').read_text());files={'frozen_source/'+n:(SOURCE/n).read_bytes() for n in [*inventory,'FILES.json']}
assert all(hashlib.sha256(files['frozen_source/'+n]).hexdigest()==m['sha256'] for n,m in inventory.items())
files.update({'gate/'+p.name:p.read_bytes() for p in GATE.iterdir() if p.suffix in ('.py','.json')})
files['NB14_VERIFICATION.json']=(done/'VERIFICATION.json').read_bytes();files['NB07_VERIFICATION.json']=(ROOT/'data_gate_validation/nb07_publication/VERIFICATION.json').read_bytes()
payload=base64.b64encode(json.dumps({n:base64.b64encode(b).decode() for n,b in files.items()},separators=(',',':')).encode()).decode()
cells=[]
def md(text):cells.append(nbformat.v4.new_markdown_cell(text))
def py(text):ast.parse(text);cells.append(nbformat.v4.new_code_cell(text))
md('''# Phase 3 · NB15 · TREAD-TWO existing-data readiness and evidence passport

**CPU · existing saved records · zero model forwards / zero fitting.**

NB14 completed and its 80 direct files / 52 CSVs were independently HF-verified.
The context classifier remains unpromoted. This notebook examines whether the
existing training grouping can support a complete group-held-out comparison,
separates spatial/global correction parts on the same saved training views,
and publishes a machine-readable capability record plus direct CSV evidence.
It does not launch a fourth full classifier fit, alter checkpoints or install
anything in either workstation.

## Kaggle settings and run order

1. **CPU / None**, **Internet ON**, no image/dataset attachment.
2. Enable Kaggle Secret **HF_TOKEN** with write access to
   `Shanmuk4622/tyre-wear-study` for publication.
3. Keep at least **2 GiB free output** and about **4 GB RAM**. Inputs total
   about70MB; full original model/data mirrors are not downloaded.
4. **Run All**, preserve the executed notebook and final gate receipt.

The numeric gate already runs locally: 386 training images, six conservative
training components, two conditional folds missing a whole fit class, and
188,160 algebraic saved-view rows. Scientific readiness is **false** while
the analysis integrity status is **passed**. An identical rerun does not
change that support limit. More rows/views do not create independent tyres.
''')
py('''import base64,hashlib,json,os,shutil,signal,subprocess,sys,time
from pathlib import Path
from filelock import FileLock
assert Path('/kaggle/working').exists(),'Run NB15 on Kaggle CPU.'
WORK=Path('/kaggle/working/tread_two_nb15_readiness');WORK.mkdir(parents=True,exist_ok=True)
assert shutil.disk_usage(WORK).free>=2*1024**3,'Keep2GiB free output for evidence/report/publication staging.'
payload=json.loads(base64.b64decode('__PAYLOAD__'))
with FileLock(str(WORK/'NOTEBOOK.lock'),timeout=1):
    for name,value in payload.items():
        path=(WORK/name).resolve();assert path.is_relative_to(WORK.resolve()) and '..' not in Path(name).parts
        body=base64.b64decode(value)
        if path.exists():assert path.read_bytes()==body,'Source mismatch: preserve output and inspect compatibility; do not overwrite.'
        else:path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(body)
CACHE=WORK/'cache';OUT=WORK/'report';SOURCE=WORK/'frozen_source';SDK=WORK/'publication_sdk'
def run(script,arguments,token=None):
    env=dict(os.environ);env.pop('HF_TOKEN',None);env['HF_HUB_DISABLE_IMPLICIT_TOKEN']='1'
    if token is not None:
        env['HF_TOKEN']=token
        env['PYTHONPATH']=str(SDK)+(os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    with FileLock(str(WORK/'NOTEBOOK.lock'),timeout=1):
        try:child=subprocess.Popen([sys.executable,'-B','-u',str(WORK/'gate'/script),*arguments],env=env,start_new_session=True)
        finally:env.pop('HF_TOKEN',None)
        try:
            while True:
                try:result=child.wait(timeout=15);break
                except subprocess.TimeoutExpired:pass
        except KeyboardInterrupt:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGINT)
                for i in range(180):
                    try:child.wait(timeout=5);break
                    except subprocess.TimeoutExpired:
                        if i%6==0:print('Waiting for atomic boundary or complete HF publication; keep working files.',flush=True)
            raise
        assert result==0,'Keep errors/cache/report and rerun unchanged. No training is repeated.'
print('Pinned existing-data gate ready; no training or image/model inference.')
'''.replace('__PAYLOAD__',payload))
md('''## 1 · Existing training support and algebraic correction utility

Pins NB14 review and NB07 identity metadata. Exact source hashes/grouping agree.
Only historical training rows enter the numeric analysis. Six conditional folds
withhold entire conservative components and count remaining class/reference
support; **no fold model is trained**. The unresolved mid parent co-assignment
does not assert that three confirmed original tyres are identical.

The four saved-view variants are frozen prior, bias only, spatial only and
full context. They are sums of saved logits/corrections at changing online
states, not independently trained ablations or fixed final-model cohort tests.
All original selected/final checkpoint identities remain unchanged. No blend,
threshold, temperature or checkpoint is selected here.
''')
py("run('run_gate.py',['--cache',str(CACHE),'--out',str(OUT)])")
md('''## 2 · Publish direct CSV and the evidence passport

The tested HF transport is preserved unchanged. Its SDK0.36.2 is installed into
a **private publication folder**, visible only to the publication subprocess;
this avoids replacing the newer SDK used by Kaggle's unrelated libraries.
No Torch/CUDA package is installed or changed. Existing Kaggle dependencies
are reused; only the pinned HF SDK wheel is added to this private folder.

Destination: `phase3/tread-two/readiness-gate/v1/<contract>`. Direct CSV, JSON,
PNG, HTML and source files; no ZIP/NPZ/database. Publication is additive and
occurs after this finite analysis completes. Shared request budgets, server
backoff, immutable manifests, ownership and identical-rerun skipping apply.
Catchable Stop allows the complete report flush; forced termination cannot
upload unsent files. Preserve working output and rerun unchanged.
''')
py('''if not (SDK/'READY.json').exists():
    SDK.mkdir(parents=True,exist_ok=True)
    subprocess.check_call([sys.executable,'-m','pip','install','--disable-pip-version-check','--no-deps','--upgrade','--target',str(SDK),'huggingface_hub==0.36.2'])
    env=dict(os.environ);env.pop('HF_TOKEN',None);env['PYTHONPATH']=str(SDK)+(os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    subprocess.check_call([sys.executable,'-c',"import huggingface_hub;assert huggingface_hub.__version__=='0.36.2'"],env=env)
    (SDK/'READY.json').write_text(json.dumps({'version':'0.36.2','private_publication_only':True}))
from kaggle_secrets import UserSecretsClient
token=UserSecretsClient().get_secret('HF_TOKEN')
try:run('publish_gate.py',['--out',str(OUT),'--source',str(SOURCE),'--work',str(WORK)],token=token)
finally:token=None
''')
md('''## 3 · Inspect and retain

Read the support diagram, correction parts, interactive passport and capability
JSON. This gate documents an existing-data limitation; it does not require
new photography or prove every possible model impossible. Low/mid group-held-out
comparison lacks remaining class support under the current constraints.
Classification stays research-only; frozen geometry/evidence remain available
within their documented scope. Physical depth and safety are unavailable.

Keep the executed notebook and final `GATE_PUBLICATION.json` for independent
HF verification. Do not rerun NB13/NB13A/NB14 or start another full fit to change
a metadata readiness flag. Both workstation applications remain unchanged.
''')
py("from IPython.display import display,FileLink,Image\nprint((WORK/'GATE_PUBLICATION.json').read_text())\nprint((OUT/'SUMMARY.json').read_text())\nprint((OUT/'CAPABILITIES.json').read_text())\ndisplay(Image(filename=str(OUT/'figures/independent_support.png')))\ndisplay(FileLink(str(OUT/'Evidence_Passport.html')))\ndisplay(FileLink(str(OUT/'CSV_INDEX.md')))")
for i,c in enumerate(cells):c.id=f'tread-two-nb15-{i:02d}'
nb=nbformat.v4.new_notebook(cells=cells,metadata=dict(kernelspec=dict(name='python3',display_name='Python3',language='python'),kaggle=dict(isInternetEnabled=True,isGpuEnabled=False)))
nbformat.validate(nb);path=ROOT/'notebooks/phase3_NB15_TREAD_TWO_Existing_Data_Readiness_And_Evidence_Passport.ipynb'
if path.exists():assert not any(c.get('execution_count') or c.get('outputs') for c in nbformat.read(path,as_version=4).cells if c.cell_type=='code'),'Preserve executed NB15'
nbformat.write(nb,path);out=ROOT/'tread_two/readiness_validation';out.mkdir(exist_ok=True)
receipt=dict(status='packaged_not_executed_on_Kaggle',notebook=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size,embedded_files=len(files),code_cells=4,sources={n:hashlib.sha256(b).hexdigest() for n,b in files.items()})
(out/'NOTEBOOK.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:v for k,v in receipt.items() if k!='sources'},indent=2))
