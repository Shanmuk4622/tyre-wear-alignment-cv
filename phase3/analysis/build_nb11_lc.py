"""Self-contained one-seed ATLAS-LC r2 notebook, direct CSV/native recovery."""
import ast,base64,hashlib,json,sys
from pathlib import Path
import nbformat
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];SOURCE=ROOT/'tread_two/lc_pilot'
catalog=json.loads((SOURCE/'FILES.json').read_text())
files={n:(SOURCE/n).read_bytes() for n in catalog};files['FILES.json']=(SOURCE/'FILES.json').read_bytes()
for n,m in catalog.items():assert hashlib.sha256(files[n]).hexdigest()==m['sha256'] and len(files[n])==m['bytes']
encoded=base64.b64encode(json.dumps({n:base64.b64encode(b).decode() for n,b in files.items()},separators=(',',':')).encode()).decode()
cells=[]
def md(s):cells.append(nbformat.v4.new_markdown_cell(s))
def py(s):ast.parse(s);cells.append(nbformat.v4.new_code_cell(s))
md('''# Phase 3 · NB11 · TREAD-TWO / ATLAS-LC r2

**One new measured candidate, seed1 · 60 epochs · 2,940 updates · 386 existing training images.**

NB08–NB10 are completed evidence, preserved unchanged. This is a separately
versioned hypothesis: four local-contrast texture channels, 18 distinct
training-photo reference patches, source-SHA self exclusion, gentler group
resampling and two-view appearance consistency. Its accuracy is **unmeasured**.
The frozen TREAD-ONE anchor still supplies segmentation, mask-derived boxes and
points; signed reference evidence exactly accounts for classification logits.
There are 81,274 trainable / 1,453,433 total parameters. No workstation changes.

## Before Run All

1. **GPU T4 ×2**, Internet ON.
2. Attach one **expanded** `shanmuk4622/tire-dataset-prepared-phase2` Kaggle
   dataset directly; keep it read-only. No new photos or ZIP dataset needed.
3. Enable **HF_TOKEN** in Kaggle Secrets, write access to
   `Shanmuk4622/tyre-wear-study`. Never paste a token into this notebook.
4. Start with at least **12 GiB free** in working output. The automatic measured
   storage projection must remain below16 GB within the20 GB output budget.
   Extra scratch capacity is never assumed. All scientific CSVs remain retained.
5. **Run All in order**. The actual dual-T4/live-HF recovery preflight must pass
   before measured training begins. Keep the executed notebook for verification.

GPU0 fits the single head; GPU1 evaluates an immutable copy at barriers. This
is FP32, dense microbatch1, no DDP or extra measured seed. Engineering checks
make four fixture updates and use two validation examples; they are not a second
measured study. Torch/CUDA are not installed or upgraded.
''')
py('''import base64,hashlib,importlib.metadata,json,os,signal,subprocess,sys,time
from pathlib import Path
from filelock import FileLock
assert Path('/kaggle/working').exists(),'Run NB11 on Kaggle.'
WORK=Path('/kaggle/working/tread_two_atlas_lc_r2');SOURCE=WORK/'source'
WORK.mkdir(parents=True,exist_ok=True);SOURCE.mkdir(exist_ok=True)
os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN']='1'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
try: version=importlib.metadata.version('huggingface_hub')
except importlib.metadata.PackageNotFoundError: version=None
if version!='0.36.2': subprocess.check_call([sys.executable,'-m','pip','install','--disable-pip-version-check','huggingface_hub==0.36.2'])
payload=json.loads(base64.b64decode('__PAYLOAD__'))
with FileLock(str(WORK/'NOTEBOOK.lock'),timeout=1):
    for name,value in payload.items():
        path=(SOURCE/name).resolve();assert path.is_relative_to(SOURCE.resolve()) and '..' not in Path(name).parts
        body=base64.b64decode(value)
        if path.exists(): assert path.read_bytes()==body,'Preserve old run; source compatibility review needed.'
        else: path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(body)
def launch(stage):
    from kaggle_secrets import UserSecretsClient
    environment=dict(os.environ);environment.pop('HF_TOKEN',None)
    environment['HF_TOKEN']=UserSecretsClient().get_secret('HF_TOKEN')
    with FileLock(str(WORK/'NOTEBOOK.lock'),timeout=1):
        try:
            child=subprocess.Popen([sys.executable,'-B','-u',str(SOURCE/'coordinator.py'),stage,'--work',str(WORK)],env=environment,start_new_session=True)
        finally: environment.pop('HF_TOKEN',None)
        try:
            while True:
                try: result=child.wait(timeout=15);break
                except subprocess.TimeoutExpired: pass
        except KeyboardInterrupt:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGINT)
                for i in range(180):
                    try: child.wait(timeout=5);break
                    except subprocess.TimeoutExpired:
                        if i%6==0: print('Waiting for atomic update and immediate HF flush attempt. Keep this session/files.',flush=True)
            raise
        assert result==0,'Stopped/paused. Preserve run/console.log and native/CSV files; rerun unchanged to restore the last verified completed update.'
print('Exact ATLAS-LC r2 source and pinned NB10 references ready. No measured fit started.')
'''.replace('__PAYLOAD__',encoded))
md('''## 1 · Verify data, devices and immutable protocol

This verifies every prepared-dataset checksum, role overlay, full bank-source
SHA and frozen anchor SHA. The reference patches are pinned from verified NB10,
not chosen by validation/test scores. All386 train photos/frames can be queries.
An original reference photo has its own patch blocked by full source SHA before
any appearance changes; the other five references of that class remain, scaled
by6/5. No query label enters that rule. New environment/source means a different
contract; compatible existing progress is never silently overwritten.
''')
py("launch('prepare')")
md('''## 2 · Actual dual-T4 / live-HF recovery and storage gate

The short engineering fixture compares uninterrupted execution against pause,
abrupt termination and a pinned HF snapshot restored in a fresh process. It
checks all native model/optimizer/scheduler/scaler/RNG/sample-order state and
all24 CSV families, excluding only elapsed/resource fields from equality.
The measured raw CSV byte sizes drive a conservative full-run projection.
On rerun, a compatible completed remote proof is reused.
''')
py("launch('preflight')")
md('''## 3 · Fit/resume one measured candidate, then evaluate

Each draw has two deterministic brightness/contrast/gamma views in[.8,1.2].
The frozen dense anchor runs once; the shared encoder sees both LC atlases and
the same literal reference bank. Objective:

`0.5*(CE(A)+CE(B)) +0.1*JS(PA,PB) +0.05*sum(alpha*(1-cos(qA,qB)))`.

Class/domain draws stay uniform; within a domain, group probability is
proportional to sqrt(number of images), then image uniform. Replacement is
explicit. More draws/views are telemetry, not additional independent tyres.
Selection is maximum validation observed-class BA, minimum NLL, earliest epoch.
All60 epoch weights are retained; no early stopping or test-selected export.

53,342 planned raw observations:47,040 online training views,4,860 validation,
206 selected/final,1,236 fixed diagnostic observations. Direct CSV records keep
every prediction/logit/probability, patch/reference affinity and signed evidence,
sample factors/probability, reference permissions, pair/patch consistency losses,
gradient summaries and resource readings. Native state keeps exact recovery;
all selected/final image projections remain native with conservation checks.

Normal HF snapshots every30 minutes, plus major stage completion and catchable
Stop/error. Local atomic commits happen after each completed optimizer update.
Immediate stop-flush attempts respect HF server backoff, which can delay upload.
Forced kernel death cannot push unsent state. Never delete leases or run another
writer for the same contract. New immutable namespace:
`phase3/tread-two/atlas-lc-r2/fit/<contract>`; direct CSV/JSON/PT/NPY/source/PNG/HTML,
never ZIP/NPZ/database. HTTP five-minute budgets and separate hourly commit guard
are included natively in this r2; no v1 repair overlay is needed.

Six predeclared diagnostics at both endpoints: grayscale-both, brightness0.8/1.2,
gamma0.8/1.2, top-signed-patch occlusion. Grayscale invariance is built into the
representation, not evidence that learned features generalize. Physical groove
depth, safety and causal-wear truth remain unavailable. Historical test was
already inspected and has no mid targets; treat results as development evidence.
''')
py("launch('train')")
md('''## 4 · Inspect the retained evidence

If the previous cell exits normally after completion, stopping is expected.
Rerunning unchanged restores/verifies the completed state and skips fitting.
For an exception, preserve `run/console.log`, lifecycle/HTTP logs, last remote
revision and all local files. Do not change the model/bank/recipe or erase state
to bypass a mismatch. Save this executed NB11 and return it for an independent
HF audit before any model promotion or workstation work.
''')
py('''from IPython.display import display,FileLink,Image
import pandas as pd
receipt=json.loads((WORK/'LAST_REMOTE.json').read_text());print(json.dumps(receipt,indent=2))
latest=json.loads((WORK/'LATEST_SNAPSHOT.json').read_text());SNAPSHOT=Path(latest['path'])
print((SNAPSHOT/'report/REPORT.json').read_text())
display(Image(filename=str(SNAPSHOT/'report/training_curves.png')))
display(FileLink(str(SNAPSHOT/'report/ATLAS_Evidence_Explorer.html')))
print('Completed one measured ATLAS-LC candidate. Keep all published evidence and executed notebook.')
''')
for i,c in enumerate(cells):c.id=f'atlas-lc-nb11-{i:02d}'
nb=nbformat.v4.new_notebook(cells=cells,metadata=dict(kernelspec=dict(name='python3',display_name='Python 3',language='python'),kaggle=dict(isInternetEnabled=True,isGpuEnabled=True,datasetSources=['shanmuk4622/tire-dataset-prepared-phase2'])))
nbformat.validate(nb)
path=ROOT/'notebooks/phase3_NB11_TREAD_TWO_ATLAS_LC_Resumable_Pilot.ipynb'
if path.exists():assert not any(c.get('execution_count') or c.get('outputs') for c in nbformat.read(path,as_version=4).cells if c.cell_type=='code'),'Preserve user-executed notebook'
nbformat.write(nb,path)
receipt=dict(status='packaged_not_executed_on_Kaggle',notebook=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size,code_cells=sum(c.cell_type=='code' for c in cells),embedded_files=len(files),source_files=catalog,measured_fit_started=False)
(ROOT/'tread_two/lc_validation/NOTEBOOK.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in receipt.items() if k!='source_files'},indent=2))
