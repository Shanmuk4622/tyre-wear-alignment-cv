"""Self-contained NB13 recovery: current state, attached output, or verified HF."""
import ast,base64,hashlib,json,sys
from pathlib import Path
import nbformat
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];SOURCE=ROOT/'tread_two/context_pilot'
inventory=json.loads((SOURCE/'FILES.json').read_text())
files={'source/'+n:(SOURCE/n).read_bytes() for n in inventory}
files['source/FILES.json']=(SOURCE/'FILES.json').read_bytes()
for n,m in inventory.items():assert hashlib.sha256(files['source/'+n]).hexdigest()==m['sha256']
launcher=ROOT/'analysis/nb13_recovery_launcher.py';launcher_sha=hashlib.sha256(launcher.read_bytes()).hexdigest()
files['NB13_RECOVERY/nb13_recovery_launcher_r2.py']=launcher.read_bytes()
latest=json.loads((ROOT/'tread_two/context_validation/NB13_STATUS_LATEST.json').read_text())
contract=(Path(latest['path'])/'remote/CONTRACT.json').read_bytes()
assert hashlib.sha256(contract).hexdigest()=='36f473b0886d6c2f49c416436d270198af3982230e3045da9969b0bf42169625'
files['NB13_RECOVERY/PINNED_CONTRACT.json']=contract
payload=base64.b64encode(json.dumps({n:base64.b64encode(b).decode() for n,b in files.items()},separators=(',',':')).encode()).decode()
cells=[]
def md(s):cells.append(nbformat.v4.new_markdown_cell(s))
def py(s):ast.parse(s);cells.append(nbformat.v4.new_code_cell(s))
md('''# NB13A · Recover and resume ATLAS-CONTEXT — revision 2

**31 October 2026. Fixed: missing local NB13 working files.** This notebook is
now self-contained and can run directly with **Run All**. It preserves the same
model, measured protocol and HF namespace; it never starts a fresh fit if a
matching recovery checkpoint is unavailable.

## Kaggle settings

1. Select **GPU T4 ×2** and **Internet ON**.
2. Attach the existing **expanded** `shanmuk4622/tire-dataset-prepared-phase2`
   dataset directly from Kaggle. Do not copy its images to working output.
3. Enable write-capable **HF_TOKEN** in Kaggle Secrets for
   `Shanmuk4622/tyre-wear-study`. Never paste a token in code/output.
4. Start with at least **12 GiB free** working output. Retain the 20 GB output
   budget and automatic 16 GB / free-space guards; extra scratch is not assumed.
5. **Run All**, in order. Do not run another writer for NB13 concurrently.

If the older NB13 session is still accessible, running these cells there keeps
its newer local checkpoint. You may also attach a saved original notebook output
containing the complete `run/` tree; compatible checkpoints are discovered and
copied read-only from that input. Otherwise this notebook restores the full
verified HF snapshot. At the latest independent check, HF held **update 1,294**;
the older notebook printed **2,467**. Unsent progress cannot be reconstructed
unless those older files or a saved output copy still exist.

The recovery reuses the original passed dual-T4/live-HF preflight, verifies the
same source/data/environment contract, restores optimizer/scheduler/scaler/RNG/
sample orders and CSV cursors, and resumes the next completed-update boundary.
It rejects environment or history drift instead of creating another experiment.
''')
py('''import base64,hashlib,importlib.metadata,json,os,subprocess,sys
from pathlib import Path
from filelock import FileLock
assert Path('/kaggle/working').exists(),'Run this notebook on Kaggle.'
WORK=Path('/kaggle/working/tread_two_atlas_context_r3');SOURCE=WORK/'source'
WORK.mkdir(parents=True,exist_ok=True)
os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN']='1'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
try:version=importlib.metadata.version('huggingface_hub')
except importlib.metadata.PackageNotFoundError:version=None
if version!='0.36.2':subprocess.check_call([sys.executable,'-m','pip','install','--disable-pip-version-check','huggingface_hub==0.36.2'])
payload=json.loads(base64.b64decode('__PAYLOAD__'))
with FileLock(str(WORK/'NOTEBOOK.lock'),timeout=1):
    for name,value in payload.items():
        path=(WORK/name).resolve();assert path.is_relative_to(WORK.resolve()) and '..' not in Path(name).parts
        body=base64.b64decode(value)
        if path.exists():assert path.read_bytes()==body,'Source or repair mismatch. Preserve existing files; do not replace model/runtime to bypass it.'
        else:path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(body)
def read(p):return json.loads(p.read_text())
pointer=WORK/'run/POINTER.json'
if pointer.exists():
    p=read(pointer);unit_path=(WORK/'run'/p['unit']).resolve()
    assert unit_path.is_relative_to((WORK/'run').resolve()),'Local unit path escapes this run; preserve it for review.'
    assert hashlib.sha256(unit_path.read_bytes()).hexdigest()==p['sha256'],'Local pointer integrity failed; preserve it.'
    unit=read(unit_path);print('Retained local progress:',json.dumps({k:unit[k] for k in ('seq','updates','phase')}))
else:print('No local checkpoint in this session. Recovery will look for attached saved outputs, otherwise restore verified HF progress. No fresh fit will be started.')
print('Frozen 32-file runtime plus inventory and pinned recovery bootstrap ready. No training has run in this cell.')
'''.replace('__PAYLOAD__',payload))
md('''## Restore verified progress, then resume the original run

The first stage verifies T4×2, the original environment, expanded Kaggle data
and pinned upstream weights. It reuses the HF-verified gate receipt rather than
repeating a measured model or engineering fit. It prefers a compatible local
or attached saved run, and requires exact history-prefix equality before using it.
Newer local progress is published before fitting continues. With no local run,
the complete HF checkpoint, raw CSVs, artifacts and ledger are downloaded and
verified; this can take time. Cached verified files are reused on retry.

Watch `[NB13 status]` every 15 seconds: operation, local update/stage and free
disk identify verification, download, publication and training. Native state and
all 27 scientific CSV families remain intact. Normal uploads occur every 30
minutes, at major stages and on catchable Stop/error; server backoff may delay
them. Forced kernel death cannot upload unsent progress.

Keep `run/console.log`, `NB13_RECOVERY/STATUS.json`, `EVENTS.jsonl`, and
`THREAD_STACKS.txt` if another error occurs. Environment mismatch gets a separate
JSON report. Do not erase a lease, checkpoint or partial run to bypass a check.
Actual Kaggle execution of this revised recovery remains pending. Save this
executed notebook afterward for an independent HF audit.
''')
py('''import signal,time
from kaggle_secrets import UserSecretsClient
LAUNCHER=WORK/'NB13_RECOVERY/nb13_recovery_launcher_r2.py'
assert hashlib.sha256(LAUNCHER.read_bytes()).hexdigest()=='__SHA__'
with FileLock(str(WORK/'NOTEBOOK.lock'),timeout=1):
    environment=dict(os.environ);environment.pop('HF_TOKEN',None)
    environment['HF_TOKEN']=UserSecretsClient().get_secret('HF_TOKEN')
    try:child=subprocess.Popen([sys.executable,'-B','-u',str(LAUNCHER),'--source',str(SOURCE),'--work',str(WORK),'--contract',str(WORK/'NB13_RECOVERY/PINNED_CONTRACT.json')],env=environment,start_new_session=True)
    finally:environment.pop('HF_TOKEN',None)
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
                    if i%6==0:print('Waiting for atomic progress / immediate HF flush attempt. Keep session files.',flush=True)
        raise
    assert result==0,'Preserve NB13_RECOVERY diagnostics and run/console.log. No different model/experiment was substituted.'
print('Original NB13 recovery/resume finished. Save this executed notebook for HF verification.')
if (WORK/'LAST_REMOTE.json').exists():print(json.dumps(read(WORK/'LAST_REMOTE.json'),indent=2))
'''.replace('__SHA__',launcher_sha))
nb=nbformat.v4.new_notebook(cells=cells,metadata=dict(kernelspec=dict(name='python3',display_name='Python 3',language='python'),kaggle=dict(isInternetEnabled=True,isGpuEnabled=True,datasetSources=['shanmuk4622/tire-dataset-prepared-phase2'])))
nbformat.validate(nb)
path=ROOT/'notebooks/phase3_NB13A_TREAD_TWO_CONTEXT_Existing_Session_Recovery.ipynb'
if path.exists() and any(c.get('execution_count') or c.get('outputs') for c in nbformat.read(path,as_version=4).cells if c.cell_type=='code'):
    latest_error=json.loads((ROOT/'tread_two/context_validation/nb13_status/NB13A_ERROR_LATEST.json').read_text())
    archived=Path(latest_error['path'])/'executed.ipynb'
    assert path.read_bytes()==archived.read_bytes(),'Archive supplied execution before making the requested repair'
nbformat.write(nb,path)
out=ROOT/'tread_two/context_validation/nb13_status/recovery_validation'
receipt=dict(status='packaged_not_executed_on_Kaggle',revision='r2',notebook=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),launcher_sha256=launcher_sha,
    code_cells=2,embedded_files=len(files),source_runtime_changed=False,existing_session_required=False,remote_checkpoint_required=True,
    frozen_source_inventory_sha256=hashlib.sha256(files['source/FILES.json']).hexdigest())
(out/'NOTEBOOK.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8');print(json.dumps(receipt,indent=2))
