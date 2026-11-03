"""Diagnostic/recovery notebook for the retained existing NB13 Kaggle session."""
import ast,base64,hashlib,json,sys
from pathlib import Path
import nbformat
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];launcher=ROOT/'analysis/nb13_recovery_launcher.py'
payload=base64.b64encode(launcher.read_bytes()).decode();sha=hashlib.sha256(launcher.read_bytes()).hexdigest()
cells=[]
def md(s):cells.append(nbformat.v4.new_markdown_cell(s))
def py(s):ast.parse(s);cells.append(nbformat.v4.new_code_cell(s))
md('''# NB13A · Inspect and resume the existing ATLAS-CONTEXT run

**31 October 2026. Use the existing stopped Kaggle NB13 session and retain its files.**
The saved notebook reached local update **2,467**, while pinned HF recovery was
at **1,294**. Do not reset that session or assume 2,467 is already on HF.

**Copy the two code cells below into the bottom of your existing Kaggle NB13,
then run those cells in order.** Opening this as a separate new Kaggle session
does not carry across the old working files; the recovery refuses an empty run.
Keep T4×2, Internet, the same expanded dataset and authorized HF_TOKEN Secret.
Do not run the old training cell simultaneously.

The first cell only reports local evidence. The second verifies the exact
32-file frozen runtime, requests safe exit of any leftover owned NB13 processes,
checks local/remote history, publishes ahead local progress **before** further
training, then resumes the original contract. It does not select a new model,
alter source hashes, change a scientific CSV or start another measured seed.

The precise reason the previous execution stopped is not established from the
saved output: no traceback was saved. Long verification/snapshot/transport paths
had no continuous status. This recovery adds phase messages every15 seconds,
bounded timeouts for otherwise unbounded HTTP sends, and local thread stacks
every3 minutes. It retains existing rate limits, backoff and owner fencing.
''')
py('''import hashlib,json,re,shutil
from pathlib import Path
WORK=Path('/kaggle/working/tread_two_atlas_context_r3');SOURCE=WORK/'source'
assert (WORK/'run/POINTER.json').exists(), 'Existing NB13 working files are required. Keep the old session; do not reset it.'
def read(p): return json.loads(p.read_text())
def safe(s): return re.sub(r'hf_[A-Za-z0-9_]+','[REDACTED]',s)
cfg=read(WORK/'CONFIG.json');p=read(WORK/'run/POINTER.json');unit=read(WORK/'run'/p['unit'])
assert cfg['protocol']=='36f473b0886d6c2f49c416436d270198af3982230e3045da9969b0bf42169625','Wrong run; preserve it for review.'
assert hashlib.sha256((WORK/'run'/p['unit']).read_bytes()).hexdigest()==p['sha256'],'Latest unit changed; preserve for review.'
print('LOCAL committed progress:',json.dumps({k:unit[k] for k in ('seq','updates','phase')}))
if (WORK/'LAST_REMOTE.json').exists():print('Last recorded HF receipt:',json.dumps(read(WORK/'LAST_REMOTE.json'),indent=2))
print('Free bytes:',shutil.disk_usage(WORK).free)
for name in ['console.log','LIFECYCLE.jsonl','REMOTE_HISTORY.jsonl','HTTP_REQUESTS.jsonl']:
    path=WORK/'run'/name
    if path.exists():
        with path.open('rb') as f:f.seek(max(0,path.stat().st_size-10000));tail=f.read().decode('utf-8',errors='replace')
        print('\\n'+name+' (last lines)\\n'+safe('\\n'.join(tail.splitlines()[-8:])))
print('Files remain untouched by this diagnostic cell. Save its output.')
''')
md('''## Verify, publish retained progress and resume

This cell runs only against the retained original directory and audited
experiment. Stop/error handling forwards SIGINT and allows atomic state plus
immediate HF flush attempts. Server backoff can delay uploads. Forced kernel
death cannot push unsent state. Keep this session alive during publication.

Watch `[NB13 status]`: `operation` identifies local verification, HF calls,
snapshot export or training; `local_cursor` reports the saved update and stage.
Diagnostic files live under `NB13_RECOVERY/`; stack captures show where waiting
occurs without recording local variables or secrets. Save this executed recovery
and original NB13, especially if another stop occurs. The new transport behavior
is locally fixture-tested; actual Kaggle recovery is not claimed yet.
''')
py('''import base64,os,signal,subprocess,sys,time
from filelock import FileLock
from kaggle_secrets import UserSecretsClient
assert (WORK/'run/POINTER.json').exists(),'Do not start a new empty session.'
RECOVERY=WORK/'NB13_RECOVERY';RECOVERY.mkdir(exist_ok=True)
LAUNCHER=RECOVERY/'nb13_recovery_launcher.py';body=base64.b64decode('__PAYLOAD__')
assert hashlib.sha256(body).hexdigest()=='__SHA__'
if LAUNCHER.exists():assert LAUNCHER.read_bytes()==body,'Preserve the earlier recovery source and inspect it before changing versions.'
else:LAUNCHER.write_bytes(body)
with FileLock(str(WORK/'NOTEBOOK.lock'),timeout=1):
    environment=dict(os.environ);environment.pop('HF_TOKEN',None)
    environment['HF_TOKEN']=UserSecretsClient().get_secret('HF_TOKEN')
    try:child=subprocess.Popen([sys.executable,'-B','-u',str(LAUNCHER),'--source',str(SOURCE),'--work',str(WORK)],env=environment,start_new_session=True)
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
                    if i%6==0:print('Waiting for safe local state / HF flush attempt. Keep this session and files.',flush=True)
        raise
    assert result==0,'Preserve NB13_RECOVERY/STATUS.json, EVENTS.jsonl, THREAD_STACKS.txt and run/console.log. No new experiment was started.'
print('Original NB13 recovery/resume completed. Save both executed notebooks for a pinned HF audit.')
if (WORK/'LAST_REMOTE.json').exists():print(json.dumps(read(WORK/'LAST_REMOTE.json'),indent=2))
'''.replace('__PAYLOAD__',payload).replace('__SHA__',sha))
nb=nbformat.v4.new_notebook(cells=cells,metadata=dict(kernelspec=dict(name='python3',display_name='Python 3',language='python'),kaggle=dict(isInternetEnabled=True,isGpuEnabled=True)))
nbformat.validate(nb)
path=ROOT/'notebooks/phase3_NB13A_TREAD_TWO_CONTEXT_Existing_Session_Recovery.ipynb'
if path.exists():assert not any(c.get('execution_count') or c.get('outputs') for c in nbformat.read(path,as_version=4).cells if c.cell_type=='code'),'Preserve supplied execution'
nbformat.write(nb,path)
out=ROOT/'tread_two/context_validation/nb13_status/recovery_validation';out.mkdir(parents=True,exist_ok=True)
receipt=dict(status='packaged_not_executed_on_Kaggle',notebook=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),launcher_sha256=sha,code_cells=2,source_runtime_changed=False,existing_session_required=True)
(out/'NOTEBOOK.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8');print(json.dumps(receipt,indent=2))
