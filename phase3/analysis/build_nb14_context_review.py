"""Self-contained CPU saved-evidence review; preserve completed NB13 sources."""
import ast,base64,hashlib,json,sys
from pathlib import Path
import nbformat
assert Path(sys.prefix).name=='cv_conda'
ROOT=Path(__file__).resolve().parents[1];source=ROOT/'tread_two/context_pilot';review=ROOT/'tread_two/context_review'
done=Path(json.loads((ROOT/'tread_two/nb13_completion/LATEST.json').read_text())['path'])
assert json.loads((done/'VERIFICATION.json').read_text())['status']=='passed'
assert json.loads((done/'LEDGER_VERIFICATION.json').read_text())['status']=='passed'
assert json.loads((ROOT/'tread_two/context_analysis_report/ANALYSIS.json').read_text())['status']=='passed'
catalog=json.loads((source/'FILES.json').read_text());files={'frozen_source/'+n:(source/n).read_bytes() for n in list(catalog)+['FILES.json']}
for n,m in catalog.items():assert hashlib.sha256(files['frozen_source/'+n]).hexdigest()==m['sha256']
files.update({'review/'+p.relative_to(review).as_posix():p.read_bytes() for p in review.rglob('*') if p.is_file() and p.suffix in ('.py','.csv','.json')})
files['FIT_MANIFEST.json']=(done/'REMOTE_MANIFEST.json').read_bytes();files['LOCAL_FULL_AUDIT.json']=(done/'VERIFICATION.json').read_bytes()
encoded=base64.b64encode(json.dumps({n:base64.b64encode(b).decode() for n,b in files.items()},separators=(',',':')).encode()).decode()
cells=[]
def md(s):cells.append(nbformat.v4.new_markdown_cell(s))
def py(s):ast.parse(s);cells.append(nbformat.v4.new_code_cell(s))
md('''# Phase 3 · NB14 · ATLAS-CONTEXT completed evidence review

**CPU · zero fitting / zero new model forwards · direct CSV publication.**

NB13/NB13A completed one 60-epoch / 2,940-update candidate under the original
protocol. Validation selected epoch7; selected/final historical endpoint BA
remains50% (62 low→mid,41 high→high). This notebook audits the saved predictions,
paired objective, frozen LC prior, learned correction, full endpoint attention,
source exclusions, layout probes and failures. It does not change selection or
install a model in a workstation.

## Settings

1. **None / CPU**, Internet ON. **No tyre dataset or GPU attachment needed.**
2. Enable Kaggle Secret **HF_TOKEN**, write access to
   `Shanmuk4622/tyre-wear-study`, for the publication cell.
3. Keep **4 GiB free** working output and about **8 GB RAM** available. The
   review consumes a hash-verified evidence subset, not the full5.65GB fit.
   Original native model weights are not loaded for this review.
4. **Run All**. Cached verified inputs are reused after restart. Preserve the
   executed notebook and publication receipt for independent verification.

The full local audit separately checked all published objects/raw schemas/counts,
the CSV/native ledger and all60epoch states. Its receipt is embedded. This CPU
review independently checks its consumed subset and generates additional CSV,
figures and interactive evidence; it does not recount every large raw family.
More rows/probes are not additional independent tyres.
''')
py('''import base64,hashlib,importlib.metadata,json,os,signal,subprocess,sys,time,shutil
from pathlib import Path
from filelock import FileLock
assert Path('/kaggle/working').exists(),'Run NB14 on Kaggle CPU.'
WORK=Path('/kaggle/working/tread_two_nb14_context_review');WORK.mkdir(parents=True,exist_ok=True)
assert shutil.disk_usage(WORK).free>=4*1024**3,'Keep4GiB free for evidence cache/report/publication staging.'
os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN']='1'
try:version=importlib.metadata.version('huggingface_hub')
except importlib.metadata.PackageNotFoundError:version=None
if version!='0.36.2':subprocess.check_call([sys.executable,'-m','pip','install','--disable-pip-version-check','huggingface_hub==0.36.2'])
payload=json.loads(base64.b64decode('__PAYLOAD__'))
with FileLock(str(WORK/'NOTEBOOK.lock'),timeout=1):
    for n,v in payload.items():
        target=(WORK/n).resolve();assert target.is_relative_to(WORK.resolve()) and '..' not in Path(n).parts
        body=base64.b64decode(v)
        if target.exists():assert target.read_bytes()==body,'Preserve earlier source/output for compatibility review.'
        else:target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(body)
CACHE=WORK/'cache';OUT=WORK/'report';SOURCE=WORK/'frozen_source'
def run(script,arguments,token=None):
    environment=dict(os.environ);environment.pop('HF_TOKEN',None)
    if token is not None:environment['HF_TOKEN']=token
    with FileLock(str(WORK/'NOTEBOOK.lock'),timeout=1):
        try:child=subprocess.Popen([sys.executable,'-B','-u',str(WORK/'review'/script),*arguments],env=environment,start_new_session=True)
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
                        if i%6==0:print('Waiting for cache/table boundary or current HF publication; keep files.',flush=True)
            raise
        assert result==0,'Preserve cache/report and errors. Rerun unchanged; there is no training to repeat.'
print('Pinned CPU review ready. No dataset download, model fitting or inference is started.')
'''.replace('__PAYLOAD__',encoded))
md('''## 1 · Audit saved science and build the complete review

This checks53,754 observations,23,520 paired objectives including correction
regularization,source-SHA permissions,seeds/factors,all saved corrections and
1,687,552 endpoint attention values. All1,854 endpoint/probe cases retain separate
frozen-reference and context accounting. Peers are pinned descriptive comparisons
on already inspected data; no independent generalization claim or deployment
threshold is fitted.

The explorer filters endpoint,probe,target,error/flip,explained class and image.
Switch between combined evidence,frozen LC evidence and context correction.
For original endpoints, select attention layer/head and hover matrix cells for
probabilities. Full probe/training attention was not saved; the viewer does not
invent it. Reverse-band/column probes are token interventions,not new photos.

Input downloads are hash cached; tables are atomic. A stopped finite review
rebuilds derived tables on rerun without any optimizer state. Its completed
publication happens in the next major cell. This is separate from NB13's
completed-update training recovery and30-minute snapshots.
''')
py("run('review_context.py',['--cache',str(CACHE),'--manifest',str(WORK/'FIT_MANIFEST.json'),'--out',str(OUT)])")
md('''## 2 · Publish additive direct records

Destination: `phase3/tread-two/atlas-context-r3/review/<contract>`. The completed
fit is read-only. All new tables are direct CSV; metadata/code/figures/HTML are
direct files. Never ZIP,NPZ or a database. The inherited publisher retains shared
request budgets,server backoff,80-object batches,verified immutable objects,
CAS ownership and identical-publication skipping.

Catchable Stop allows a complete review flush to finish; server delays may apply.
Forced kernel death cannot publish unsent files. Rerun unchanged to verify/skip
an identical completed publication. Only the authorized Kaggle Secret is used.
''')
py("from kaggle_secrets import UserSecretsClient\ntoken=UserSecretsClient().get_secret('HF_TOKEN')\ntry:\n    run('publish_context_review.py',['--out',str(OUT),'--source',str(SOURCE),'--manifest',str(WORK/'FIT_MANIFEST.json'),'--work',str(WORK)],token=token)\nfinally:\n    token=None")
md('''## 3 · Inspect and keep the executed notebook

Read the actual results and comparisons without promoting the classifier.
Low recognition remains unresolved; labels are mileage proxies,not physical
wear/depth/safety truth. Attention and conserved signed evidence are internal
accounting,not causal explanations. No new photos or workstation changes.

Save executed NB14 and the final review receipt for an independent HF audit.
Do not rerun the completed NB13/NB13A or choose another test-scoring checkpoint.
''')
py("from IPython.display import display,FileLink,Image\nprint((WORK/'REVIEW_PUBLICATION.json').read_text())\nprint((OUT/'README.md').read_text())\ndisplay(Image(filename=str(OUT/'figures/context_and_attention.png')))\ndisplay(FileLink(str(OUT/'ATLAS_Failure_Explorer.html')))\ndisplay(FileLink(str(OUT/'CSV_INDEX.md')))")
for i,c in enumerate(cells):c.id=f'atlas-context-nb14-{i:02d}'
nb=nbformat.v4.new_notebook(cells=cells,metadata=dict(kernelspec=dict(name='python3',display_name='Python3',language='python'),kaggle=dict(isInternetEnabled=True,isGpuEnabled=False)))
nbformat.validate(nb);path=ROOT/'notebooks/phase3_NB14_TREAD_TWO_CONTEXT_Completed_Review.ipynb'
if path.exists():assert not any(c.get('execution_count') or c.get('outputs') for c in nbformat.read(path,as_version=4).cells if c.cell_type=='code'),'Preserve executed NB14'
nbformat.write(nb,path);out=ROOT/'tread_two/context_review_validation';out.mkdir(exist_ok=True)
receipt=dict(status='packaged_not_executed_on_Kaggle',notebook=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size,embedded_files=len(files),code_cells=4,
    sources={n:hashlib.sha256(b).hexdigest() for n,b in files.items()},model_forwards=0,training_updates=0)
(out/'NOTEBOOK.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:v for k,v in receipt.items() if k!='sources'},indent=2))
